import numpy as np
import awkward as ak
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import tqdm
import time
import os
from collections import defaultdict, Counter

import random
from contextlib import nullcontext
import sklearn.metrics as m
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from utils.logger import _logger
from utils.nn.tools import (
    _concat,
    AllGather,
)
from utils.import_tools import import_module

ParT = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2024Plus.py'), 'ParT')

import torch.distributed as dist

class ParticleTransformer_dual_cls(ParT.ParticleTransformer):
    """
    在保持 ParticleTransformer2024Plus 不变的前提下，扩展一个可选功能：
    - 在经过 encoder 的自注意力堆叠（例如 8 个 blocks）后，支持两条独立的 class attention 分支，
      分别用于分类与对比学习（CLIP），而非共享一套 class attention blocks。

    兼容性：
    - 默认行为与原版完全一致（dual_cls_blocks=False）。
    - 当 dual_cls_blocks=True 且 num_cls_tokens>=2 时：
      使用第 1 个 cls token 经过原有 self.cls_blocks（用于分类），
      第 2 个 cls token 经过一套独立的 self.cls_blocks_aux（用于对比学习），
      最终返回形状为 (batch, 2, embed_dim) 的 class token 表征，与现有 wrapper 的接口保持一致。
    """

    def __init__(self, **kwargs) -> None:
        # 取出自定义开关，避免传递给父类
        dual_cls_blocks = kwargs.pop('dual_cls_blocks', False)
        super().__init__(**kwargs)
        self.dual_cls_blocks = dual_cls_blocks

        # 仅在开启双分支且存在 cls_blocks 时，克隆一套独立的 class blocks
        if self.dual_cls_blocks and (self.cls_blocks is not None):
            # 深拷贝，确保两条分支参数独立
            self.cls_blocks_aux = copy.deepcopy(self.cls_blocks)
        else:
            self.cls_blocks_aux = None

        # 若开启双分支，则为两条分支分别注册独立的 cls token 参数（具备不同的名字）
        if self.dual_cls_blocks:
            embed_dim = self.cls_token.shape[-1]
            main_init = self.cls_token[:, 0:1, :].clone().detach()
            aux_init = self.cls_token[:, 0:1, :].clone().detach()
            self.cls_token = nn.Parameter(main_init)
            self.cls_token_aux = nn.Parameter(aux_init)

    def _forward_aggregator(self, x, padding_mask):
        # 若未开启双分支，沿用原版逻辑
        if not self.dual_cls_blocks or (self.cls_blocks is None):
            return super()._forward_aggregator(x, padding_mask)

        # 双分支：分别使用独立命名的两个 cls token（主/辅）
        with torch.autocast('cuda', enabled=self.use_amp):
            bsz = x.size(0)
            cls_token = self.cls_token.expand(bsz, -1, -1)
            cls_token_aux = self.cls_token_aux.expand(bsz, -1, -1)

            # 主分支（用于分类）：使用原有 cls_blocks
            for block in self.cls_blocks:
                cls_token = block(x, x_cls=cls_token, padding_mask=padding_mask)

            # 辅分支（用于对比学习）：使用独立的 cls_blocks_aux
            if self.cls_blocks_aux is None:
                self.cls_blocks_aux = copy.deepcopy(self.cls_blocks)
            for block in self.cls_blocks_aux:
                cls_token_aux = block(x, x_cls=cls_token_aux, padding_mask=padding_mask)

            # 分别归一化并各自返回 (batch, embed_dim)
            x_main = self.norm(cls_token.squeeze(1))
            x_aux = self.norm(cls_token_aux.squeeze(1))
        return x_main, x_aux

'''
This code is adapted from Sophon's official repository: https://github.com/jet-universe/sophon/blob/main/networks/example_ParticleTransformer_sophon.py
Additional features:
 - a univeral wrapper for Sophon models for CLIP-only, CLIP+classification and post-CLIP fine-tuning
 - new eval/test utilities: custom BkgRej maker; custom label_cls_nodes and label_stored for saving outpout nodes
'''

class FFN(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, fc_params=[], bias_last=True):
        super().__init__()
        fcs = []
        in_dim = input_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, output_dim, bias=bias_last))
        self.fc = nn.Sequential(*fcs)
    
    def forward(self, x):
        return self.fc(x)


class ParticleTransformerSophonCLIPWrapper(torch.nn.Module):
    '''
        A univeral wrapper for Sophon models for CLIP-only, CLIP+classification and post-CLIP fine-tuning
    '''

    def __init__(self, **kwargs) -> None:
        super().__init__()
        gen_model_kw = kwargs.pop('gen_model_kw')
        clip_kw = kwargs.pop('clip_kw')
        self.clip_mode = clip_kw['mode']
        self.clip_share_token = clip_kw['share_token']

        assert self.clip_mode in ['clip-only', 'clip-with-cls', 'clip-with-gencls', 'clip-finetune', 'cls-only', 'gencls-only'], 'Invalid mode %s' % self.clip_mode
        if self.clip_mode in ['clip-only', 'clip-with-cls', 'clip-with-gencls']:

            # remove FC in the model and use outer FC
            fc_params = kwargs.get('fc_params', None)
            gen_fc_params = gen_model_kw.get('fc_params', None)
            kwargs['fc_params'] = None
            gen_model_kw['fc_params'] = None

            # 当 dual_cls_blocks 启用时，无视 share_token；不需要通过 num_cls_tokens 提供第二个 token
            # 若未启用 dual，则仍按旧逻辑：仅在不共享 token 时为主分支提供第二个 cls token
            if self.clip_mode == 'clip-with-cls' and (not clip_kw.get('dual_cls_blocks', False)) and (not self.clip_share_token):
                kwargs['num_cls_tokens'] = 2
            if self.clip_mode == 'clip-with-gencls' and not self.clip_share_token:
                gen_model_kw['num_cls_tokens'] = 2

            # initialize model
            # 仅在 clip-with-cls 模式下支持 dual_cls_blocks（来自 clip_kw 配置）；启用后忽略 share_token
            if self.clip_mode == 'clip-with-cls' and clip_kw.get('dual_cls_blocks', False):
                kwargs['dual_cls_blocks'] = True
            self.mod = ParticleTransformer_dual_cls(**kwargs)
            self.gen = ParticleTransformer_dual_cls(**gen_model_kw)

            # 标记 dual 是否启用，便于 forward 中无视 share_token
            self.dual_cls_blocks = clip_kw.get('dual_cls_blocks', False) if self.clip_mode == 'clip-with-cls' else False

            # define outer FCs
            self.mod_proj = FFN(input_dim=kwargs['embed_dims'][-1], output_dim=clip_kw['proj_dim'], fc_params=clip_kw['main_cont_fc_parmas'], bias_last=False)
            self.gen_proj = FFN(input_dim=gen_model_kw['embed_dims'][-1], output_dim=clip_kw['proj_dim'], fc_params=clip_kw['gen_cont_fc_parmas'], bias_last=False)
            if self.clip_mode == 'clip-with-cls':
                # also define FC for classification
                assert fc_params is not None, 'fc_params must be provided for clip-with-cls mode'
                self.mod_fc = FFN(input_dim=kwargs['embed_dims'][-1], output_dim=kwargs['num_classes'], fc_params=fc_params, bias_last=True)
            if self.clip_mode == 'clip-with-gencls':
                assert fc_params is not None, 'fc_params must be provided in "gen_model_kw" for clip-with-gencls mode'
                self.gen_fc = FFN(input_dim=gen_model_kw['embed_dims'][-1], output_dim=kwargs['num_classes'], fc_params=gen_fc_params, bias_last=True)

        elif self.clip_mode in ['cls-only', 'clip-finetune']:

            self.mod = ParticleTransformer_dual_cls(**kwargs)

            # initialize model for clip-finetune mode
            assert clip_kw['init_path'] is not None, 'init_path must be provided for clip-finetune mode'
            init_model_state = {}
            for k, v in torch.load(clip_kw['init_path'], map_location='cpu').items():
                if k.startswith('mod.'):
                    if k == 'mod.cls_token' and v.shape[1] == 2:
                        # special treatment for cls_token
                        # if in shape (1, 2, dim), i.e. two class tokens, one for classification, and one for CLIP contrastive loss, only take the first one
                        init_model_state[k.replace('mod.', '', 1)] = v[:, 0:1]
                    else:
                        init_model_state[k.replace('mod.', '', 1)] = v
                if k.startswith('mod_fc.') and clip_kw['init_opts'].get('load_fc', False):
                    # this only exists if the CLIP model is trained with clip-with-cls mode
                    init_model_state[k.replace('mod_fc.', '', 1)] = v

            missing, unexpected = self.mod.load_state_dict(init_model_state, strict=False)
            _logger.info('Loaded model state from %s: missing keys %s, unexpected keys %s' % (clip_kw['init_path'], missing, unexpected))

        elif self.clip_mode in ['gencls-only']:
            self.gen = ParticleTransformer_dual_cls(**gen_model_kw)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', 'gen.cls_token'}

    def forward(self, *args):
        '''
            args: a list of inputs, provided by the YAML card. should be:
              - points, features, lorentz_vectors, mask, gen_points, gen_features, gen_lorentz_vectors, gen_mask (for clip-only, clip-with-cls, clip-with-gencls)
              - points, features, lorentz_vectors, mask (for cls-only and clip-finetune)
              - gen_points, gen_features, gen_lorentz_vectors, gen_mask (for gencls-only)
            Output: logits, x_mod, x_gen
              - logits: (batch, num_classes), the output logits of 
                  1. main ParT if the mode has cls (clip-with-cls, cls-only, clip-finetune)
                  2. gen ParT if the mode has gencls (gencls-only)
                  3. None if the mode is clip-only
              - x_mod: (batch, proj_dim), the latent features of main ParT to compute contrastive loss
              - x_gen: (batch, proj_dim), the latent features of gen ParT to compute contrastive loss
        '''
        # return self.mod(features, v=lorentz_vectors, mask=mask) # not using the default foward implementation. Should add emport_embed flag

        if self.clip_mode in ['clip-only', 'clip-with-cls', 'clip-with-gencls']:
            points, features, lorentz_vectors, mask, gen_points, gen_features, gen_lorentz_vectors, gen_mask = args

            x_mod_out = self.mod(features, v=lorentz_vectors, mask=mask)
            x_gen_out = self.gen(gen_features, v=gen_lorentz_vectors, mask=gen_mask)

            # 工具函数：从可能的 (tensor or tuple or (bsz,2,dim)) 中取分类分支与对比分支
            def split_outputs(x):
                if isinstance(x, tuple) and len(x) == 2:
                    return x[0], x[1]
                if torch.is_tensor(x):
                    if x.ndim == 3 and x.size(1) == 2:
                        return x[:, 0], x[:, 1]
                    elif x.ndim == 2:
                        return x, x
                raise AssertionError('Unexpected output shape/type: %s' % (str(type(x)) + ((' ' + str(tuple(x.shape))) if torch.is_tensor(x) else '')))

            # FC (for classifications) and projections (for contrastive loss)
            if self.clip_mode == 'clip-with-cls':
                # 若启用 dual，则无视 share_token：主分支强制分为分类与对比两路
                if getattr(self, 'dual_cls_blocks', False):
                    x_mod_cls, x_mod_clip = split_outputs(x_mod_out)
                    logits = self.mod_fc(x_mod_cls)
                    x_mod = self.mod_proj(x_mod_clip)

                    if isinstance(x_gen_out, tuple) and len(x_gen_out) == 2:
                        _, x_gen_clip = x_gen_out
                    elif torch.is_tensor(x_gen_out) and x_gen_out.ndim == 3 and x_gen_out.size(1) == 2:
                        x_gen_clip = x_gen_out[:, 1]
                    else:
                        x_gen_clip = x_gen_out
                    x_gen = self.gen_proj(x_gen_clip)
                else:
                    if not self.clip_share_token:
                        # 主模型：分类与对比两路分支
                        x_mod_cls, x_mod_clip = split_outputs(x_mod_out)
                        logits = self.mod_fc(x_mod_cls)
                        x_mod = self.mod_proj(x_mod_clip)

                        # gen模型：只用于对比分支，若返回双分支，取对比路
                        if isinstance(x_gen_out, tuple) and len(x_gen_out) == 2:
                            _, x_gen_clip = x_gen_out
                        elif torch.is_tensor(x_gen_out) and x_gen_out.ndim == 3 and x_gen_out.size(1) == 2:
                            x_gen_clip = x_gen_out[:, 1]
                        else:
                            x_gen_clip = x_gen_out
                        x_gen = self.gen_proj(x_gen_clip)
                    else:
                        # 共享一套表征
                        assert torch.is_tensor(x_mod_out) and x_mod_out.ndim == 2, 'Invalid shape %s' % str(getattr(x_mod_out, 'shape', None))
                        logits = self.mod_fc(x_mod_out)
                        x_mod = self.mod_proj(x_mod_out)
                        x_gen = self.gen_proj(x_gen_out if torch.is_tensor(x_gen_out) else x_gen_out[1] if isinstance(x_gen_out, tuple) else x_gen_out)

            elif self.clip_mode == 'clip-with-gencls':
                if not self.clip_share_token:
                    # 生成模型：分类与对比两路分支
                    x_gen_cls, x_gen_clip = split_outputs(x_gen_out)
                    logits = self.gen_fc(x_gen_cls)
                    x_gen = self.gen_proj(x_gen_clip)

                    # 主模型：只用于对比分支
                    if isinstance(x_mod_out, tuple) and len(x_mod_out) == 2:
                        _, x_mod_clip = x_mod_out
                    elif torch.is_tensor(x_mod_out) and x_mod_out.ndim == 3 and x_mod_out.size(1) == 2:
                        x_mod_clip = x_mod_out[:, 1]
                    else:
                        x_mod_clip = x_mod_out
                    x_mod = self.mod_proj(x_mod_clip)
                else:
                    # 共享一套表征
                    assert torch.is_tensor(x_gen_out) and x_gen_out.ndim == 2, 'Invalid shape %s' % str(getattr(x_gen_out, 'shape', None))
                    logits = self.gen_fc(x_gen_out)
                    x_mod = self.mod_proj(x_mod_out if torch.is_tensor(x_mod_out) else x_mod_out[1] if isinstance(x_mod_out, tuple) else x_mod_out)
                    x_gen = self.gen_proj(x_gen_out)

            elif self.clip_mode == 'clip-only':
                logits = None
                # 仅对比学习：若为双分支，取对比路
                if isinstance(x_mod_out, tuple) and len(x_mod_out) == 2:
                    x_mod = self.mod_proj(x_mod_out[1])
                elif torch.is_tensor(x_mod_out) and x_mod_out.ndim == 3 and x_mod_out.size(1) == 2:
                    x_mod = self.mod_proj(x_mod_out[:, 1])
                else:
                    x_mod = self.mod_proj(x_mod_out)

                if isinstance(x_gen_out, tuple) and len(x_gen_out) == 2:
                    x_gen = self.gen_proj(x_gen_out[1])
                elif torch.is_tensor(x_gen_out) and x_gen_out.ndim == 3 and x_gen_out.size(1) == 2:
                    x_gen = self.gen_proj(x_gen_out[:, 1])
                else:
                    x_gen = self.gen_proj(x_gen_out)

        elif self.clip_mode in ['cls-only', 'clip-finetune']:
            points, features, lorentz_vectors, mask = args
            logits = self.mod(features, v=lorentz_vectors, mask=mask)
            x_mod = None
            x_gen = None

        elif self.clip_mode in ['gencls-only']:
            gen_points, gen_features, gen_lorentz_vectors, gen_mask = args
            logits = self.gen(gen_features, v=gen_lorentz_vectors, mask=gen_mask)
            x_mod = None
            x_gen = None

        return logits, x_mod, x_gen


class CLIPLoss(torch.nn.Module):
    '''
        Computes the CLIP loss and classification loss
    '''

    def __init__(self, clip_mode=None, beta=1., alpha=1., label_smoothing=0.05):
        super().__init__()
        self.clip_mode = clip_mode
        self.beta = beta
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        if clip_mode in ['clip-only', 'clip-with-cls', 'clip-with-gencls']:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, logits, x_mod, x_gen, labels):
        '''
            logits: (batch, out_dim), the output logits of main ParT for standard classification.
            x_mod: (batch, proj_dim), the latent features of main ParT to compute contrastive loss
            x_gen: (batch, proj_dim), the latent features of GEN-level ParT to compute contrastive loss
            labels: (batch,), labels for classification
        '''
        # compute classification loss
        if logits is not None:
            loss_cls = F.cross_entropy(logits.float(), labels, label_smoothing=self.label_smoothing)
        else:
            loss_cls = torch.tensor(0., device=labels.device)

        # CLIP constrastive learning
        # normalize the features
        if x_mod is not None:
            denom_mod = x_mod.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            denom_gen = x_gen.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x_mod = x_mod / denom_mod
            x_gen = x_gen / denom_gen
            x_mod = torch.nan_to_num(x_mod)
            x_gen = torch.nan_to_num(x_gen)

            # compute cosine similarity
            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            logits_cont_2d = logit_scale * (x_mod.float() @ x_gen.float().t()) # (batch, batch)
            logits_cont_2d = torch.nan_to_num(logits_cont_2d, neginf=-100.0, posinf=100.0).clamp(min=-100.0, max=100.0)
            logits_cont_2d_t = logits_cont_2d.t()
            indices = torch.arange(x_mod.size(0)).to(x_mod.device) # (batch,)

            loss_cont = F.cross_entropy(logits_cont_2d, indices) + F.cross_entropy(logits_cont_2d_t, indices)
        else:
            loss_cont = torch.tensor(0., device=labels.device)

        loss = self.beta * loss_cls + self.alpha * loss_cont
        return loss, loss_cls, loss_cont


class PCGrad:
    """
    一个轻量的 PCGrad 包装器，包装任意 PyTorch 优化器，实现梯度冲突手术（Projecting Conflicting Gradients）。
    用法：
      - opt = PCGrad(torch.optim.Adam(...))
      - opt.pc_backward([loss_task1, loss_task2], scaler=grad_scaler)
      - grad_scaler.step(opt) / opt.step()
    说明：
      - 当提供 scaler 时，会在 pc_backward 内部对当前梯度做 unscale_，随后再执行手术与聚合；
      - pc_backward 会负责零梯度、逐任务回传与聚合后的赋值，外部仅需调用 step 即可。
    """
    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optim = optimizer
        self.is_pcgrad = True

    @property
    def param_groups(self):
        return self._optim.param_groups

    @property
    def defaults(self):
        return self._optim.defaults

    def zero_grad(self, set_to_none: bool = False):
        try:
            self._optim.zero_grad(set_to_none=set_to_none)
        except TypeError:
            # 兼容不支持 set_to_none 的优化器（如 Lookahead）
            for group in self._optim.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad.detach_()
                            p.grad.zero_()

    def step(self, *args, **kwargs):
        return self._optim.step(*args, **kwargs)

    def _get_trainable_params(self):
        trainable_params = []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    trainable_params.append(p)
        return trainable_params

    @torch.no_grad()
    def _project_conflicts_inplace(self, grads_list):
        """
        对每个任务的梯度执行两两冲突投影：
          若 <g_i, g_j> < 0，则 g_i <- g_i - <g_i, g_j> / ||g_j||^2 * g_j
        grads_list: List[List[Tensor or None]]，外层为任务，内层按参数顺序存储梯度。
        """
        num_tasks = len(grads_list)
        order_indices = list(range(num_tasks))
        for i in range(num_tasks):
            random.shuffle(order_indices)
            for j in order_indices:
                if j == i:
                    continue
                gi_list = grads_list[i]
                gj_list = grads_list[j]
                for k in range(len(gi_list)):
                    gi = gi_list[k]
                    gj = gj_list[k]
                    if gi is None or gj is None:
                        continue
                    gij = torch.dot(gi.flatten(), gj.flatten())
                    if gij < 0:
                        denom = gj.norm().pow(2).clamp_min(1e-12)
                        gi.add_(gj, alpha=-(gij / denom))

    def pc_backward(self, objectives, scaler: torch.cuda.amp.GradScaler = None, ddp_model=None):
        """
        对多个目标（任务）依次回传、收集梯度，进行 PCGrad 手术并聚合到参数的 .grad。
        - objectives: 可迭代的标量损失张量列表（已按需要乘好权重，比如 beta/alpha）
        - scaler: 可选 GradScaler；若提供，将使用其 scale/backward/unscale_ 逻辑
        - ddp_model: 可选 DDP/DP 模型句柄；本实现不再多次 backward，而是用 autograd.grad 抽取梯度，随后手动 all-reduce
        """
        # 仅保留需要梯度的目标
        valid_objectives = [obj for obj in objectives if hasattr(obj, "requires_grad") and obj.requires_grad]
        if len(valid_objectives) == 0:
            return

        params = self._get_trainable_params()
        task_grads = []  # List[List[Tensor or None]]
        use_ddp = (ddp_model is not None)

        # 使用 autograd.grad 逐任务提取梯度，避免 DDP reducer 在多次 backward 中重复就绪
        for idx, obj in enumerate(valid_objectives):
            scaled_obj = scaler.scale(obj) if scaler is not None else obj
            grads = torch.autograd.grad(
                outputs=scaled_obj,
                inputs=params,
                retain_graph=(idx < len(valid_objectives) - 1),
                allow_unused=True
            )
            grads_this_task = []
            for g in grads:
                grads_this_task.append(None if g is None else g.detach().clone())
            task_grads.append(grads_this_task)

        # 执行冲突投影
        self._project_conflicts_inplace(task_grads)

        # 聚合并写回到 param.grad
        self.zero_grad(set_to_none=True)
        for p_idx, p in enumerate(params):
            accumulated = None
            for t in range(len(task_grads)):
                g = task_grads[t][p_idx]
                if g is None:
                    continue
                accumulated = g if accumulated is None else (accumulated.add_(g))
            if accumulated is not None:
                p.grad = accumulated

        # AMP: 若使用 GradScaler，此时 p.grad 仍是 scaled grads，需要先 unscale 再进行分布式聚合
        if scaler is not None:
            scaler.unscale_(self)

        # 分布式：显式 all-reduce 聚合后的梯度，保证多卡一致
        if use_ddp and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                for p in params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(world_size)
        # 将非常数梯度中的 NaN/Inf 清零，避免污染权重
        for p in params:
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)


def get_model(data_config, **kwargs):

    # default configurations
    cfg = dict(
        input_dim=len(data_config.input_dicts.get('pf_features', [])),
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        # gen model kwargs
        gen_model_kw=dict(),
        # clip kwargs
        clip_kw=dict()
    )
    cfg['gen_model_kw'].update(
        input_dim=len(data_config.input_dicts.get('gen_features', [])),
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[64, 64, 64],
        pair_embed_dims=[32, 32, 32],
        num_heads=4,
        num_layers=4,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=None,
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg['clip_kw'].update(
        mode='clip-only',
        proj_dim=128,
        share_token=False,
        dual_cls_blocks=False,  # 开关放到 clip_kw 中，仅在 clip-with-cls 模式下生效
        exclude=[],
        main_cont_fc_parmas=[],
        gen_cont_fc_parmas=[],
        init_path=None,
        init_opts=dict(),
        beta=1., # loss weight for cls loss
        alpha=1., # loss weight for contrastive loss
    )

    # update configurations
    for k, v in kwargs.pop('gen_model_kw', dict()).items():
        assert k in cfg['gen_model_kw'], 'Invalid key %s in "gen_model_kw"' % k
        cfg['gen_model_kw'][k] = v
    for k, v in kwargs.pop('clip_kw', dict()).items():
        assert k in cfg['clip_kw'], 'Invalid key %s in "clip_kw"' % k
        cfg['clip_kw'][k] = v
    for k, v in kwargs.items():
        assert k in cfg, 'Invalid key %s' % k
        cfg[k] = v
    
    # 在 clip-with-cls 模式下，dual_cls_blocks 完全由 clip_kw 控制；若未提供则默认 False
    if cfg['clip_kw']['mode'] == 'clip-with-cls':
        cfg['clip_kw']['dual_cls_blocks'] = cfg['clip_kw'].get('dual_cls_blocks', False)

    _logger.info('Model config: %s' % str(cfg))

    # remove the eval/test-time related options from cfg
    eval_kw = cfg.pop('eval_kw', dict())
    cfg.pop('label_cls_nodes', None)
    cfg.pop('label_stored', None)

    model = ParticleTransformerSophonCLIPWrapper(**cfg)
    
    # set eval_kw for ROC curve configuration
    model.eval_kw = eval_kw

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    clip_kw = kwargs.get('clip_kw')
    return CLIPLoss(clip_mode=clip_kw['mode'], beta=clip_kw['beta'], alpha=clip_kw['alpha'])


def get_train_fn(data_config, **kwargs):
    return train_classification_sophon_clip


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_classification_sophon_clip


def get_save_fn(data_config, **kwargs):
    return save_classification_sophon_clip


# Customized training and evaluation functions for Sophon
# functions are adapted from https://github.com/hqucms/weaver-core/blob/main/weaver/utils/nn/tools.py

def train_classification_sophon_clip(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, extra_args=None):
    model.train()

    # 用 PCGrad 包装优化器（只包装一次）
    if not hasattr(opt, 'is_pcgrad'):
        opt = PCGrad(opt)

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_cont = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev) # label is obtained from inputs
            entry_count += label.shape[0]
            opt.zero_grad()
            # 判断当前模式是否包含分类或对比任务
            clip_mode = getattr(model.module, 'clip_mode', None) if hasattr(model, 'module') else getattr(model, 'clip_mode', None)
            has_cls_task = clip_mode in ['clip-with-cls', 'cls-only', 'clip-finetune']
            has_cont_task = clip_mode in ['clip-only', 'clip-with-cls', 'clip-with-gencls']

            # 当使用 PCGrad 时，分别做独立前向以避免 DDP 同图多次反传
            if hasattr(opt, 'is_pcgrad') and opt.is_pcgrad:
                objectives = []
                loss_cls = torch.tensor(0., device=label.device)
                loss_cont = torch.tensor(0., device=label.device)

                if has_cls_task:
                    with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                        logits, _, _ = model(*inputs)
                        loss_cls = F.cross_entropy(logits, label)
                    objectives.append(loss_func.beta * loss_cls)

                if has_cont_task:
                    with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                        _, x_mod, x_gen = model(*inputs)
                        # 仅计算对比损失分量
                        _, _, loss_cont = loss_func(None, x_mod, x_gen, label)
                    objectives.append(loss_func.alpha * loss_cont)

                # 过滤非有限的目标，避免 NaN 传播
                objectives = [obj for obj in objectives if torch.isfinite(obj)]
                if len(objectives) > 0:
                    if grad_scaler is None:
                        opt.pc_backward(objectives, scaler=None, ddp_model=model)
                        # 梯度裁剪（提高稳定性）
                        try:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        except Exception:
                            pass
                        opt.step()
                        # 约束温度参数范围，避免对比 logits 失控
                        try:
                            if hasattr(loss_func, 'logit_scale'):
                                with torch.no_grad():
                                    loss_func.logit_scale.data.clamp_(min=np.log(1/100.0), max=np.log(100.0))
                        except Exception:
                            pass
                    else:
                        opt.pc_backward(objectives, scaler=grad_scaler, ddp_model=model)
                        # AMP 下：已在 pc_backward 内执行 unscale_，可进行裁剪
                        try:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        except Exception:
                            pass
                        grad_scaler.step(opt)
                        grad_scaler.update()
                        # 约束温度参数范围
                        try:
                            if hasattr(loss_func, 'logit_scale'):
                                with torch.no_grad():
                                    loss_func.logit_scale.data.clamp_(min=np.log(1/100.0), max=np.log(100.0))
                        except Exception:
                            pass

                # 汇总本次 loss（用于日志，不用于反向）
                loss = (loss_func.beta * loss_cls + loss_func.alpha * loss_cont).detach()
            else:
                # 非 PCGrad 路径：一次前向一次反向
                with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                    logits, x_mod, x_gen = model(*inputs)
                    loss, loss_cls, loss_cont = loss_func(logits, x_mod, x_gen, label)
                    # 检查非有限 loss，避免更新
                    if not torch.isfinite(loss):
                        _logger.info('Skip step due to non-finite loss (loss=%.3e, cls=%.3e, cont=%.3e)' % (loss, loss_cls, loss_cont))
                        opt.zero_grad()
                    else:
                        if grad_scaler is None:
                            loss.backward()
                            # 裁剪
                            try:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            except Exception:
                                pass
                            opt.step()
                        else:
                            grad_scaler.scale(loss).backward()
                            # 先 unscale 再裁剪
                            try:
                                grad_scaler.unscale_(opt)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            except Exception:
                                pass
                            grad_scaler.step(opt)
                            grad_scaler.update()
                    # 约束温度参数范围
                    try:
                        if hasattr(loss_func, 'logit_scale'):
                            with torch.no_grad():
                                loss_func.logit_scale.data.clamp_(min=np.log(1/100.0), max=np.log(100.0))
                    except Exception:
                        pass

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()
            loss_cls = loss_cls.item()
            loss_cont = loss_cont.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            total_loss += loss
            total_loss_cls += loss_cls
            total_loss_cont += loss_cont
            tq_dict = {
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'LossCls': '%.5f' % loss_cls,
                'LossCont': '%.5f' % loss_cont,
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
            }

            if logits is not None:
                _, preds = logits.max(1)
                correct = (preds == label).sum().item()
                total_correct += correct
                tq_dict.update({
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count),
                })
            
            tq.set_postfix(tq_dict)

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Max CUDA memory: %.1f MB' % (torch.cuda.max_memory_allocated(dev) / 1024.**2,))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("LossCls/train (epoch)", total_loss_cls / num_batches, epoch),
            ("LossCont/train (epoch)", total_loss_cont / num_batches, epoch),
        ])
        if logits is not None:
            tb_helper.write_scalars([
                ("Acc/train (epoch)", total_correct / count, epoch),
            ])

        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification_sophon_clip(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, extra_args=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_cont = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    eval_kw = model.module.eval_kw \
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.eval_kw
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                logits, x_mod, x_gen = model(*inputs)
                if loss_func is not None:
                    loss, loss_cls, loss_cont = loss_func(logits, x_mod, x_gen, label)
                else: # for test mode
                    loss, loss_cls, loss_cont = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

                # all-reduce the loss
                loss = AllGather.apply(loss.unsqueeze(0)).mean().item()
                loss_cls = AllGather.apply(loss_cls.unsqueeze(0)).mean().item()
                loss_cont = AllGather.apply(loss_cont.unsqueeze(0)).mean().item()

                # all-gather the scores and labels
                if logits is not None:
                    logits = AllGather.apply(logits)
                    scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))
                y = {k: AllGather.apply(v.to(dev)) for k, v in y.items()}
                label = y[data_config.label_names[0]].long().to(dev) # gathered label
                for k, v in y.items():
                    labels[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.numpy(force=True))

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                total_loss_cls += loss_cls * num_examples
                total_loss_cont += loss_cont * num_examples
                tq_dict = {
                    'LossCls': '%.5f' % loss_cls,
                    'LossCont': '%.5f' % loss_cont,
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                }

                if logits is not None:
                    _, preds = logits.max(1)
                    correct = (preds == label).sum().item()
                    total_correct += correct
                    tq_dict.update({
                        'Acc': '%.5f' % (correct / num_examples),
                        'AvgAcc': '%.5f' % (total_correct / count),
                    })

                tq.set_postfix(tq_dict)

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("LossCls/%s (epoch)" % tb_mode, total_loss_cls / count, epoch),
            ("LossCont/%s (epoch)" % tb_mode, total_loss_cont / count, epoch),
        ])
        if logits is not None:
            tb_helper.write_scalars([
                ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ])

    if logits is not None:
        scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper and logits is not None and for_training:
        truth_label = labels['truth_label']
        scores_dict, flag_dict = {}, {}
        
        # Default ROC curve configuration for cls_0, cls_1, cls_2
        roc_kwargs_default = {
            'label_inds_map': {
                'Xbb': [0],
                'Xcc': [1], 
                'QCD': list(range(161, 188)),
            },
            'comp_list': [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        }
        
        # Use provided roc_kw or fall back to default
        roc_kwargs = eval_kw.get('roc_kw', roc_kwargs_default)
        
        for name, inds in roc_kwargs.get('label_inds_map').items():
            flag_dict[name] = np.any([truth_label == i for i in inds], axis=0)
            scores_dict[name] = np.sum(scores[:, inds], axis=1)
            print(name, flag_dict[name].shape, scores_dict[name].shape)
        comp_list = roc_kwargs.get('comp_list') # e.g. [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        bkgrej = {}

        f, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linestyle='--', color='gray', label='Random guess')

        for name_sig, name_bkg in comp_list:
            discr = scores_dict[name_sig] / (scores_dict[name_sig] + scores_dict[name_bkg])
            discr_sig, discr_bkg = discr[flag_dict[name_sig]], discr[flag_dict[name_bkg]]
            fpr, tpr, _ = m.roc_curve(
                np.concatenate([np.ones_like(discr_sig), np.zeros_like(discr_bkg)]),
                np.concatenate([discr_sig, discr_bkg])
            )
            ax.plot(tpr, fpr, label='%s vs %s (AUC=%.4f)' % (name_sig, name_bkg, m.auc(fpr, tpr)))
            bkgrej[(name_sig, name_bkg)] = np.interp(0.3, tpr, 1. / np.maximum(fpr, 1e-10)) # bkgrej at eff_sig=30%
        ax.legend()
        ax.set_xlabel('True positive rate (signal eff.)', ha='right', x=1.0); ax.set_ylabel('False positive rate (BKG eff.)', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_helper.writer.add_figure('ROC/%s/epoch%s' % (tb_mode, str(epoch).zfill(4)), f)

        # write bkgrej values
        for name_sig, name_bkg in comp_list:
            tb_helper.write_scalars([
                ('BkgRej_%s_vs_%s/%s (epoch)' % (name_sig, name_bkg, tb_mode), bkgrej[(name_sig, name_bkg)], epoch),
            ])


    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if logits is not None:
            if len(scores) != entry_count:
                if len(labels_counts):
                    labels_counts = np.concatenate(labels_counts)
                    scores = ak.unflatten(scores, labels_counts)
                    for k, v in labels.items():
                        labels[k] = ak.unflatten(v, labels_counts)
                else:
                    assert (count % entry_count == 0)
                    scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                    for k, v in labels.items():
                        labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers


def save_classification_sophon_clip(args, data_config, scores, labels, observers):
    import ast
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}

    num_classes = network_options['num_classes']

    # default Sophon labels
    label_sophon_default = ["label_X_bb", "label_X_cc", "label_X_ss", "label_X_qq", "label_X_bc", "label_X_cs", "label_X_bq", "label_X_cq", "label_X_sq", "label_X_gg", "label_X_ee", "label_X_mm", "label_X_tauhtaue", "label_X_tauhtaum", "label_X_tauhtauh", "label_X_YY_bbbb", "label_X_YY_bbcc", "label_X_YY_bbss", "label_X_YY_bbqq", "label_X_YY_bbgg", "label_X_YY_bbee", "label_X_YY_bbmm", "label_X_YY_bbtauhtaue", "label_X_YY_bbtauhtaum", "label_X_YY_bbtauhtauh", "label_X_YY_bbb", "label_X_YY_bbc", "label_X_YY_bbs", "label_X_YY_bbq", "label_X_YY_bbg", "label_X_YY_bbe", "label_X_YY_bbm", "label_X_YY_cccc", "label_X_YY_ccss", "label_X_YY_ccqq", "label_X_YY_ccgg", "label_X_YY_ccee", "label_X_YY_ccmm", "label_X_YY_cctauhtaue", "label_X_YY_cctauhtaum", "label_X_YY_cctauhtauh", "label_X_YY_ccb", "label_X_YY_ccc", "label_X_YY_ccs", "label_X_YY_ccq", "label_X_YY_ccg", "label_X_YY_cce", "label_X_YY_ccm", "label_X_YY_ssss", "label_X_YY_ssqq", "label_X_YY_ssgg", "label_X_YY_ssee", "label_X_YY_ssmm", "label_X_YY_sstauhtaue", "label_X_YY_sstauhtaum", "label_X_YY_sstauhtauh", "label_X_YY_ssb", "label_X_YY_ssc", "label_X_YY_sss", "label_X_YY_ssq", "label_X_YY_ssg", "label_X_YY_sse", "label_X_YY_ssm", "label_X_YY_qqqq", "label_X_YY_qqgg", "label_X_YY_qqee", "label_X_YY_qqmm", "label_X_YY_qqtauhtaue", "label_X_YY_qqtauhtaum", "label_X_YY_qqtauhtauh", "label_X_YY_qqb", "label_X_YY_qqc", "label_X_YY_qqs", "label_X_YY_qqq", "label_X_YY_qqg", "label_X_YY_qqe", "label_X_YY_qqm", "label_X_YY_gggg", "label_X_YY_ggee", "label_X_YY_ggmm", "label_X_YY_ggtauhtaue", "label_X_YY_ggtauhtaum", "label_X_YY_ggtauhtauh", "label_X_YY_ggb", "label_X_YY_ggc", "label_X_YY_ggs", "label_X_YY_ggq", "label_X_YY_ggg", "label_X_YY_gge", "label_X_YY_ggm", "label_X_YY_bee", "label_X_YY_cee", "label_X_YY_see", "label_X_YY_qee", "label_X_YY_gee", "label_X_YY_bmm", "label_X_YY_cmm", "label_X_YY_smm", "label_X_YY_qmm", "label_X_YY_gmm", "label_X_YY_btauhtaue", "label_X_YY_ctauhtaue", "label_X_YY_stauhtaue", "label_X_YY_qtauhtaue", "label_X_YY_gtauhtaue", "label_X_YY_btauhtaum", "label_X_YY_ctauhtaum", "label_X_YY_stauhtaum", "label_X_YY_qtauhtaum", "label_X_YY_gtauhtaum", "label_X_YY_btauhtauh", "label_X_YY_ctauhtauh", "label_X_YY_stauhtauh", "label_X_YY_qtauhtauh", "label_X_YY_gtauhtauh", "label_X_YY_qqqb", "label_X_YY_qqqc", "label_X_YY_qqqs", "label_X_YY_bbcq", "label_X_YY_ccbs", "label_X_YY_ccbq", "label_X_YY_ccsq", "label_X_YY_sscq", "label_X_YY_qqbc", "label_X_YY_qqbs", "label_X_YY_qqcs", "label_X_YY_bcsq", "label_X_YY_bcs", "label_X_YY_bcq", "label_X_YY_bsq", "label_X_YY_csq", "label_X_YY_bcev", "label_X_YY_csev", "label_X_YY_bqev", "label_X_YY_cqev", "label_X_YY_sqev", "label_X_YY_qqev", "label_X_YY_bcmv", "label_X_YY_csmv", "label_X_YY_bqmv", "label_X_YY_cqmv", "label_X_YY_sqmv", "label_X_YY_qqmv", "label_X_YY_bctauev", "label_X_YY_cstauev", "label_X_YY_bqtauev", "label_X_YY_cqtauev", "label_X_YY_sqtauev", "label_X_YY_qqtauev", "label_X_YY_bctaumv", "label_X_YY_cstaumv", "label_X_YY_bqtaumv", "label_X_YY_cqtaumv", "label_X_YY_sqtaumv", "label_X_YY_qqtaumv", "label_X_YY_bctauhv", "label_X_YY_cstauhv", "label_X_YY_bqtauhv", "label_X_YY_cqtauhv", "label_X_YY_sqtauhv", "label_X_YY_qqtauhv", "label_QCD_bbccss", "label_QCD_bbccs", "label_QCD_bbcc", "label_QCD_bbcss", "label_QCD_bbcs", "label_QCD_bbc", "label_QCD_bbss", "label_QCD_bbs", "label_QCD_bb", "label_QCD_bccss", "label_QCD_bccs", "label_QCD_bcc", "label_QCD_bcss", "label_QCD_bcs", "label_QCD_bc", "label_QCD_bss", "label_QCD_bs", "label_QCD_b", "label_QCD_ccss", "label_QCD_ccs", "label_QCD_cc", "label_QCD_css", "label_QCD_cs", "label_QCD_c", "label_QCD_ss", "label_QCD_s", "label_QCD_light"]

    label_cls_nodes = network_options.get('label_cls_nodes', label_sophon_default)
    label_stored = network_options.get('label_stored', label_cls_nodes) # by default, store all classification node scores

    output = {}
    output['cls_index'] = labels['truth_label'] # classes can be too many, only store the index
    for idx, label_name in enumerate(label_cls_nodes):
        if label_name in label_stored:
            output[label_name] = (labels['truth_label'] == idx)
            output['score_' + label_name] = scores[:, idx]

    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        assert v.ndim == 1
        output[k] = v
    for k, v in observers.items():
        assert v.ndim == 1
        output[k] = v
    
    for k in output.keys():
        print(k, output[k])

    return output
