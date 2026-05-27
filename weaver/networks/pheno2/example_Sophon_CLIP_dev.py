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
                    # projection heads are numerically sensitive under AMP; force fp32 here
                    with torch.cuda.amp.autocast(enabled=False):
                        x_mod = self.mod_proj(x_mod_clip.float())

                    if isinstance(x_gen_out, tuple) and len(x_gen_out) == 2:
                        _, x_gen_clip = x_gen_out
                    elif torch.is_tensor(x_gen_out) and x_gen_out.ndim == 3 and x_gen_out.size(1) == 2:
                        x_gen_clip = x_gen_out[:, 1]
                    else:
                        x_gen_clip = x_gen_out
                    with torch.cuda.amp.autocast(enabled=False):
                        x_gen = self.gen_proj(x_gen_clip.float())
                else:
                    if not self.clip_share_token:
                        # 主模型：分类与对比两路分支
                        x_mod_cls, x_mod_clip = split_outputs(x_mod_out)
                        logits = self.mod_fc(x_mod_cls)
                        with torch.cuda.amp.autocast(enabled=False):
                            x_mod = self.mod_proj(x_mod_clip.float())

                        # gen模型：只用于对比分支，若返回双分支，取对比路
                        if isinstance(x_gen_out, tuple) and len(x_gen_out) == 2:
                            _, x_gen_clip = x_gen_out
                        elif torch.is_tensor(x_gen_out) and x_gen_out.ndim == 3 and x_gen_out.size(1) == 2:
                            x_gen_clip = x_gen_out[:, 1]
                        else:
                            x_gen_clip = x_gen_out
                        with torch.cuda.amp.autocast(enabled=False):
                            x_gen = self.gen_proj(x_gen_clip.float())
                    else:
                        # 共享一套表征
                        assert torch.is_tensor(x_mod_out) and x_mod_out.ndim == 2, 'Invalid shape %s' % str(getattr(x_mod_out, 'shape', None))
                        logits = self.mod_fc(x_mod_out)
                        with torch.cuda.amp.autocast(enabled=False):
                            x_mod = self.mod_proj(x_mod_out.float())
                            _xg = x_gen_out if torch.is_tensor(x_gen_out) else x_gen_out[1] if isinstance(x_gen_out, tuple) else x_gen_out
                            x_gen = self.gen_proj(_xg.float())

            elif self.clip_mode == 'clip-with-gencls':
                if not self.clip_share_token:
                    # 生成模型：分类与对比两路分支
                    x_gen_cls, x_gen_clip = split_outputs(x_gen_out)
                    logits = self.gen_fc(x_gen_cls)
                    with torch.cuda.amp.autocast(enabled=False):
                        x_gen = self.gen_proj(x_gen_clip.float())

                    # 主模型：只用于对比分支
                    if isinstance(x_mod_out, tuple) and len(x_mod_out) == 2:
                        _, x_mod_clip = x_mod_out
                    elif torch.is_tensor(x_mod_out) and x_mod_out.ndim == 3 and x_mod_out.size(1) == 2:
                        x_mod_clip = x_mod_out[:, 1]
                    else:
                        x_mod_clip = x_mod_out
                    with torch.cuda.amp.autocast(enabled=False):
                        x_mod = self.mod_proj(x_mod_clip.float())
                else:
                    # 共享一套表征
                    assert torch.is_tensor(x_gen_out) and x_gen_out.ndim == 2, 'Invalid shape %s' % str(getattr(x_gen_out, 'shape', None))
                    logits = self.gen_fc(x_gen_out)
                    with torch.cuda.amp.autocast(enabled=False):
                        _xm = x_mod_out if torch.is_tensor(x_mod_out) else x_mod_out[1] if isinstance(x_mod_out, tuple) else x_mod_out
                        x_mod = self.mod_proj(_xm.float())
                        x_gen = self.gen_proj(x_gen_out.float())

            elif self.clip_mode == 'clip-only':
                logits = None
                # 仅对比学习：若为双分支，取对比路
                if isinstance(x_mod_out, tuple) and len(x_mod_out) == 2:
                    with torch.cuda.amp.autocast(enabled=False):
                        x_mod = self.mod_proj(x_mod_out[1].float())
                elif torch.is_tensor(x_mod_out) and x_mod_out.ndim == 3 and x_mod_out.size(1) == 2:
                    with torch.cuda.amp.autocast(enabled=False):
                        x_mod = self.mod_proj(x_mod_out[:, 1].float())
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        x_mod = self.mod_proj(x_mod_out.float())

                if isinstance(x_gen_out, tuple) and len(x_gen_out) == 2:
                    with torch.cuda.amp.autocast(enabled=False):
                        x_gen = self.gen_proj(x_gen_out[1].float())
                elif torch.is_tensor(x_gen_out) and x_gen_out.ndim == 3 and x_gen_out.size(1) == 2:
                    with torch.cuda.amp.autocast(enabled=False):
                        x_gen = self.gen_proj(x_gen_out[:, 1].float())
                else:
                    with torch.cuda.amp.autocast(enabled=False):
                        x_gen = self.gen_proj(x_gen_out.float())

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

    def __init__(self, clip_mode=None, beta=1., alpha=1., soften=0., softenqcd=0.):
        super().__init__()
        self.clip_mode = clip_mode
        self.beta = beta
        self.alpha = alpha
        self.soften = float(soften)
        self.softenqcd = float(softenqcd)
        assert 0. <= self.soften <= 1., 'soften must be in [0, 1], got %s' % str(self.soften)
        assert 0. <= self.softenqcd <= 1., 'softenqcd must be in [0, 1], got %s' % str(self.softenqcd)
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
            loss_cls = F.cross_entropy(logits, labels)
        else:
            loss_cls = torch.tensor(0., device=labels.device)

        # CLIP constrastive learning (force fp32 for numerical stability under AMP)
        if x_mod is not None:
            with torch.cuda.amp.autocast(enabled=False):
                x_mod = x_mod.float()
                x_gen = x_gen.float()
                eps = 1e-6
                x_mod = x_mod / x_mod.norm(dim=-1, keepdim=True).clamp_min(eps)
                x_gen = x_gen / x_gen.norm(dim=-1, keepdim=True).clamp_min(eps)

                # compute cosine similarity
                # CLIP-style stabilization: clamp the exp(scale) to avoid overflow (esp. under AMP)
                logit_scale = self.logit_scale.float().exp().clamp(max=100.0)
                logits_cont_2d = logit_scale * (x_mod @ x_gen.t()) # (batch, batch)
                logits_cont_2d_t = logits_cont_2d.t()

            # ---- debug (limited prints): detect non-finite / extreme values ----
            if not hasattr(self, '_dbg_nonfinite_cnt'):
                self._dbg_nonfinite_cnt = 0
            if self._dbg_nonfinite_cnt < 20:
                for _n, _t in (('x_mod', x_mod), ('x_gen', x_gen), ('logits_cont_2d', logits_cont_2d)):
                    _finite = torch.isfinite(_t)
                    if not bool(_finite.all()):
                        self._dbg_nonfinite_cnt += 1
                        _logger.warning(
                            '[CLIPLoss debug] %s nonfinite=%d/%d absmax=%s min=%s max=%s dtype=%s logit_scale=%.3e',
                            _n,
                            int((~_finite).sum().item()),
                            int(_t.numel()),
                            str(_t.detach().abs().amax().item()),
                            str(_t.detach().min().item()),
                            str(_t.detach().max().item()),
                            str(_t.dtype),
                            float(logit_scale.detach().item()),
                        )

            if self.soften > 0. or self.softenqcd > 0.:
                # soft target matrix: diagonal=1, selected same-label off-diagonal=soften, then row-normalize.
                same_label = labels.view(-1, 1).eq(labels.view(1, -1))
                target = same_label.to(dtype=logits_cont_2d.dtype) * self.soften
                if self.softenqcd > 0.:
                    qcd_label = labels.ge(161) & labels.lt(188)
                    same_qcd_label = same_label & qcd_label.view(-1, 1) & qcd_label.view(1, -1)
                    target = torch.where(
                        same_qcd_label,
                        torch.full_like(target, self.softenqcd),
                        target,
                    )
                target.fill_diagonal_(1.)
                target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)

                log_prob = F.log_softmax(logits_cont_2d, dim=1)
                log_prob_t = F.log_softmax(logits_cont_2d_t, dim=1)
                # use the row-normalized target distribution for both directions (do NOT transpose after normalization)
                loss_cont = -(target * log_prob).sum(dim=1).mean() - (target * log_prob_t).sum(dim=1).mean()
            else:
                indices = torch.arange(x_mod.size(0), device=x_mod.device) # (batch,)
                loss_cont = F.cross_entropy(logits_cont_2d, indices) + F.cross_entropy(logits_cont_2d_t, indices)
        else:
            loss_cont = torch.tensor(0., device=labels.device)

        loss = self.beta * loss_cls + self.alpha * loss_cont
        return loss, loss_cls, loss_cont


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
        soften=0.,  # only when >0: same-label off-diagonal entries in CLIP target matrix
        softenqcd=0.,  # only when >0: same-label softening for labels in range(161, 188)
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
    return CLIPLoss(
        clip_mode=clip_kw['mode'],
        beta=clip_kw['beta'],
        alpha=clip_kw['alpha'],
        soften=clip_kw.get('soften', 0.),
        softenqcd=clip_kw.get('softenqcd', 0.),
    )


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
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                logits, x_mod, x_gen = model(*inputs)
                loss, loss_cls, loss_cont = loss_func(logits, x_mod, x_gen, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

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
    # --- Minimal feature: dump gen-encoder vectors (used for contrastive loss) for the first N eval batches ---
    # Single-GPU use case; this is for downstream dimensionality reduction studies.
    dump_gen_vec_max_batches = int(eval_kw.get('dump_gen_vec_max_batches', 0))
    active = False
    dump_gen_vec_path = eval_kw.get(
        'dump_gen_vec_path',
        os.path.abspath(f'gen_encoder_vectors_eval_epoch{int(epoch):04d}_first{dump_gen_vec_max_batches}batches.txt')
    )
    dump_gen_vec_enabled = bool(eval_kw.get('dump_gen_vec_enabled', active)) and for_training and (dump_gen_vec_max_batches > 0)
    dump_gen_vec_fh = None
    if dump_gen_vec_enabled:
        # overwrite per-eval call (typically per epoch)
        dump_gen_vec_fh = open(dump_gen_vec_path, 'w', encoding='utf-8')

    # --- Minimal feature: dump x_mod vectors (used for contrastive loss) for the first N eval batches ---
    dump_mod_vec_max_batches = int(eval_kw.get('dump_mod_vec_max_batches', dump_gen_vec_max_batches))
    dump_mod_vec_path = eval_kw.get(
        'dump_mod_vec_path',
        os.path.abspath(f'mod_encoder_vectors_eval_epoch{int(epoch):04d}_first{dump_mod_vec_max_batches}batches.txt')
    )
    dump_mod_vec_enabled = bool(eval_kw.get('dump_mod_vec_enabled', active)) and for_training and (dump_mod_vec_max_batches > 0)
    dump_mod_vec_fh = None
    if dump_mod_vec_enabled:
        dump_mod_vec_fh = open(dump_mod_vec_path, 'w', encoding='utf-8')

    # --- Minimal feature: dump raw logits (pre-softmax) for the first N eval batches ---
    dump_logits_max_batches = int(eval_kw.get('dump_logits_max_batches', dump_gen_vec_max_batches))
    dump_logits_path = eval_kw.get(
        'dump_logits_path',
        os.path.abspath(f'logits_eval_epoch{int(epoch):04d}_first{dump_logits_max_batches}batches.txt')
    )
    dump_logits_enabled = bool(eval_kw.get('dump_logits_enabled', active)) and for_training and (dump_logits_max_batches > 0)
    dump_logits_fh = None
    if dump_logits_enabled:
        dump_logits_fh = open(dump_logits_path, 'w', encoding='utf-8')
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

                # Dump gen vectors for the first N batches (each line: "<label>\t<vec0> <vec1> ...")
                if dump_gen_vec_fh is not None and num_batches < dump_gen_vec_max_batches and (x_gen is not None):
                    x_gen_cpu = x_gen.detach().float().cpu().numpy()
                    label_cpu = label.detach().cpu().numpy()
                    for lb, vec in zip(label_cpu, x_gen_cpu):
                        dump_gen_vec_fh.write(str(int(lb)) + "\t" + " ".join(f"{v:.6e}" for v in vec.tolist()) + "\n")

                # Dump mod vectors for the first N batches (each line: "<label>\t<vec0> <vec1> ...")
                if dump_mod_vec_fh is not None and num_batches < dump_mod_vec_max_batches and (x_mod is not None):
                    x_mod_cpu = x_mod.detach().float().cpu().numpy()
                    label_cpu = label.detach().cpu().numpy()
                    for lb, vec in zip(label_cpu, x_mod_cpu):
                        dump_mod_vec_fh.write(str(int(lb)) + "\t" + " ".join(f"{v:.6e}" for v in vec.tolist()) + "\n")

                # Dump logits for the first N batches (each line: "<label>\t<logit0> <logit1> ...")
                if dump_logits_fh is not None and num_batches < dump_logits_max_batches and (logits is not None):
                    logits_cpu = logits.detach().float().cpu().numpy()
                    label_cpu = label.detach().cpu().numpy()
                    for lb, vec in zip(label_cpu, logits_cpu):
                        dump_logits_fh.write(str(int(lb)) + "\t" + " ".join(f"{v:.6e}" for v in vec.tolist()) + "\n")

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
    if dump_gen_vec_fh is not None:
        dump_gen_vec_fh.close()
        _logger.info('Dumped gen encoder vectors for first %d eval batches to %s', dump_gen_vec_max_batches, dump_gen_vec_path)
    if dump_mod_vec_fh is not None:
        dump_mod_vec_fh.close()
        _logger.info('Dumped mod encoder vectors for first %d eval batches to %s', dump_mod_vec_max_batches, dump_mod_vec_path)
    if dump_logits_fh is not None:
        dump_logits_fh.close()
        _logger.info('Dumped logits for first %d eval batches to %s', dump_logits_max_batches, dump_logits_path)

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
                #'QCD': list(range(161, 188)),
            },
            #'comp_list': [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
            'comp_list': [('Xcc', 'Xbb')] # ROC curves for A vs B
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
