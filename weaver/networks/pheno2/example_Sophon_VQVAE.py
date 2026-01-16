import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict, Counter

from vqtorch.nn import VectorQuant

from utils.logger import _logger
from utils.nn.tools import (
    _concat,
    AllGather,
)
from utils.import_tools import import_module

ParticleTransformer = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformer


class FFN(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, fc_params=None, bias_last=True):
        super().__init__()
        fc_params = fc_params or []
        fcs = []
        in_dim = input_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, output_dim, bias=bias_last))
        self.fc = nn.Sequential(*fcs)

    def forward(self, x):
        return self.fc(x)


class ParticleTransformerSophonVQVAEWrapper(nn.Module):
    """
    VQVAE 包装（全局 EMA codebook）：
    - pf 分支：encoder + 分类头 -> z_e
    - gen 分支：可选聚合，但 codebook 为全局 EMA，不再逐事件生成
    - QCD 样本：强制使用 code idx 0
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        gen_model_kw = kwargs.pop('gen_model_kw')
        vq_kw = kwargs.pop('vq_kw')

        # cfg
        self.qcd_label_range = tuple(vq_kw.get('qcd_label_range', (161, 188)))
        self.logits_weight = vq_kw.get('logits_weight', 1.0)
        codebook_dim = vq_kw.get('codebook_dim', gen_model_kw['embed_dims'][-1])
        self.codebook_size = vq_kw.get('codebook_size', 16)  # 全局 codebook 条目数
        self.loss_gen_align_weight = vq_kw.get('loss_gen_align_weight', 1.0)
        self.loss_pf_align_weight = vq_kw.get('loss_pf_align_weight', 1.0)
        self.entropy_reg = vq_kw.get('entropy_reg', 0.0)
        self.kmeans_warmup_steps = int(vq_kw.get('kmeans_warmup_steps', 0) or 0)

        # 关闭内置 FC，改用外部头
        fc_params = kwargs.get('fc_params', None)
        kwargs['fc_params'] = None
        gen_model_kw['fc_params'] = None

        # 初始化 encoder
        self.mod = ParticleTransformer(**kwargs)
        self.gen = ParticleTransformer(**gen_model_kw)

        # 头部
        self.mod_fc = FFN(
            input_dim=kwargs['embed_dims'][-1],
            output_dim=kwargs['num_classes'],
            fc_params=fc_params or [],
            bias_last=True,
        )
        self.mod_proj = FFN(
            input_dim=kwargs['embed_dims'][-1],
            output_dim=codebook_dim,
            fc_params=vq_kw.get('enc_proj', []),
            bias_last=False,
        )
        self.gen_proj = FFN(
            input_dim=gen_model_kw['embed_dims'][-1],
            output_dim=codebook_dim,
            fc_params=vq_kw.get('gen_proj', []),
            bias_last=False,
        )

        # QCD 专用 codebook
        self.qcd_code_idx = 0  # EMA 版：强制 QCD 使用 idx=0

        # vqtorch VectorQuant layer
        self.vq = VectorQuant(
            feature_size=codebook_dim,
            num_codes=self.codebook_size,
            beta=vq_kw.get('beta', 0.95),
            sync_nu=vq_kw.get('sync_nu', 2.0),
            affine_lr=vq_kw.get('affine_lr', 2.0),
            affine_groups=vq_kw.get('affine_groups', 1),
            replace_freq=vq_kw.get('replace_freq', 0),
            kmeans_init=vq_kw.get('kmeans_init', False),
            norm=vq_kw.get('norm', 'none'),
            cb_norm=vq_kw.get('cb_norm', 'none'),
            dim=vq_kw.get('dim', -1),
            code_vector_size=vq_kw.get('code_vector_size', None),
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', 'gen.cls_token'}

    def _build_qcd_mask(self, labels, logits=None):
        labels_for_qcd = labels
        if labels_for_qcd is None and logits is not None:
            labels_for_qcd = logits.detach().softmax(dim=-1).argmax(dim=-1)
        qcd_mask = None
        if labels_for_qcd is not None:
            qcd_mask = (labels_for_qcd >= self.qcd_label_range[0]) & (labels_for_qcd < self.qcd_label_range[1])
        return qcd_mask

    def forward(self, *args, labels=None):
        """
        args: points, features, lorentz_vectors, mask, gen_points, gen_features, gen_lorentz_vectors, gen_mask
        返回: logits, vq_out(dict)
        """
        points, features, lorentz_vectors, mask, gen_points, gen_features, gen_lorentz_vectors, gen_mask = args

        x_cls = self.mod(features, v=lorentz_vectors, mask=mask)  # (B, Dm)
        gen_cls = self.gen(gen_features, v=gen_lorentz_vectors, mask=gen_mask)  # (B, Dg)
        gen_cls_proj = self.gen_proj(gen_cls)  # (B, D)

        logits = self.mod_fc(x_cls) if self.mod_fc is not None else None

        # QCD mask
        qcd_mask = self._build_qcd_mask(labels, logits=logits)

        # 量化（全局 EMA codebook）
        z_e = self.mod_proj(x_cls)
        z_q_raw, vq_dict = self.vq(gen_cls_proj)
        code_idx = vq_dict.get('q', None)
        z_q = z_q_raw

        # Enforce QCD constraint on codes:
        # - QCD must use code 0
        # - non-QCD is forbidden to use code 0
        # This avoids trivial collapse where most samples map to a single reserved code.
        if qcd_mask is not None and code_idx is not None:
            codebook = self.vq.get_codebook()  # (K, D), includes affine transform if enabled
            if codebook.size(0) > 1:
                z = gen_cls_proj.detach()  # (B, D)
                dist = (
                    z.pow(2).sum(dim=1, keepdim=True)
                    + codebook.pow(2).sum(dim=1).unsqueeze(0)
                    - 2.0 * (z @ codebook.t())
                )  # (B, K)
                dist = dist.clone()
                dist[qcd_mask, 1:] = 1e30   # QCD -> force 0
                dist[~qcd_mask, 0] = 1e30   # non-QCD -> forbid 0
                code_idx = dist.argmin(dim=1)  # (B,)

                # rebuild quantized output & VQ loss using constrained assignments
                z_q_embed = F.embedding(code_idx, codebook)      # (B, D)
                z_q_group = z_q_embed.unsqueeze(1)               # (B, 1, D) for groups=1
                vq_dict['q'] = code_idx.unsqueeze(1)             # (B, 1)
                vq_dict['z_q'] = z_q_group
                vq_dict['loss'] = self.vq.compute_loss(vq_dict['z'], z_q_group).mean()
                z_q = gen_cls_proj + (z_q_embed - gen_cls_proj).detach()
            else:
                # codebook_size == 1: cannot forbid code 0
                code_idx = code_idx.view(code_idx.shape[0], -1)[:, 0]
        elif code_idx is not None:
            # normalize to (B,)
            code_idx = code_idx.view(code_idx.shape[0], -1)[:, 0]
        else:
            code_idx = None

        vq_out = {
            'z_e': z_e,
            'gen_proj': gen_cls_proj,
            'z_q': z_q,
            'code_idx': code_idx,
            'codebook_size': self.codebook_size,
            'is_qcd': qcd_mask,
            'vq_loss': vq_dict.get('loss', None),
            'vq_dict': vq_dict,
        }
        return logits, vq_out


class VQVAELoss(nn.Module):
    """
    logits_weight 控制分类分量；loss_pf_align_weight / loss_gen_align_weight 控制 pf/gen 与 code 对齐；
    entropy_reg 鼓励均衡使用码本。
    """

    def __init__(self, logits_weight=1.0, vq_weight=1.0,
                 loss_pf_align_weight=1.0, loss_gen_align_weight=1.0,
                 entropy_reg=0.0, codebook_size=None,
                 vq_loss_update_freq: int = 1):
        super().__init__()
        self.logits_weight = logits_weight
        self.vq_weight = vq_weight
        self.loss_pf_align_weight = loss_pf_align_weight
        self.loss_gen_align_weight = loss_gen_align_weight
        self.entropy_reg = entropy_reg
        self.codebook_size = codebook_size
        self.vq_loss_update_freq = int(vq_loss_update_freq) if vq_loss_update_freq is not None else 1

    def forward(self, logits, vq_out, labels, step: int | None = None):
        device = vq_out['z_e'].device
        if logits is not None and self.logits_weight != 0:
            loss_cls = F.cross_entropy(logits, labels)
        else:
            loss_cls = torch.tensor(0., device=device)

        vq_loss = vq_out.get('vq_loss', None)
        if vq_loss is None:
            vq_loss = vq_out.get('vq_dict', {}).get('loss', None) if 'vq_dict' in vq_out else None
        if vq_loss is None:
            vq_loss = torch.tensor(0., device=device)
        if self.training and step is not None and self.vq_loss_update_freq and self.vq_loss_update_freq > 1:
            if (int(step) % self.vq_loss_update_freq) != 0:
                # zero-out VQ loss on non-update steps to avoid codebook collapse
                vq_loss = vq_loss * 0.0

        z_e = vq_out['z_e']
        gen_proj = vq_out['gen_proj']
        z_q = vq_out['z_q']
        code_idx = vq_out['code_idx']
        codebook_size = vq_out.get('codebook_size', self.codebook_size)

        loss_pf_align = F.mse_loss(z_e, z_q.detach())
        loss_gen_align = F.mse_loss(gen_proj, z_q.detach())

        loss_entropy = torch.tensor(0., device=device)
        if self.entropy_reg > 0 and codebook_size is not None and code_idx is not None:
            with torch.no_grad():
                one_hot = F.one_hot(code_idx.view(-1), codebook_size).float()
                probs = one_hot.mean(dim=0).clamp(min=1e-9)
                entropy = -(probs * probs.log()).sum()
            loss_entropy = self.entropy_reg * (-entropy)  # 惩罚低熵

        loss_vq = (
            vq_loss
            + self.loss_pf_align_weight * loss_pf_align
            + self.loss_gen_align_weight * loss_gen_align
            + loss_entropy
        )

        loss = self.logits_weight * loss_cls + self.vq_weight * loss_vq

        return loss, loss_cls, loss_vq, loss_pf_align, loss_gen_align, loss_entropy


def get_model(data_config, **kwargs):
    # 默认配置与 CLIP 对齐
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
        # vq kwargs
        vq_kw=dict(),
    )
    cfg['gen_model_kw'].update(
        input_dim=len(data_config.input_dicts.get('gen_features', [])),
        num_classes=None,
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
        trim=True,
        for_inference=False,
    )
    cfg['vq_kw'].update(
        codebook_dim=64,  # 与 gen 分支最后一层对齐
        codebook_size=16,
        logits_weight=1.0,
        vq_weight=1.0,
        loss_pf_align_weight=1.0,
        loss_gen_align_weight=1.0,
        ema_decay=0.99,
        ema_eps=1e-5,
        entropy_reg=1e-3,
        enc_proj=[],
        gen_proj=[],
        qcd_label_range=(161, 188),
        beta=0.95,
        sync_nu=0.0,
        affine_lr=0.0,
        affine_groups=1,
        replace_freq=0,
        kmeans_init=False,
        norm='none',
        cb_norm='none',
        dim=-1,
        code_vector_size=None,
        vq_loss_update_freq=4,
        # kmeans_init warmup: run a few no-grad forward passes on random-initialized model
        # so vqtorch can initialize the codebook from the latent distribution.
        kmeans_warmup_steps=0,
    )

    # 用户覆盖
    for k, v in kwargs.pop('gen_model_kw', dict()).items():
        assert k in cfg['gen_model_kw'], 'Invalid key %s in "gen_model_kw"' % k
        cfg['gen_model_kw'][k] = v
    for k, v in kwargs.pop('vq_kw', dict()).items():
        assert k in cfg['vq_kw'], 'Invalid key %s in "vq_kw"' % k
        cfg['vq_kw'][k] = v
    for k, v in kwargs.items():
        assert k in cfg, 'Invalid key %s' % k
        cfg[k] = v

    _logger.info('Model config: %s' % str(cfg))

    eval_kw = cfg.pop('eval_kw', dict())
    cfg.pop('label_cls_nodes', None)
    cfg.pop('label_stored', None)

    model = ParticleTransformerSophonVQVAEWrapper(**cfg)
    model.eval_kw = eval_kw

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax', 'code_idx'],
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            'softmax': {0: 'N'},
            'code_idx': {0: 'N'},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    vq_kw = kwargs.get('vq_kw', {})
    return VQVAELoss(
        logits_weight=vq_kw.get('logits_weight', 1.0),
        vq_weight=vq_kw.get('vq_weight', 1.0),
        loss_pf_align_weight=vq_kw.get('loss_pf_align_weight', 1.0),
        loss_gen_align_weight=vq_kw.get('loss_gen_align_weight', 1.0),
        entropy_reg=vq_kw.get('entropy_reg', 0.0),
        codebook_size=vq_kw.get('codebook_size', None),
        vq_loss_update_freq=vq_kw.get('vq_loss_update_freq', 4),
    )


def get_train_fn(data_config, **kwargs):
    return train_classification_vqvae


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_classification_vqvae


def get_save_fn(data_config, **kwargs):
    return save_classification_vqvae


# 训练与评估
def train_classification_vqvae(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, extra_args=None):
    model.train()
    if loss_func is not None:
        loss_func.train()

    data_config = train_loader.dataset.config
    label_counter = Counter()
    total_loss = total_loss_cls = total_loss_vq = total_loss_pf_align = total_loss_gen_align = 0
    num_batches = total_correct = entry_count = count = 0
    total_codebook_util = 0

    import time
    import tqdm

    # Optional: kmeans_init warmup for vqtorch codebook on a randomly initialized model.
    # vqtorch's `kmeans_init=True` uses a forward hook; running a few no-grad forwards first
    # stabilizes training and matches "initialize codebook with K-means on latent space".
    if epoch == 0:
        _m = model.module if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model
        warmup_steps = int(getattr(_m, 'kmeans_warmup_steps', 0) or 0)
        vq_layer = getattr(_m, 'vq', None)
        need_kmeans = (
            warmup_steps > 0
            and vq_layer is not None
            and hasattr(vq_layer, 'data_initialized')
            and int(vq_layer.data_initialized.item()) == 0
            and not getattr(_m, '_vq_kmeans_warmup_done', False)
        )
        if need_kmeans:
            rank0 = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank0 = (torch.distributed.get_rank() == 0)
            if rank0:
                _logger.info('Running vqtorch kmeans_init warmup: %d step(s)', warmup_steps)
                prev_mode = _m.training
                _m.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(False):
                    it = iter(train_loader)
                    for _ in range(warmup_steps):
                        X, y, _ = next(it)
                        inputs = [X[k].to(dev) for k in data_config.input_names]
                        label = y[data_config.label_names[0]].long().to(dev)
                        _m(*inputs, labels=label)
                _m.train(prev_mode)
                _m._vq_kmeans_warmup_done = True
            # sync codebook (and any buffers) across ranks
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
                for p in _m.parameters():
                    torch.distributed.broadcast(p.data, src=0)
                for b in _m.buffers():
                    torch.distributed.broadcast(b.data, src=0)
                torch.distributed.barrier()

    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for batch_idx, (X, y, _) in enumerate(tq):
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                logits, vq_out = model(*inputs, labels=label)
                loss, loss_cls, loss_vq, loss_pf_align, loss_gen_align, loss_entropy = loss_func(
                    logits, vq_out, label, step=batch_idx
                )
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            # detach for logging
            loss_val = loss.item()
            loss_cls_val = loss_cls.item()
            loss_vq_val = loss_vq.item()
            loss_pf_align_val = loss_pf_align.item()
            loss_gen_align_val = loss_gen_align.item()
            loss_entropy_val = loss_entropy.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples

            total_loss += loss_val
            total_loss_cls += loss_cls_val
            total_loss_vq += loss_vq_val
            total_loss_pf_align += loss_pf_align_val
            total_loss_gen_align += loss_gen_align_val

            # 计算codebook利用率：当前batch中使用的唯一code数量 / codebook_size
            code_idx = vq_out.get('code_idx', None)
            codebook_size = vq_out.get('codebook_size', loss_func.codebook_size)
            if code_idx is not None and codebook_size is not None:
                code_idx = code_idx.detach()
                unique_codes = torch.unique(code_idx).numel()
                codebook_util = unique_codes / codebook_size
                total_codebook_util += codebook_util
            else:
                codebook_util = 0.0

            tq_dict = {
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                #'AvgLossCls': '%.5f' % (total_loss_cls / num_batches),
                #'AvgLossVQ': '%.5f' % (total_loss_vq / num_batches),
                #'AvgLossPf': '%.5f' % (total_loss_pf_align / num_batches),
                'AvgLossGen': '%.5f' % (total_loss_gen_align / num_batches),
                'Loss': '%.5f' % loss_val,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'CodebookUtil': '%.3f' % codebook_util,
            }

            if logits is not None and loss_func.logits_weight != 0:
                _, preds = logits.max(1)
                correct = (preds == label).sum().item()
                total_correct += correct
                tq_dict.update({
                    #'Acc': '%.5f' % (correct / num_examples),
                    #'AvgAcc': '%.5f' % (total_correct / count),
                })

            tq.set_postfix(tq_dict)

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / max(count, 1)))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Max CUDA memory: %.1f MB' % (torch.cuda.max_memory_allocated(dev) / 1024.**2,))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("LossCls/train (epoch)", total_loss_cls / num_batches, epoch),
            ("LossVQ/train (epoch)", total_loss_vq / num_batches, epoch),
            ("LossPf/train (epoch)", total_loss_pf_align / num_batches, epoch),
            ("LossGen/train (epoch)", total_loss_gen_align / num_batches, epoch),
        ])
        if logits is not None and loss_func.logits_weight != 0:
            tb_helper.write_scalars([
                ("Acc/train (epoch)", total_correct / max(count, 1), epoch),
            ])
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification_vqvae(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                                  eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                                  tb_helper=None, extra_args=None):
    model.eval()
    if loss_func is not None:
        loss_func.eval()

    data_config = test_loader.dataset.config
    label_counter = Counter()
    total_loss = total_loss_cls = total_loss_vq = 0
    num_batches = total_correct = entry_count = count = 0
    total_codebook_util = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)

    import time
    import tqdm

    start_time = time.time()
    eval_kw = model.module.eval_kw \
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.eval_kw
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]

                logits, vq_out = model(*inputs, labels=label if for_training else None)
                if loss_func is not None:
                    loss, loss_cls, loss_vq, loss_pf_align, loss_gen_align, loss_entropy = loss_func(
                        logits, vq_out, label, step=None
                    )
                else:
                    loss, loss_cls, loss_vq, loss_pf_align, loss_gen_align, loss_entropy = (
                        torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.))

                # all-reduce
                loss = AllGather.apply(loss.unsqueeze(0)).mean().item()
                loss_cls = AllGather.apply(loss_cls.unsqueeze(0)).mean().item()
                loss_vq = AllGather.apply(loss_vq.unsqueeze(0)).mean().item()
                loss_pf_align = AllGather.apply(loss_pf_align.unsqueeze(0)).mean().item()
                loss_gen_align = AllGather.apply(loss_gen_align.unsqueeze(0)).mean().item()
                loss_entropy = AllGather.apply(loss_entropy.unsqueeze(0)).mean().item()

                # gather logits/code_idx/labels
                if logits is not None:
                    logits = AllGather.apply(logits)
                    scores.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                code_idx = vq_out.get('code_idx', None)
                if code_idx is not None:
                    code_idx = AllGather.apply(code_idx)
                    labels['vq_code_idx'].append(code_idx.cpu().numpy())

                y = {k: AllGather.apply(v.to(dev)) for k, v in y.items()}
                label = y[data_config.label_names[0]].long().to(dev)
                for k, v in y.items():
                    labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.numpy(force=True))

                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy().tolist())

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                total_loss_cls += loss_cls * num_examples
                total_loss_vq += loss_vq * num_examples

                tq_dict = {
                    'LossCls': '%.5f' % loss_cls,
                    'LossVQ': '%.5f' % loss_vq,
                    'LossPf': '%.5f' % loss_pf_align,
                    'LossGen': '%.5f' % loss_gen_align,
                    'LossEnt': '%.5f' % loss_entropy,
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                }

                if logits is not None and loss_func is not None and loss_func.logits_weight != 0:
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
            ("LossVQ/%s (epoch)" % tb_mode, total_loss_vq / count, epoch),
        ])
        if logits is not None and loss_func is not None and loss_func.logits_weight != 0:
            tb_helper.write_scalars([
                ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
            ])

    if logits is not None:
        scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    if for_training:
        return total_correct / max(count, 1)
    else:
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / max(count, 1), scores if logits is not None else None, labels, observers


def save_classification_vqvae(args, data_config, scores, labels, observers):
    import ast
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}

    num_classes = network_options['num_classes']
    label_cls_nodes = network_options.get('label_cls_nodes', None)
    if label_cls_nodes is None:
        label_cls_nodes = [f'label_{i}' for i in range(num_classes)]
    label_stored = network_options.get('label_stored', label_cls_nodes)

    output = {}
    output['cls_index'] = labels['truth_label']
    output['vq_code_idx'] = labels.get('vq_code_idx')

    if scores is not None:
        for idx, label_name in enumerate(label_cls_nodes):
            if label_name in label_stored:
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

