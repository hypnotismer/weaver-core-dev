import os
import math
import torch
import torch.nn as nn
from torch import Tensor
import tqdm
import time
from collections import defaultdict, Counter
from functools import partial
import numpy as np

from utils.logger import _logger
from utils.nn.tools import (
    train_regression,
    evaluate_regression,
    evaluate_metrics,
    _flatten_preds,
    _flatten_label,
    _concat
)
from utils.import_tools import import_module

# Default soft-sculpt configuration (overridable via -o sculpt_kw {...})
# sculpt_on: 'an' | 'train_bkg' | 'both'
VALID_SCULPT_ON = ('an', 'train_bkg', 'both')
VALID_BIN_WEIGHTING = ('occupancy', 'uniform')
DEFAULT_SCULPT_KW = {
    'enable': True,
    'sculpt_on': 'an',
    'lambda': 1.0,
    'lambda_train': None,  # None -> same as lambda
    # Soft-sculpt constraint starts at epoch >= warmup_epochs (0-based weaver epoch).
    # e.g. warmup_epochs=10 -> epochs 0..9 train without sculpt; from epoch 10 onward apply it.
    'warmup_epochs': 0,
    # Soft cut/no-cut PDF-ratio flatness (aligned with plot_mass_sculpting_xggg).
    'tau': 0.02,
    'pass_fracs': [0.70, 0.50, 0.30],  # keep fractions (= bkg rej 30/50/70%)
    'score_index': 0,  # label_xggg
    # AN: high-mass bins; train_bkg: full train selection mass range (yaml fj_sdmass 20-360)
    'mass_bins': list(range(180, 361, 10)),
    'mass_bins_train': list(range(20, 361, 10)),
    'train_bkg_cls_indices': [1, 2],  # top, qcd under label_cls_nodes
    'mass_source': 'finetune_parts',  # soft/hard sculpt on finetune parts reg mass
    'reg_index': 1,  # preds_reg[:,1] == target_parts_mass_factor
    # Rolling buffer approximates global WP thresholds / bin stats (avoids per-batch noise).
    'buffer_size': 65536,
    'min_bin_count': 32,
    'bin_weighting': 'occupancy',  # 'occupancy' | 'uniform' (occupancy uses Σ event weights)
    # Per-event reweight: look up data_config.reweight_hists via Z[fj_pt,fj_sdmass]
    # + class label (no dataset.py change). After accept/reject sampling, ones is
    # often preferable — set use_event_weight: False in that case.
    'use_event_weight': True,
    'weight_branch': 'weight_',  # optional precomputed branch in Z, if present
}

_SCULPT_NO_REG_WARNED = False


def _resolve_sculpt_kw(sculpt_kw):
    if sculpt_kw is None:
        return None
    cfg = dict(DEFAULT_SCULPT_KW)
    cfg.update(sculpt_kw)
    sculpt_on = cfg.get('sculpt_on', 'an')
    if sculpt_on not in VALID_SCULPT_ON:
        raise ValueError(
            f"sculpt_kw['sculpt_on'] must be one of {VALID_SCULPT_ON}, got {sculpt_on!r}")
    weighting = cfg.get('bin_weighting', 'occupancy')
    if weighting not in VALID_BIN_WEIGHTING:
        raise ValueError(
            f"sculpt_kw['bin_weighting'] must be one of {VALID_BIN_WEIGHTING}, got {weighting!r}")
    cfg['buffer_size'] = int(cfg.get('buffer_size', 65536) or 65536)
    cfg['min_bin_count'] = int(cfg.get('min_bin_count', 32) or 32)
    return cfg


def finetune_reg_mass(mass_corr, jet_pt, jet_corr_pt, fj_mass):
    """AK15 regression mass; matches plot_mass_sculpting_xggg.regression_mass."""
    jet_pt = jet_pt.reshape(-1).to(dtype=mass_corr.dtype)
    jet_corr_pt = jet_corr_pt.reshape(-1).to(dtype=mass_corr.dtype)
    fj_mass = fj_mass.reshape(-1).to(dtype=mass_corr.dtype)
    mass_corr = mass_corr.reshape(-1)
    raw_factor = 1.0 - jet_pt / jet_corr_pt.clamp(min=1e-6)
    ak15_mass = fj_mass * jet_corr_pt / jet_pt.clamp(min=1e-6)
    return mass_corr * ak15_mass * (1.0 - raw_factor)


class ScoreMassBuffer:
    """Fixed-capacity rolling buffer of detached (score, mass, event_weight)."""

    def __init__(self, capacity):
        self.capacity = max(int(capacity), 1)
        self.scores = np.empty(0, dtype=np.float32)
        self.masses = np.empty(0, dtype=np.float32)
        self.weights = np.empty(0, dtype=np.float32)

    def clear(self):
        self.scores = np.empty(0, dtype=np.float32)
        self.masses = np.empty(0, dtype=np.float32)
        self.weights = np.empty(0, dtype=np.float32)

    @property
    def n(self):
        return int(self.scores.shape[0])

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().float().reshape(-1).cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(x, dtype=np.float32).reshape(-1)

    def append(self, scores, masses, weights=None):
        scores = self._to_numpy(scores)
        masses = self._to_numpy(masses)
        if scores is None or masses is None or scores.size == 0:
            return
        if weights is None:
            weights = np.ones(scores.shape[0], dtype=np.float32)
        else:
            weights = self._to_numpy(weights)
            if weights.shape[0] != scores.shape[0]:
                raise ValueError('ScoreMassBuffer.append: weights length mismatch')
        self.scores = np.concatenate([self.scores, scores])
        self.masses = np.concatenate([self.masses, masses])
        self.weights = np.concatenate([self.weights, weights])
        if self.scores.shape[0] > self.capacity:
            self.scores = self.scores[-self.capacity:]
            self.masses = self.masses[-self.capacity:]
            self.weights = self.weights[-self.capacity:]

    def as_tensors(self, device, dtype):
        if self.n == 0:
            return None, None, None
        s = torch.as_tensor(self.scores, device=device, dtype=dtype)
        m = torch.as_tensor(self.masses, device=device, dtype=dtype)
        w = torch.as_tensor(self.weights, device=device, dtype=dtype)
        return s, m, w


def _make_sculpt_buffers(sculpt_kw):
    if sculpt_kw is None or not sculpt_kw.get('enable', False):
        return None, None
    cap = int(sculpt_kw.get('buffer_size', 65536))
    return ScoreMassBuffer(cap), ScoreMassBuffer(cap)


def _lookup_reweight_from_hists(Z, label_cls, n, data_config):
    """Numpy reweight factors from data_config.reweight_hists (same logic as weaver _build_weights)."""
    hists = getattr(data_config, 'reweight_hists', None)
    branches = getattr(data_config, 'reweight_branches', None)
    bins = getattr(data_config, 'reweight_bins', None)
    classes = getattr(data_config, 'reweight_classes', None)
    if not hists or not branches or not bins or not classes:
        return None
    if Z is None or branches[0] not in Z or branches[1] not in Z:
        return None
    x = np.asarray(Z[branches[0]].detach().cpu().numpy() if torch.is_tensor(Z[branches[0]])
                   else Z[branches[0]], dtype=np.float64).reshape(-1)
    yv = np.asarray(Z[branches[1]].detach().cpu().numpy() if torch.is_tensor(Z[branches[1]])
                    else Z[branches[1]], dtype=np.float64).reshape(-1)
    if x.shape[0] != n or yv.shape[0] != n:
        return None
    if torch.is_tensor(label_cls):
        cls = label_cls.detach().cpu().numpy().reshape(-1)
    else:
        cls = np.asarray(label_cls).reshape(-1)
    if cls.shape[0] != n:
        return None

    x_bins, y_bins = bins
    discard = getattr(data_config, 'reweight_discard_under_overflow', True)
    wgt = np.zeros(n, dtype=np.float32)
    x_idx = np.clip(np.digitize(x, x_bins) - 1, 0, len(x_bins) - 2)
    y_idx = np.clip(np.digitize(yv, y_bins) - 1, 0, len(y_bins) - 2)
    in_range = np.ones(n, dtype=bool)
    if discard:
        in_range = (
            (x >= min(x_bins)) & (x <= max(x_bins)) &
            (yv >= min(y_bins)) & (yv <= max(y_bins))
        )
    for i, label in enumerate(classes):
        hist = hists.get(label) if isinstance(hists, dict) else None
        if hist is None:
            continue
        hist = np.asarray(hist, dtype=np.float32)
        pos = (cls == i) & in_range
        if not np.any(pos):
            continue
        wgt[pos] = hist[x_idx[pos], y_idx[pos]]
    return wgt


def _read_event_weights(Z, n, device, dtype, sculpt_kw, data_config=None, label_cls=None):
    """Return per-event reweight factors (detached). Falls back to ones.

    Priority:
      1) Z[weight_branch] if present (e.g. predict/observers)
      2) lookup data_config.reweight_hists with Z[fj_pt,fj_sdmass] + label_cls
      3) ones
    """
    ones = torch.ones(n, device=device, dtype=dtype)
    if sculpt_kw is None or not sculpt_kw.get('use_event_weight', True):
        return ones
    branch = sculpt_kw.get('weight_branch', 'weight_')
    if Z is not None and branch in Z:
        w = Z[branch].to(device=device, dtype=dtype).reshape(-1)
        if w.numel() == n:
            return w.clamp(min=0.0).detach()
        _logger.warning(
            'sculpt weight branch %s length %d != batch %d; trying hist lookup', branch, w.numel(), n)
    if data_config is not None and label_cls is not None:
        w_np = _lookup_reweight_from_hists(Z, label_cls, n, data_config)
        if w_np is not None:
            return torch.as_tensor(w_np, device=device, dtype=dtype).clamp(min=0.0)
    return ones


def _weighted_quantile(values, weights, q):
    """Weighted quantile; values/weights 1-D, q in [0,1]. No grad through result."""
    values = values.detach()
    weights = weights.detach().clamp(min=0.0)
    if values.numel() == 0:
        return values.new_zeros(())
    if values.numel() == 1:
        return values[0]
    v, order = torch.sort(values)
    w = weights[order]
    cdf = torch.cumsum(w, dim=0)
    total = cdf[-1].clamp(min=1e-8)
    cdf = cdf / total
    # first index where cdf >= q
    idx = torch.searchsorted(cdf, values.new_tensor(float(q)))
    idx = int(idx.clamp(max=v.numel() - 1).item())
    return v[idx]


def soft_pdf_ratio_sculpt_loss(logits, mass, event_mask, sculpt_kw, mass_bins=None,
                               buffer=None, event_weights=None):
    """Soft cut/no-cut PDF-ratio flatness (plot-aligned), with optional rolling buffer.

    Histograms use per-event reweight ``event_weights`` (Σw, not raw counts):
      h_all_b = Σ_i w_i 1_bin,  h_pass_b = Σ_i w_i softpass_i 1_bin
    History in ``buffer`` is detached; current-batch scores keep gradients.
    """
    if sculpt_kw is None or not sculpt_kw.get('enable', False):
        return logits.new_zeros(())

    mask = event_mask.reshape(-1).bool()
    n_cur = int(mask.sum().item())
    if n_cur < 1 and (buffer is None or buffer.n < 8):
        return logits.sum() * 0.0

    score_index = int(sculpt_kw.get('score_index', 0))
    tau = float(sculpt_kw.get('tau', 0.02))
    pass_fracs = sculpt_kw.get('pass_fracs', [0.70, 0.50, 0.30])
    min_bin_count = float(sculpt_kw.get('min_bin_count', 32) or 32)
    weighting = sculpt_kw.get('bin_weighting', 'occupancy')
    if mass_bins is None:
        mass_bins = sculpt_kw.get('mass_bins', list(range(180, 361, 10)))

    s = torch.softmax(logits, dim=-1)[:, score_index]
    s_cur = s[mask]
    m_cur = mass.reshape(-1).to(device=s.device, dtype=s.dtype)[mask]
    if event_weights is None:
        ew_cur = torch.ones_like(s_cur)
    else:
        ew_cur = event_weights.reshape(-1).to(device=s.device, dtype=s.dtype)[mask].clamp(min=0.0)

    s_hist = m_hist = ew_hist = None
    if buffer is not None and buffer.n > 0:
        s_hist, m_hist, ew_hist = buffer.as_tensors(s.device, s.dtype)

    # Threshold from detached history+current (weighted quantile), no grad through t.
    if s_hist is not None and s_cur.numel() > 0:
        s_for_thr = torch.cat([s_hist, s_cur.detach()], dim=0)
        ew_for_thr = torch.cat([ew_hist, ew_cur.detach()], dim=0)
    elif s_hist is not None:
        s_for_thr = s_hist
        ew_for_thr = ew_hist
    else:
        if s_cur.numel() < 8:
            return s.sum() * 0.0
        s_for_thr = s_cur.detach()
        ew_for_thr = ew_cur.detach()

    if s_for_thr.numel() < 8:
        if buffer is not None and s_cur.numel() > 0:
            buffer.append(s_cur, m_cur, ew_cur)
        return s.sum() * 0.0

    bin_edges = torch.as_tensor(mass_bins, device=s.device, dtype=s.dtype)
    n_bins = len(mass_bins) - 1

    losses = []
    for p in pass_fracs:
        p = float(p)
        t = _weighted_quantile(s_for_thr, ew_for_thr, 1.0 - p)

        # Soft pass * event weight; history detached, current differentiable via softpass.
        parts_sw = []  # sample_weight * soft_pass
        parts_ew = []  # sample_weight
        parts_idx = []
        if s_hist is not None:
            soft_h = torch.sigmoid((s_hist - t) / max(tau, 1e-6))
            idx_hist = torch.bucketize(m_hist, bin_edges) - 1
            valid_h = (idx_hist >= 0) & (idx_hist < n_bins)
            parts_sw.append((ew_hist * soft_h)[valid_h])
            parts_ew.append(ew_hist[valid_h])
            parts_idx.append(idx_hist[valid_h])
        if s_cur.numel() > 0:
            soft_c = torch.sigmoid((s_cur - t) / max(tau, 1e-6))
            idx_cur = torch.bucketize(m_cur, bin_edges) - 1
            valid_c = (idx_cur >= 0) & (idx_cur < n_bins)
            parts_sw.append((ew_cur * soft_c)[valid_c])
            parts_ew.append(ew_cur[valid_c])
            parts_idx.append(idx_cur[valid_c])
        if not parts_sw:
            continue
        sw_all = torch.cat(parts_sw, dim=0)
        ew_all = torch.cat(parts_ew, dim=0)
        idx_v = torch.cat(parts_idx, dim=0)
        if idx_v.numel() == 0:
            continue

        h_all = torch.zeros(n_bins, device=s.device, dtype=s.dtype)
        h_pass = torch.zeros(n_bins, device=s.device, dtype=s.dtype)
        h_all.scatter_add_(0, idx_v, ew_all)
        h_pass.scatter_add_(0, idx_v, sw_all)

        # min_bin_count interpreted as minimum total event-weight in the bin
        keep = h_all >= min_bin_count
        if not bool(keep.any()):
            continue
        h_all_k = h_all[keep]
        h_pass_k = h_pass[keep]
        pdf_all = h_all_k / h_all_k.sum().clamp(min=1e-8)
        pdf_pass = h_pass_k / h_pass_k.sum().clamp(min=1e-8)
        R = pdf_pass / pdf_all.clamp(min=1e-8)
        sq = (R - 1.0) ** 2
        if weighting == 'occupancy':
            wt = h_all_k / h_all_k.sum()
            losses.append((sq * wt).sum())
        else:
            losses.append(sq.mean())

    if buffer is not None and s_cur.numel() > 0:
        buffer.append(s_cur, m_cur, ew_cur)

    if not losses:
        return s.sum() * 0.0
    return torch.stack(losses).mean()


# Backward-compatible name used by older call sites / docs.
def soft_sculpt_loss(logits, mass, event_mask, sculpt_kw, mass_bins=None, buffer=None, event_weights=None):
    return soft_pdf_ratio_sculpt_loss(
        logits, mass, event_mask, sculpt_kw, mass_bins=mass_bins, buffer=buffer,
        event_weights=event_weights)


def _sculpt_lambdas(sculpt_kw):
    lam = float(sculpt_kw.get('lambda', 1.0))
    lam_train = sculpt_kw.get('lambda_train', None)
    if lam_train is None:
        lam_train = lam
    else:
        lam_train = float(lam_train)
    return lam, lam_train


def _sculpt_constraint_active(sculpt_kw, epoch):
    """True when soft-sculpt should be added to the training loss this epoch."""
    if sculpt_kw is None or not sculpt_kw.get('enable', False):
        return False
    warmup = int(sculpt_kw.get('warmup_epochs', 0) or 0)
    return int(epoch) >= warmup


def _train_bkg_sculpt_mask(label_cls, tag_mask, sculpt_kw):
    """Mask for soft sculpt on training backgrounds (default: top|qcd), excluding AN sculpt events."""
    indices = sculpt_kw.get('train_bkg_cls_indices', [1, 2])
    label_cls = label_cls.reshape(-1)
    tag_mask = tag_mask.reshape(-1).bool()
    bkg = torch.zeros_like(tag_mask)
    for idx in indices:
        bkg = bkg | (label_cls == int(idx))
    return tag_mask & bkg


def _dual_sculpt_losses(logits, sculpt_mass, is_sculpt, label_cls, tag_mask, sculpt_kw,
                        buffer_an=None, buffer_train=None, event_weights=None):
    """Return (loss_an, loss_train) according to sculpt_on."""
    zero = logits.new_zeros(())
    if sculpt_kw is None or not sculpt_kw.get('enable', False) or sculpt_mass is None:
        return zero, zero
    sculpt_on = sculpt_kw.get('sculpt_on', 'an')
    loss_an = zero
    loss_train = zero
    if sculpt_on in ('an', 'both') and is_sculpt is not None:
        loss_an = soft_pdf_ratio_sculpt_loss(
            logits, sculpt_mass, is_sculpt, sculpt_kw,
            mass_bins=sculpt_kw.get('mass_bins'),
            buffer=buffer_an,
            event_weights=event_weights,
        )
    if sculpt_on in ('train_bkg', 'both'):
        mask_train = _train_bkg_sculpt_mask(label_cls, tag_mask, sculpt_kw)
        loss_train = soft_pdf_ratio_sculpt_loss(
            logits, sculpt_mass, mask_train, sculpt_kw,
            mass_bins=sculpt_kw.get('mass_bins_train'),
            buffer=buffer_train,
            event_weights=event_weights,
        )
    return loss_an, loss_train


def _np_weighted_quantile(values, weights, q):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = np.clip(weights[order], 0.0, None)
    cdf = np.cumsum(w)
    total = cdf[-1]
    if total <= 0:
        return float(np.quantile(values, q))
    cdf = cdf / total
    return float(v[min(int(np.searchsorted(cdf, q)), len(v) - 1)])


def hard_sculpt_metrics(scores, masses, pass_fracs, mass_bins, min_bin_count=1, weights=None):
    """Hard cut/no-cut PDF-ratio metrics matching plot_mass_sculpting_xggg.

    When ``weights`` is given, histograms / quantiles use Σw (not raw counts).
    Returns dict with per-WP mae_|R-1| and mse_(R-1)^2 plus means.
    """
    scores = np.asarray(scores, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(scores, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != scores.shape[0]:
            weights = np.ones_like(scores, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
    out = {}
    maes, mses = [], []
    for p in pass_fracs:
        p = float(p)
        key = f'pass_{int(round(p * 100)):02d}'
        if scores.size < 8:
            out[f'{key}_mae'] = float('nan')
            out[f'{key}_mse'] = float('nan')
            continue
        t = _np_weighted_quantile(scores, weights, 1.0 - p)
        passed = scores >= t
        h_all, _ = np.histogram(masses, bins=mass_bins, weights=weights)
        h_pass, _ = np.histogram(masses[passed], bins=mass_bins, weights=weights[passed])
        valid = h_all >= float(min_bin_count)
        if not np.any(valid) or h_pass.sum() <= 0:
            out[f'{key}_mae'] = float('nan')
            out[f'{key}_mse'] = float('nan')
            continue
        pdf_all = h_all.astype(np.float64) / max(h_all.sum(), 1e-10)
        pdf_pass = h_pass.astype(np.float64) / max(h_pass.sum(), 1e-10)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.ones_like(pdf_all)
            R[valid] = pdf_pass[valid] / np.maximum(pdf_all[valid], 1e-10)
        mae = float(np.mean(np.abs(R[valid] - 1.0)))
        mse = float(np.mean((R[valid] - 1.0) ** 2))
        out[f'{key}_mae'] = mae
        out[f'{key}_mse'] = mse
        maes.append(mae)
        mses.append(mse)
    out['mae_mean'] = float(np.mean(maes)) if maes else float('nan')
    out['mse_mean'] = float(np.mean(mses)) if mses else float('nan')
    return out


def hard_sculpt_mae(scores, masses, pass_fracs, mass_bins, weights=None):
    """Hard cut/no-cut MAE(|R-1|) for validation monitoring (numpy)."""
    metrics = hard_sculpt_metrics(
        scores, masses, pass_fracs, mass_bins, min_bin_count=1, weights=weights)
    out = {}
    for p in pass_fracs:
        key = f'pass_{int(round(float(p) * 100)):02d}'
        out[key] = metrics.get(f'{key}_mae', float('nan'))
    return out


def _log_hard_sculpt_metrics(scores, masses, sculpt_kw, mass_bins, log_prefix, epoch,
                             tb_helper=None, weights=None):
    if scores is None or masses is None:
        return
    scores = np.asarray(scores)
    masses = np.asarray(masses)
    if scores.size < 8:
        return
    min_bin = float(sculpt_kw.get('min_bin_count', 32) or 1)
    metrics = hard_sculpt_metrics(
        scores, masses,
        sculpt_kw.get('pass_fracs', [0.70, 0.50, 0.30]),
        mass_bins,
        min_bin_count=min_bin,
        weights=weights,
    )
    _logger.info(
        '%s/hard mae_mean=%.5f mse_mean=%.5f (n=%d, sumw=%.1f, min_bin_w=%.3g, finetune_parts_regmass)',
        log_prefix, metrics['mae_mean'], metrics['mse_mean'], scores.size,
        float(np.sum(weights)) if weights is not None else float(scores.size), min_bin)
    for k, v in metrics.items():
        if k in ('mae_mean', 'mse_mean'):
            continue
        if np.isfinite(v):
            _logger.info('%s/%s: %.5f', log_prefix, k, v)
    if tb_helper:
        tb_helper.write_scalars([
            (f'{log_prefix}/{k} (epoch)', v, epoch)
            for k, v in metrics.items() if np.isfinite(v)
        ])


def _log_hard_sculpt_mae(scores_list, masses_list, sculpt_kw, mass_bins, log_prefix, epoch,
                         tb_helper=None, weights_list=None):
    if not scores_list:
        return
    s_all = np.concatenate(scores_list)
    m_all = np.concatenate(masses_list)
    w_all = np.concatenate(weights_list) if weights_list else None
    _log_hard_sculpt_metrics(
        s_all, m_all, sculpt_kw, mass_bins, log_prefix, epoch, tb_helper, weights=w_all)


def _log_buffer_hard_sculpt(buffer, sculpt_kw, mass_bins, log_prefix, epoch, tb_helper=None):
    if buffer is None or buffer.n < 8:
        return
    _log_hard_sculpt_metrics(
        buffer.scores, buffer.masses, sculpt_kw, mass_bins, log_prefix, epoch, tb_helper,
        weights=buffer.weights)


def _tag_subset_hybrid_loss(loss_func, logits, preds_reg, label_cls, label_reg, tag_mask):
    """Run existing hybrid/cls loss only on non-sculpt (tagger) events."""
    if tag_mask is None:
        return loss_func(logits, preds_reg, label_cls, label_reg)

    tag_mask = tag_mask.reshape(-1).bool()
    if not bool(tag_mask.any()):
        zero = logits.sum() * 0.0
        return zero, {'cls': 0.0, 'reg': 0.0}

    logits_t = logits[tag_mask]
    label_cls_t = label_cls[tag_mask]
    preds_reg_t = preds_reg[tag_mask] if preds_reg is not None and preds_reg.numel() > 0 else preds_reg
    label_reg_t = label_reg[tag_mask] if label_reg is not None else None
    return loss_func(logits_t, preds_reg_t, label_cls_t, label_reg_t)


def _read_sculpt_from_Z(Z, dev):
    """Return is_sculpt and kinematics needed for finetune reg mass."""
    if Z is None or 'is_sculpt' not in Z:
        return None, None
    is_sculpt = Z['is_sculpt'].to(dev).float().reshape(-1)
    need = ('jet_pt', 'jet_corr_pt', 'fj_mass')
    if not all(k in Z for k in need):
        return is_sculpt, None
    kin = {
        'jet_pt': Z['jet_pt'].to(dev).float().reshape(-1),
        'jet_corr_pt': Z['jet_corr_pt'].to(dev).float().reshape(-1),
        'fj_mass': Z['fj_mass'].to(dev).float().reshape(-1),
    }
    return is_sculpt, kin


def _compute_sculpt_mass(preds_reg, kin, sculpt_kw, logits_for_zero=None):
    """Build finetune parts reg mass, or None if reg head / kinematics unavailable."""
    global _SCULPT_NO_REG_WARNED
    if kin is None or sculpt_kw is None or not sculpt_kw.get('enable', False):
        return None
    reg_index = int(sculpt_kw.get('reg_index', 1))
    if preds_reg is None or preds_reg.ndim < 2 or preds_reg.shape[1] <= reg_index:
        if not _SCULPT_NO_REG_WARNED:
            _logger.warning(
                'sculpt mass_source=finetune_parts requires preds_reg[:, %d]; '
                'soft/hard sculpt skipped (e.g. cls-only mode).', reg_index)
            _SCULPT_NO_REG_WARNED = True
        return None
    return finetune_reg_mass(
        preds_reg[:, reg_index],
        kin['jet_pt'],
        kin['jet_corr_pt'],
        kin['fj_mass'],
    )


ParticleTransformerTagger_ncoll = import_module(os.path.join(os.path.dirname(__file__), 'ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformerTagger_ncoll


class ParticleTransformerTaggerForFinetune(nn.Module):
    def __init__(self, finetune_kw=dict(), **kwargs) -> None:
        '''
            finetune_kw (dict): fine-tuning configurations
            - mode (str): fine-tuning mode, 'cls' for classification, 'hybrid' for classification + regression,
              'reg.guass' for regression with Gaussian NLL loss
            - input_highlevel_dim (int): dimension of the high-level input features
            - target_inds: list of target indices for the fine-tuning; can be a list of integers, a single integer, 'all', None
            - num_ft_nodes (int): number of output nodes of the external FC layer
            - freeze_main_params (bool): whether to freeze the main model parameters
            - fc_params (list): list of tuples (dim, dropout) of the FC layers
            - fc_suff_kw (dict): suffix FC configurations
                 - append_after (str): 'output', 'hidden', 'fc.0'
                 - params (list): list of tuples (dim, dropout) of the FC layers
        '''

        super().__init__()
        self.for_inference = kwargs.get('for_inference')

        # main model
        self.main = ParticleTransformerTagger_ncoll(**kwargs)

        # external FC
        self.mode = finetune_kw.get('mode') # mode of fine-tuning, determine which loss function etc to use
        self.input_highlevel_dim = finetune_kw.get('input_highlevel_dim')
        self.target_inds = finetune_kw.get('target_inds')
        if self.target_inds == 'all':
            self.target_inds = list(range(kwargs['num_classes']))
        elif isinstance(self.target_inds, int):
            self.target_inds = [self.target_inds]
        self.target_inds_opt = finetune_kw.get('target_inds_opt', None)

        self.num_ft_nodes = finetune_kw.get('num_ft_nodes')
        self.num_ft_cls_nodes = finetune_kw.get('num_ft_cls_nodes', self.num_ft_nodes)
        self.target_reg_inds = finetune_kw.get('target_reg_inds', None)
        if self.mode == 'hybrid' and self.target_reg_inds is None:
            self.target_reg_inds = list(range(finetune_kw['num_cls_nodes'], kwargs['num_classes']))
        if isinstance(self.target_reg_inds, int):
            self.target_reg_inds = [self.target_reg_inds]
        self.num_ft_reg_nodes = finetune_kw.get('num_ft_reg_nodes', len(self.target_reg_inds) if self.target_reg_inds is not None else 0)
        self.freeze_main_params = finetune_kw.get('freeze_main_params', True)

        # If requested, fully freeze the main model parameters so DDP does not
        # expect gradients for them. This avoids "unused parameter" issues when
        # fine-tuning with external heads that bypass parts of the main network.
        if self.freeze_main_params:
            for p in self.main.parameters():
                p.requires_grad = False

        fc_params = finetune_kw.get('fc_params')
        self.fc_suff_kw = finetune_kw.get('fc_suff_kw', None)

        fcs = []
        in_dim = kwargs['embed_dims'][-1] + self.input_highlevel_dim # concat high-level input dims to the embed layer
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, self.num_ft_nodes)) # dim -> num_ft_nodes
        self.fc = nn.Sequential(*fcs)

        # suffix FC after the main model; appended after output (slicing by target_inds) or the last hidden layer
        if self.fc_suff_kw is not None:
            fcs = []
            append_after = self.fc_suff_kw.get('append_after', 'output')
            if append_after == 'output':
                in_dim = len(self.target_inds)
            elif append_after == 'hidden':
                in_dim = kwargs['embed_dims'][-1]
            elif append_after == 'fc.0':
                in_dim = kwargs['fc_params'][0][0]
            else:
                raise ValueError('Invalid append_after value')
            for out_dim, drop_rate in self.fc_suff_kw.get('params'):
                fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, self.num_ft_nodes)) # dim -> num_ft_nodes
            self.fc_suff = nn.Sequential(*fcs)
        else:
            self.fc_suff = None

    def forward(self, *args):
        if self.freeze_main_params:
            # freeze the main model
            # this is important as it also freezes the running stats of the batchnorm layers
            self.main.eval()

        # process main model
        if self.input_highlevel_dim > 0:
            output, x = self.main(*args[:-1])
            xcat = torch.cat([x, args[-1].squeeze(2)], dim=1)
        else:
            output, x = self.main(*args)
            xcat = x

        if self.mode == 'hybrid':
            if self.target_inds is None:
                output_cls = output.new_zeros((output.size(0), self.num_ft_cls_nodes))
            else:
                output_cls = output[:, self.target_inds]
                if self.target_inds_opt == 'sum':
                    output_cls = output_cls.sum(dim=1, keepdim=True)
            output_reg = output[:, self.target_reg_inds]
            output = torch.cat([output_cls, output_reg], dim=1)

            with torch.autocast('cuda', enabled=self.main.use_amp):
                if self.fc_suff is not None:
                    append_after = self.fc_suff_kw.get('append_after')
                    if append_after == 'output':
                        output_resid = self.fc_suff(output)
                    elif append_after == 'hidden':
                        output_resid = self.fc_suff(x)
                    elif append_after == 'fc.0':
                        output_resid = self.main.part.fc[0](x)
                        output_resid = self.fc_suff(output_resid)
                    else:
                        raise ValueError('Invalid append_after value')
                else:
                    output_resid = 0
                output_resid = output_resid + self.fc(xcat)

            output = output + output_resid

            if self.for_inference:
                output_cls, output_reg = output.split([self.num_ft_cls_nodes, self.num_ft_reg_nodes], dim=1)
                output_cls = torch.softmax(output_cls, dim=1)
                output = torch.cat([output_cls, output_reg], dim=1)

            if self.training and torch.is_grad_enabled():
                with torch.autocast('cuda', enabled=self.main.use_amp):
                    zero = torch.zeros((), device=output.device, dtype=output.dtype)
                    for p in self.parameters():
                        if p.requires_grad:
                            zero = zero + (p.view(-1)[0] * 0.0)
                    output = output + zero
            return output

        # slicing the output
        if self.target_inds is not None:
            output = output[:, self.target_inds]
            if self.target_inds_opt == 'sum':
                output = output.sum(dim=1, keepdim=True)
        else:
            output = 0

        # process suffix FC (if valid) 
        # -> modify "output" if new layers (fc_suff) are appended after the main model output / after intermediate hidden/fc.0 layers
        with torch.autocast('cuda', enabled=self.main.use_amp):
            if self.fc_suff is not None:
                append_after = self.fc_suff_kw.get('append_after')
                if append_after == 'output':
                    output = self.fc_suff(output)
                elif append_after == 'hidden':
                    output = self.fc_suff(x)
                elif append_after == 'fc.0':
                    output = self.main.part.fc[0](x)
                    output = self.fc_suff(output)
                else:
                    raise ValueError('Invalid append_after value')

        # process FC
        with torch.autocast('cuda', enabled=self.main.use_amp):
            output_fc = self.fc(xcat)

        # use FC nodes as residual to main outputs
        # note for the special treatment for different fine-tuning modes
        if self.mode == 'reg.guass':
            mu, log_var = output_fc.split(1, dim=1)
            # mu as the residual to the main model output (massCorr + massCorrResid)
            mu = mu + output
            output = torch.cat([mu, log_var], dim=1)
        # elif self.mode == 'reg.guass.fixvar':
        #     mu = output_fc
        #     # mu as the residual to the main model output (massCorr + massCorrResid)
        #     mu = mu + output
        #     log_var = (torch.zeros_like(mu) + 1).log()
        #     output = torch.cat([mu, log_var], dim=1)
        else:
            # FC output as the residual to the main model output
            output = output + output_fc

        if self.for_inference:
            if self.mode == 'cls':
                output = torch.softmax(output, dim=1)
        
        # Ensure all trainable parameters participate in the autograd graph on every forward.
        # This avoids DistributedDataParallel error: "Expected to have finished reduction..."
        # which can happen when some parameters (e.g. heads not used under certain fine-tuning
        # configurations) do not receive gradients in a particular iteration.
        # By adding a zero-valued dependency on every trainable parameter, we keep numerical
        # outputs unchanged while guaranteeing graph connectivity for DDP.
        if self.training and torch.is_grad_enabled():
            with torch.autocast('cuda', enabled=self.main.use_amp):
                zero = torch.zeros((), device=output.device, dtype=output.dtype)
                for p in self.parameters():
                    if p.requires_grad:
                        # zero-valued dependency; does not change output but keeps params in graph
                        zero = zero + (p.view(-1)[0] * 0.0)
                output = output + zero
        return output


def get_model(data_config, **kwargs):
    assert 'num_nodes' in kwargs, 'num_nodes must be provided'
    assert 'num_cls_nodes' in kwargs, 'num_cls_nodes must be provided'
    num_nodes = kwargs.pop('num_nodes')
    num_cls_nodes = kwargs.pop('num_cls_nodes')
    label_cls_nodes = kwargs.pop('label_cls_nodes', None)
    label_stored = kwargs.pop('label_stored', None)
    reg_kw = kwargs.pop('reg_kw', dict())
    finetune_kw = kwargs.pop('finetune_kw', None)
    eval_kw = kwargs.pop('eval_kw', dict())
    # consumed by get_train_fn / get_evaluate_fn; do not pass into the network ctor
    kwargs.pop('sculpt_kw', None)

    # use SwiGLU-default setup
    cfg = dict(
        input_dims=tuple(map(lambda x: len(data_config.input_dicts[x]), ['cpf_features', 'npf_features', 'sv_features'])),
        share_embed=False,
        num_classes=num_nodes,
        # network configurations
        pair_input_type='pp',
        pair_input_dim=4,
        pair_extra_dim=0,
        use_pair_norm=False,
        remove_self_pair=False,
        use_pre_activation_pair=True,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={},
        fc_params=(),
        activation='gelu',
        # GloParT wrapper configurations
        # input_highlevel_dim=len(data_config.input_dicts.get('jet_features', [])),
        # use_external_fc=False,
        # misc
        trim=True,
        for_inference=False,
    )

    if kwargs.pop('use_swiglu_config', False):
        cfg.update(
            block_params={"scale_attn_mask": True, "scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
            cls_block_params={"scale_attn": False, "scale_fc": False, "scale_heads": False, "scale_resids": False, "activation": "swiglu"},
        )
    if kwargs.pop('use_pair_norm_config', False):
        cfg.update(
            use_pair_norm=True,
            pair_input_dim=6,
        )

    cfg.update(**kwargs)

    if finetune_kw is None:
        model = ParticleTransformerTagger_ncoll(**cfg)
    else:
        # finetune mode
        assert finetune_kw.get('mode') is not None, 'mode must be provided in finetune_kw'
        finetune_kw.update(
            input_highlevel_dim=len(data_config.input_dicts.get('jet_features', [])),
            num_cls_nodes=num_cls_nodes,
        )
        cfg.update(
            finetune_kw=finetune_kw,
            return_embed=True, # return the last embed layer before FC
        )
        model = ParticleTransformerTaggerForFinetune(**cfg)

    # set special args
    model.num_nodes = num_nodes
    model.num_cls_nodes = finetune_kw.get('num_ft_cls_nodes', num_cls_nodes) if finetune_kw is not None and finetune_kw.get('mode') == 'hybrid' else num_cls_nodes
    model.eval_kw = eval_kw

    _logger.info('Model config: %s' % str(cfg))

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['output'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'output': {0: 'N'}}},
    }

    return model, model_info


class LogCoshLoss(torch.nn.L1Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', split_reg=False):
        super(LogCoshLoss, self).__init__(None, None, reduction)
        self.split_reg = split_reg

    def forward(self, input, target_reg, target_cls=None, n_cls=None):

        if not self.split_reg:
            x = input - target_reg # dim: (B, n_reg)
            loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
        
        else:
            # calculate regression loss for each class separately
            # input: (N, C * n_reg), target_reg: (N, n_reg), target_cls: (N)
            n_reg = target_reg.shape[1]
            input = input.view(-1, n_cls, n_reg)
            target_reg = target_reg.view(-1, 1, n_reg)
            target_cls = torch.nn.functional.one_hot(target_cls, num_classes=n_cls).bool().view(-1, n_cls, 1)
            x = input - target_reg # dim: (B, n_cls, n_reg)
            loss = x + torch.nn.functional.softplus(-2. * x) - math.log(2)
            loss = (loss * target_cls).sum(dim=1) # dim: (B, n_reg)
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean(dim=0)
        elif self.reduction == 'sum':
            return loss.sum(dim=0)


class HybridLoss(torch.nn.Module):

    def __init__(self, reduction='mean', gamma=1., split_reg=False):
        super().__init__()
        self.loss_cls_fn = torch.nn.CrossEntropyLoss()
        self.loss_reg_fn = LogCoshLoss(reduction=reduction, split_reg=split_reg)
        self.gamma = gamma

    def forward(self, input_cls, input_reg, target_cls, target_reg):

        loss_cls = self.loss_cls_fn(input_cls, target_cls)
        loss_reg = self.loss_reg_fn(input_reg, target_reg, target_cls=target_cls, n_cls=input_cls.shape[1])
        loss = loss_cls + self.gamma * loss_reg.sum()
        loss_dict = {'cls': loss_cls.item(), 'reg': loss_reg.sum().item()}
        loss_dict.update({f'reg_{i}': loss_reg[i].item() for i in range(loss_reg.shape[0])})
        return loss, loss_dict


class ComposedHybridLoss(torch.nn.Module):

    def __init__(self, reduction='mean', gamma=1., composed_split_reg=None, as_resid_of=None):
        # composed_split_reg: lists of True/False; True means using this regression target for split-class regression
        # as_resid_of: the index of the unified regression target that the split-class regression target is a residual of
        split = composed_split_reg
        unifd = [True] * len(split) # enable all regression targets for unified regression
        assert any(unifd) and any(split), 'At least one regression target must be used for unified and split regression'

        super().__init__()
        self.loss_cls_fn = torch.nn.CrossEntropyLoss()
        self.loss_reg_unifd_fn = LogCoshLoss(reduction=reduction, split_reg=False)
        self.loss_reg_split_fn = LogCoshLoss(reduction=reduction, split_reg=True)
        self.unifd = unifd
        self.split = split
        self.num_unifd = sum(unifd)
        self.gamma = gamma
        self.as_resid_of = as_resid_of

    def forward(self, input_cls, input_reg, target_cls, target_reg):

        loss_cls = self.loss_cls_fn(input_cls, target_cls)

        # regression inputs
        input_reg_unifd = input_reg[:, :self.num_unifd]
        input_reg_split = input_reg[:, self.num_unifd:]

        # compute unified regression loss
        n_target_reg = target_reg.shape[1]
        loss_reg_unifd = self.loss_reg_unifd_fn(input_reg_unifd, target_reg[:, self.unifd])

        # compute split-class regression loss. Only do split-class regression for specific targets defined by composed_split_reg
        if self.as_resid_of:
            # the split-class reg node is a residual node to the unified reg node
            input_reg_split = input_reg_split + input_reg_unifd[:, self.as_resid_of]
        loss_reg_split = self.loss_reg_split_fn(input_reg_split, target_reg[:, self.split], target_cls=target_cls, n_cls=input_cls.shape[1])

        loss = loss_cls + self.gamma * (loss_reg_unifd.sum() + loss_reg_split.sum())
        loss_dict = {'cls': loss_cls.item(), 'reg_unifd': loss_reg_unifd.sum().item(), 'reg_split': loss_reg_split.sum().item(), 'reg': loss_reg_unifd.sum().item() + loss_reg_split.sum().item()}
        return loss, loss_dict


class CrossEntropyLossHybridWrapper(torch.nn.CrossEntropyLoss):
    def forward(self, *args):
        if len(args) == 4:
            input_cls, _, target_cls, _ = args
        else:
            input_cls, target_cls = args
        loss = super().forward(input_cls, target_cls)
        return loss, {'cls': loss.item(), 'reg': 0.}


class GuassianNLLLoss(torch.nn.GaussianNLLLoss):
    def forward(self, input, target):
        mu, log_var = input.split(1, dim=-1)
        return super().forward(mu.squeeze(-1), target, log_var.squeeze(-1).exp()) # must ensure input and target have the same shape...


def get_loss(data_config, **kwargs):
    if kwargs.get('finetune_kw', None) is None:
        reg_kw = kwargs.get('reg_kw', dict())
        gamma = reg_kw.get('gamma', 1)
        split_reg = reg_kw.get('split_reg', False)
        composed_split_reg = reg_kw.get('composed_split_reg', None)
        as_resid_of = reg_kw.get('as_resid_of', False)
        if gamma == 0:
            return CrossEntropyLossHybridWrapper()
        else:
            if composed_split_reg is None:
                return HybridLoss(gamma=gamma, split_reg=split_reg)
            else:
                return ComposedHybridLoss(gamma=gamma, composed_split_reg=composed_split_reg, as_resid_of=as_resid_of)
    else:
        # fine-tune mode, determine the loss function based on the mode
        mode = kwargs.get('finetune_kw').get('mode')
        if mode == 'cls':
            return nn.CrossEntropyLoss()
        elif mode == 'hybrid':
            reg_kw = kwargs.get('reg_kw', dict())
            gamma = reg_kw.get('gamma', 1)
            split_reg = reg_kw.get('split_reg', False)
            composed_split_reg = reg_kw.get('composed_split_reg', None)
            as_resid_of = reg_kw.get('as_resid_of', False)
            if gamma == 0:
                return CrossEntropyLossHybridWrapper()
            elif composed_split_reg is None:
                return HybridLoss(gamma=gamma, split_reg=split_reg)
            else:
                return ComposedHybridLoss(gamma=gamma, composed_split_reg=composed_split_reg, as_resid_of=as_resid_of)
        elif mode == 'reg':
            return LogCoshLoss()
        elif mode == 'reg.mse':
            return nn.MSELoss()
        elif mode == 'reg.guass':
            return GuassianNLLLoss()
        else:
            return None


def get_train_fn(data_config, **kwargs):
    sculpt_kw = _resolve_sculpt_kw(kwargs.get('sculpt_kw', None))
    finetune_kw = kwargs.get('finetune_kw', None)
    if finetune_kw is None:
        return partial(train_hybrid, sculpt_kw=sculpt_kw)
    else:
        mode = finetune_kw.get('mode')
        if mode == 'cls':
            return partial(train_classification, sculpt_kw=sculpt_kw)
        elif mode == 'hybrid':
            return partial(train_hybrid, sculpt_kw=sculpt_kw)
        elif mode in ['reg', 'reg.mse']:
            return train_regression
        elif mode == 'reg.guass':
            return train_guass_regression


def get_evaluate_fn(data_config, **kwargs):
    sculpt_kw = _resolve_sculpt_kw(kwargs.get('sculpt_kw', None))
    finetune_kw = kwargs.get('finetune_kw', None)
    if finetune_kw is None:
        return partial(evaluate_hybrid, sculpt_kw=sculpt_kw)
    else:
        mode = finetune_kw.get('mode')
        if mode == 'cls':
            return partial(evaluate_classification, sculpt_kw=sculpt_kw)
        elif mode == 'hybrid':
            return partial(evaluate_hybrid, sculpt_kw=sculpt_kw)
        elif mode in ['reg', 'reg.mse']:
            return evaluate_regression
        elif mode == 'reg.guass':
            return evaluate_guass_regression


def get_save_fn(data_config, **kwargs):
    finetune_kw = kwargs.get('finetune_kw', None)
    if finetune_kw is None:
        return save_hybrid
    else:
        mode = finetune_kw.get('mode')
        if mode == 'hybrid':
            return save_hybrid
        if mode == 'reg.guass':
            return save_guass_regression
        return None


#### ================== Custom train/eval/save functions ================== ####

def train_hybrid(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, train_loss=None, tb_helper=None, sculpt_kw=None):
    model.train()

    data_config = train_loader.dataset.config
    sculpt_kw = _resolve_sculpt_kw(sculpt_kw)
    sculpt_active = _sculpt_constraint_active(sculpt_kw, epoch)
    buffer_an, buffer_train = _make_sculpt_buffers(sculpt_kw)
    if sculpt_kw and sculpt_kw.get('enable', False):
        warmup = int(sculpt_kw.get('warmup_epochs', 0) or 0)
        if sculpt_active:
            _logger.info(
                'Soft-sculpt constraint ON (epoch=%d >= warmup_epochs=%d); '
                'pdf-ratio loss buffer_size=%d min_bin_count=%d tau=%.3g',
                epoch, warmup,
                int(sculpt_kw.get('buffer_size', 65536)),
                int(sculpt_kw.get('min_bin_count', 32)),
                float(sculpt_kw.get('tau', 0.02)))
        else:
            _logger.info(
                'Soft-sculpt warm-up: constraint OFF (epoch=%d < warmup_epochs=%d); '
                'sculpt loss logged only', epoch, warmup)

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    total_loss_sculpt = 0
    total_loss_sculpt_an = 0
    total_loss_sculpt_train = 0
    total_loss_reg_i = defaultdict(float)
    total_loss_reg_split = 0
    total_loss_reg_unifd = 0
    num_batches = 0
    total_correct = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    count_tag = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, Z in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            # for classification
            label_cls = y['truth_label'].long() # _label_ -> truth_label
            label_counter.update(label_cls.cpu().numpy())
            label_cls = label_cls.to(dev)
            n_cls = model.module.num_cls_nodes if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.num_cls_nodes
            num_examples = label_cls.shape[0]

            is_sculpt, sculpt_kin = _read_sculpt_from_Z(Z, dev)
            if is_sculpt is None:
                tag_mask = torch.ones(num_examples, dtype=torch.bool, device=dev)
            else:
                tag_mask = ~(is_sculpt > 0.5)

            # for regression
            if len(data_config.label_names) > 1:
                label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]] # can support multiple regression target
                label_reg = torch.cat(label_reg, dim=1)
            else:
                label_reg = None
            n_reg_target = len(data_config.label_names) - 1

            opt.zero_grad()
            # with torch.autograd.detect_anomaly():
            with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits = model_output[:, :n_cls]
                preds_reg = model_output[:, n_cls:]
                loss_tag, loss_monitor = _tag_subset_hybrid_loss(
                    loss_func, logits, preds_reg, label_cls, label_reg, tag_mask)
                event_w = _read_event_weights(
                    Z, num_examples, logits.device, logits.dtype, sculpt_kw,
                    data_config=data_config, label_cls=label_cls)
                if sculpt_active:
                    sculpt_mass = _compute_sculpt_mass(preds_reg, sculpt_kin, sculpt_kw)
                    loss_sculpt_an, loss_sculpt_train = _dual_sculpt_losses(
                        logits, sculpt_mass, is_sculpt, label_cls, tag_mask, sculpt_kw,
                        buffer_an=buffer_an, buffer_train=buffer_train, event_weights=event_w)
                    lam, lam_train = _sculpt_lambdas(sculpt_kw) if sculpt_kw else (1.0, 1.0)
                    loss_sculpt = loss_sculpt_an + loss_sculpt_train
                    loss = loss_tag + lam * loss_sculpt_an + lam_train * loss_sculpt_train
                else:
                    # warm-up: monitor sculpt only, do not backprop through it
                    with torch.no_grad():
                        sculpt_mass = _compute_sculpt_mass(preds_reg, sculpt_kin, sculpt_kw)
                        loss_sculpt_an, loss_sculpt_train = _dual_sculpt_losses(
                            logits, sculpt_mass, is_sculpt, label_cls, tag_mask, sculpt_kw,
                            buffer_an=buffer_an, buffer_train=buffer_train, event_weights=event_w)
                        loss_sculpt = loss_sculpt_an + loss_sculpt_train
                    loss = loss_tag
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds_cls = logits.max(1)
            loss = loss.item()
            loss_sculpt_an_val = float(loss_sculpt_an.item()) if torch.is_tensor(loss_sculpt_an) else float(loss_sculpt_an)
            loss_sculpt_train_val = float(loss_sculpt_train.item()) if torch.is_tensor(loss_sculpt_train) else float(loss_sculpt_train)
            loss_sculpt_val = loss_sculpt_an_val + loss_sculpt_train_val

            num_batches += 1
            count += num_examples
            n_tag = int(tag_mask.sum().item())
            count_tag += n_tag
            if n_tag > 0:
                correct = (preds_cls[tag_mask] == label_cls[tag_mask]).sum().item()
            else:
                correct = 0
 
            total_loss += loss
            total_loss_cls += loss_monitor['cls']
            total_loss_reg += loss_monitor.get('reg', 0.0)
            total_loss_sculpt += loss_sculpt_val
            total_loss_sculpt_an += loss_sculpt_an_val
            total_loss_sculpt_train += loss_sculpt_train_val
            if 'reg_split' in loss_monitor:
                total_loss_reg_split += loss_monitor['reg_split']
                total_loss_reg_unifd += loss_monitor['reg_unifd']
            elif n_reg_target > 1:
                for i in range(n_reg_target):
                    if f'reg_{i}' in loss_monitor:
                        total_loss_reg_i[i] += loss_monitor[f'reg_{i}']
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss_monitor['cls'],
                'LossReg': '%.5f' % loss_monitor.get('reg', 0.0),
                'LossSculptSoftAN': '%.5f' % loss_sculpt_an_val,
                'LossSculptSoftTrain': '%.5f' % loss_sculpt_train_val,
                'LossSculptSoft': '%.5f' % loss_sculpt_val,
                'LossTot': '%.5f' % loss,
                'Acc': '%.5f' % (correct / max(n_tag, 1)),
            })

            # stop writing to tensorboard after 500 batches
            if tb_helper and num_batches < 500:
                tb_helper.write_scalars([
                    ("Loss/train", loss_monitor['cls'], tb_helper.batch_train_count + num_batches), # to compare cls loss to previous loss
                    ("LossReg/train", loss_monitor.get('reg', 0.0), tb_helper.batch_train_count + num_batches),
                    ("LossSculptSoftAN/train", loss_sculpt_an_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptSoftTrain/train", loss_sculpt_train_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptSoft/train", loss_sculpt_val, tb_helper.batch_train_count + num_batches),
                    # keep old names for TB continuity
                    ("LossSculptAN/train", loss_sculpt_an_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptTrain/train", loss_sculpt_train_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculpt/train", loss_sculpt_val, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / max(n_tag, 1), tb_helper.batch_train_count + num_batches),
                    ])
                if 'reg_split' in loss_monitor:
                    tb_helper.write_scalars([
                        ("LossRegSplit/train", loss_monitor['reg_split'], tb_helper.batch_train_count + num_batches),
                        ("LossRegUnifd/train", loss_monitor['reg_unifd'], tb_helper.batch_train_count + num_batches),
                        ])
                elif n_reg_target > 1:
                    for i in range(n_reg_target):
                        if f'reg_{i}' in loss_monitor:
                            tb_helper.write_scalars([
                                (f"LossReg{i}/train", loss_monitor[f'reg_{i}'], tb_helper.batch_train_count + num_batches),
                                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info(
        'Train AvgLoss: %.5f, AvgLossReg: %.5f, AvgLossSculptSoftAN: %.5f, AvgLossSculptSoftTrain: %.5f, '
        'AvgLossSculptSoft: %.5f, AvgLossTot: %.5f, AvgAcc: %.5f' %
        (total_loss_cls / num_batches, total_loss_reg / num_batches,
         total_loss_sculpt_an / num_batches, total_loss_sculpt_train / num_batches,
         total_loss_sculpt / num_batches, total_loss / num_batches, total_correct / max(count_tag, 1)))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    if sculpt_kw and sculpt_kw.get('enable', False):
        _log_buffer_hard_sculpt(
            buffer_an, sculpt_kw, sculpt_kw.get('mass_bins', list(range(180, 361, 10))),
            'SculptHard_train_AN', epoch, tb_helper)
        _log_buffer_hard_sculpt(
            buffer_train, sculpt_kw, sculpt_kw.get('mass_bins_train', list(range(20, 361, 10))),
            'SculptHard_train_trainbkg', epoch, tb_helper)

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss_cls / num_batches, epoch), # to compare cls loss to previous loss
            ("LossReg/train (epoch)", total_loss_reg / num_batches, epoch),
            ("LossSculptSoftAN/train (epoch)", total_loss_sculpt_an / num_batches, epoch),
            ("LossSculptSoftTrain/train (epoch)", total_loss_sculpt_train / num_batches, epoch),
            ("LossSculptSoft/train (epoch)", total_loss_sculpt / num_batches, epoch),
            ("LossSculptAN/train (epoch)", total_loss_sculpt_an / num_batches, epoch),
            ("LossSculptTrain/train (epoch)", total_loss_sculpt_train / num_batches, epoch),
            ("LossSculpt/train (epoch)", total_loss_sculpt / num_batches, epoch),
            ("LossTot/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / max(count_tag, 1), epoch),
            ])
        if 'reg_split' in loss_monitor:
            tb_helper.write_scalars([
                ("LossRegSplit/train (epoch)", total_loss_reg_split / num_batches, epoch),
                ("LossRegUnifd/train (epoch)", total_loss_reg_unifd / num_batches, epoch),
                ])
        elif n_reg_target > 1:
            for i in range(n_reg_target):
                tb_helper.write_scalars([
                    (f"LossReg{i}/train (epoch)", total_loss_reg_i[i] / num_batches, epoch),
                    ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_hybrid(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics_cls=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                        eval_metrics_reg=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                          'mean_gamma_deviance'],
                        tb_helper=None, sculpt_kw=None):
    model.eval()

    data_config = test_loader.dataset.config
    sculpt_kw = _resolve_sculpt_kw(sculpt_kw)

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_reg = 0
    total_loss_reg_i = defaultdict(float)
    total_loss_reg_split = 0
    total_loss_reg_unifd = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    count_tag = 0
    scores_cls = []
    scores_reg = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    sculpt_scores = []
    sculpt_masses = []
    sculpt_weights = []
    sculpt_scores_train = []
    sculpt_masses_train = []
    sculpt_weights_train = []
    start_time = time.time()
    model_embed_output_array = []
    label_cls_array = []
    eval_kw = model.module.eval_kw \
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.eval_kw
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                # for classification
                label_cls = y['truth_label'].long() # _label_ -> truth_label
                entry_count += label_cls.shape[0]
                num_examples = label_cls.shape[0]
                label_counter.update(label_cls.cpu().numpy())
                label_cls = label_cls.to(dev)
                n_cls = model.module.num_cls_nodes if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.num_cls_nodes

                is_sculpt, sculpt_kin = _read_sculpt_from_Z(Z, dev)
                if is_sculpt is None:
                    tag_mask = torch.ones(num_examples, dtype=torch.bool, device=dev)
                else:
                    tag_mask = ~(is_sculpt > 0.5)

                # for regression
                if len(data_config.label_names) > 1:
                    label_reg = [y[n].float().to(dev).unsqueeze(1) for n in data_config.label_names[1:]]
                    label_reg = torch.cat(label_reg, dim=1)
                else:
                    label_reg = None
                n_reg_target = len(data_config.label_names) - 1

                model_output = model(*inputs)

                logits = model_output[:, :n_cls].float()
                preds_reg = model_output[:, n_cls:].float()

                if not for_training:
                    scores_cls.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    scores_reg.append(preds_reg.detach().cpu().numpy())
                    for k, v in y.items():
                        labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())
                if for_training and eval_kw.get('roc_kw', None):
                    # for making ROC curves — only tagger events
                    if tag_mask.any():
                        scores_cls.append(torch.softmax(logits[tag_mask], dim=1).detach().cpu().numpy())
                        labels['truth_label'].append(y['truth_label'][tag_mask.cpu()].cpu().numpy())

                if for_training and sculpt_kw and sculpt_kw.get('enable', False):
                    sculpt_mass = _compute_sculpt_mass(preds_reg, sculpt_kin, sculpt_kw)
                    if sculpt_mass is not None:
                        score_index = int(sculpt_kw.get('score_index', 0))
                        s = torch.softmax(logits, dim=1)[:, score_index]
                        event_w = _read_event_weights(
                            Z, num_examples, logits.device, logits.dtype, sculpt_kw,
                            data_config=data_config, label_cls=label_cls)
                        sculpt_on = sculpt_kw.get('sculpt_on', 'an')
                        if sculpt_on in ('an', 'both') and is_sculpt is not None:
                            sculpt_mask = is_sculpt > 0.5
                            if sculpt_mask.any():
                                sculpt_scores.append(s[sculpt_mask].detach().cpu().numpy())
                                sculpt_masses.append(sculpt_mass[sculpt_mask].detach().cpu().numpy())
                                sculpt_weights.append(event_w[sculpt_mask].detach().cpu().numpy())
                        if sculpt_on in ('train_bkg', 'both'):
                            train_mask = _train_bkg_sculpt_mask(label_cls, tag_mask, sculpt_kw)
                            if train_mask.any():
                                sculpt_scores_train.append(s[train_mask].detach().cpu().numpy())
                                sculpt_masses_train.append(sculpt_mass[train_mask].detach().cpu().numpy())
                                sculpt_weights_train.append(event_w[train_mask].detach().cpu().numpy())

                _, preds_cls = logits.max(1)
                n_tag = int(tag_mask.sum().item())
                if for_training:
                    loss, loss_monitor = _tag_subset_hybrid_loss(
                        loss_func, logits, preds_reg, label_cls, label_reg, tag_mask)
                    loss = loss.item()
                else:
                    loss, loss_monitor = 0., {'cls': 0., 'reg': 0.}

                num_batches += 1
                count += num_examples
                count_tag += n_tag
                correct = (preds_cls[tag_mask] == label_cls[tag_mask]).sum().item() if n_tag > 0 else 0
                total_correct += correct
                total_loss += loss * max(n_tag, 1)
                total_loss_cls += loss_monitor['cls'] * max(n_tag, 1)
                total_loss_reg += loss_monitor.get('reg', 0.0) * max(n_tag, 1)
                if 'reg_split' in loss_monitor:
                    total_loss_reg_split += loss_monitor['reg_split'] * max(n_tag, 1)
                    total_loss_reg_unifd += loss_monitor['reg_unifd'] * max(n_tag, 1)
                elif n_reg_target > 1 and for_training:
                    for i in range(n_reg_target):
                        if f'reg_{i}' in loss_monitor:
                            total_loss_reg_i[i] += loss_monitor[f'reg_{i}'] * max(n_tag, 1)

                tq.set_postfix({
                    'Loss': '%.5f' % loss_monitor['cls'],
                    'LossReg': '%.5f' % loss_monitor.get('reg', 0.0),
                    'LossTot': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / max(count_tag, 1)),
                    'Acc': '%.5f' % (correct / max(n_tag, 1)),
                    'AvgAcc': '%.5f' % (total_correct / max(count_tag, 1)),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if for_training and sculpt_kw and sculpt_kw.get('enable', False):
        _log_hard_sculpt_mae(
            sculpt_scores, sculpt_masses, sculpt_kw,
            sculpt_kw.get('mass_bins', list(range(180, 361, 10))),
            'SculptHard_val_AN', epoch, tb_helper, weights_list=sculpt_weights)
        _log_hard_sculpt_mae(
            sculpt_scores_train, sculpt_masses_train, sculpt_kw,
            sculpt_kw.get('mass_bins_train', list(range(20, 361, 10))),
            'SculptHard_val_trainbkg', epoch, tb_helper, weights_list=sculpt_weights_train)

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        denom = max(count_tag, 1)
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss_cls / denom, epoch),
            ("LossReg/%s (epoch)" % tb_mode, total_loss_reg / denom, epoch),
            ("LossTot/%s (epoch)" % tb_mode, total_loss / denom, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / denom, epoch),
            ])
        if 'reg_split' in loss_monitor:
            tb_helper.write_scalars([
                ("LossRegSplit/%s (epoch)" % tb_mode, total_loss_reg_split / denom, epoch),
                ("LossRegUnifd/%s (epoch)" % tb_mode, total_loss_reg_unifd / denom, epoch),
                ])
        elif n_reg_target > 1:
            for i in range(n_reg_target):
                tb_helper.write_scalars([
                    (f"LossReg{i}/{tb_mode} (epoch)", total_loss_reg_i[i] / denom, epoch),
                    ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper and for_training and eval_kw.get('roc_kw', None) and scores_cls:
        roc_kwargs = eval_kw['roc_kw']
        scores_cls_arr = np.concatenate(scores_cls)
        truth_label = np.concatenate(labels['truth_label'])
        scores_dict, flag_dict = {}, {}
        for name, inds in roc_kwargs.get('label_inds_map').items():
            flag_dict[name] = np.any([truth_label == i for i in inds], axis=0)
            scores_dict[name] = np.sum(scores_cls_arr[:, inds], axis=1)
            print(name, flag_dict[name].shape, scores_dict[name].shape)
        comp_list = roc_kwargs.get('comp_list') # e.g. [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        bkgrej = {}

        import matplotlib.pyplot as plt
        import sklearn.metrics as m
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
        ax.set_xlabel('Signal eff.', ha='right', x=1.0); ax.set_ylabel('BKG eff.', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.writer.add_figure('ROC/%s (epoch)' % tb_mode, f, epoch)

        # write bkgrej values
        for name_sig, name_bkg in comp_list:
            tb_helper.write_scalars([
                ('BkgRej_%s_vs_%s/%s (epoch)' % (name_sig, name_bkg, tb_mode), bkgrej[(name_sig, name_bkg)], epoch),
            ])

    if not for_training:
        scores_cls = np.concatenate(scores_cls)
        scores_reg = np.concatenate(scores_reg)
        labels = {k: _concat(v) for k, v in labels.items()}
        metric_results_cls = evaluate_metrics(labels['truth_label'], scores_cls, eval_metrics=eval_metrics_cls)
        _logger.info('Evaluation metric for cls: \n%s', '\n'.join(
            ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results_cls.items()]))

    if for_training:
        return total_loss / max(count_tag, 1)
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / max(count, 1), (scores_cls, scores_reg), labels, observers


def train_classification(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None, sculpt_kw=None):
    """Classification training with optional AN soft-sculpt loss (sculpt events skipped for CE)."""
    model.train()
    data_config = train_loader.dataset.config
    sculpt_kw = _resolve_sculpt_kw(sculpt_kw)
    sculpt_active = _sculpt_constraint_active(sculpt_kw, epoch)
    buffer_an, buffer_train = _make_sculpt_buffers(sculpt_kw)
    if sculpt_kw and sculpt_kw.get('enable', False):
        warmup = int(sculpt_kw.get('warmup_epochs', 0) or 0)
        if sculpt_active:
            _logger.info(
                'Soft-sculpt constraint ON (epoch=%d >= warmup_epochs=%d); '
                'pdf-ratio loss buffer_size=%d min_bin_count=%d tau=%.3g',
                epoch, warmup,
                int(sculpt_kw.get('buffer_size', 65536)),
                int(sculpt_kw.get('min_bin_count', 32)),
                float(sculpt_kw.get('tau', 0.02)))
        else:
            _logger.info(
                'Soft-sculpt warm-up: constraint OFF (epoch=%d < warmup_epochs=%d); '
                'sculpt loss logged only', epoch, warmup)

    label_counter = Counter()
    total_loss = 0
    total_loss_cls = 0
    total_loss_sculpt = 0
    total_loss_sculpt_an = 0
    total_loss_sculpt_train = 0
    num_batches = 0
    total_correct = 0
    count = 0
    count_tag = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, Z in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long()
            try:
                label_mask = y[data_config.label_names[0] + '_mask'].bool()
            except KeyError:
                label_mask = None
            label = _flatten_label(label, label_mask)
            num_examples = label.shape[0]
            label_counter.update(label.cpu().numpy())
            label = label.to(dev)

            is_sculpt, sculpt_kin = _read_sculpt_from_Z(Z, dev)
            if is_sculpt is None:
                tag_mask = torch.ones(num_examples, dtype=torch.bool, device=dev)
            else:
                tag_mask = ~(is_sculpt > 0.5)

            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits = _flatten_preds(model_output, label_mask)
                if tag_mask.any():
                    loss_cls = loss_func(logits[tag_mask], label[tag_mask])
                else:
                    loss_cls = logits.sum() * 0.0
                # cls-only heads have no preds_reg -> sculpt skipped unless reg available
                event_w = _read_event_weights(
                    Z, num_examples, logits.device, logits.dtype, sculpt_kw,
                    data_config=data_config, label_cls=label)
                if sculpt_active:
                    sculpt_mass = _compute_sculpt_mass(None, sculpt_kin, sculpt_kw)
                    loss_sculpt_an, loss_sculpt_train = _dual_sculpt_losses(
                        logits, sculpt_mass, is_sculpt, label, tag_mask, sculpt_kw,
                        buffer_an=buffer_an, buffer_train=buffer_train, event_weights=event_w)
                    lam, lam_train = _sculpt_lambdas(sculpt_kw) if sculpt_kw else (1.0, 1.0)
                    loss_sculpt = loss_sculpt_an + loss_sculpt_train
                    loss = loss_cls + lam * loss_sculpt_an + lam_train * loss_sculpt_train
                else:
                    with torch.no_grad():
                        sculpt_mass = _compute_sculpt_mass(None, sculpt_kin, sculpt_kw)
                        loss_sculpt_an, loss_sculpt_train = _dual_sculpt_losses(
                            logits, sculpt_mass, is_sculpt, label, tag_mask, sculpt_kw,
                            buffer_an=buffer_an, buffer_train=buffer_train, event_weights=event_w)
                        loss_sculpt = loss_sculpt_an + loss_sculpt_train
                    loss = loss_cls
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            loss_val = loss.item()
            loss_cls_val = float(loss_cls.item()) if torch.is_tensor(loss_cls) else float(loss_cls)
            loss_sculpt_an_val = float(loss_sculpt_an.item()) if torch.is_tensor(loss_sculpt_an) else float(loss_sculpt_an)
            loss_sculpt_train_val = float(loss_sculpt_train.item()) if torch.is_tensor(loss_sculpt_train) else float(loss_sculpt_train)
            loss_sculpt_val = loss_sculpt_an_val + loss_sculpt_train_val

            num_batches += 1
            count += num_examples
            n_tag = int(tag_mask.sum().item())
            count_tag += n_tag
            correct = (preds[tag_mask] == label[tag_mask]).sum().item() if n_tag > 0 else 0
            total_loss += loss_val
            total_loss_cls += loss_cls_val
            total_loss_sculpt += loss_sculpt_val
            total_loss_sculpt_an += loss_sculpt_an_val
            total_loss_sculpt_train += loss_sculpt_train_val
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss_cls_val,
                'LossSculptSoftAN': '%.5f' % loss_sculpt_an_val,
                'LossSculptSoftTrain': '%.5f' % loss_sculpt_train_val,
                'LossSculptSoft': '%.5f' % loss_sculpt_val,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / max(n_tag, 1)),
                'AvgAcc': '%.5f' % (total_correct / max(count_tag, 1))})

            if tb_helper and num_batches < 500:
                tb_helper.write_scalars([
                    ("lr/train", scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'], tb_helper.batch_train_count + num_batches),
                    ("Loss/train", loss_cls_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptSoftAN/train", loss_sculpt_an_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptSoftTrain/train", loss_sculpt_train_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptSoft/train", loss_sculpt_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptAN/train", loss_sculpt_an_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculptTrain/train", loss_sculpt_train_val, tb_helper.batch_train_count + num_batches),
                    ("LossSculpt/train", loss_sculpt_val, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / max(n_tag, 1), tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, inputs=(X, y), model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info(
        'Train AvgLoss: %.5f, AvgLossSculptSoftAN: %.5f, AvgLossSculptSoftTrain: %.5f, '
        'AvgLossSculptSoft: %.5f, AvgAcc: %.5f' % (
            total_loss_cls / num_batches, total_loss_sculpt_an / num_batches,
            total_loss_sculpt_train / num_batches, total_loss_sculpt / num_batches,
            total_correct / max(count_tag, 1)))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    if sculpt_kw and sculpt_kw.get('enable', False):
        _log_buffer_hard_sculpt(
            buffer_an, sculpt_kw, sculpt_kw.get('mass_bins', list(range(180, 361, 10))),
            'SculptHard_train_AN', epoch, tb_helper)
        _log_buffer_hard_sculpt(
            buffer_train, sculpt_kw, sculpt_kw.get('mass_bins_train', list(range(20, 361, 10))),
            'SculptHard_train_trainbkg', epoch, tb_helper)

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss_cls / num_batches, epoch),
            ("LossSculptSoftAN/train (epoch)", total_loss_sculpt_an / num_batches, epoch),
            ("LossSculptSoftTrain/train (epoch)", total_loss_sculpt_train / num_batches, epoch),
            ("LossSculptSoft/train (epoch)", total_loss_sculpt / num_batches, epoch),
            ("LossSculptAN/train (epoch)", total_loss_sculpt_an / num_batches, epoch),
            ("LossSculptTrain/train (epoch)", total_loss_sculpt_train / num_batches, epoch),
            ("LossSculpt/train (epoch)", total_loss_sculpt / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / max(count_tag, 1), epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        tb_helper.batch_train_count += num_batches
        tb_helper.train_loss = total_loss_cls / num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()

    return total_loss / num_batches


def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            best_val_metrics='acc',
                            tb_helper=None, sculpt_kw=None):
    """Classification eval with sculpt CE masking and val hard-sculpt MAE."""
    model.eval()
    data_config = test_loader.dataset.config
    sculpt_kw = _resolve_sculpt_kw(sculpt_kw)

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    count_tag = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    sculpt_scores = []
    sculpt_masses = []
    sculpt_weights = []
    sculpt_scores_train = []
    sculpt_masses_train = []
    sculpt_weights_train = []
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long()
                entry_count += label.shape[0]
                try:
                    label_mask = y[data_config.label_names[0] + '_mask'].bool()
                except KeyError:
                    label_mask = None
                if not for_training and label_mask is not None:
                    labels_counts.append(np.squeeze(label_mask.numpy().sum(axis=-1)))
                label = _flatten_label(label, label_mask)
                num_examples = label.shape[0]
                label_counter.update(label.cpu().numpy())
                label = label.to(dev)

                is_sculpt, sculpt_kin = _read_sculpt_from_Z(Z, dev)
                if is_sculpt is None:
                    tag_mask = torch.ones(num_examples, dtype=torch.bool, device=dev)
                else:
                    tag_mask = ~(is_sculpt > 0.5)

                model_output = model(*inputs)
                logits = _flatten_preds(model_output, label_mask).float()

                if not for_training:
                    scores.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                    for k, v in y.items():
                        labels[k].append(_flatten_label(v, label_mask).cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                if for_training and sculpt_kw and sculpt_kw.get('enable', False):
                    sculpt_mass = _compute_sculpt_mass(None, sculpt_kin, sculpt_kw)
                    if sculpt_mass is not None:
                        score_index = int(sculpt_kw.get('score_index', 0))
                        s = torch.softmax(logits, dim=1)[:, score_index]
                        event_w = _read_event_weights(
                            Z, num_examples, logits.device, logits.dtype, sculpt_kw,
                            data_config=data_config, label_cls=label)
                        sculpt_on = sculpt_kw.get('sculpt_on', 'an')
                        if sculpt_on in ('an', 'both') and is_sculpt is not None:
                            sculpt_mask = is_sculpt > 0.5
                            if sculpt_mask.any():
                                sculpt_scores.append(s[sculpt_mask].detach().cpu().numpy())
                                sculpt_masses.append(sculpt_mass[sculpt_mask].detach().cpu().numpy())
                                sculpt_weights.append(event_w[sculpt_mask].detach().cpu().numpy())
                        if sculpt_on in ('train_bkg', 'both'):
                            train_mask = _train_bkg_sculpt_mask(label, tag_mask, sculpt_kw)
                            if train_mask.any():
                                sculpt_scores_train.append(s[train_mask].detach().cpu().numpy())
                                sculpt_masses_train.append(sculpt_mass[train_mask].detach().cpu().numpy())
                                sculpt_weights_train.append(event_w[train_mask].detach().cpu().numpy())

                _, preds = logits.max(1)
                n_tag = int(tag_mask.sum().item())
                if for_training and loss_func is not None:
                    if n_tag > 0:
                        loss = loss_func(logits[tag_mask], label[tag_mask]).item()
                    else:
                        loss = 0.0
                else:
                    loss = 0.0

                num_batches += 1
                count += num_examples
                count_tag += n_tag
                correct = (preds[tag_mask] == label[tag_mask]).sum().item() if n_tag > 0 else 0
                total_loss += loss * max(n_tag, 1)
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / max(count_tag, 1)),
                    'Acc': '%.5f' % (correct / max(n_tag, 1)),
                    'AvgAcc': '%.5f' % (total_correct / max(count_tag, 1))})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if for_training and sculpt_kw and sculpt_kw.get('enable', False):
        _log_hard_sculpt_mae(
            sculpt_scores, sculpt_masses, sculpt_kw,
            sculpt_kw.get('mass_bins', list(range(180, 361, 10))),
            'SculptHard_val_AN', epoch, tb_helper, weights_list=sculpt_weights)
        _log_hard_sculpt_mae(
            sculpt_scores_train, sculpt_masses_train, sculpt_kw,
            sculpt_kw.get('mass_bins_train', list(range(20, 361, 10))),
            'SculptHard_val_trainbkg', epoch, tb_helper, weights_list=sculpt_weights_train)

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / max(count_tag, 1), epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / max(count_tag, 1), epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    if not for_training:
        scores = np.concatenate(scores)
        labels = {k: _concat(v) for k, v in labels.items()}
        metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
        _logger.info('Evaluation metrics:\n%s', '\n'.join(['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        if best_val_metrics == 'acc':
            return total_correct / max(count_tag, 1)
        return total_loss / max(count_tag, 1)
    else:
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / max(count, 1), scores, labels, observers


def save_hybrid(args, data_config, scores, labels, observers):
    import ast
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    reg_kw = network_options.get('reg_kw', dict())
    split_reg = reg_kw.get('split_reg', False)
    composed_split_reg = reg_kw.get('composed_split_reg', None)
    label_cls_nodes = network_options.get('label_cls_nodes', None)
    label_stored = network_options.get('label_stored', None)
    assert label_cls_nodes is not None, 'label_cls_nodes must be provided as a network option in the test mode'

    if label_stored is None:
        label_stored = [
            "label_Top_bWcs", "label_Top_bWqq", "label_Top_bWc", "label_Top_bWs", "label_Top_bWq", "label_Top_bWev", "label_Top_bWmv", "label_Top_bWtauev", "label_Top_bWtaumv", "label_Top_bWtauhv", "label_Top_Wcs", "label_Top_Wqq", "label_Top_Wev", "label_Top_Wmv", "label_Top_Wtauev", "label_Top_Wtaumv", "label_Top_Wtauhv",
            "label_Top_bWpcs", "label_Top_bWpqq", "label_Top_bWpc", "label_Top_bWps", "label_Top_bWpq", "label_Top_bWpev", "label_Top_bWpmv", "label_Top_bWptauev", "label_Top_bWptaumv", "label_Top_bWptauhv", "label_Top_Wpcs", "label_Top_Wpqq", "label_Top_Wpev", "label_Top_Wpmv", "label_Top_Wptauev", "label_Top_Wptaumv", "label_Top_Wptauhv",
            "label_Top_bWmcs", "label_Top_bWmqq", "label_Top_bWmc", "label_Top_bWms", "label_Top_bWmq", "label_Top_bWmev", "label_Top_bWmmv", "label_Top_bWmtauev", "label_Top_bWmtaumv", "label_Top_bWmtauhv", "label_Top_Wmcs", "label_Top_Wmqq", "label_Top_Wmev", "label_Top_Wmmv", "label_Top_Wmtauev", "label_Top_Wmtaumv", "label_Top_Wmtauhv",
            "label_H_bb", "label_H_cc", "label_H_ss", "label_H_qq", "label_H_bc", "label_Hp_bc", "label_Hm_bc", "label_H_bs", "label_H_cs", "label_Hp_cs", "label_Hm_cs", "label_H_gg", "label_H_aa", "label_H_ee", "label_H_mm", "label_H_tauhtaue", "label_H_tauhtaum", "label_H_tauhtauh", "label_H_WW_cscs", "label_H_WW_csqq", "label_H_WW_qqqq", "label_H_WW_csc", "label_H_WW_css", "label_H_WW_csq", "label_H_WW_qqc", "label_H_WW_qqs", "label_H_WW_qqq", "label_H_WW_csev", "label_H_WW_qqev", "label_H_WW_csmv", "label_H_WW_qqmv", "label_H_WW_cstauev", "label_H_WW_qqtauev", "label_H_WW_cstaumv", "label_H_WW_qqtaumv", "label_H_WW_cstauhv", "label_H_WW_qqtauhv", 
            "label_H_WxWx_cscs", "label_H_WxWx_csqq", "label_H_WxWx_qqqq", "label_H_WxWx_csc", "label_H_WxWx_css", "label_H_WxWx_csq", "label_H_WxWx_qqc", "label_H_WxWx_qqs", "label_H_WxWx_qqq", "label_H_WxWx_csev", "label_H_WxWx_qqev", "label_H_WxWx_csmv", "label_H_WxWx_qqmv", "label_H_WxWx_cstauev", "label_H_WxWx_qqtauev", "label_H_WxWx_cstaumv", "label_H_WxWx_qqtaumv", "label_H_WxWx_cstauhv", "label_H_WxWx_qqtauhv", 
            "label_H_WxWxStar_cscs", "label_H_WxWxStar_csqq", "label_H_WxWxStar_qqqq", "label_H_WxWxStar_csc", "label_H_WxWxStar_css", "label_H_WxWxStar_csq", "label_H_WxWxStar_qqc", "label_H_WxWxStar_qqs", "label_H_WxWxStar_qqq", "label_H_WxWxStar_csev", "label_H_WxWxStar_qqev", "label_H_WxWxStar_csmv", "label_H_WxWxStar_qqmv", "label_H_WxWxStar_cstauev", "label_H_WxWxStar_qqtauev", "label_H_WxWxStar_cstaumv", "label_H_WxWxStar_qqtaumv", "label_H_WxWxStar_cstauhv", "label_H_WxWxStar_qqtauhv", 
            "label_QCD_bb", "label_QCD_cc", "label_QCD_b", "label_QCD_c", "label_QCD_others"
            ]

    output = {}
    scores_cls, scores_reg = scores
    assert scores_cls.shape[1] == len(label_cls_nodes), 'Number of classification nodes does not match'

    # write regression nodes
    if len(data_config.label_names) > 1:
        if composed_split_reg is not None:
            idx_reg = 0
            # write unified regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                # do unified regression (always true for this script)
                print('write unified regression nodes:', name)
                output[name] = labels[name]
                output['output_' + name] = scores_reg[:, idx_reg]
                idx_reg += 1
            # write split regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                if composed_split_reg[idx-1]: # do split regression
                    print('write split regression nodes:', name)
                    if name not in output:
                        output[name] = labels[name]
                    for idx_cls, label_name in enumerate(label_cls_nodes):
                        if label_name not in label_stored:
                            continue
                        output['output_' + name + '_' + label_name] = scores_reg[:, idx_reg + idx_cls]
                    idx_reg += len(label_cls_nodes)

        elif split_reg:
            # write split regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                output[name] = labels[name]
                for idx_cls, label_name in enumerate(label_cls_nodes):
                    if label_name not in label_stored:
                        continue
                    output['output_' + name + '_' + label_name] = scores_reg[:, (idx-1) * len(label_cls_nodes) + idx_cls]
        else:
            # write normal (unified) regression nodes
            for idx in range(1, len(data_config.label_names)):
                name = data_config.label_names[idx]
                output[name] = labels[name]
                output['output_' + name] = scores_reg[:, idx-1]

    # write classification nodes
    output['cls_index'] = labels['truth_label'] # classes can be too many, only store the index
    for idx, label_name in enumerate(label_cls_nodes):
        if label_name not in label_stored:
            continue
        output['score_' + label_name] = scores_cls[:, idx]

    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        assert v.ndim == 1
        output[k] = v

    for k, v in observers.items():
        assert v.ndim == 1
        output[k] = v

    return output


#### ================== Custom train/eval/save functions for guassian NLL regression ================== ####

def train_guass_regression(model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None, tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].float()
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                preds = model_output[:, 0]
                loss = loss_func(model_output, label)
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

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - label
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                # 'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                # 'AvgMSE': '%.5f' % (sum_sqr_err / count),
                # 'MAE': '%.5f' % (abs_err / num_examples),
                # 'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_guass_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        train_loss=None,
                        tb_helper=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    scores_logvar = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)
                preds = model_output[:, 0].float()

                scores.append(preds.detach().cpu().numpy())
                scores_logvar.append(model_output[:, 1].float().detach().cpu().numpy())
                for k, v in y.items():
                    labels[k].append(v.cpu().numpy())
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.cpu().numpy())

                loss = 0 if loss_func is None else loss_func(model_output, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=num_batches,
                                                mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
            ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    scores_logvar = np.concatenate(scores_logvar)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, (scores, scores_logvar), labels, observers


def save_guass_regression(args, data_config, scores, labels, observers):
    scores_mu, scores_logvar = scores
    output = {}
    name = data_config.label_names[0]
    output[name] = labels[name]
    output['output_' + name] = scores_mu
    output['output_' + name + '_sigma'] = np.exp(scores_logvar / 2) ## convert logvar to sigma
    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        assert v.ndim == 1
        output[k] = v
    for k, v in observers.items():
        assert v.ndim == 1
        output[k] = v
    return output
