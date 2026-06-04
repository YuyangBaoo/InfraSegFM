# -*- coding: utf-8 -*-
"""Segmentation metrics (domain-agnostic).

Supported metrics
-----------------
- dsc: Dice similarity coefficient (empty-aware)
- hd:  95th percentile Hausdorff distance (surface-to-surface)
- nsd: Normalized surface dice (tolerance in pixels)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _to_numpy_bool(x: torch.Tensor) -> np.ndarray:
    x = x.detach().to("cpu")
    if x.dtype != torch.bool:
        x = x > 0.5
    return x.numpy().astype(bool)


def _surface(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask

    if _HAS_SCIPY:
        er = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return np.logical_xor(mask, er)

    up = np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    dn = np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    lf = np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    rt = np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    interior = mask & up & dn & lf & rt
    return mask & (~interior)


def dice_empty_aware(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred[:, 0]
    if target.dim() == 4 and target.size(1) == 1:
        target = target[:, 0]

    pred = pred.bool()
    target = target.bool()

    dims = tuple(range(1, pred.dim()))
    pred_sum = pred.sum(dims)
    target_sum = target.sum(dims)

    both_empty = (pred_sum == 0) & (target_sum == 0)
    one_empty = (pred_sum == 0) ^ (target_sum == 0)

    inter = (pred & target).sum(dims).float()
    denom = pred_sum.float() + target_sum.float()
    dsc = (2.0 * inter + eps) / (denom + eps)

    dsc = torch.where(both_empty, torch.ones_like(dsc), dsc)
    dsc = torch.where(one_empty, torch.zeros_like(dsc), dsc)
    return dsc


def _hd95_one(pred: np.ndarray, gt: np.ndarray) -> float:
    sp = _surface(pred)
    sg = _surface(gt)

    if sp.sum() == 0 or sg.sum() == 0:
        return float("nan")

    if _HAS_SCIPY:
        dt_g = distance_transform_edt(~sg)
        dt_p = distance_transform_edt(~sp)
        d_p2g = dt_g[sp]
        d_g2p = dt_p[sg]
        p95_1 = float(np.percentile(d_p2g, 95))
        p95_2 = float(np.percentile(d_g2p, 95))
        return max(p95_1, p95_2)

    p_pts = np.argwhere(sp).astype(np.float32)
    g_pts = np.argwhere(sg).astype(np.float32)
    if p_pts.size == 0 or g_pts.size == 0:
        return float("nan")

    dp = np.sqrt(((p_pts[:, None, :] - g_pts[None, :, :]) ** 2).sum(axis=2))
    d1 = np.percentile(dp.min(axis=1), 95)
    d2 = np.percentile(dp.min(axis=0), 95)
    return float(max(d1, d2))


def _nsd_one(pred: np.ndarray, gt: np.ndarray, tol: float) -> float:
    sp = _surface(pred)
    sg = _surface(gt)

    if sp.sum() == 0 and sg.sum() == 0:
        return 1.0
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0

    if _HAS_SCIPY:
        dt_g = distance_transform_edt(~sg)
        dt_p = distance_transform_edt(~sp)
        d_p2g = dt_g[sp]
        d_g2p = dt_p[sg]
        hit_p = float((d_p2g <= tol).sum())
        hit_g = float((d_g2p <= tol).sum())
        denom = float(sp.sum() + sg.sum())
        return (hit_p + hit_g) / max(1.0, denom)

    p_pts = np.argwhere(sp).astype(np.float32)
    g_pts = np.argwhere(sg).astype(np.float32)
    if p_pts.size == 0 or g_pts.size == 0:
        return 0.0

    d = np.sqrt(((p_pts[:, None, :] - g_pts[None, :, :]) ** 2).sum(axis=2))
    hit_p = (d.min(axis=1) <= tol).sum()
    hit_g = (d.min(axis=0) <= tol).sum()
    denom = float(sp.sum() + sg.sum())
    return float((hit_p + hit_g) / max(1.0, denom))


class SegmentMetrics(nn.Module):
    """Batch metrics wrapper.

    Input:
      pred:   (B,1,H,W) or (B,H,W) bool/float
      target: (B,1,H,W) or (B,H,W) bool/float

    Output:
      dict[name] -> Tensor[B]
    """

    def __init__(self, metric_name: Sequence[str], nsd_tolerance: float = 2.0):
        super().__init__()
        self.metric_name = [str(m).lower() for m in metric_name]
        self.nsd_tolerance = float(nsd_tolerance)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if "dsc" in self.metric_name:
            out["dsc"] = dice_empty_aware(pred, target)

        need_hd = "hd" in self.metric_name or "hd95" in self.metric_name
        need_nsd = "nsd" in self.metric_name
        if not (need_hd or need_nsd):
            return out

        if pred.dim() == 4 and pred.size(1) == 1:
            pred_ = pred[:, 0]
        else:
            pred_ = pred
        if target.dim() == 4 and target.size(1) == 1:
            target_ = target[:, 0]
        else:
            target_ = target

        b = pred_.shape[0]
        hd_vals: List[float] = []
        nsd_vals: List[float] = []

        pred_np = _to_numpy_bool(pred_)
        tgt_np = _to_numpy_bool(target_)

        for i in range(b):
            p = pred_np[i]
            g = tgt_np[i]
            if need_hd:
                hd_vals.append(_hd95_one(p, g))
            if need_nsd:
                nsd_vals.append(_nsd_one(p, g, tol=self.nsd_tolerance))

        device = pred.device
        if need_hd:
            out["hd"] = torch.tensor(hd_vals, device=device, dtype=torch.float32)
        if need_nsd:
            out["nsd"] = torch.tensor(nsd_vals, device=device, dtype=torch.float32)

        return out
