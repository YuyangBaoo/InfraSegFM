# -*- coding: utf-8 -*-
"""
Style router utilities (per-image features) for MoE gating.

We intentionally keep this lightweight and deterministic:
- no extra dependencies
- computed from input tensor already in memory

img: torch.Tensor, shape (B,C,H,W), float in [0,1] (or any bounded range)
returns: torch.Tensor, shape (B, style_dim)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-12)
    return mag

@torch.no_grad()
def extract_style_features(img: torch.Tensor, style_dim: int = 16) -> torch.Tensor:
    """
    Produce a compact feature vector capturing illumination, texture/edge density, and contrast.

    Base dims (<=16):
      - per-channel mean (3)
      - per-channel std  (3)
      - global contrast (p95-p5) (1)
      - edge mean (1)
      - edge std  (1)
      - saturation-ish proxy: std across channels mean (1)
      - optional: histogram-like moments (up to fill style_dim)
    """
    if img.ndim != 4:
        raise ValueError(f"img must be 4D (B,C,H,W), got {tuple(img.shape)}")
    B, C, H, W = img.shape
    x = img.float()

    if C == 1:
        x3 = x.repeat(1,3,1,1)
    elif C >= 3:
        x3 = x[:, :3]
    else:
        x3 = x[:, :1].repeat(1,3,1,1)

    # (B,3)
    mean = x3.mean(dim=(2,3))
    std  = x3.std(dim=(2,3), unbiased=False)

    # grayscale
    gray = (0.2989*x3[:,0:1] + 0.5870*x3[:,1:2] + 0.1140*x3[:,2:3])
    # robust contrast
    flat = gray.flatten(1)
    p5  = torch.quantile(flat, 0.05, dim=1)
    p95 = torch.quantile(flat, 0.95, dim=1)
    contrast = (p95 - p5).unsqueeze(1)  # (B,1)

    # edges
    edges = _sobel_edges(gray)
    e_mean = edges.mean(dim=(2,3))  # (B,1)
    e_std  = edges.std(dim=(2,3), unbiased=False)  # (B,1)

    # channel dispersion proxy
    ch_std_mean = x3.std(dim=1, unbiased=False).mean(dim=(1,2)).unsqueeze(1)  # (B,1)

    feats = torch.cat([mean, std, contrast, e_mean, e_std, ch_std_mean], dim=1)  # (B,3+3+1+1+1+1=10)

    # Fill remaining dims with simple moments if needed
    if feats.shape[1] < style_dim:
        # add skew/kurtosis-like moments on grayscale
        m = flat.mean(dim=1, keepdim=True)
        v = (flat - m).pow(2).mean(dim=1, keepdim=True) + 1e-12
        s = ((flat - m).pow(3).mean(dim=1, keepdim=True) / v.sqrt().pow(3)).clamp(-10,10)
        k = ((flat - m).pow(4).mean(dim=1, keepdim=True) / v.pow(2)).clamp(0,100)
        extra = torch.cat([m, v, s, k], dim=1)  # (B,4)
        feats = torch.cat([feats, extra], dim=1)

    if feats.shape[1] > style_dim:
        feats = feats[:, :style_dim]

    # normalize to roughly comparable scale (optional but helps)
    feats = torch.tanh(feats)
    return feats
