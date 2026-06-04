# -*- coding: utf-8 -*-
"""Mixture-of-Experts (MoE) auxiliary losses.

This repo's MoE gate produces per-sample expert probabilities of shape [B, E]
at multiple transformer layers. The utilities here implement two lightweight
regularizers that are cheap and effective at preventing expert collapse:

1) Load-balance loss (Switch-style): encourages both *importance* (soft usage)
   and *load* (hard top-1 assignment frequency) to be balanced across experts.
2) Entropy bonus: encourages non-degenerate gates early in training.

All functions are torchscript-friendly and require no changes to your dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class MoELossOutput:
    total: torch.Tensor
    lb: torch.Tensor
    ent: torch.Tensor
    per_layer: List[Dict[str, torch.Tensor]]


def _safe_entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Mean entropy over batch for probabilities p of shape [B, E]."""
    p = torch.clamp(p, eps, 1.0)
    return -(p * p.log()).sum(dim=1).mean()


def moe_aux_loss(
    gates: List[torch.Tensor],
    lb_coef: float = 0.01,
    ent_coef: float = 0.01,
    topk_for_load: int = 1,
    eps: float = 1e-9,
) -> MoELossOutput:
    """Compute MoE auxiliary loss from a list of gate probabilities.

    Args:
        gates: list of [B, E] tensors from multiple layers.
        lb_coef: coefficient for load-balance loss.
        ent_coef: coefficient for entropy bonus (implemented as -entropy in loss).
        topk_for_load: how to compute hard load. 1 means top-1.
        eps: numerical stability.

    Returns:
        MoELossOutput where `total` is ready to add to main loss.
    """
    if (gates is None) or (len(gates) == 0) or (lb_coef == 0.0 and ent_coef == 0.0):
        z = torch.tensor(0.0)
        return MoELossOutput(total=z, lb=z, ent=z, per_layer=[])

    lb_all = []
    ent_all = []
    per_layer: List[Dict[str, torch.Tensor]] = []

    for g in gates:
        if g is None:
            continue
        # g: [B, E]
        B, E = g.shape

        # Importance: average soft probability per expert
        importance = g.mean(dim=0)  # [E]

        # Load: frequency of hard assignment (top-1 by default)
        if topk_for_load <= 1:
            top_idx = torch.argmax(g, dim=1)  # [B]
            load = torch.bincount(top_idx, minlength=E).float() / max(float(B), 1.0)
        else:
            # Count each of top-k as 1/k credit
            _, top_idx = torch.topk(g, k=min(topk_for_load, E), dim=1)
            load = torch.zeros(E, device=g.device, dtype=torch.float)
            for j in range(top_idx.shape[1]):
                load.scatter_add_(0, top_idx[:, j], torch.ones(B, device=g.device, dtype=torch.float))
            load = load / (max(float(B), 1.0) * float(top_idx.shape[1]))

        # Switch-style load-balance:
        # minimized when both importance and load are uniform.
        lb = (importance * load).sum() * float(E)

        ent = _safe_entropy(g, eps=eps)

        lb_all.append(lb)
        ent_all.append(ent)
        per_layer.append(
            {
                "importance": importance.detach(),
                "load": load.detach(),
                "lb": lb.detach(),
                "entropy": ent.detach(),
            }
        )

    if len(lb_all) == 0:
        z = torch.tensor(0.0)
        return MoELossOutput(total=z, lb=z, ent=z, per_layer=[])

    lb_mean = torch.stack(lb_all).mean()
    ent_mean = torch.stack(ent_all).mean()

    # We *maximize* entropy, so in loss we add -entropy
    total = lb_coef * lb_mean + (-ent_coef) * ent_mean
    return MoELossOutput(total=total, lb=lb_mean, ent=ent_mean, per_layer=per_layer)
