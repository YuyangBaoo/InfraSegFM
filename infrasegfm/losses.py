# -*- coding: utf-8 -*-
"""Loss utilities (domain-agnostic)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Dice + BCE (with logits) for binary segmentation.

    Supports per-sample output when reduction='none'.
    """

    def __init__(
        self,
        sigmoid: bool = True,
        squared_pred: bool = True,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean | sum | none")

        self.sigmoid = bool(sigmoid)
        self.squared_pred = bool(squared_pred)
        self.reduction = reduction
        self.pos_weight = pos_weight
        self.lambda_dice = float(lambda_dice)
        self.lambda_bce = float(lambda_bce)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(f"input/target shapes must match, got {logits.shape} vs {target.shape}")

        target_f = target.float()
        probs = torch.sigmoid(logits) if self.sigmoid else logits
        reduce_dims = tuple(range(1, probs.dim()))

        # Dice (per-sample)
        inter = (probs * target_f).sum(dim=reduce_dims)
        if self.squared_pred:
            denom = (probs * probs).sum(dim=reduce_dims) + (target_f * target_f).sum(dim=reduce_dims)
        else:
            denom = probs.sum(dim=reduce_dims) + target_f.sum(dim=reduce_dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice

        # BCE (per-sample)
        bce_map = F.binary_cross_entropy_with_logits(
            logits,
            target_f,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        bce_loss = bce_map.mean(dim=reduce_dims)

        total = self.lambda_dice * dice_loss + self.lambda_bce * bce_loss

        if self.reduction == "none":
            return total
        if self.reduction == "sum":
            return total.sum()
        return total.mean()
