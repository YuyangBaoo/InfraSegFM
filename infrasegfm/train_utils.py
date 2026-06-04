"""Small training and evaluation helpers shared by CLI tools."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(data: Dict, label: torch.Tensor, device: torch.device):
    for key, value in list(data.items()):
        if torch.is_tensor(value):
            data[key] = value.to(device, non_blocking=True)
        elif key == "context":
            data[key] = tuple(v.to(device, non_blocking=True) if torch.is_tensor(v) else v for v in value)
    return data, label.to(device, non_blocking=True)


def select_best_mask(logits: torch.Tensor, label: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Select the highest-Dice candidate mask for each sample."""

    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
    pred = (torch.sigmoid(logits) > float(threshold)).float()
    target = (label > 0).float()

    scores = []
    for idx in range(pred.shape[1]):
        p = pred[:, idx : idx + 1]
        dims = tuple(range(1, p.dim()))
        inter = (p * target).sum(dim=dims)
        denom = p.sum(dim=dims) + target.sum(dim=dims)
        dsc = (2.0 * inter + 1e-6) / (denom + 1e-6)
        both_empty = (p.sum(dim=dims) == 0) & (target.sum(dim=dims) == 0)
        dsc = torch.where(both_empty, torch.ones_like(dsc), dsc)
        scores.append(dsc)
    best = torch.stack(scores, dim=1).argmax(dim=1)
    return pred[torch.arange(pred.shape[0], device=pred.device), best].unsqueeze(1)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_torch_checkpoint(path: Union[str, Path], map_location: str = "cpu"):
    """Load a trusted tensor checkpoint without triggering PyTorch's pickle warning."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
