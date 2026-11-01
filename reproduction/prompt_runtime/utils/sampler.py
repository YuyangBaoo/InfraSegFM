# -*- coding: utf-8 -*-
"""Samplers for highly imbalanced multi-task datasets.

Why this file exists
--------------------
InfraSeg-style data is organized as many TaskFolders with very different numbers
of samples (e.g., Handheld_ConcreteCrack >> IndustrialLineScan_RailCrack).
If we simply shuffle all samples, the largest tasks dominate every epoch and
the model (and especially the MoE gate) tends to overfit the majority style.

However, *pure* task-balanced sampling (each TaskFolder equally likely) can be
too aggressive on extremely imbalanced data: head tasks become under-trained
unless you also increase training steps a lot. This file therefore provides a
**mixed sampler** that smoothly interpolates between:
  - natural sampling (alpha=0)
  - fully task-balanced sampling (alpha=1)

Use build_task_mixed_sampler(...) from pretrain/finetune.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence, Optional

import torch
from torch.utils.data import WeightedRandomSampler


def build_task_balanced_sampler(task_folders: Sequence[str]) -> WeightedRandomSampler:
    """Backward-compatible: pure task-balanced sampler (alpha=1.0, power=1.0)."""
    return build_task_mixed_sampler(task_folders, alpha=1.0, power=1.0)


def build_task_prefix_mixed_sampler(
    task_folders: Sequence[str],
    dataset_prefixes: Sequence[str],
    alpha_task: float = 0.25,
    alpha_prefix: float = 0.25,
    power_task: float = 1.0,
    power_prefix: float = 1.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a sampler that balances both TaskFolder and DatasetPrefix.

    Motivation
    ----------
    In CivilDB-style data, a single TaskFolder can contain many DatasetPrefixes
    (different acquisition sources / styles). If we only balance by TaskFolder,
    the head prefixes inside the head task still dominate and the MoE gate
    can collapse toward that dominant style.

    This sampler mixes three components:
      1) natural sampling (all weights = 1)
      2) task-balanced (inverse freq by TaskFolder)
      3) (task,prefix)-balanced (inverse freq by joint key)

    Args:
        task_folders: TaskFolder name per sample index, length N.
        dataset_prefixes: DatasetPrefix per sample index, length N.
        alpha_task: mixing weight for task-balanced component in [0,1].
        alpha_prefix: mixing weight for (task,prefix)-balanced component in [0,1].
            Note: the final weight is:
                w = (1-alpha_task-alpha_prefix)*1 + alpha_task*w_task + alpha_prefix*w_joint
            so you should keep alpha_task+alpha_prefix <= 1.
        power_task: exponent for task inverse-frequency.
        power_prefix: exponent for joint inverse-frequency.
        num_samples: drawn per epoch (replacement). Default N.

    Returns:
        WeightedRandomSampler.
    """
    if task_folders is None or len(task_folders) == 0:
        raise ValueError("task_folders is empty; cannot build sampler")
    if dataset_prefixes is None or len(dataset_prefixes) != len(task_folders):
        raise ValueError("dataset_prefixes must have same length as task_folders")

    # Clamp
    alpha_task = float(alpha_task)
    alpha_prefix = float(alpha_prefix)
    alpha_task = 0.0 if alpha_task < 0.0 else (1.0 if alpha_task > 1.0 else alpha_task)
    alpha_prefix = 0.0 if alpha_prefix < 0.0 else (1.0 if alpha_prefix > 1.0 else alpha_prefix)
    if alpha_task + alpha_prefix > 1.0:
        s = alpha_task + alpha_prefix
        alpha_task /= s
        alpha_prefix /= s

    power_task = float(power_task)
    power_prefix = float(power_prefix)
    power_task = 0.0 if power_task < 0.0 else power_task
    power_prefix = 0.0 if power_prefix < 0.0 else power_prefix

    counts_task = Counter(task_folders)
    joint_keys = [f"{t}||{p}" for t, p in zip(task_folders, dataset_prefixes)]
    counts_joint = Counter(joint_keys)

    w_base = torch.ones(len(task_folders), dtype=torch.float)
    w_task = torch.tensor([1.0 / (float(counts_task[t]) ** power_task) for t in task_folders], dtype=torch.float)
    w_joint = torch.tensor([1.0 / (float(counts_joint[k]) ** power_prefix) for k in joint_keys], dtype=torch.float)

    weights = (1.0 - alpha_task - alpha_prefix) * w_base + alpha_task * w_task + alpha_prefix * w_joint

    if num_samples is None or int(num_samples) <= 0:
        num_samples = len(task_folders)
    else:
        num_samples = int(num_samples)

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def build_task_mixed_sampler(
    task_folders: Sequence[str],
    alpha: float = 0.25,
    power: float = 1.0,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a sampler that mixes natural + task-balanced sampling.

    Args:
        task_folders: list/sequence of TaskFolder names, length N (one per sample).
        alpha: mixing factor in [0, 1].
            - 0.0 => plain sampling (each sample weight = 1)
            - 1.0 => pure task-balanced sampling (each task gets ~equal probability)
            Recommended for your CivilDB scale: 0.1~0.3 as a default starting point.
        power: strength of inverse-frequency reweighting.
            weight_balanced = 1 / (count(task) ** power)
            - 1.0 is standard inverse frequency
            - 0.5 is milder
        num_samples: samples drawn per epoch (with replacement). Default: N.

    Returns:
        WeightedRandomSampler with replacement.
    """
    if task_folders is None or len(task_folders) == 0:
        raise ValueError("task_folders is empty; cannot build sampler")

    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)
    power = float(power)
    power = 0.0 if power < 0.0 else power

    counts = Counter(task_folders)
    w_base = torch.ones(len(task_folders), dtype=torch.float)
    w_bal = torch.tensor([1.0 / (float(counts[t]) ** power) for t in task_folders], dtype=torch.float)

    weights = (1.0 - alpha) * w_base + alpha * w_bal

    if num_samples is None or int(num_samples) <= 0:
        num_samples = len(task_folders)
    else:
        num_samples = int(num_samples)

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
