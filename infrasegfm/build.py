"""Factory helpers for SAM and InfraSegFM."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .model import InfraSegFM
from .segment_anything import sam_model_registry


def build_sam(
    model_type: str = "vit_b",
    checkpoint: Optional[str] = None,
    image_size: int = 256,
    keep_resolution: bool = True,
):
    """Build a SAM backbone.

    Args:
        model_type: One of ``vit_b``, ``vit_l``, or ``vit_h``.
        checkpoint: Optional path to a SAM checkpoint. If omitted, the model is
            randomly initialized; this is useful only for smoke tests.
        image_size: Square input size used by the SAM image encoder.
        keep_resolution: Whether the SAM mask decoder upsamples masks to the
            encoder input resolution. InfraSegFM uses ``True`` in the paper code.
    """

    model_type = str(model_type)
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown SAM model_type={model_type!r}. Valid values: {sorted(sam_model_registry)}")

    ckpt = None if checkpoint in (None, "", "none", "None") else str(Path(checkpoint))
    return sam_model_registry[model_type](image_size=int(image_size), keep_resolution=bool(keep_resolution), checkpoint=ckpt)


def build_infrasegfm(
    model_type: str = "vit_b",
    sam_checkpoint: Optional[str] = None,
    image_size: int = 256,
    keep_resolution: bool = True,
    bottleneck_dim: int = 16,
    embedding_dim: int = 16,
    expert_num: int = 4,
    adapter_positions: Optional[Sequence[int]] = None,
    gate_topk: int = 2,
    gate_temperature: float = 1.0,
    gate_noise_std: float = 0.0,
    style_use_std: bool = True,
    style_dropout: float = 0.10,
) -> InfraSegFM:
    """Build the InfraSegFM model with a SAM backbone."""

    sam = build_sam(model_type=model_type, checkpoint=sam_checkpoint, image_size=image_size, keep_resolution=keep_resolution)
    return InfraSegFM(
        sam=sam,
        bottleneck_dim=bottleneck_dim,
        embedding_dim=embedding_dim,
        expert_num=expert_num,
        pos=adapter_positions,
        gate_topk=gate_topk,
        gate_temperature=gate_temperature,
        gate_noise_std=gate_noise_std,
        style_use_std=style_use_std,
        style_dropout=style_dropout,
    )
