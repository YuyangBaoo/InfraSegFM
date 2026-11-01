# -*- coding: utf-8 -*-
"""Verify the shared-adapter control is parameter-matched to the routed MoE.

Builds InfraSegFM in both modes on CPU (meta-free, no data needed) and reports
trainable parameters overall and along the encoder adaptation path.

Run from J:\InfraSegFM:
    .venv\Scripts\python.exe count_params_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from segment_anything import sam_model_registry  # noqa: E402
from model.infra_segfm import InfraSegFM, matched_shared_hidden_dim  # noqa: E402


def build(mode: str, bn: int = 16):
    ckpt = os.path.join("playground", "SAM", "sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_b"](image_size=256, keep_resolution=True, checkpoint=ckpt)
    return InfraSegFM(sam, bn, 16, 4, adapter_mode=mode)


def report(model, name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("adapter_" in n or "style_proj" in n)
    )
    embed = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and ("platform_embed" in n or "context_embed" in n)
    )
    decoder = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "mask_decoder" in n
    )
    print(f"[{name}] total={total:,} trainable={trainable:,}")
    print(f"    encoder adapters (adapter_*/style_proj): {adapter:,}")
    print(f"    conditioning embeddings (trainable):     {embed:,}")
    print(f"    mask decoder:                            {decoder:,}")
    return adapter


if __name__ == "__main__":
    w = matched_shared_hidden_dim(16, 16, 4)
    print(f"matched shared hidden width: {w}")
    a_moe = report(build("moe"), "moe   ")
    a_sh = report(build("shared"), "shared")
    a_medsa = report(build("medsa"), "medsa ")
    report(build("lora"), "lora  ")
    a_fullL = report(build("moe", bn=133), "FullL ")
    dL = a_fullL - a_medsa
    print(f"FullL vs MedSA adapter-path diff: {dL:+,} ({100.0 * dL / max(a_medsa, 1):+.3f}%)")
    diff = a_sh - a_moe
    print(f"\nadapter-path param diff (shared - moe): {diff:+,} ({100.0 * diff / max(a_moe, 1):+.3f}%)")
    tol = 0.005
    ok = abs(diff) / max(a_moe, 1) <= tol
    print("PARAM-MATCH-OK" if ok else "PARAM-MATCH-FAIL (>0.5% difference)")
    sys.exit(0 if ok else 1)
