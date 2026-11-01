# -*- coding: utf-8 -*-
"""Per-sample gate-confidence probe for InfraSegFM.

Dumps one CSV row per image: file, task, mean over adapted blocks of
(gate max-prob, gate entropy). Used to design/validate confidence-adaptive
routing on the *validation* split only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from os.path import join
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import InfraSegDataset
from model import InfraSegFM
from segment_anything import sam_model_registry


def _safe_load_moe_image_encoder_only(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    # Checkpoint keys are sam-relative (saved from self.sam.state_dict()).
    cur = model.sam.state_dict()
    load = {k: v for k, v in state_dict.items() if k in cur and cur[k].shape == v.shape}
    assert len(load) > 100, f"suspiciously few matched keys: {len(load)} - prefix mismatch?"
    n_gate = sum(1 for k in load if "adapter_gate" in k)
    print(f"[load] matched {len(load)} keys ({n_gate} gate tensors)")
    cur.update(load)
    model.sam.load_state_dict(cur, strict=False)


def main() -> None:
    p = argparse.ArgumentParser("Per-sample gate probe")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_type", type=str, default="vit_b")
    p.add_argument("--moe_ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.moe_ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    gate_w = next(v for k, v in state.items() if k.endswith("adapter_gate.weight"))
    expert_num, gate_in = gate_w.shape
    embedding_dim = gate_in // 4
    bottleneck_dim = 16
    for k, v in state.items():
        if k.endswith("adapter_down.0.weight"):
            bottleneck_dim = int(v.shape[0])
            break

    sam_ckpt = join(args.checkpoint, {"vit_b": "sam_vit_b_01ec64.pth",
                                      "vit_l": "sam_vit_l_0b3195.pth",
                                      "vit_h": "sam_vit_h_4b8939.pth"}[args.model_type])
    sam = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=sam_ckpt)
    net = InfraSegFM(sam, bottleneck_dim=bottleneck_dim, embedding_dim=embedding_dim,
                     expert_num=expert_num).to(device)
    net = nn.DataParallel(net)
    _safe_load_moe_image_encoder_only(net.module, state)
    net.eval()

    ds = InfraSegDataset(args.data_root, train=False)
    if getattr(args, "max_samples", 0) and args.max_samples > 0 and len(ds) > args.max_samples:
        import numpy as _np
        idx = _np.linspace(0, len(ds) - 1, args.max_samples).astype(int).tolist()
        ds.gt_paths = [ds.gt_paths[i] for i in idx]
        if hasattr(ds, "task_folders"):
            ds.task_folders = [ds.task_folders[i] for i in idx]
        if hasattr(ds, "meta_indices"):
            ds.meta_indices = [ds.meta_indices[i] for i in idx]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    rows = []
    with torch.no_grad():
        for data, _gt in dl:
            names = list(data["name"])
            for k, v in list(data.items()):
                if torch.is_tensor(v):
                    data[k] = v.to(device)
                elif isinstance(v, (tuple, list)) and len(v) and torch.is_tensor(v[0]):
                    data[k] = tuple(x.to(device) for x in v)
            out = net(data, return_gates=True)
            gates = out.get("gates", [])
            if not gates:
                raise RuntimeError("no gates returned")
            # stack: (L, B, E) -> per-sample stats averaged over layers
            g = torch.stack([x.float() for x in gates], dim=0)
            top1 = g.max(dim=-1).values.mean(dim=0)                    # (B,)
            ent = (-(g.clamp_min(1e-12) * g.clamp_min(1e-12).log()).sum(-1)).mean(0)  # (B,)
            for i, name in enumerate(names):
                nm = os.path.normpath(str(name))
                parts = nm.replace("\\", "/").split("/")
                task = ""
                for j, seg in enumerate(parts):
                    if seg in ("npy_gts", "npy_imgs") and j > 0:
                        task = parts[j - 1]
                        break
                rows.append((nm, task, float(top1[i]), float(ent[i])))

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "task", "gate_top1_mean", "gate_entropy_mean"])
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
