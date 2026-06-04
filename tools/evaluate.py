"""Evaluate an InfraSegFM checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrasegfm import InfraSegDataset, build_infrasegfm
from infrasegfm.metrics import SegmentMetrics
from infrasegfm.train_utils import ensure_dir, load_torch_checkpoint, move_batch_to_device, select_best_mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, required=True, help="Split root or task folder to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, default="", help="Path to SAM checkpoint used by the trained model.")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=50, help="Paper workflow default.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default="runs/eval")
    parser.add_argument("--metrics", nargs="+", default=["dsc", "hd"], help="Metrics: dsc, hd/hd95, nsd.")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[eval] CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    ckpt = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    moe_cfg = ckpt.get("moe_cfg", {}) if isinstance(ckpt, dict) else {}
    model = build_infrasegfm(
        model_type=args.model_type,
        sam_checkpoint=args.sam_checkpoint,
        image_size=args.image_size,
        bottleneck_dim=int(saved_args.get("bottleneck_dim", 16)),
        embedding_dim=int(saved_args.get("embedding_dim", 16)),
        expert_num=int(saved_args.get("expert_num", 4)),
        gate_topk=int(saved_args.get("moe_topk", moe_cfg.get("moe_topk", 2))),
        gate_temperature=float(saved_args.get("moe_temp", moe_cfg.get("moe_temp", 1.0))),
        gate_noise_std=float(saved_args.get("moe_noise_std", moe_cfg.get("moe_noise_std", 0.0))),
        style_use_std=bool(saved_args.get("moe_style_bn", moe_cfg.get("moe_style_bn", 1))),
        style_dropout=float(saved_args.get("moe_style_dropout", moe_cfg.get("moe_style_dropout", 0.10))),
    ).to(device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_parameters(state)
    model.eval()

    dataset = InfraSegDataset(args.data_root, train=False, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    metric_names = []
    for metric in args.metrics:
        name = "hd" if metric.lower() == "hd95" else metric.lower()
        if name in {"dsc", "hd", "nsd"} and name not in metric_names:
            metric_names.append(name)
    metrics = SegmentMetrics(metric_names).to(device)
    rows = []
    by_task = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for data, label in tqdm(loader, desc="Evaluating"):
            names = data["name"]
            tasks = data["task_folder"]
            data, label = move_batch_to_device(data, label, device)
            logits = model(data)
            pred = select_best_mask(logits, label, threshold=args.threshold)
            batch_metrics = metrics(pred, label)
            for i, name in enumerate(names):
                row = {"name": name, "task": tasks[i]}
                for key, values in batch_metrics.items():
                    value = float(values[i].detach().cpu().item())
                    row[key] = value
                    by_task[tasks[i]][key].append(value)
                rows.append(row)

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "task"] + metric_names)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task"] + metric_names)
        writer.writeheader()
        all_values = defaultdict(list)
        for task, task_metrics in sorted(by_task.items()):
            row = {"task": task}
            for key in metric_names:
                vals = [v for v in task_metrics.get(key, []) if v == v]
                row[key] = sum(vals) / max(len(vals), 1)
                all_values[key].extend(vals)
            writer.writerow(row)
        row = {"task": "OVERALL"}
        for key in metric_names:
            vals = all_values.get(key, [])
            row[key] = sum(vals) / max(len(vals), 1)
        writer.writerow(row)

    print(f"Metrics written to {csv_path}")
    print(f"Summary written to {summary_path}")
    print(row)


if __name__ == "__main__":
    main()
