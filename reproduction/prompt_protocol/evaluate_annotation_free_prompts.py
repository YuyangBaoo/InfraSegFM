"""Isolated, read-only checkpoint evaluation with annotation-free prompts.

The frozen model and quality head are loaded through the audited formal-test
driver; labels are used only after inference to calculate Dice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torchvision.transforms import Resize


def load_driver(path: Path):
    spec = importlib.util.spec_from_file_location("formal_test_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import formal-test driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean(values):
    return statistics.fmean(values) if values else None


def summarize(rows, elapsed_seconds, forward_seconds):
    settings = defaultdict(list)
    for row in rows:
        settings[row["setting"]].append(row)
    result = {}
    for setting, items in settings.items():
        task_rows = defaultdict(list)
        for item in items:
            task_rows[item["task"]].append(item)
        result[setting] = {
            "samples": len(items),
            "tasks": len(task_rows),
            "sample_mean_candidate_0_dice": mean([r["candidate_0_dice"] for r in items]),
            "sample_mean_raw_score_dice": mean([r["raw_score_dice"] for r in items]),
            "sample_mean_quality_head_dice": mean([r["quality_head_dice"] for r in items]),
            "sample_mean_oracle_upper_bound_dice": mean([r["oracle_upper_bound_dice"] for r in items]),
            "task_macro_candidate_0_dice": mean([
                mean([r["candidate_0_dice"] for r in group]) for group in task_rows.values()
            ]),
            "task_macro_quality_head_dice": mean([
                mean([r["quality_head_dice"] for r in group]) for group in task_rows.values()
            ]),
        }
    return {
        "conditions": result,
        "samples_total": len(rows),
        "elapsed_seconds_evaluation": elapsed_seconds,
        "elapsed_seconds_forward_only": forward_seconds,
        "samples_per_second_overall": len(rows) / elapsed_seconds if elapsed_seconds else None,
        "samples_per_second_forward_only": len(rows) / forward_seconds if forward_seconds else None,
    }


def run(args):
    if args.prompt_mode == "empty" and args.batch_size != 1:
        raise ValueError("The audited SAM prompt encoder requires batch_size=1 for empty prompts")
    driver = load_driver(args.driver or (args.formal_root / "code" / "evaluate_quality_head_formal_test.py"))
    profile = driver.profile_config(args.profile)
    code_root = args.code_root or (args.formal_root / "code" / "original")
    paths = driver.profile_paths(args.formal_root, profile)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    source_hashes = driver.assert_source_fingerprint(code_root)
    head_state, head_info = driver.load_final_head(paths["head"], profile)
    model, model_info = driver.load_exact_model(
        args.formal_root, code_root, profile, paths["checkpoint"], args.sam_checkpoint, device
    )
    if head_info.get("source_non_head_state_sha256") != model_info["source_non_head_state_sha256"]:
        raise RuntimeError("Quality head and model checkpoint do not match")
    tuned_head = driver.clone_tuned_head(model, head_state, device)
    if args.prompt_mode == "empty":
        original_prompt_forward = model.sam.prompt_encoder.forward

        def empty_prompt_forward(*, points, boxes, masks):
            return original_prompt_forward(points=None, boxes=None, masks=None)

        model.sam.prompt_encoder.forward = empty_prompt_forward
    driver.prepare_imports(code_root)
    from segment_anything.utils.transforms import ResizeLongestSide

    image_size = int(model.sam.image_encoder.img_size)
    image_transform = Resize((image_size, image_size), antialias=True)
    box_transform = ResizeLongestSide(image_size)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    start = time.monotonic()
    forward_seconds = 0.0
    task_progress = []
    with torch.inference_mode():
        for setting in profile["settings"]:
            spec = driver.TEST_SETTINGS[setting]
            setting_root = driver.test_setting_root(args.eval_root, setting)
            for task_dir in driver.discover_task_dirs(setting_root):
                dataset, routing = driver.make_dataset(task_dir)
                loader = driver.make_loader(dataset, args.batch_size, args.workers)
                task = task_dir.relative_to(setting_root).as_posix()
                processed = 0
                for data, label_cpu in loader:
                    if args.limit_per_task and processed >= args.limit_per_task:
                        break
                    names = [str(value) for value in data["name"]]
                    height, width = [int(v) for v in data["img"].shape[-2:]]
                    # Full-model forward still expects the box key. For the empty
                    # condition the prompt-encoder wrapper discards it completely.
                    data["box"] = torch.tensor(
                        [[0.0, 0.0, float(width - 1), float(height - 1)]] * len(names),
                        dtype=data["box"].dtype,
                    )
                    driver.transform_batch(data, image_size, box_transform, image_transform, device)
                    torch.cuda.synchronize(device)
                    forward_start = time.monotonic()
                    masks, raw_scores, features = driver.capture_one_forward(model, data)
                    torch.cuda.synchronize(device)
                    forward_seconds += time.monotonic() - forward_start
                    quality_scores = tuned_head(features)[:, 1:]
                    if not torch.isfinite(quality_scores).all().item():
                        raise RuntimeError("Quality-head score contains NaN/Inf")
                    dice = driver.candidate_dice(
                        masks, label_cpu.to(device, non_blocking=True),
                        antialias=bool(spec["metric_resize_antialias"]),
                    )
                    raw_idx = raw_scores.argmax(dim=1)
                    quality_idx = quality_scores.argmax(dim=1)
                    oracle_dice, oracle_idx = dice.max(dim=1)
                    row_idx = torch.arange(len(names), device=device)
                    raw_dice = dice[row_idx, raw_idx]
                    quality_dice = dice[row_idx, quality_idx]
                    for index, name in enumerate(names):
                        if args.limit_per_task and processed >= args.limit_per_task:
                            break
                        sample_id = driver.sample_id_from_name(name, setting_root)
                        rows.append({
                            "profile": args.profile,
                            "setting": setting,
                            "task": task,
                            "sample_id": sample_id,
                            "prompt_mode": args.prompt_mode,
                            "candidate_0_dice": float(dice[index, 0]),
                            "raw_score_selected_index": int(raw_idx[index]),
                            "raw_score_dice": float(raw_dice[index]),
                            "quality_head_selected_index": int(quality_idx[index]),
                            "quality_head_dice": float(quality_dice[index]),
                            "oracle_upper_bound_selected_index": int(oracle_idx[index]),
                            "oracle_upper_bound_dice": float(oracle_dice[index]),
                        })
                        processed += 1
                task_progress.append({"setting": setting, "task": task, "samples": processed,
                                      "routing_cleared": routing["routing_metadata_cleared_before_batch_emission"]})
                print(f"{args.profile} {setting} {task}: {processed} samples", flush=True)

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    result = summarize(rows, elapsed, forward_seconds)
    result.update({
        "schema": "infrasegfm-annotation-free-prompt-evaluation-v2",
        "profile": args.profile,
        "prompt_mode": args.prompt_mode,
        "meaning": (
            "No GT-derived localization prompt; a full-image box is supplied to the model."
            if args.prompt_mode == "fixed_full_image_box" else
            "The SAM prompt encoder receives no point, box, or mask prompt; batch size is one."
        ),
        "limit_per_task": args.limit_per_task,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": torch.cuda.get_device_name(device),
        "source_files_sha256": source_hashes,
        "model_checkpoint_sha256": model_info["checkpoint_sha256"],
        "head_state_sha256": head_info.get("head_state_sha256"),
        "task_progress": task_progress,
        "per_sample_sha256": None,
    })
    csv_path = args.output_dir / "per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result["per_sample_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"profile": args.profile, "samples": len(rows),
                      "elapsed_seconds": elapsed, "forward_seconds": forward_seconds,
                      "settings": result["conditions"]}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--driver", type=Path, help="formal-test driver path")
    parser.add_argument("--code-root", type=Path, help="frozen runtime source path")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("full_v1", "full_v2", "shared_v1", "shared_v2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=("fixed_full_image_box", "empty"),
                        default="fixed_full_image_box")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
