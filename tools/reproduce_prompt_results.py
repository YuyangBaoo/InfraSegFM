"""Recompute saved prompt-result summaries from the released CSV archive.

This command aggregates existing per-image Dice records; it does not train a
model or perform inference.  Formal GT-box/perturbation observations and the
later full-image-box/empty-prompt observations retain separate scope labels.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
import zipfile

PREFIX = "03_inference_routing/"
FIELDS = ["scope", "model", "profile", "version", "seed", "setting", "target_data_percent", "prompt_condition",
          "box_perturbation_percent", "selector", "offline_reference", "samples", "tasks",
          "image_mean_dice", "task_macro_dice", "image_mean_iou"]


class Sum:
    """Blockwise fsum keeps the calculation precise with bounded memory."""
    def __init__(self):
        self.block, self.partials, self.count = [], [], 0

    def add(self, value):
        self.block.append(value)
        self.count += 1
        if len(self.block) == 4096:
            self.partials.append(math.fsum(self.block))
            self.block.clear()

    def mean(self):
        if not self.count:
            raise ValueError("Cannot average an empty group")
        return math.fsum(self.partials + [math.fsum(self.block)]) / self.count


def records(archive, member):
    with io.TextIOWrapper(archive.open(PREFIX + member), encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def score(row, field):
    try:
        value = float(row[field])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Missing/invalid {field}: {row.get('sample_id', row)}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Nonfinite/out-of-range {field}: {value}")
    return value


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def reproduce(data_zip, output_dir):
    groups, contexts = {}, {}
    counts, checks, max_delta = defaultdict(int), 0, 0.0

    def check(actual, expected, label):
        nonlocal checks, max_delta
        a, e = float(actual), float(expected)
        if not math.isfinite(a) or not math.isfinite(e):
            raise ValueError(f"Nonfinite comparison: {label}")
        delta = abs(a - e)
        checks += 1
        max_delta = max(max_delta, delta)
        if delta > 1e-10:
            raise ValueError(f"Summary mismatch {label}: {a} versus {e}")

    def add(context, selector, task, dice, iou=None):
        key = tuple(context.items()) + (("selector", selector),)
        if key not in groups:
            groups[key] = {"dice": Sum(), "iou": Sum(), "tasks": defaultdict(Sum)}
            contexts[key] = {**context, "selector": selector}
        state = groups[key]
        state["dice"].add(dice)
        state["tasks"][task].add(dice)
        if iou is not None:
            state["iou"].add(iou)

    with zipfile.ZipFile(data_zip) as archive:
        formal = list(records(archive, "01_Formal_Condition_Summary.csv"))
        formal_ref = {(r["profile"], r["setting"], r["box_perturbation_percent"]): r for r in formal}
        baseline = list(records(archive, "10_Supplemental_Baseline_Summary.csv"))
        base_ref = {(r["model"], r["seed"], r["setting"], r["prompt_condition"], r["candidate_selector"]): r for r in baseline}
        shared = list(records(archive, "11_Supplemental_Full_Shared_Summary.csv"))
        shared_ref = {(r["profile"], r["setting"], r["prompt_mode"]): r for r in shared}
        if len(formal_ref) != len(formal) or len(base_ref) != len(baseline) or len(shared_ref) != len(shared):
            raise ValueError("Duplicate reference-condition key")

        for row in records(archive, "06_Formal_Per_Image_Results.csv"):
            key = row["profile"], row["setting"], row["box_perturbation_percent"]
            if key not in formal_ref:
                raise ValueError(f"Unexpected formal condition: {key}")
            candidates = [score(row, f"candidate_{i}_dice") for i in range(3)]
            raw = [float(row[f"raw_score_{i}"]) for i in range(3)]
            tuned = [float(row[f"tuned_score_{i}"]) for i in range(3)]
            if not all(math.isfinite(v) for v in raw + tuned):
                raise ValueError("Nonfinite predicted quality score")
            for index, values, field in [("raw_selected_index", raw, "raw_selected_dice"),
                                         ("tuned_selected_index", tuned, "tuned_selected_dice"),
                                         ("oracle_selected_index", candidates, "oracle_dice")]:
                expected_index = argmax(values)
                if int(row[index]) != expected_index:
                    raise ValueError(f"Candidate selection differs: {key}, {row['sample_id']}, {index}")
                check(score(row, field), candidates[expected_index], f"{key}/{field}")
            context = {"scope": "formal_quality_head", "model": row["model_family"], "profile": row["profile"],
                       "version": row["version"], "seed": row["source_seed"], "setting": row["setting"],
                       "target_data_percent": row["target_data_percent"],
                       "prompt_condition": "GT_derived_box", "box_perturbation_percent": row["box_perturbation_percent"]}
            for selector, field in [("fixed_candidate_0", "candidate_0_dice"),
                                    ("original_quality_scores", "raw_selected_dice"),
                                    ("finalized_quality_head", "tuned_selected_dice"),
                                    ("oracle_GT_Dice_offline", "oracle_dice")]:
                add(context, selector, row["task"], score(row, field))
            counts["formal_per_image"] += 1

        for row in records(archive, "12_Supplemental_Baseline_Per_Image.csv"):
            key = row["model"], row["seed"], row["setting"], row["prompt_condition"], row["candidate_selector"]
            ref = base_ref.get(key)
            if ref is None:
                raise ValueError(f"Unexpected baseline condition: {key}")
            context = {"scope": row["source_batch"], "model": row["model"], "profile": "",
                       "version": ref["source_version"], "seed": row["seed"], "setting": row["setting"],
                       "target_data_percent": row["ratio"],
                       "prompt_condition": row["prompt_condition"], "box_perturbation_percent": ""}
            add(context, row["candidate_selector"], row["task"], score(row, "Dice"), score(row, "IoU"))
            counts["supplemental_baseline_per_image"] += 1

        for row in records(archive, "13_Supplemental_Full_Shared_Per_Image.csv"):
            key = row["profile"], row["setting"], row["prompt_mode"]
            if key not in shared_ref:
                raise ValueError(f"Unexpected Full/Shared condition: {key}")
            context = {"scope": row["source_batch"], "model": "Full" if row["profile"].startswith("full_") else "Shared",
                       "profile": row["profile"], "version": row["profile"].rsplit("_", 1)[-1], "seed": "",
                       "target_data_percent": formal_ref[(row["profile"], row["setting"], "0")]["target_data_percent"],
                       "setting": row["setting"], "prompt_condition": row["prompt_mode"], "box_perturbation_percent": ""}
            for selector, field in [("fixed_candidate_0", "candidate_0_dice"),
                                    ("original_quality_scores", "raw_score_dice"),
                                    ("finalized_quality_head", "quality_head_dice"),
                                    ("oracle_GT_Dice_offline", "oracle_upper_bound_dice")]:
                add(context, selector, row["task"], score(row, field))
            counts["supplemental_full_shared_per_image"] += 1

    output = []
    formal_fields = {"fixed_candidate_0": "candidate_0", "finalized_quality_head": "tuned", "oracle_GT_Dice_offline": "oracle"}
    shared_fields = {"fixed_candidate_0": "candidate_0", "original_quality_scores": "raw_score", "finalized_quality_head": "quality_head", "oracle_GT_Dice_offline": "oracle_upper_bound"}
    for key, state in groups.items():
        row = contexts[key]
        row.update(samples=state["dice"].count, tasks=len(state["tasks"]), image_mean_dice=state["dice"].mean(),
                   task_macro_dice=math.fsum(v.mean() for v in state["tasks"].values()) / len(state["tasks"]),
                   image_mean_iou=state["iou"].mean() if state["iou"].count else "",
                   offline_reference="true" if "oracle" in row["selector"].lower() else "false")
        if row["scope"] == "formal_quality_head":
            ref = formal_ref[(row["profile"], row["setting"], row["box_perturbation_percent"])]
            check(row["samples"], ref["samples"], "formal samples"); check(row["tasks"], ref["tasks"], "formal tasks")
            if row["selector"] in formal_fields:
                field = formal_fields[row["selector"]]
                check(row["image_mean_dice"], ref[field + "_sample_mean_dice"], "formal image Dice")
                check(row["task_macro_dice"], ref[field + "_task_macro_dice"], "formal task Dice")
        elif row["profile"]:
            ref = shared_ref[(row["profile"], row["setting"], row["prompt_condition"])]; field = shared_fields[row["selector"]]
            check(row["samples"], ref["samples"], "supplemental Full/Shared samples"); check(row["tasks"], ref["tasks"], "supplemental Full/Shared tasks")
            check(row["image_mean_dice"], ref["sample_mean_" + field + "_dice"], "supplemental Full/Shared image Dice")
            if "task_macro_" + field + "_dice" in ref:
                check(row["task_macro_dice"], ref["task_macro_" + field + "_dice"], "supplemental Full/Shared task Dice")
        else:
            ref = base_ref[(row["model"], row["seed"], row["setting"], row["prompt_condition"], row["selector"])]
            check(row["samples"], ref["sample_count"], "baseline samples")
            check(row["image_mean_dice"], ref["Dice"], "baseline image Dice"); check(row["image_mean_iou"], ref["IoU"], "baseline image IoU")
        output.append(row)
    expected_groups = len(formal_ref) * 4 + len(shared_ref) * 4 + len(base_ref)
    if len(output) != expected_groups:
        raise ValueError(f"Missing output condition: {len(output)} versus {expected_groups}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reproduced_prompt_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(output, key=lambda r: tuple(str(r[f]) for f in FIELDS[:10])))
    result = {"status": "PASS", "operation": "aggregate_saved_CSV_results", "model_execution": False,
              "source_rows": dict(counts), "summary_rows": len(output), "checks": checks, "max_absolute_difference": max_delta}
    (output_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, required=True, help="InfraSegFM_All_Experimental_Data.zip (48-CSV release)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.data_zip, args.output_dir)


if __name__ == "__main__":
    main()
