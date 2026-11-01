"""Launch the frozen Full/Shared prompt protocols with explicit asset bindings.

The launcher itself uses Python's standard library. ``profiles``, ``assets``
and ``--dry-run`` do not import a model or execute inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from _prompt_profiles import CONFIG

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reproduction" / "prompt_protocol"
RUNTIME = ROOT / "reproduction" / "prompt_runtime"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_files(items: list[dict], base: Path) -> list[dict]:
    checked = []
    for item in items:
        path = base / item["path"]
        result = {"path": str(path), "expected_sha256": item["sha256"]}
        if not path.is_file():
            result["status"] = "missing"
        else:
            result["actual_sha256"] = digest(path)
            result["status"] = "verified" if result["actual_sha256"] == item["sha256"] else "hash_mismatch"
        checked.append(result)
    return checked


def runtime_files(config: dict) -> list[dict]:
    return [{"path": p, "sha256": h} for p, h in config["frozen_source_files"].items()]


def model_assets(config: dict, profiles: list[str]) -> list[dict]:
    items = []
    for name in profiles:
        profile = config["profiles"][name]
        items.extend([
            {"path": "inputs/models/" + profile["checkpoint_file"], "sha256": profile["checkpoint_sha256"]},
            {"path": "inputs/heads/" + profile["head_file"], "sha256": profile["head_file_sha256"]},
        ])
    items.extend(config["runtime_compatibility_assets"])
    return items


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["profiles", "assets", "prepare", "preflight", "evaluate", "aggregate", "full-image", "empty"])
    parser.add_argument("--workdir", type=Path, help="isolated protocol directory with inputs/models, inputs/heads and inputs/provenance")
    parser.add_argument("--profile", choices=["full_v1", "full_v2", "shared_v1", "shared_v2"])
    parser.add_argument("--eval-root", type=Path, help="directory containing ID/ and ODD/CrossSite, ODD/CrossTask, ODD/RealWorld")
    parser.add_argument("--source-project-root", type=Path, help="original source root containing playground/InfraSegDBv1 and InfraSegDBv2 development splits")
    parser.add_argument("--sam-checkpoint", type=Path, help="official sam_vit_b_01ec64.pth")
    parser.add_argument("--output-dir", type=Path, help="new output directory for full-image or empty-space prompts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16, help="annotation-free runner batch; empty prompt requires 1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="show the exact command and asset status without importing torch or executing it")
    return parser, parser.parse_args()


def main() -> int:
    parser, args = parse_args()
    config = CONFIG
    if args.mode == "profiles":
        print(json.dumps(config, indent=2))
        return 0
    if args.workdir is None:
        parser.error("--workdir is required for this mode")
    workdir = args.workdir.resolve()
    profiles = [args.profile] if args.profile else list(config["profiles"])
    runtime = inspect_files(runtime_files(config), RUNTIME)
    if args.mode == "assets":
        checked = inspect_files(model_assets(config, profiles), workdir)
        result = {"runtime": runtime, "assets": checked}
        print(json.dumps(result, indent=2))
        return 0 if all(r["status"] == "verified" for r in runtime + checked) else 2
    require_profile = args.mode in {"preflight", "evaluate", "full-image", "empty"}
    if require_profile and not args.profile:
        parser.error("--profile is required for this mode")
    if args.workers != 2:
        parser.error("the recorded protocol uses --workers 2")
    if args.mode == "empty" and args.batch_size != 1:
        parser.error("empty spatial prompts require --batch-size 1")
    if args.mode in {"full-image", "empty"} and args.device != "cuda:0":
        parser.error("the annotation-free runner uses cuda:0; select the visible GPU with CUDA_VISIBLE_DEVICES")
    driver = PROTOCOL / "evaluate_quality_head_formal_test.py"
    if args.mode in {"full-image", "empty"}:
        cmd = [sys.executable, "-B", str(PROTOCOL / "evaluate_annotation_free_prompts.py"), "--formal-root", str(workdir),
               "--driver", str(driver), "--code-root", str(RUNTIME), "--profile", args.profile,
               "--prompt-mode", "empty" if args.mode == "empty" else "fixed_full_image_box",
               "--batch-size", str(args.batch_size), "--workers", "2"]
    else:
        cmd = [sys.executable, "-B", str(driver), args.mode, "--workdir", str(workdir)]
        if args.mode != "aggregate":
            cmd += ["--code-root", str(RUNTIME)]
        if require_profile:
            cmd += ["--profile", args.profile, "--device", args.device]
    required = []
    if args.mode in {"prepare", "evaluate", "full-image", "empty"}:
        required.append("eval_root")
    if args.mode == "preflight":
        required.append("source_project_root")
    if require_profile:
        required.append("sam_checkpoint")
    if args.mode in {"full-image", "empty"}:
        required.append("output_dir")
    for name in required:
        value = getattr(args, name)
        if value is None:
            parser.error("--" + name.replace("_", "-") + " is required for this mode")
        cmd += ["--" + name.replace("_", "-"), str(value.resolve())]
    assets = inspect_files(model_assets(config, [args.profile]), workdir) if require_profile else []
    if require_profile and args.sam_checkpoint.is_file():
        assets += inspect_files([{"path": args.sam_checkpoint.resolve().name, "sha256": config["sam_checkpoint_sha256"]}], args.sam_checkpoint.resolve().parent)
    elif require_profile:
        assets.append({"path": str(args.sam_checkpoint.resolve()), "status": "missing"})
    status = {"mode": args.mode, "command": cmd, "runtime": runtime, "assets": assets}
    if args.dry_run:
        status["status"] = "planned_command_only_no_inference"
        print(json.dumps(status, indent=2))
        return 0
    failed = [r for r in runtime + assets if r["status"] != "verified"]
    if failed:
        print(json.dumps({"status": "required_assets_not_verified", "files": failed}, indent=2), file=sys.stderr)
        return 2
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
