# -*- coding: utf-8 -*-
"""One-command workflow runner for InfraSegFM full and ablation experiments.

The runner calls the existing scripts in this repository:
  - pretrain.py
  - evaluate_internal.py
  - evaluate_external.py
  - finetune.py

It defines two high-level workflows:
  1) --mode full:       the normal complete InfraSegFM pipeline, no ablation.
  2) --mode ablations:  a compact three-ablation pipeline:
        NoSynBridgeCrack, NoStyle, NoRoutePrior.

Use --dry_run first to print all commands without launching long jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

# Force UTF-8 on stdout/stderr. When output is redirected on Windows, Python
# falls back to the ANSI codepage (e.g. GBK), which cannot encode U+FFFD
# replacement characters coming from child-process output and crashes the
# whole workflow with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from typing import Dict, List, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.runtime_logging import append_jsonl


@dataclass(frozen=True)
class Variant:
    name: str
    paper_name: str
    description: str
    extra_train_args: Sequence[str]
    extra_eval_args: Sequence[str]


VARIANTS: Dict[str, Variant] = {
    "Full": Variant(
        name="Full",
        paper_name="Full InfraSegFM",
        description="Complete InfraSegFM with SynBridgeCrack, online style routing, and task-first routing prior.",
        extra_train_args=(),
        extra_eval_args=(),
    ),
    "NoSynBridgeCrack": Variant(
        name="NoSynBridgeCrack",
        paper_name="w/o SynBridgeCrack",
        description="Data ablation: exclude paired .npy samples whose image or mask path contains 'SynBridgeCrack'.",
        extra_train_args=("--exclude_source_keywords", "SynBridgeCrack"),
        extra_eval_args=(),
    ),
    "NoStyle": Variant(
        name="NoStyle",
        paper_name="w/o online style",
        description="Routing ablation: zero out the online style embedding before the MoE gate.",
        extra_train_args=("--moe_disable_style", "1"),
        extra_eval_args=("--moe_disable_style", "1"),
    ),
    "NoRoutePrior": Variant(
        name="NoRoutePrior",
        paper_name="w/o task-first routing prior",
        description="Prior ablation: disable offline preassignment, route-supervision loss, and few-shot early forced routing.",
        extra_train_args=("--moe_preassign", "0", "--moe_route_sup_coef", "0", "--moe_warmup_epochs", "0"),
        extra_eval_args=(),
    ),
    "SharedAdapter": Variant(
        name="SharedAdapter",
        paper_name="parameter-matched shared adapter",
        description="Control: replace the routed experts with a single parameter-matched shared adapter (no gate, no style, no routing prior).",
        extra_train_args=(
            "--adapter_mode", "shared",
            "--moe_preassign", "0",
            "--moe_route_sup_coef", "0",
            "--moe_warmup_epochs", "0",
            "--moe_lb_coef", "0",
            "--moe_ent_coef", "0",
        ),
        extra_eval_args=("--adapter_mode", "shared"),
    ),
    "FullOpt": Variant(
        name="FullOpt",
        paper_name="InfraSegFM (optimized recipe)",
        description="Same architecture and parameter count as Full; training recipe adds cosine LR decay and task-identity dropout p=0.15 for open-set routing.",
        extra_train_args=("--lr_schedule", "cosine", "--task_drop_p", "0.15"),
        extra_eval_args=(),
    ),
    "FullL": Variant(
        name="FullL",
        paper_name="InfraSegFM-L (capacity-matched to Med-SA)",
        description="Scaled InfraSegFM: bottleneck 16->133 so encoder adapter params match Med-SA (589,745 vs 591,744 per block, -0.34%). Same routing, same protocol.",
        extra_train_args=("--bottleneck_dim", "133"),
        extra_eval_args=("--bottleneck_dim", "133"),
    ),
    "MedSA": Variant(
        name="MedSA",
        paper_name="Med-SA (reimpl.)",
        description="Baseline: Med-SA serial adapters (space adapter after attention + scaled MLP adapter), same data/prompt/eval protocol.",
        extra_train_args=(
            "--adapter_mode", "medsa",
            "--moe_preassign", "0",
            "--moe_route_sup_coef", "0",
            "--moe_warmup_epochs", "0",
            "--moe_lb_coef", "0",
            "--moe_ent_coef", "0",
        ),
        extra_eval_args=("--adapter_mode", "medsa"),
    ),
    "SAMLoRA": Variant(
        name="SAMLoRA",
        paper_name="SAM-LoRA (SAMed-style, reimpl.)",
        description="Baseline: SAMed-style LoRA (r=4 on q/v projections) + mask decoder tuning, same data/prompt/eval protocol.",
        extra_train_args=(
            "--adapter_mode", "lora",
            "--moe_preassign", "0",
            "--moe_route_sup_coef", "0",
            "--moe_warmup_epochs", "0",
            "--moe_lb_coef", "0",
            "--moe_ent_coef", "0",
        ),
        extra_eval_args=("--adapter_mode", "lora"),
    ),
}

SHIFT_TO_VERSION = {"CrossSite": "v1", "RealWorld": "v1", "CrossTask": "v2"}
SHIFT_TO_DATA = {
    "CrossSite": "InfraSegDBCrossSite",
    "RealWorld": "InfraSegDBRealWorld",
    "CrossTask": "InfraSegDBCrossTask",
}


def _flag(enabled: bool, name: str) -> List[str]:
    return [name] if enabled else []


def _ids(xs: Sequence[int]) -> List[str]:
    return [str(int(x)) for x in xs]


def _j(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


def _task_name(run_root: str, variant: str, leaf: str) -> str:
    # Forward slashes are accepted by Windows Python and are normalized by os.path.join in called scripts.
    return f"{run_root}/{variant}/{leaf}"


def _weight(root: str, run_root: str, variant: str, leaf: str) -> str:
    return _j(root, run_root, variant, leaf, "model_best.pth")


def _final_epoch_marker(root: str, run_root: str, variant: str, leaf: str, epochs: int) -> str:
    last_epoch = max(int(epochs) - 1, 0)
    return _j(root, run_root, variant, leaf, f"model_{last_epoch}.pth")


def _cache_name(variant: Variant) -> str:
    return f"_moe_routing_cache_{variant.name}.json"


def _exists(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _workflow_status_file(args: argparse.Namespace) -> str:
    return _j(args.root, args.run_root, f"workflow_status_{args.mode}.jsonl")


def _safe_log_stem(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        elif ch in ("\\", "/", ":", " "):
            keep.append("_")
    stem = "".join(keep).strip("._")
    return stem[:140] or "command"


def _command_log_file(args: argparse.Namespace, cmd: List[str], marker: str | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_log_stem(marker or "_".join(cmd[:3]))
    return _j(args.root, args.run_root, "workflow_command_logs", f"{stamp}_{stem}.log")


def _record_workflow_event(args: argparse.Namespace, event: str, cmd: List[str], marker: str | None, **extra) -> None:
    append_jsonl(
        _workflow_status_file(args),
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "marker": marker,
            "marker_exists": _exists(marker) if marker else None,
            "cmd": cmd,
            "cwd": os.getcwd(),
            "dry_run": bool(getattr(args, "dry_run", False)),
            **extra,
        },
    )


def _should_echo_child_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "%|" in stripped and "it/s" in stripped:
        return False
    if stripped.startswith("Epoch [") and (" Train:" in stripped or " Val:" in stripped) and "it/s" in stripped:
        return False
    return True


def _run(cmd: List[str], args: argparse.Namespace, *, marker: str | None, log_file: str) -> None:
    line = " ".join(cmd)
    command_log = _command_log_file(args, cmd, marker)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if marker and args.skip_existing and _exists(marker):
        print(f"[SKIP] {marker}")
        print(f"       {line}")
        _record_workflow_event(args, "skip_existing", cmd, marker)
        return
    print("\n[RUN] " + line)
    if args.dry_run:
        _record_workflow_event(args, "dry_run", cmd, marker, command_log=command_log)
        return
    _record_workflow_event(args, "start", cmd, marker, command_log=command_log)
    os.makedirs(os.path.dirname(command_log), exist_ok=True)
    with open(command_log, "a", encoding="utf-8", errors="replace") as lf:
        lf.write(f"[RUN] {line}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for out_line in proc.stdout:
                if _should_echo_child_line(out_line):
                    print(out_line, end="", flush=True)
                lf.write(out_line)
                lf.flush()
            returncode = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            returncode = proc.wait()
            _record_workflow_event(args, "interrupted", cmd, marker, returncode=returncode, command_log=command_log)
            raise
    if returncode != 0:
        # A child can crash during interpreter teardown (CUDA vs. tensorboard/tqdm
        # daemon threads -> Windows 0xC0000005 access violation) AFTER its real
        # output was already written to disk. If the step's expected marker exists,
        # the work is actually complete; log a warning and continue instead of
        # aborting the whole workflow. (pretrain/finetune markers are the final
        # model_N.pth, only written on genuine completion; eval markers are the
        # result CSV.) A genuine mid-run crash leaves no marker and still raises.
        if marker and _exists(marker):
            print(f"[WARN] child exited {returncode} but marker present; treating as success: {marker}", flush=True)
            _record_workflow_event(args, "nonzero_exit_marker_ok", cmd, marker, returncode=returncode, command_log=command_log)
        else:
            _record_workflow_event(args, "failed", cmd, marker, returncode=returncode, command_log=command_log)
            raise subprocess.CalledProcessError(returncode, cmd)
    else:
        _record_workflow_event(args, "done", cmd, marker, command_log=command_log)


def _base(args: argparse.Namespace) -> List[str]:
    return [
        "--checkpoint", args.checkpoint,
        "--model_type", args.model_type,
        "--device", args.device,
        "--device_ids", *_ids(args.device_ids),
        "--num_workers", str(args.num_workers),
        "--pin_memory", str(args.pin_memory),
    ]


def _train(args: argparse.Namespace, *, pretrain: bool) -> List[str]:
    return [
        "--num_epochs", str(args.pretrain_epochs if pretrain else args.finetune_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr if pretrain else args.ft_lr),
        "--weight_decay", str(args.weight_decay),
        "--mask_decoder_lr_mult", str(args.mask_decoder_lr_mult),
        "--task_balanced_sampling", str(args.task_balanced_sampling),
        "--task_sampling_alpha", str(args.task_sampling_alpha),
        "--task_sampling_power", str(args.task_sampling_power),
        "--save_every", str(args.save_every),
        "--heartbeat_interval", str(args.heartbeat_interval),
        "--disable_tqdm", str(args.disable_tqdm),
        *_flag(args.use_amp, "--use_amp"),
    ]


def _eval(args: argparse.Namespace) -> List[str]:
    return [
        "--checkpoint", args.checkpoint,
        "--model_type", args.model_type,
        "--device", args.device,
        "--device_ids", *_ids(args.device_ids),
        "--batch_size", str(args.eval_batch_size),
        "--num_workers", str(args.num_workers),
        "--pin_memory", str(args.pin_memory),
        "--metric", *args.metric,
        *_flag(args.vis, "--vis"),
    ]


def _variant_train_args(v: Variant, args: argparse.Namespace) -> List[str]:
    out = ["--moe_cache_name", _cache_name(v), *v.extra_train_args]
    if args.rebuild_cache:
        out.append("--moe_preassign_force")
    return out


def _variant_eval_args(v: Variant) -> List[str]:
    return list(v.extra_eval_args)


def pretrain_version(args: argparse.Namespace, v: Variant, version: str, log_file: str) -> None:
    data = "InfraSegDBv1" if version == "v1" else "InfraSegDBv2"
    leaf = f"InfraSegFM{version}"
    marker = _final_epoch_marker(args.root, args.run_root, v.name, leaf, args.pretrain_epochs)
    cmd = [
        sys.executable, "pretrain.py",
        *_base(args), *_train(args, pretrain=True),
        "--data_path", _j(args.root, data),
        "--work_dir", args.root,
        "--task_name", _task_name(args.run_root, v.name, leaf),
        *_variant_train_args(v, args),
    ]
    _run(cmd, args, marker=marker, log_file=log_file)


def eval_id(args: argparse.Namespace, v: Variant, log_file: str) -> None:
    weight = _weight(args.root, args.run_root, v.name, "InfraSegFMv1")
    marker = _j(os.path.dirname(weight), "internal", "ID.csv")
    cmd = [
        sys.executable, "evaluate_internal.py",
        *_eval(args),
        "--model_weight", weight,
        "--data_path", _j(args.root, "eval", "ID"),
        *_variant_eval_args(v),
    ]
    _run(cmd, args, marker=marker, log_file=log_file)


def _external_marker(out_dir: str, shift: str) -> str:
    # evaluate_external.py writes <basename(data_path)>-<shift_type with / replaced by _>.csv
    return _j(out_dir, f"eval-ODD_{shift}.csv")


def eval_external(args: argparse.Namespace, v: Variant, shift: str, ratio: int, weight: str, log_file: str) -> None:
    version = SHIFT_TO_VERSION[shift]
    result_name = f"train{ratio}Result"
    out_dir = _j(args.root, args.run_root, v.name, f"InfraSegFM{version}_finetune_{shift}", result_name)
    cmd = [
        sys.executable, "evaluate_external.py",
        *_eval(args),
        "--model_weight", weight,
        "--data_path", _j(args.root, "eval"),
        "--shift_type", f"ODD/{shift}",
        "--out_dir", out_dir,
        *_variant_eval_args(v),
    ]
    _run(cmd, args, marker=_external_marker(out_dir, shift), log_file=log_file)


def finetune(args: argparse.Namespace, v: Variant, shift: str, ratio: int, log_file: str) -> str:
    version = SHIFT_TO_VERSION[shift]
    base_weight = _weight(args.root, args.run_root, v.name, f"InfraSegFM{version}")
    leaf = f"InfraSegFM{version}_finetune_{shift}/train{ratio}"
    marker = _final_epoch_marker(args.root, args.run_root, v.name, leaf, args.finetune_epochs)
    cmd = [
        sys.executable, "finetune.py",
        *_base(args), *_train(args, pretrain=False),
        "--data_path", _j(args.root, SHIFT_TO_DATA[shift]),
        "--train_split", f"train{ratio}",
        "--val_split", "val",
        "--resume", base_weight,
        "--work_dir", args.root,
        "--task_name", _task_name(args.run_root, v.name, leaf),
        *_variant_train_args(v, args),
    ]
    _run(cmd, args, marker=marker, log_file=log_file)
    return _weight(args.root, args.run_root, v.name, leaf)


def variants_for(args: argparse.Namespace) -> List[Variant]:
    if args.mode == "full":
        return [VARIANTS["Full"]]
    if args.variants:
        return [VARIANTS[x] for x in args.variants]
    return [VARIANTS["NoSynBridgeCrack"], VARIANTS["NoStyle"], VARIANTS["NoRoutePrior"]]


def main() -> None:
    p = argparse.ArgumentParser("Run InfraSegFM full or ablation workflow")
    p.add_argument("--mode", choices=["full", "ablations"], required=True)
    p.add_argument("--variants", nargs="*", choices=[k for k in VARIANTS if k != "Full"], default=None)
    p.add_argument("--checkpoint", default="./playground/SAM")
    p.add_argument("--model_type", default="vit_b")
    p.add_argument("--root", default="./playground")
    p.add_argument("--run_root", default="ablation_runs")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device_ids", nargs="+", type=int, default=[0])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", type=int, default=0, choices=[0, 1])
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--finetune_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=80)
    p.add_argument("--eval_batch_size", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ft_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--mask_decoder_lr_mult", type=float, default=0.1)
    p.add_argument("--task_balanced_sampling", type=int, default=1, choices=[0, 1])
    p.add_argument("--task_sampling_alpha", type=float, default=0.25)
    p.add_argument("--task_sampling_power", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=0,
                   help="numbered training checkpoints: 0 saves only the final model_N.pth; >0 saves every N epochs")
    p.add_argument("--heartbeat_interval", type=int, default=100,
                   help="write pretrain/finetune heartbeat JSON every N batches; 0 disables batch heartbeats")
    p.add_argument("--disable_tqdm", type=int, default=0, choices=[0, 1],
                   help="disable tqdm progress bars in pretrain/finetune; heartbeat status remains available")
    p.add_argument("--metric", nargs="+", default=["dsc", "hd"])
    p.add_argument("--shifts", nargs="+", choices=["CrossSite", "RealWorld", "CrossTask"], default=["CrossSite", "RealWorld", "CrossTask"])
    p.add_argument("--ratios", nargs="+", type=int, default=[0, 5, 15, 25, 50, 100])
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    os.makedirs(_j(args.root, args.run_root), exist_ok=True)
    log_file = _j(args.root, args.run_root, f"commands_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    manifest = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "variants": [asdict(v) for v in variants_for(args)],
        "args": vars(args),
    }
    with open(_j(args.root, args.run_root, f"manifest_{args.mode}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    for v in variants_for(args):
        print(f"\n========== {v.name}: {v.description} ==========")
        pretrain_version(args, v, "v1", log_file)
        eval_id(args, v, log_file)
        if "CrossTask" in args.shifts:
            pretrain_version(args, v, "v2", log_file)
        for shift in args.shifts:
            for ratio in args.ratios:
                if int(ratio) == 0:
                    version = SHIFT_TO_VERSION[shift]
                    weight = _weight(args.root, args.run_root, v.name, f"InfraSegFM{version}")
                    eval_external(args, v, shift, 0, weight, log_file)
                else:
                    ft_weight = finetune(args, v, shift, int(ratio), log_file)
                    eval_external(args, v, shift, int(ratio), ft_weight, log_file)
    print("\n[Done] All requested commands completed.")
    print(f"Command log: {log_file}")


if __name__ == "__main__":
    main()
