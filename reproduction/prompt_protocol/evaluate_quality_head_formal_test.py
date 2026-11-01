"""Frozen, test-only evaluation for finalized InfraSegFM IoU-quality heads.

This driver deliberately has no training code.  It reconstructs exactly one
historical Full/Shared checkpoint, verifies the selected six-tensor quality
head against its recorded hashes, and evaluates the three *returned*
multimask candidates under three fixed reporting rules:

* fixed returned candidate 0;
* argmax of the finalized, separately loaded quality head; and
* GT-best candidate (offline oracle only).

The quality head is kept in a clone during normal evaluation.  Consequently a
single full-model forward produces the shared three candidate masks, while the
three reporting rules only select from those same candidates.  The original
full checkpoint and copied source tree are never modified.

The supported workflow is intentionally narrow:

``prepare``
    Create one immutable final-test membership and deterministic prompt cache.
``preflight``
    On three source-development samples, prove that temporarily overlaying the
    finalized head changes score output only and leaves candidate masks intact.
``evaluate``
    Evaluate one fixed profile on its protocol-assigned final-test settings.
``aggregate``
    Validate all four completed profiles and create paired Full--Shared tables.

No subcommand trains, fine-tunes, selects a head from test results, creates a
new seed, or changes candidate order/threshold/preprocessing/precision.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize


EPS = 1e-6
SCHEMA = "infrasegfm-quality-head-formal-test-v1"
HEAD_PREFIX = "mask_decoder.iou_prediction_head."
SAM_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
# ``build_sam`` resizes SAM-B's 1024-pixel positional tensors when constructing
# the recorded 256-pixel model.  The 3090 source run used torch 2.7.1+cu118;
# the formal-test server uses torch 2.5.1.  Their interpolation kernels produce
# slightly different values, even though the source files and original SAM-B
# checkpoint match.  This small, hash-pinned payload contains only the nine
# source-runtime interpolation outputs from the audited 3090 build.  It lets
# the server reconstruct the same frozen segmentation state; it contains no
# complete-model or IoU-head tensors and is never trained or changed.
SAM_RUNTIME_OVERLAY_SCHEMA = "infrasegfm-sam-b-256-interpolation-overlay-v1"
SAM_RUNTIME_OVERLAY_FILE = "sam_b_256_interpolation_overlay.pt"
SAM_RUNTIME_OVERLAY_MANIFEST = "sam_b_256_interpolation_overlay.manifest.json"
SAM_RUNTIME_OVERLAY_SHA256 = "336bedafb02f9f440a6d4f8f390de07bcf7de467c94ef468e2ee8a5e28f679bc"
SAM_RUNTIME_OVERLAY_MANIFEST_SHA256 = "209a41bf3033fad7fd45dac61cec0d930be0fcd60b787848dfbd1cb657db4a34"
SAM_RUNTIME_OVERLAY_SOURCE_TORCH = "2.7.1+cu118"
SAM_RUNTIME_OVERLAY_KEYS: tuple[str, ...] = (
    "image_encoder.pos_embed",
    "image_encoder.blocks.2.attn.rel_pos_h",
    "image_encoder.blocks.2.attn.rel_pos_w",
    "image_encoder.blocks.5.attn.rel_pos_h",
    "image_encoder.blocks.5.attn.rel_pos_w",
    "image_encoder.blocks.8.attn.rel_pos_h",
    "image_encoder.blocks.8.attn.rel_pos_w",
    "image_encoder.blocks.11.attn.rel_pos_h",
    "image_encoder.blocks.11.attn.rel_pos_w",
)
PROMPT_SEED = 26_092_001
PERTURBATION_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)

# These are the exact frozen source files used for quality-head selection on
# the 3090.  The server runs an isolated source copy with this fingerprint.
EXPECTED_SOURCE_FILES = {
    "model/infra_segfm.py": "2210f47b9ea81fd76e5fb1f044b071ff895ce6808f6fd48ecce4e2abdb5e26f5",
    "segment_anything/modeling/mask_decoder.py": "e268fae12dc719ea30b8dbeafb5f4534a4a4f93026286e104a8b0ee49bc9a790",
    "segment_anything/build_sam.py": "7832a8055f3aca1d7410e93b1e839114b02a178f7144acb94fb2042669840309",
    "data/dataset.py": "0c24a5edf88c070ddce8ca2585b76158ac67c93a8a2fa28bf879c10a07582a48",
    "pretrain.py": "7c505c8a3e9443da1d3563fd26e0a9e8546f2b7cbbe9b1e1c5a6d3191c0e8324",
    "evaluate_internal.py": "e387baeeb67e66a383261717a184d073c54cc7bf79b9bd48515857b50588243b",
}

# The paths are relative to the isolated delivery directory.  A launcher must
# place only the four pre-existing full checkpoints and four small head files
# here.  There is deliberately no generic --checkpoint override.
PROFILES: Dict[str, Dict[str, Any]] = {
    "full_v1": {
        "model_label": "Full v1",
        "family": "Full",
        "version": "v1",
        "adapter_mode": "moe",
        "settings": ("ID", "CrossSite", "RealWorld"),
        "source_data_relative": "playground/InfraSegDBv1",
        "expected_source_dev_samples": 28_626,
        "expected_source_dev_tasks": 37,
        "checkpoint_file": "full_v1_model_best.pth",
        "checkpoint_sha256": "a6712c32bc9cb46465a689388de4e6a611fbb7188ece45150db932019751a618",
        "historical_checkpoint_epoch": 25,
        "head_file": "full_v1/quality_head_best.pt",
        "head_file_sha256": "c19a9501266cd4afa6d8e0fae7613f1eccc475eb035f17af0267f33bf0fe7030",
        "head_state_sha256": "66524a41646d7bb02d728894688c8b85a8aa21ce492b134c5be2b5ca81cef8d1",
        "selected_head_epoch": 7,
        "evaluation_batch_size": 32,
        "legacy_head_binding": {
            "head_bytes": 541553,
            "selected_config_sha256": "b51176b47492c3da912ec9803fa0d7fc21d8356561c3b073ccdb48c60538f6a1",
            "source_non_head_state_sha256": "0f2d9d0c88928fe1ae92d08c472c224823c4bf903fcfc6a77fb235799bf9ca86",
        },
    },
    "full_v2": {
        "model_label": "Full v2",
        "family": "Full",
        "version": "v2",
        "adapter_mode": "moe",
        "settings": ("CrossTask",),
        "source_data_relative": "playground/InfraSegDBv2",
        "expected_source_dev_samples": 27_312,
        "expected_source_dev_tasks": 32,
        "checkpoint_file": "full_v2_model_best.pth",
        "checkpoint_sha256": "c44fc5ecc71188d59033a174ca48f8ca2de1a4445923ea760ea1d0bb6544c405",
        "historical_checkpoint_epoch": 29,
        "head_file": "full_v2/quality_head_best.pt",
        "head_file_sha256": "205c05eb10ff7a9777ba1f97756137d7a08215dbbf6193b8dbc0f1204894ed8f",
        "head_state_sha256": "550f81613cd5f2c1fe9b91617b8131eca7a92c8e45b9b72f0ac6ba2563f8052c",
        "selected_head_epoch": 11,
        "evaluation_batch_size": 64,
    },
    "shared_v1": {
        "model_label": "Shared v1",
        "family": "Shared",
        "version": "v1",
        "adapter_mode": "shared",
        "settings": ("ID", "CrossSite", "RealWorld"),
        "source_data_relative": "playground/InfraSegDBv1",
        "expected_source_dev_samples": 28_626,
        "expected_source_dev_tasks": 37,
        "checkpoint_file": "shared_v1_model_best.pth",
        "checkpoint_sha256": "580fa10aa411ca9f3df59c56be60db2e9f260a345cde1985fafa4ce9e4578768",
        "historical_checkpoint_epoch": 23,
        "head_file": "shared_v1/quality_head_best.pt",
        "head_file_sha256": "48c7ae3a040a0d73fcb4fa6d145d62426c3ec9f317ba843badbad7166709a564",
        "head_state_sha256": "f47224f3ce17105afc4fedc1e3ec248871dac96b68f6b26cba5577135e4452b1",
        "selected_head_epoch": 2,
        "evaluation_batch_size": 64,
    },
    "shared_v2": {
        "model_label": "Shared v2",
        "family": "Shared",
        "version": "v2",
        "adapter_mode": "shared",
        "settings": ("CrossTask",),
        "source_data_relative": "playground/InfraSegDBv2",
        "expected_source_dev_samples": 27_312,
        "expected_source_dev_tasks": 32,
        "checkpoint_file": "shared_v2_model_best.pth",
        "checkpoint_sha256": "d86eddbbde51a268963e99eed9d1a2d7edac70ef4e4b3870a38339e908fbe8b0",
        "historical_checkpoint_epoch": 29,
        "head_file": "shared_v2/quality_head_best.pt",
        "head_file_sha256": "0923e6b2c40e69ee1363dca0d1cfc08bc52fa2f91754d7b080267107810fbe66",
        "head_state_sha256": "e35c6072355c834a580343ace5c5f17dc0077e601cea1e59e987cb636b6898bf",
        "selected_head_epoch": 6,
        "evaluation_batch_size": 64,
    },
}

TEST_SETTINGS: Dict[str, Dict[str, Any]] = {
    "ID": {
        "relative": "ID",
        "expected_tasks": 38,
        "expected_samples": 35_879,
        "metric_resize_antialias": True,
        "metric_variant": "evaluate_internal: bilinear, align_corners=False, antialias=True",
        "protocol_scope": "v1 ID final test; 0% target data",
    },
    "CrossSite": {
        "relative": "ODD/CrossSite",
        "expected_tasks": 2,
        "expected_samples": 3_013,
        "metric_resize_antialias": False,
        "metric_variant": "evaluate_external: bilinear, align_corners=False",
        "protocol_scope": "v1 CrossSite final test; 0% target data",
    },
    "CrossTask": {
        "relative": "ODD/CrossTask",
        "expected_tasks": 6,
        "expected_samples": 1_653,
        "metric_resize_antialias": False,
        "metric_variant": "evaluate_external: bilinear, align_corners=False",
        "protocol_scope": "v2 CrossTask final test; 0% target data",
    },
    "RealWorld": {
        "relative": "ODD/RealWorld",
        "expected_tasks": 4,
        "expected_samples": 907,
        "metric_resize_antialias": False,
        "metric_variant": "evaluate_external: bilinear, align_corners=False",
        "protocol_scope": "v1 RealWorld final test; 0% target data",
    },
}

# Development-only reference values are carried forward solely to describe
# whether the already locked development benefit has the same direction on the
# untouched final test.  They are never consulted by ``evaluate``.
DEVELOPMENT_REFERENCE = {
    "full_v1": {"candidate_0_macro": 0.6730008195050154, "tuned_macro": 0.6894517137665795,
                "candidate_0_micro": 0.6826274991035461, "tuned_micro": 0.6873140335083008},
    "full_v2": {"candidate_0_macro": 0.669210, "tuned_macro": 0.685027,
                "candidate_0_micro": 0.684828, "tuned_micro": 0.689548},
    "shared_v1": {"candidate_0_macro": 0.643031, "tuned_macro": 0.662346,
                  "candidate_0_micro": 0.669906, "tuned_micro": 0.673942},
    "shared_v2": {"candidate_0_macro": 0.645367, "tuned_macro": 0.665734,
                  "candidate_0_micro": 0.672628, "tuned_micro": 0.677982},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        require(torch.is_tensor(value), f"State value is not a tensor: {key}")
        cpu = value.detach().to("cpu").contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch releases before weights_only.
        return torch.load(path, map_location="cpu")


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, torch.Tensor):
        return clean_json(value.detach().to("cpu").tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(clean_json(value), handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(clean_json(row))
    os.replace(temporary, path)


def setup_logger(path: Path, label: str) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"quality_head_formal.{label}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def prepare_imports(code_root: Path) -> None:
    code_root = code_root.resolve()
    text = str(code_root)
    if text not in sys.path:
        sys.path.insert(0, text)
    os.chdir(text)
    sys.dont_write_bytecode = True


def source_files(code_root: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in EXPECTED_SOURCE_FILES:
        path = code_root / relative
        require(path.is_file(), f"Missing frozen source file: {path}")
        result[relative] = sha256(path)
    return result


def assert_source_fingerprint(code_root: Path) -> Dict[str, str]:
    actual = source_files(code_root)
    require(actual == EXPECTED_SOURCE_FILES, f"Frozen source fingerprint mismatch: {actual}")
    return actual


def source_non_head_digest(model: Any) -> str:
    return canonical_state_sha256({
        key: value for key, value in model.sam.state_dict().items()
        if HEAD_PREFIX not in key
    })


def load_sam_runtime_overlay(workdir: Path, sam_checkpoint: Path) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load the narrow, source-runtime-only SAM interpolation compatibility payload.

    The payload is deliberately checked independently from every complete
    checkpoint and quality head.  It only restores values that are omitted by
    the historical complete-model checkpoints and therefore originate from the
    frozen SAM-B 1024-to-256 construction path.
    """
    provenance = workdir / "inputs" / "provenance"
    overlay_path = provenance / SAM_RUNTIME_OVERLAY_FILE
    manifest_path = provenance / SAM_RUNTIME_OVERLAY_MANIFEST
    require(overlay_path.is_file(), f"Missing source-runtime SAM overlay: {overlay_path}")
    require(manifest_path.is_file(), f"Missing source-runtime SAM overlay manifest: {manifest_path}")
    require(sha256(overlay_path) == SAM_RUNTIME_OVERLAY_SHA256,
            "Source-runtime SAM overlay hash mismatch")
    require(sha256(manifest_path) == SAM_RUNTIME_OVERLAY_MANIFEST_SHA256,
            "Source-runtime SAM overlay manifest hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Source-runtime SAM overlay manifest is not valid JSON") from error
    require(isinstance(manifest, Mapping), "Source-runtime SAM overlay manifest is malformed")
    require(manifest.get("schema") == SAM_RUNTIME_OVERLAY_SCHEMA,
            "Source-runtime SAM overlay manifest schema mismatch")
    require(manifest.get("scope")
            == "Exact frozen SAM-B tensors produced by the original 3090 build_sam resize path; only source-runtime compatibility, no quality-head or complete-model tensor.",
            "Source-runtime SAM overlay manifest scope mismatch")
    require(str(manifest.get("source_sam_checkpoint_sha256")) == SAM_SHA256
            and sha256(sam_checkpoint) == SAM_SHA256,
            "Source-runtime SAM overlay binds a different SAM-B checkpoint")
    require(int(manifest.get("image_size", -1)) == 256
            and manifest.get("keep_resolution") is True
            and str(manifest.get("source_torch_version")) == SAM_RUNTIME_OVERLAY_SOURCE_TORCH,
            "Source-runtime SAM overlay construction context mismatch")
    require(tuple(manifest.get("keys", ())) == SAM_RUNTIME_OVERLAY_KEYS,
            "Source-runtime SAM overlay manifest key order mismatch")
    require(str(manifest.get("overlay_file")) == SAM_RUNTIME_OVERLAY_FILE
            and str(manifest.get("overlay_file_sha256")) == SAM_RUNTIME_OVERLAY_SHA256
            and int(manifest.get("overlay_file_bytes", -1)) == overlay_path.stat().st_size,
            "Source-runtime SAM overlay manifest does not bind this exact payload")

    payload = torch_load(overlay_path)
    require(isinstance(payload, Mapping) and payload.get("schema") == SAM_RUNTIME_OVERLAY_SCHEMA,
            "Source-runtime SAM overlay payload schema mismatch")
    require(str(payload.get("source_sam_checkpoint_sha256")) == SAM_SHA256
            and int(payload.get("image_size", -1)) == 256
            and payload.get("keep_resolution") is True
            and str(payload.get("source_torch_version")) == SAM_RUNTIME_OVERLAY_SOURCE_TORCH,
            "Source-runtime SAM overlay payload construction context mismatch")
    require(tuple(payload.get("keys", ())) == SAM_RUNTIME_OVERLAY_KEYS,
            "Source-runtime SAM overlay payload key order mismatch")
    raw_state = payload.get("state")
    require(isinstance(raw_state, Mapping) and set(raw_state) == set(SAM_RUNTIME_OVERLAY_KEYS),
            "Source-runtime SAM overlay state keys mismatch")
    manifest_tensor_hashes = manifest.get("tensor_sha256")
    require(isinstance(manifest_tensor_hashes, Mapping)
            and set(manifest_tensor_hashes) == set(SAM_RUNTIME_OVERLAY_KEYS),
            "Source-runtime SAM overlay tensor provenance is incomplete")
    state: Dict[str, torch.Tensor] = {}
    for key in SAM_RUNTIME_OVERLAY_KEYS:
        value = raw_state[key]
        require(torch.is_tensor(value) and value.dtype == torch.float32
                and torch.isfinite(value).all().item(),
                f"Source-runtime SAM overlay tensor is invalid: {key}")
        copied = value.detach().to("cpu").contiguous().clone()
        require(tensor_sha256(copied) == str(manifest_tensor_hashes[key]),
                f"Source-runtime SAM overlay tensor hash mismatch: {key}")
        state[key] = copied
    return state, {
        "kind": "source_runtime_sam_b_1024_to_256_interpolation_compatibility",
        "overlay": str(overlay_path),
        "overlay_sha256": SAM_RUNTIME_OVERLAY_SHA256,
        "manifest": str(manifest_path),
        "manifest_sha256": SAM_RUNTIME_OVERLAY_MANIFEST_SHA256,
        "source_torch_version": SAM_RUNTIME_OVERLAY_SOURCE_TORCH,
        "sam_checkpoint_sha256": SAM_SHA256,
        "image_size": 256,
        "keep_resolution": True,
        "tensor_keys": list(SAM_RUNTIME_OVERLAY_KEYS),
        "tensor_sha256": {key: str(manifest_tensor_hashes[key]) for key in SAM_RUNTIME_OVERLAY_KEYS},
    }


def apply_sam_runtime_overlay(sam: Any, state: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Atomically copy only the audited source-runtime tensors into SAM-B."""
    destination = sam.state_dict()
    require(set(state) == set(SAM_RUNTIME_OVERLAY_KEYS), "Unexpected SAM runtime overlay keys")
    before: Dict[str, str] = {}
    with torch.no_grad():
        for key in SAM_RUNTIME_OVERLAY_KEYS:
            require(key in destination, f"SAM runtime overlay destination key is absent: {key}")
            source = state[key]
            target = destination[key]
            require(tuple(target.shape) == tuple(source.shape) and target.dtype == source.dtype,
                    f"SAM runtime overlay tensor shape/dtype mismatch: {key}")
            before[key] = tensor_sha256(target)
            target.copy_(source)
            require(torch.equal(target, source), f"SAM runtime overlay copy failed: {key}")
    after = {key: tensor_sha256(destination[key]) for key in SAM_RUNTIME_OVERLAY_KEYS}
    return {
        "applied": True,
        "tensor_count": len(SAM_RUNTIME_OVERLAY_KEYS),
        "server_pre_overlay_tensor_sha256": before,
        "source_runtime_tensor_sha256": after,
    }


def full_head_state(model: Any) -> Dict[str, torch.Tensor]:
    state = {
        key: value.detach().to("cpu").clone()
        for key, value in model.sam.state_dict().items()
        if HEAD_PREFIX in key
    }
    require(len(state) == 6, f"Expected six original IoU-head tensors, got {len(state)}")
    require(all(torch.isfinite(value).all().item() for value in state.values()), "Original IoU head has NaN/Inf")
    return dict(sorted(state.items()))


def local_head_state(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if all(str(key).startswith(HEAD_PREFIX) for key in state):
        state = {str(key)[len(HEAD_PREFIX):]: value for key, value in state.items()}
    result = {str(key): value.detach().to("cpu").clone() for key, value in state.items()}
    require(len(result) == 6, f"Expected six local IoU-head tensors, got {len(result)}")
    require(all(not key.startswith(HEAD_PREFIX) for key in result), "IoU-head prefix normalization failed")
    require(all(torch.isfinite(value).all().item() for value in result.values()), "Finalized IoU head has NaN/Inf")
    return dict(sorted(result.items()))


def profile_config(name: str) -> Dict[str, Any]:
    require(name in PROFILES, f"Unknown profile: {name}")
    return dict(PROFILES[name])


def profile_paths(workdir: Path, profile: Mapping[str, Any]) -> Dict[str, Path]:
    paths = {
        "checkpoint": workdir / "inputs" / "models" / str(profile["checkpoint_file"]),
        "head": workdir / "inputs" / "heads" / str(profile["head_file"]),
    }
    for label, path in paths.items():
        require(path.is_file(), f"Missing {label} for {profile['model_label']}: {path}")
    return paths


def load_full_v1_head_binding(
    head_path: Path,
    profile: Mapping[str, Any],
    payload: Mapping[str, Any],
    selected: Mapping[str, Any],
    actual_head_hash: str,
    state_hash: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate the fixed Full v1 head against its recorded binding."""
    recorded = profile.get("legacy_head_binding")
    require(isinstance(recorded, Mapping), "Full v1 head binding is malformed")
    locked_full_v1 = PROFILES["full_v1"]
    identity_fields = (
        "model_label", "family", "version", "adapter_mode", "checkpoint_file",
        "checkpoint_sha256", "historical_checkpoint_epoch", "head_file",
        "head_file_sha256", "head_state_sha256", "selected_head_epoch",
    )
    require(all(profile.get(key) == locked_full_v1.get(key) for key in identity_fields),
            "Full v1 head profile differs from its recorded binding")
    require(dict(recorded) == locked_full_v1["legacy_head_binding"],
            "Full v1 compatibility metadata differs from its recorded binding")
    require(str(payload.get("schema")) == "infrasegfm-quality-head-tuning-v1-head-only-weight",
            "Full v1 head payload schema mismatch")
    require(payload.get("compatibility") is None,
            "Full v1 profile binding expects its original head payload format")
    require(actual_head_hash == str(profile["head_file_sha256"])
            and sha256(head_path) == actual_head_hash
            and head_path.stat().st_size == int(recorded["head_bytes"]),
            "Full v1 head file binding mismatch")
    require(state_hash == str(profile["head_state_sha256"]),
            "Full v1 head state binding mismatch")
    require(str(payload.get("source_checkpoint_sha256")) == str(profile["checkpoint_sha256"]),
            "Full v1 head source checkpoint binding mismatch")
    require(int(selected.get("best_epoch", -1))
            == int(profile["selected_head_epoch"]),
            "Full v1 head selected epoch binding mismatch")
    selected_hash = hashlib.sha256(json.dumps(
        clean_json(selected), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    require(selected_hash == str(recorded["selected_config_sha256"]),
            "Full v1 head selected configuration binding mismatch")

    compatibility = {
        "source_non_head_state_sha256": str(recorded["source_non_head_state_sha256"]),
        "candidate_score_layout": "three returned candidates map to raw IoU-score indices [1:4]",
        "source_files_sha256": dict(EXPECTED_SOURCE_FILES),
    }
    binding = {
        "kind": "embedded_full_v1_binding",
        "head_file_sha256": actual_head_hash,
        "head_state_sha256": state_hash,
        "selected_config_sha256": selected_hash,
        "payload_compatibility_present": False,
    }
    return compatibility, binding


def load_final_head(head_path: Path, profile: Mapping[str, Any]) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    actual_hash = sha256(head_path)
    require(actual_hash == profile["head_file_sha256"], f"Finalized head file hash mismatch: {actual_hash}")
    payload = torch_load(head_path)
    require(isinstance(payload, Mapping), "Finalized head payload is malformed")
    state_raw = payload.get("state")
    require(isinstance(state_raw, Mapping), "Finalized head payload lacks state")
    state = local_head_state(state_raw)
    state_hash = canonical_state_sha256(state)
    require(state_hash == profile["head_state_sha256"], f"Finalized head state hash mismatch: {state_hash}")
    require(
        str(payload.get("source_checkpoint_sha256")) == str(profile["checkpoint_sha256"]),
        "Finalized head was selected from a different complete checkpoint",
    )
    selected = payload.get("selected_config")
    require(isinstance(selected, Mapping), "Finalized head lacks selected configuration")
    payload_compatibility = payload.get("compatibility")
    if isinstance(payload_compatibility, Mapping):
        compatibility = dict(payload_compatibility)
        compatibility_binding: Dict[str, Any] = {
            "kind": "payload_compatibility",
            "payload_compatibility_present": True,
        }
    else:
        require(payload_compatibility is None,
                "Finalized head compatibility block is malformed")
        # Full v1 keeps its compatibility metadata in the profile binding.
        require("legacy_head_binding" in profile,
                "Finalized head lacks the required compatibility block")
        compatibility, compatibility_binding = load_full_v1_head_binding(
            head_path, profile, payload, selected, actual_hash, state_hash,
        )

    def declared_compatibility(key: str) -> Any:
        if key in payload:
            return payload[key]
        return compatibility.get(key)

    require(
        str(declared_compatibility("candidate_score_layout"))
        == "three returned candidates map to raw IoU-score indices [1:4]",
        "Finalized head candidate-score layout is not the audited three-candidate mapping",
    )
    require(
        int(selected.get("best_epoch", -1)) == int(profile["selected_head_epoch"]),
        "Finalized head epoch differs from the locked development selection",
    )
    payload_sources = declared_compatibility("source_files_sha256")
    require(isinstance(payload_sources, Mapping) and dict(payload_sources) == EXPECTED_SOURCE_FILES,
            "Finalized head was produced with a different frozen source fingerprint")
    return state, {
        "file": str(head_path),
        "file_sha256": actual_hash,
        "state_sha256": state_hash,
        "payload_schema": payload.get("schema"),
        "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
        "source_non_head_state_sha256": declared_compatibility("source_non_head_state_sha256"),
        "selected_config": dict(selected),
        "candidate_score_layout": declared_compatibility("candidate_score_layout"),
        "compatibility_binding": compatibility_binding,
    }


def load_exact_model(
    workdir: Path,
    code_root: Path,
    profile: Mapping[str, Any],
    checkpoint: Path,
    sam_checkpoint: Path,
    device: torch.device,
) -> tuple[Any, Dict[str, Any]]:
    """Build one recorded architecture and require every saved tensor to match."""
    prepare_imports(code_root)
    from model import InfraSegFM
    from segment_anything import sam_model_registry

    require(sha256(checkpoint) == profile["checkpoint_sha256"], "Complete checkpoint hash mismatch")
    require(sam_checkpoint.is_file(), f"SAM-B checkpoint is missing: {sam_checkpoint}")
    require(sha256(sam_checkpoint) == SAM_SHA256, "SAM-B checkpoint hash mismatch")
    checkpoint_payload = torch_load(checkpoint)
    require(isinstance(checkpoint_payload, Mapping) and isinstance(checkpoint_payload.get("model"), Mapping),
            "Complete checkpoint has no model state")
    saved = checkpoint_payload["model"]
    tensor_state = {key: value for key, value in saved.items() if torch.is_tensor(value)}
    require(not (set(SAM_RUNTIME_OVERLAY_KEYS) & set(tensor_state)),
            "Complete checkpoint unexpectedly contains source-runtime overlay tensors")
    overlay_state, overlay_info = load_sam_runtime_overlay(workdir, sam_checkpoint)
    sam = sam_model_registry["vit_b"](image_size=256, keep_resolution=True, checkpoint=str(sam_checkpoint))
    model = InfraSegFM(
        sam,
        16,
        16,
        4,
        gate_topk=2,
        gate_temperature=1.0,
        gate_noise_std=0.0,
        style_use_std=True,
        style_dropout=0.1,
        disable_style=False,
        adapter_mode=str(profile["adapter_mode"]),
        task_drop_p=0.0,
    )
    model.load_parameters(saved)
    # Apply after the historical checkpoint path.  The nine pinned keys are
    # required to be absent from that checkpoint, so this exact source-runtime
    # restoration cannot overwrite any recorded complete-model tensor.
    overlay_apply = apply_sam_runtime_overlay(model.sam, overlay_state)
    loaded = model.sam.state_dict()
    missing = sorted(key for key in tensor_state if key not in loaded)
    wrong_shape = sorted(key for key, value in tensor_state.items()
                         if key in loaded and tuple(value.shape) != tuple(loaded[key].shape))
    different = sorted(key for key, value in tensor_state.items()
                       if key in loaded and tuple(value.shape) == tuple(loaded[key].shape)
                       and not torch.equal(value, loaded[key]))
    require(not missing and not wrong_shape and not different,
            f"{profile['model_label']} checkpoint did not load exactly")
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    non_head_digest = source_non_head_digest(model)
    return model, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "sam_checkpoint": str(sam_checkpoint),
        "sam_checkpoint_sha256": SAM_SHA256,
        "checkpoint_epoch_recorded": checkpoint_payload.get("epoch", profile["historical_checkpoint_epoch"]),
        "checkpoint_tensor_count": len(tensor_state),
        "all_checkpoint_tensors_loaded_exactly": True,
        "source_runtime_compatibility": {**overlay_info, "application": overlay_apply},
        "original_head_state_sha256": canonical_state_sha256(local_head_state(full_head_state(model))),
        "source_non_head_state_sha256": non_head_digest,
        "constructor": {
            "adapter_mode": profile["adapter_mode"],
            "bottleneck_dim": 16,
            "embedding_dim": 16,
            "expert_num": 4,
            "gate_topk": 2,
            "gate_temperature": 1.0,
            "gate_noise_std": 0.0,
            "style_use_std": True,
            "style_dropout": 0.1,
            "disable_style": False,
            "task_drop_p": 0.0,
            "precision": "FP32",
            "model_eval": True,
        },
    }


def clone_tuned_head(model: Any, state: Mapping[str, torch.Tensor], device: torch.device) -> Any:
    head = copy.deepcopy(model.sam.mask_decoder.iou_prediction_head).to(device)
    head.load_state_dict(dict(state), strict=True)
    head.eval()
    require(not head.training, "Finalized quality head clone is not in eval mode")
    require(canonical_state_sha256(head.state_dict()) == canonical_state_sha256(state),
            "Finalized quality head clone did not load exactly")
    return head


def clear_dataset_routing(dataset: Any) -> Dict[str, Any]:
    before = getattr(dataset, "_moe_assign", {}) or {}
    path = getattr(dataset, "_moe_cache_path", None)
    dataset._moe_assign = {}
    if hasattr(dataset, "_moe_cache_path"):
        dataset._moe_cache_path = None
    return {
        "routing_assignments_discovered_before_clear": int(len(before)),
        "routing_cache_path_discovered_before_clear": str(path) if path else None,
        "routing_assignments_active": int(len(getattr(dataset, "_moe_assign", {}) or {})),
        "routing_metadata_cleared_before_batch_emission": True,
    }


def make_dataset(task_root: Path) -> tuple[Any, Dict[str, Any]]:
    """Create a source dataset without allowing a persisted routing cache to force experts."""
    from data.dataset import InfraTaskDataset

    kwargs = {
        "train": False,
        "exclude_source_keywords": (),
        "moe_cache_name": "_quality_head_formal_test_no_cache.json",
    }
    dataset = InfraTaskDataset(str(task_root), **kwargs)
    routing = clear_dataset_routing(dataset)
    require(len(dataset) > 0, f"No samples in task folder: {task_root}")
    return dataset, routing


def discover_task_dirs(setting_root: Path) -> list[Path]:
    require(setting_root.is_dir(), f"Final-test setting root is missing: {setting_root}")
    found: list[Path] = []
    for current, directories, _files in os.walk(setting_root):
        here = Path(current)
        if (here / "npy_imgs").is_dir() and (here / "npy_gts").is_dir():
            found.append(here)
            directories[:] = []  # A task leaf cannot contain another task leaf.
    result = sorted(found, key=lambda item: item.relative_to(setting_root).as_posix())
    require(result, f"No task directories with npy_imgs/npy_gts under {setting_root}")
    return result


def sample_id_from_name(name: str, setting_root: Path) -> str:
    path = Path(name).resolve()
    try:
        return path.relative_to(setting_root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Dataset sample is outside its final-test root: {path}") from exc


def deterministic_perturb_box(
    raw_box: Sequence[float], sample_id: str, level: float, height: int, width: int, seed: int,
) -> list[float]:
    """Audited four-coordinate deterministic box perturbation used in quality-head validation."""
    require(level in PERTURBATION_LEVELS, f"Unsupported perturbation level: {level}")
    x1, y1, x2, y2 = [float(value) for value in raw_box]
    if level != 0.0:
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        raw_seed = hashlib.sha256(f"{seed}|{level:.4f}|{sample_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(raw_seed[:8], "little", signed=False))
        dx1, dx2 = rng.uniform(-level * box_w, level * box_w, size=2)
        dy1, dy2 = rng.uniform(-level * box_h, level * box_h, size=2)
        x1 = min(max(x1 + float(dx1), 0.0), float(width - 1))
        x2 = min(max(x2 + float(dx2), 0.0), float(width - 1))
        y1 = min(max(y1 + float(dy1), 0.0), float(height - 1))
        y2 = min(max(y2 + float(dy2), 0.0), float(height - 1))
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    if hi_x - lo_x < 1.0:
        hi_x = min(float(width - 1), lo_x + 1.0)
        lo_x = max(0.0, hi_x - 1.0)
    if hi_y - lo_y < 1.0:
        hi_y = min(float(height - 1), lo_y + 1.0)
        lo_y = max(0.0, hi_y - 1.0)
    return [float(lo_x), float(lo_y), float(hi_x), float(hi_y)]


def prompt_sha256(setting: str, task: str, sample_id: str, level: float, raw_box: Sequence[float], box: Sequence[float], seed: int) -> str:
    payload = {
        "setting": setting,
        "task": task,
        "sample_id": sample_id,
        "level": f"{level:.4f}",
        "seed": int(seed),
        "raw_box": [round(float(value), 8) for value in raw_box],
        "perturbed_box": [round(float(value), 8) for value in box],
    }
    packed = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def test_setting_root(eval_root: Path, setting: str) -> Path:
    require(setting in TEST_SETTINGS, f"Unknown final-test setting: {setting}")
    return (eval_root / str(TEST_SETTINGS[setting]["relative"])).resolve()


def prompt_cache_paths(workdir: Path) -> Dict[str, Path]:
    root = workdir / "inputs" / "provenance"
    return {
        "members": root / "test_members.csv",
        "prompts": root / "prompt_cache.csv",
        "manifest": root / "prompt_cache_manifest.json",
        "weight_mapping": root / "weight_mapping.json",
    }


def write_frozen_run_files(workdir: Path, source_hashes: Mapping[str, str]) -> None:
    """Write the package-facing configuration and credential-free command record."""
    config_path = workdir / "config" / "formal_test_config.json"
    command_path = workdir / "commands" / "run_formal_test.txt"
    require(not config_path.exists() and not command_path.exists(),
            "Frozen command/config record already exists; do not overwrite the formal protocol")
    atomic_json(config_path, {
        "schema": SCHEMA + "-configuration",
        "scope": "Frozen existing checkpoints and finalized quality heads only; no training, seed expansion, or model expansion",
        "source_files_sha256": dict(source_hashes),
        "sam_checkpoint_sha256": SAM_SHA256,
        "precision": "FP32",
        "target_data_percent": 0,
        "prompt_seed": PROMPT_SEED,
        "box_perturbation_percent": [0, 5, 10, 20],
        "candidate_contract": {
            "returned_candidates": 3,
            "returned_index_to_raw_iou_index": [1, 2, 3],
            "actual_selector": "argmax(finalized_iou_head_scores)",
            "fixed_baseline": "returned candidate 0",
            "oracle": "offline GT Dice only",
        },
        "metrics": {
            "threshold": "sigmoid(mask_logits) > 0.5",
            "empty_rule": "both empty -> 1; GT empty and prediction nonempty -> 0",
            "setting_variants": {setting: TEST_SETTINGS[setting]["metric_variant"] for setting in TEST_SETTINGS},
        },
        "profiles": {
            name: {
                "model_label": config["model_label"], "family": config["family"], "version": config["version"],
                "adapter_mode": config["adapter_mode"], "checkpoint_file": config["checkpoint_file"],
                "checkpoint_sha256": config["checkpoint_sha256"], "head_file": config["head_file"],
                "head_file_sha256": config["head_file_sha256"], "head_state_sha256": config["head_state_sha256"],
                "selected_head_epoch": config["selected_head_epoch"], "settings": list(config["settings"]),
                "evaluation_batch_size": config["evaluation_batch_size"], "workers": 2,
            }
            for name, config in PROFILES.items()
        },
    })
    command = """# Frozen InfraSegFM quality-head formal-test commands (no training).
# Set these four paths once in the isolated server directory.
TEST_ROOT=/path/to/formal_test
SOURCE_PROJECT=/path/to/source_project
EVAL_ROOT=${SOURCE_PROJECT}/playground/eval
SAM_B=${SOURCE_PROJECT}/weights/sam_vit_b_01ec64.pth
PYTHON=python
DRIVER=${TEST_ROOT}/code/evaluate_quality_head_formal_test.py
CODE_ROOT=${TEST_ROOT}/code/original

# Build the one common final-test member/prompt cache before any GPU evaluation.
${PYTHON} -B ${DRIVER} prepare --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --eval-root ${EVAL_ROOT}

# Verify every finalized small head against its matching complete checkpoint on three source-development samples.
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -B ${DRIVER} preflight --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --source-project-root ${SOURCE_PROJECT} --sam-checkpoint ${SAM_B} --profile full_v1 --device cuda:0
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -B ${DRIVER} preflight --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --source-project-root ${SOURCE_PROJECT} --sam-checkpoint ${SAM_B} --profile full_v2 --device cuda:0
CUDA_VISIBLE_DEVICES=3 ${PYTHON} -B ${DRIVER} preflight --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --source-project-root ${SOURCE_PROJECT} --sam-checkpoint ${SAM_B} --profile shared_v1 --device cuda:0
CUDA_VISIBLE_DEVICES=4 ${PYTHON} -B ${DRIVER} preflight --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --source-project-root ${SOURCE_PROJECT} --sam-checkpoint ${SAM_B} --profile shared_v2 --device cuda:0

# Run the fixed protocol.  Each command evaluates only its prescribed settings and uses its one finalized head for all four prompt perturbations.
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -B ${DRIVER} evaluate --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --eval-root ${EVAL_ROOT} --sam-checkpoint ${SAM_B} --profile full_v1 --device cuda:0
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -B ${DRIVER} evaluate --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --eval-root ${EVAL_ROOT} --sam-checkpoint ${SAM_B} --profile full_v2 --device cuda:0
CUDA_VISIBLE_DEVICES=3 ${PYTHON} -B ${DRIVER} evaluate --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --eval-root ${EVAL_ROOT} --sam-checkpoint ${SAM_B} --profile shared_v1 --device cuda:0
CUDA_VISIBLE_DEVICES=4 ${PYTHON} -B ${DRIVER} evaluate --workdir ${TEST_ROOT} --code-root ${CODE_ROOT} --eval-root ${EVAL_ROOT} --sam-checkpoint ${SAM_B} --profile shared_v2 --device cuda:0

# Generate the 32-condition summary and exact paired Full--Shared tables after all four profile jobs complete.
${PYTHON} -B ${DRIVER} aggregate --workdir ${TEST_ROOT}
"""
    atomic_text(command_path, command)


def prepare_prompt_cache(args: argparse.Namespace) -> None:
    workdir = args.workdir.resolve()
    code_root = args.code_root.resolve()
    eval_root = args.eval_root.resolve()
    paths = prompt_cache_paths(workdir)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("Prompt cache already exists; retain it as the immutable common Full/Shared prompt list")
    source_hashes = assert_source_fingerprint(code_root)
    prepare_imports(code_root)
    logger = setup_logger(workdir / "logs" / "prepare_prompts.log", "prepare")
    member_rows: list[Dict[str, Any]] = []
    prompt_rows: list[Dict[str, Any]] = []
    task_counts: Dict[str, int] = {}
    routing_records: list[Dict[str, Any]] = []
    start = time.monotonic()

    for setting, spec in TEST_SETTINGS.items():
        setting_root = test_setting_root(eval_root, setting)
        task_dirs = discover_task_dirs(setting_root)
        setting_count = 0
        observed_tasks: set[str] = set()
        for task_dir in task_dirs:
            dataset, routing = make_dataset(task_dir)
            task_dir_relative = task_dir.relative_to(setting_root).as_posix()
            routing_records.append({"setting": setting, "task_dir": task_dir_relative, **routing})
            for index in range(len(dataset)):
                data, _label = dataset[index]
                model_task_folder = str(data["task_folder"])
                require(model_task_folder == task_dir.name,
                        f"Task metadata changed under {task_dir}: {model_task_folder}")
                # The setting-relative task directory is the reporting ID.  It
                # stays unique when an ODD layout contains nested leaves with
                # the same basename; the model itself still receives the
                # original taxonomy leaf through ``data['task_folder']``.
                task = task_dir_relative
                sample_id = sample_id_from_name(str(data["name"]), setting_root)
                raw_box = [float(value) for value in data["box"].reshape(-1).tolist()]
                require(len(raw_box) == 4, f"Malformed box for {setting}/{sample_id}")
                height, width = int(data["img"].shape[-2]), int(data["img"].shape[-1])
                require(height > 1 and width > 1, f"Invalid image dimensions for {setting}/{sample_id}")
                member_rows.append({
                    "setting": setting,
                    "task": task,
                    "model_task_folder": model_task_folder,
                    "task_dir": task_dir_relative,
                    "sample_id": sample_id,
                    "image_height": height,
                    "image_width": width,
                    "raw_box_x1": raw_box[0], "raw_box_y1": raw_box[1],
                    "raw_box_x2": raw_box[2], "raw_box_y2": raw_box[3],
                })
                for level in PERTURBATION_LEVELS:
                    box = deterministic_perturb_box(raw_box, sample_id, level, height, width, args.prompt_seed)
                    prompt_rows.append({
                        "setting": setting,
                        "task": task,
                        "model_task_folder": model_task_folder,
                        "task_dir": task_dir_relative,
                        "sample_id": sample_id,
                        "box_perturbation_percent": int(round(level * 100)),
                        "prompt_seed": int(args.prompt_seed),
                        "image_height": height,
                        "image_width": width,
                        "raw_box_x1": raw_box[0], "raw_box_y1": raw_box[1],
                        "raw_box_x2": raw_box[2], "raw_box_y2": raw_box[3],
                        "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
                        "prompt_sha256": prompt_sha256(setting, task, sample_id, level, raw_box, box, args.prompt_seed),
                    })
                setting_count += 1
                observed_tasks.add(task)
        require(setting_count == int(spec["expected_samples"]),
                f"{setting} final-test sample count mismatch: {setting_count}")
        require(len(observed_tasks) == int(spec["expected_tasks"]),
                f"{setting} final-test task count mismatch: {len(observed_tasks)}")
        task_counts[setting] = setting_count
        logger.info("prepared %s: %d samples, %d tasks", setting, setting_count, len(observed_tasks))

    member_keys = [(row["setting"], row["task"], row["sample_id"]) for row in member_rows]
    prompt_keys = [(row["setting"], row["task"], row["sample_id"], row["box_perturbation_percent"]) for row in prompt_rows]
    require(len(member_keys) == len(set(member_keys)), "Duplicate final-test member in prompt cache")
    require(len(prompt_keys) == len(set(prompt_keys)), "Duplicate final-test prompt in prompt cache")
    require(len(prompt_rows) == len(member_rows) * len(PERTURBATION_LEVELS), "Incomplete perturbation cache")
    member_fields = list(member_rows[0].keys())
    prompt_fields = list(prompt_rows[0].keys())
    write_csv_atomic(paths["members"], member_fields, member_rows)
    write_csv_atomic(paths["prompts"], prompt_fields, prompt_rows)
    manifest = {
        "schema": SCHEMA + "-prompt-cache",
        "scope": "Final-test membership and deterministic GT-box prompts only; no score/result is stored here",
        "source_files_sha256": source_hashes,
        "eval_root": str(eval_root),
        "prompt_seed": int(args.prompt_seed),
        "perturbation_levels_percent": [int(round(value * 100)) for value in PERTURBATION_LEVELS],
        "perturbation_definition": "each xyxy coordinate has an independently deterministic uniform shift within ± level × raw box width/height, then clamp and non-degenerate repair",
        "members": {"path": str(paths["members"]), "rows": len(member_rows), "sha256": sha256(paths["members"])},
        "prompts": {"path": str(paths["prompts"]), "rows": len(prompt_rows), "sha256": sha256(paths["prompts"])},
        "settings": {
            setting: {**TEST_SETTINGS[setting], "observed_samples": task_counts[setting]}
            for setting in TEST_SETTINGS
        },
        "routing_metadata": routing_records,
        "elapsed_seconds": time.monotonic() - start,
    }
    atomic_json(paths["manifest"], manifest)
    atomic_json(paths["weight_mapping"], {
        "schema": SCHEMA + "-weight-mapping",
        "scope": "Fixed mapping selected on source development data before final testing; final-test outcomes cannot alter this mapping",
        "source_files_sha256": source_hashes,
        "sam_checkpoint_sha256": SAM_SHA256,
        "profiles": {
            name: {
                "model_label": config["model_label"],
                "family": config["family"],
                "version": config["version"],
                "complete_checkpoint_relative": f"inputs/models/{config['checkpoint_file']}",
                "complete_checkpoint_sha256": config["checkpoint_sha256"],
                "historical_checkpoint_epoch": config["historical_checkpoint_epoch"],
                "quality_head_relative": f"inputs/heads/{config['head_file']}",
                "quality_head_file_sha256": config["head_file_sha256"],
                "quality_head_state_sha256": config["head_state_sha256"],
                "selected_quality_head_epoch": config["selected_head_epoch"],
                "formal_test_settings": list(config["settings"]),
                "target_data_percent": 0,
                "precision": "FP32",
                "evaluation_batch_size": config["evaluation_batch_size"],
            }
            for name, config in PROFILES.items()
        },
        "prompt_cache": {"path": str(paths["prompts"]), "sha256": sha256(paths["prompts"])},
        "test_members": {"path": str(paths["members"]), "sha256": sha256(paths["members"])},
    })
    write_frozen_run_files(workdir, source_hashes)
    logger.info("prompt cache complete: %s", paths["manifest"])


def load_prompt_cache(workdir: Path, code_root: Path) -> tuple[Dict[tuple[str, str, str, int], Dict[str, Any]], Dict[str, Any]]:
    paths = prompt_cache_paths(workdir)
    for path in paths.values():
        require(path.is_file(), f"Missing immutable prompt-cache file: {path}")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    require(manifest.get("schema") == SCHEMA + "-prompt-cache", "Prompt-cache schema mismatch")
    require(manifest.get("source_files_sha256") == EXPECTED_SOURCE_FILES, "Prompt cache source fingerprint mismatch")
    require(sha256(paths["members"]) == manifest["members"]["sha256"], "Prompt member list hash mismatch")
    require(sha256(paths["prompts"]) == manifest["prompts"]["sha256"], "Prompt cache hash mismatch")
    require(int(manifest.get("prompt_seed")) == PROMPT_SEED, "Prompt cache seed differs from locked protocol")
    require(assert_source_fingerprint(code_root) == EXPECTED_SOURCE_FILES, "Current source fingerprint mismatch")
    index: Dict[tuple[str, str, str, int], Dict[str, Any]] = {}
    with paths["prompts"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["setting"], row["task"], row["sample_id"], int(row["box_perturbation_percent"]))
            require(key not in index, f"Duplicate prompt key: {key}")
            index[key] = row
    expected_rows = sum(int(spec["expected_samples"]) for spec in TEST_SETTINGS.values()) * len(PERTURBATION_LEVELS)
    require(len(index) == expected_rows, f"Prompt-cache row count mismatch: {len(index)}")
    return index, manifest


def make_loader(dataset: Any, batch_size: int, workers: int) -> DataLoader:
    # Evaluation is deterministic: no shuffle, no prompt jitter, no worker RNG
    # whose state can alter the cached prompt values.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=False, drop_last=False)


def transform_batch(data: MutableMapping[str, Any], image_size: int, box_transform: Any, image_transform: Any, device: torch.device) -> None:
    if data["img"].shape[-1] != image_size:
        data["box"] = box_transform.apply_boxes_torch(
            data["box"].reshape(-1, 2, 2), data["img"].shape[-2:]
        ).reshape(-1, 4)
        data["img"] = image_transform(data["img"])
    data["img"] = data["img"].to(device, non_blocking=True)
    data["box"] = data["box"].to(device, non_blocking=True)


def candidate_dice(masks: torch.Tensor, label: torch.Tensor, *, antialias: bool) -> torch.Tensor:
    """Apply the established setting-specific resize, threshold and empty rules."""
    if masks.shape[-2:] != label.shape[-2:]:
        kwargs: Dict[str, Any] = {"size": label.shape[-2:], "mode": "bilinear", "align_corners": False}
        if antialias:
            kwargs["antialias"] = True
        masks = F.interpolate(masks, **kwargs)
    prediction = torch.sigmoid(masks) > 0.5
    target = label > 0
    target = target.expand(-1, prediction.shape[1], -1, -1)
    intersection = (prediction & target).flatten(2).sum(dim=2).float()
    pred_area = prediction.flatten(2).sum(dim=2).float()
    target_area = target.flatten(2).sum(dim=2).float()
    total = pred_area + target_area
    dice = (2.0 * intersection + EPS) / (total + EPS)
    both_empty = (pred_area == 0) & (target_area == 0)
    gt_empty_pred_nonempty = (target_area == 0) & (pred_area > 0)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    dice = torch.where(gt_empty_pred_nonempty, torch.zeros_like(dice), dice)
    require(torch.isfinite(dice).all().item(), "Candidate Dice contains NaN/Inf")
    return dice


def capture_one_forward(model: Any, data: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return three masks, returned original scores, and the head input from one full forward."""
    captured: Dict[str, torch.Tensor] = {}
    decoder_calls = 0
    head_calls = 0

    def head_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        nonlocal head_calls
        head_calls += 1
        require(len(inputs) == 1 and torch.is_tensor(inputs[0]), "IoU head input is malformed")
        captured["features"] = inputs[0].detach().clone()

    def decoder_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal decoder_calls
        decoder_calls += 1
        require(isinstance(output, tuple) and len(output) == 2, "Mask decoder output is malformed")
        captured["masks"] = output[0].detach().clone()
        captured["scores"] = output[1].detach().clone()

    head_handle = model.sam.mask_decoder.iou_prediction_head.register_forward_pre_hook(head_pre_hook)
    decoder_handle = model.sam.mask_decoder.register_forward_hook(decoder_hook)
    try:
        formal_masks = model(data)
    finally:
        head_handle.remove()
        decoder_handle.remove()
    require(decoder_calls == 1 and head_calls == 1, "Expected exactly one decoder/head call per full forward")
    require(set(captured) == {"features", "masks", "scores"}, "Forward capture incomplete")
    masks, scores, features = captured["masks"], captured["scores"], captured["features"]
    require(torch.equal(formal_masks, masks), "InfraSegFM.forward masks differ from decoder masks")
    require(masks.ndim == 4 and masks.shape[1] == 3 and tuple(scores.shape) == tuple(masks.shape[:2]),
            "Expected three aligned returned candidate masks and scores")
    require(torch.isfinite(masks).all().item() and torch.isfinite(scores).all().item() and torch.isfinite(features).all().item(),
            "Full-model forward produced NaN/Inf")
    direct_scores = model.sam.mask_decoder.iou_prediction_head(features)[:, 1:]
    require(torch.equal(direct_scores, scores), "Raw IoU output indices [1:4] do not align with returned candidates")
    return masks, scores, features


def source_dev_preflight(args: argparse.Namespace) -> None:
    workdir = args.workdir.resolve()
    code_root = args.code_root.resolve()
    source_project_root = args.source_project_root.resolve()
    profile = profile_config(args.profile)
    out_path = workdir / "outputs" / "preflight" / f"{args.profile}.json"
    require(not out_path.exists(), f"Preflight already exists and is immutable: {out_path}")
    source_hashes = assert_source_fingerprint(code_root)
    paths = profile_paths(workdir, profile)
    state, head_info = load_final_head(paths["head"], profile)
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "Preflight requires the assigned CUDA GPU")
    logger = setup_logger(workdir / "logs" / f"preflight_{args.profile}.log", f"preflight.{args.profile}")
    model, model_info = load_exact_model(workdir, code_root, profile, paths["checkpoint"], args.sam_checkpoint.resolve(), device)
    expected_non_head = head_info.get("source_non_head_state_sha256")
    require(isinstance(expected_non_head, str) and expected_non_head == model_info["source_non_head_state_sha256"],
            "Finalized head compatibility digest does not match the loaded frozen segmentation model")
    tuned_clone = clone_tuned_head(model, state, device)
    prepare_imports(code_root)
    from data.dataset import InfraSegDataset
    from segment_anything.utils.transforms import ResizeLongestSide

    dev_root = source_project_root / str(profile["source_data_relative"]) / "val"
    require(dev_root.is_dir(), f"Source development split missing: {dev_root}")
    dataset = InfraSegDataset(str(dev_root), train=False, exclude_source_keywords=(),
                              moe_cache_name="_quality_head_formal_test_no_cache.json")
    routing = clear_dataset_routing(dataset)
    task_count = len(set(str(item) for item in dataset.task_folders))
    require(len(dataset) == int(profile["expected_source_dev_samples"]), "Unexpected source-development sample count")
    require(task_count == int(profile["expected_source_dev_tasks"]), "Unexpected source-development task count")
    loader = make_loader(dataset, batch_size=1, workers=0)
    image_size = int(model.sam.image_encoder.img_size)
    image_transform = Resize((image_size, image_size), antialias=True)
    box_transform = ResizeLongestSide(image_size)
    original_state = local_head_state(full_head_state(model))
    entries: list[Dict[str, Any]] = []
    original_digest = canonical_state_sha256(original_state)
    start = time.monotonic()
    with torch.inference_mode():
        for index, (data, _label) in enumerate(loader):
            if index >= 3:
                break
            transform_batch(data, image_size, box_transform, image_transform, device)
            model.sam.mask_decoder.iou_prediction_head.load_state_dict(original_state, strict=True)
            model.eval()
            baseline_masks, baseline_scores, baseline_features = capture_one_forward(model, data)
            clone_scores = tuned_clone(baseline_features)[:, 1:]
            require(torch.isfinite(clone_scores).all().item(), "Tuned clone score has NaN/Inf")
            model.sam.mask_decoder.iou_prediction_head.load_state_dict(state, strict=True)
            model.eval()
            overlay_masks, overlay_scores, overlay_features = capture_one_forward(model, data)
            require(torch.equal(baseline_masks, overlay_masks), "Quality-head overlay changed candidate masks")
            require(torch.equal(baseline_features, overlay_features), "Quality-head overlay changed decoder input feature")
            require(torch.equal(clone_scores, overlay_scores), "Tuned clone does not reproduce temporary overlay scores")
            entries.append({
                "index": index,
                "sample_id": str(data["name"][0]),
                "task": str(data["task_folder"][0]),
                "candidate_masks_sha256": tensor_sha256(baseline_masks),
                "original_scores_sha256": tensor_sha256(baseline_scores),
                "tuned_scores_sha256": tensor_sha256(overlay_scores),
                "candidate_masks_bitwise_unchanged": True,
                "original_selected_index": int(baseline_scores.argmax(dim=1).item()),
                "tuned_selected_index": int(overlay_scores.argmax(dim=1).item()),
            })
    model.sam.mask_decoder.iou_prediction_head.load_state_dict(original_state, strict=True)
    require(canonical_state_sha256(model.sam.mask_decoder.iou_prediction_head.state_dict()) == original_digest,
            "Preflight did not restore the original complete-checkpoint head")
    require(source_non_head_digest(model) == model_info["source_non_head_state_sha256"],
            "Preflight altered frozen non-head state")
    require(len(entries) == 3, "Preflight did not cover three source-development samples")
    report = {
        "schema": SCHEMA + "-preflight",
        "profile": {"key": args.profile, **profile},
        "source_files_sha256": source_hashes,
        "model": model_info,
        "finalized_head": head_info,
        "source_development_split": {
            "root": str(dev_root), "samples": len(dataset), "tasks": task_count,
            "routing": routing, "used_only_for_migration_check": True,
        },
        "checks": {
            "candidate_score_layout_raw_indices": "returned candidate 0/1/2 map to raw IoU output 1/2/3",
            "candidate_masks_bitwise_unchanged_under_temporary_head_overlay": True,
            "tuned_clone_equals_temporary_overlay_scores": True,
            "original_head_restored_after_preflight": True,
            "non_head_state_unchanged": True,
            "checked_samples": len(entries),
        },
        "entries": entries,
        "elapsed_seconds": time.monotonic() - start,
    }
    atomic_json(out_path, report)
    logger.info("preflight passed: %s", out_path)


def require_completed_preflight(workdir: Path, profile_name: str, profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Block final-test evaluation until its matching migration proof is present."""
    path = workdir / "outputs" / "preflight" / f"{profile_name}.json"
    require(path.is_file(), f"Formal evaluation requires a completed migration preflight: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("schema") == SCHEMA + "-preflight", "Preflight schema mismatch")
    require(report.get("profile", {}).get("key") == profile_name, "Preflight profile mismatch")
    require(report.get("source_files_sha256") == EXPECTED_SOURCE_FILES, "Preflight source fingerprint mismatch")
    checks = report.get("checks")
    require(isinstance(checks, Mapping) and checks.get("checked_samples") == 3,
            "Preflight did not cover the required three source-development samples")
    for key in (
        "candidate_masks_bitwise_unchanged_under_temporary_head_overlay",
        "tuned_clone_equals_temporary_overlay_scores",
        "original_head_restored_after_preflight",
        "non_head_state_unchanged",
    ):
        require(checks.get(key) is True, f"Preflight did not pass required check: {key}")
    finalized = report.get("finalized_head")
    model = report.get("model")
    require(isinstance(finalized, Mapping) and isinstance(model, Mapping), "Preflight proof is incomplete")
    require(finalized.get("file_sha256") == profile["head_file_sha256"], "Preflight used a different small head file")
    require(finalized.get("state_sha256") == profile["head_state_sha256"], "Preflight used a different small head state")
    require(model.get("checkpoint_sha256") == profile["checkpoint_sha256"], "Preflight used a different complete checkpoint")
    return {"path": str(path), "sha256": sha256(path), "checked_samples": 3}


def float_from(row: Mapping[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Prompt cache has invalid numeric field {name}") from exc


def assert_prompt_row(
    row: Mapping[str, str], *, setting: str, task: str, sample_id: str, raw_box: Sequence[float], height: int, width: int,
) -> list[float]:
    require(row["setting"] == setting and row["task"] == task and row["sample_id"] == sample_id,
            "Prompt cache member mismatch")
    require(int(row["image_height"]) == height and int(row["image_width"]) == width,
            "Prompt cache image shape mismatch")
    cached_raw = [float_from(row, key) for key in ("raw_box_x1", "raw_box_y1", "raw_box_x2", "raw_box_y2")]
    require(np.allclose(np.asarray(cached_raw), np.asarray(raw_box), rtol=0.0, atol=0.0),
            "Final-test raw GT prompt differs from immutable prompt cache")
    percent = int(row["box_perturbation_percent"])
    level = percent / 100.0
    box = [float_from(row, key) for key in ("box_x1", "box_y1", "box_x2", "box_y2")]
    expected = deterministic_perturb_box(raw_box, sample_id, level, height, width, PROMPT_SEED)
    require(np.allclose(np.asarray(box), np.asarray(expected), rtol=0.0, atol=1e-7),
            "Prompt cache perturbation differs from the locked deterministic rule")
    expected_hash = prompt_sha256(setting, task, sample_id, level, raw_box, box, PROMPT_SEED)
    require(row["prompt_sha256"] == expected_hash, "Prompt cache per-sample hash mismatch")
    return box


def empty_stats() -> Dict[str, Any]:
    return {"samples": 0, "c0_sum": 0.0, "tuned_sum": 0.0, "oracle_sum": 0.0,
            "raw_score_sum": [0.0, 0.0, 0.0], "tuned_score_sum": [0.0, 0.0, 0.0],
            "raw_selection_counts": [0, 0, 0], "tuned_selection_counts": [0, 0, 0],
            "oracle_selection_counts": [0, 0, 0]}


def update_stats(stats: MutableMapping[str, Any], *, c0: float, tuned: float, oracle: float,
                 raw_scores: Sequence[float], tuned_scores: Sequence[float], raw_index: int, tuned_index: int, oracle_index: int) -> None:
    stats["samples"] += 1
    stats["c0_sum"] += c0
    stats["tuned_sum"] += tuned
    stats["oracle_sum"] += oracle
    for index in range(3):
        stats["raw_score_sum"][index] += float(raw_scores[index])
        stats["tuned_score_sum"][index] += float(tuned_scores[index])
    stats["raw_selection_counts"][raw_index] += 1
    stats["tuned_selection_counts"][tuned_index] += 1
    stats["oracle_selection_counts"][oracle_index] += 1


def finalized_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    samples = int(stats["samples"])
    require(samples > 0, "Cannot summarize an empty condition/task")
    c0 = float(stats["c0_sum"]) / samples
    tuned = float(stats["tuned_sum"]) / samples
    oracle = float(stats["oracle_sum"]) / samples
    return {
        "samples": samples,
        "candidate_0_dice": c0,
        "tuned_quality_dice": tuned,
        "oracle_dice": oracle,
        "tuned_minus_candidate_0": tuned - c0,
        "oracle_minus_tuned": oracle - tuned,
        "raw_score_mean": [float(value) / samples for value in stats["raw_score_sum"]],
        "tuned_score_mean": [float(value) / samples for value in stats["tuned_score_sum"]],
        "raw_selection_counts": list(stats["raw_selection_counts"]),
        "tuned_selection_counts": list(stats["tuned_selection_counts"]),
        "oracle_selection_counts": list(stats["oracle_selection_counts"]),
    }


def weighted_summary(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = sum(int(item["samples"]) for item in items)
    require(total > 0, "No items for micro summary")
    return {
        "samples": total,
        "candidate_0_dice": sum(float(item["candidate_0_dice"]) * int(item["samples"]) for item in items) / total,
        "tuned_quality_dice": sum(float(item["tuned_quality_dice"]) * int(item["samples"]) for item in items) / total,
        "oracle_dice": sum(float(item["oracle_dice"]) * int(item["samples"]) for item in items) / total,
    }


def macro_summary(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    require(items, "No tasks for macro summary")
    return {
        "tasks": len(items),
        "candidate_0_dice": float(np.mean([float(item["candidate_0_dice"]) for item in items])),
        "tuned_quality_dice": float(np.mean([float(item["tuned_quality_dice"]) for item in items])),
        "oracle_dice": float(np.mean([float(item["oracle_dice"]) for item in items])),
    }


def evaluate_profile(args: argparse.Namespace) -> None:
    workdir = args.workdir.resolve()
    code_root = args.code_root.resolve()
    eval_root = args.eval_root.resolve()
    profile = profile_config(args.profile)
    output_dir = workdir / "outputs" / args.profile
    expected_outputs = (output_dir / "per_sample.csv", output_dir / "per_task.csv", output_dir / "summary.json")
    require(not any(path.exists() for path in expected_outputs),
            f"{args.profile} already has a completed/partial output; do not overwrite formal results")
    prompt_index, prompt_manifest = load_prompt_cache(workdir, code_root)
    paths = profile_paths(workdir, profile)
    state, head_info = load_final_head(paths["head"], profile)
    preflight_evidence = require_completed_preflight(workdir, args.profile, profile)
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "Formal evaluation requires the assigned CUDA GPU")
    logger = setup_logger(workdir / "logs" / f"evaluate_{args.profile}.log", f"evaluate.{args.profile}")
    model, model_info = load_exact_model(workdir, code_root, profile, paths["checkpoint"], args.sam_checkpoint.resolve(), device)
    require(head_info.get("source_non_head_state_sha256") == model_info["source_non_head_state_sha256"],
            "Finalized quality-head compatibility digest differs from the reconstructed model")
    tuned_head = clone_tuned_head(model, state, device)
    prepare_imports(code_root)
    from segment_anything.utils.transforms import ResizeLongestSide

    source_hashes_before = source_files(code_root)
    image_size = int(model.sam.image_encoder.img_size)
    image_transform = Resize((image_size, image_size), antialias=True)
    box_transform = ResizeLongestSide(image_size)
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary_csv = output_dir / "per_sample.csv.tmp"
    fields = [
        "profile", "model_family", "version", "setting", "protocol_scope", "target_data_percent",
        "box_perturbation_percent", "task", "model_task_folder", "sample_id", "prompt_sha256",
        "candidate_0_dice", "candidate_1_dice", "candidate_2_dice",
        "raw_score_0", "raw_score_1", "raw_score_2", "raw_selected_index", "raw_selected_dice",
        "tuned_score_0", "tuned_score_1", "tuned_score_2", "tuned_selected_index", "tuned_selected_dice", "tuned_dice",
        "oracle_selected_index", "oracle_dice", "oracle_only", "metric_resize_antialias", "metric_variant",
    ]
    condition_stats: Dict[tuple[str, int], Dict[str, Any]] = defaultdict(empty_stats)
    task_stats: Dict[tuple[str, int, str], Dict[str, Any]] = defaultdict(empty_stats)
    coverage: Dict[tuple[str, int], int] = Counter()
    mask_digests: Dict[tuple[str, int], hashlib._Hash] = {}
    start = time.monotonic()
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for setting in profile["settings"]:
                spec = TEST_SETTINGS[setting]
                setting_root = test_setting_root(eval_root, setting)
                task_dirs = discover_task_dirs(setting_root)
                observed_setting_samples = 0
                observed_tasks: set[str] = set()
                for task_dir in task_dirs:
                    dataset, routing = make_dataset(task_dir)
                    reporting_task = task_dir.relative_to(setting_root).as_posix()
                    logger.info("%s %s %s: %d samples; routing cleared=%d", args.profile, setting,
                                task_dir.name, len(dataset), routing["routing_assignments_discovered_before_clear"])
                    loader = make_loader(dataset, int(profile["evaluation_batch_size"]), args.workers)
                    for data, label_cpu in loader:
                        raw_height, raw_width = int(data["img"].shape[-2]), int(data["img"].shape[-1])
                        names = [str(value) for value in data["name"]]
                        model_tasks = [str(value) for value in data["task_folder"]]
                        require(all(value == task_dir.name for value in model_tasks),
                                f"Model task metadata differs inside {task_dir}")
                        tasks = [reporting_task] * len(names)
                        sample_ids = [sample_id_from_name(name, setting_root) for name in names]
                        raw_boxes = data["box"].detach().to("cpu", dtype=torch.float32).tolist()
                        for percent in (0, 5, 10, 20):
                            cached_boxes: list[list[float]] = []
                            cached_rows: list[Mapping[str, str]] = []
                            for task, sample_id, raw_box in zip(tasks, sample_ids, raw_boxes):
                                key = (setting, task, sample_id, percent)
                                require(key in prompt_index, f"Missing immutable prompt cache row: {key}")
                                cache_row = prompt_index[key]
                                cached_rows.append(cache_row)
                                cached_boxes.append(assert_prompt_row(
                                    cache_row, setting=setting, task=task, sample_id=sample_id,
                                    raw_box=raw_box, height=raw_height, width=raw_width,
                                ))
                            # Only the cached prompt is fed into the full forward.  Labels remain CPU
                            # until scores have already selected the non-oracle candidate.
                            local_data = dict(data)
                            local_data["box"] = torch.tensor(cached_boxes, dtype=data["box"].dtype)
                            transform_batch(local_data, image_size, box_transform, image_transform, device)
                            masks, raw_scores, features = capture_one_forward(model, local_data)
                            tuned_scores = tuned_head(features)[:, 1:]
                            require(torch.isfinite(tuned_scores).all().item(), "Finalized quality head produced NaN/Inf")
                            require(tuple(tuned_scores.shape) == tuple(raw_scores.shape) == (len(names), 3),
                                    "Quality score shape does not match three returned candidates")
                            raw_index = raw_scores.argmax(dim=1)
                            tuned_index = tuned_scores.argmax(dim=1)
                            label = label_cpu.to(device, non_blocking=True)
                            dice = candidate_dice(masks, label, antialias=bool(spec["metric_resize_antialias"]))
                            oracle_dice, oracle_index = dice.max(dim=1)
                            row_index = torch.arange(len(names), device=device)
                            c0_dice = dice[:, 0]
                            raw_dice = dice[row_index, raw_index]
                            tuned_dice = dice[row_index, tuned_index]
                            require(torch.isfinite(tuned_dice).all().item() and torch.isfinite(oracle_dice).all().item(),
                                    "Selected Dice contains NaN/Inf")
                            digest_key = (setting, percent)
                            if digest_key not in mask_digests:
                                mask_digests[digest_key] = hashlib.sha256()
                            mask_digests[digest_key].update(masks.detach().to("cpu").contiguous().numpy().tobytes())
                            for index, (task, sample_id, cache_row) in enumerate(zip(tasks, sample_ids, cached_rows)):
                                raw_values = [float(raw_scores[index, column]) for column in range(3)]
                                tuned_values = [float(tuned_scores[index, column]) for column in range(3)]
                                dice_values = [float(dice[index, column]) for column in range(3)]
                                raw_choice = int(raw_index[index])
                                tuned_choice = int(tuned_index[index])
                                oracle_choice = int(oracle_index[index])
                                c0 = float(c0_dice[index])
                                tuned = float(tuned_dice[index])
                                oracle = float(oracle_dice[index])
                                update_stats(condition_stats[(setting, percent)], c0=c0, tuned=tuned, oracle=oracle,
                                             raw_scores=raw_values, tuned_scores=tuned_values, raw_index=raw_choice,
                                             tuned_index=tuned_choice, oracle_index=oracle_choice)
                                update_stats(task_stats[(setting, percent, task)], c0=c0, tuned=tuned, oracle=oracle,
                                             raw_scores=raw_values, tuned_scores=tuned_values, raw_index=raw_choice,
                                             tuned_index=tuned_choice, oracle_index=oracle_choice)
                                coverage[(setting, percent)] += 1
                                writer.writerow({
                                    "profile": args.profile, "model_family": profile["family"], "version": profile["version"],
                                    "setting": setting, "protocol_scope": spec["protocol_scope"], "target_data_percent": 0,
                                    "box_perturbation_percent": percent, "task": task,
                                    "model_task_folder": model_tasks[index], "sample_id": sample_id,
                                    "prompt_sha256": cache_row["prompt_sha256"],
                                    "candidate_0_dice": dice_values[0], "candidate_1_dice": dice_values[1], "candidate_2_dice": dice_values[2],
                                    "raw_score_0": raw_values[0], "raw_score_1": raw_values[1], "raw_score_2": raw_values[2],
                                    "raw_selected_index": raw_choice, "raw_selected_dice": float(raw_dice[index]),
                                    "tuned_score_0": tuned_values[0], "tuned_score_1": tuned_values[1], "tuned_score_2": tuned_values[2],
                                    "tuned_selected_index": tuned_choice, "tuned_selected_dice": tuned, "tuned_dice": tuned,
                                    "oracle_selected_index": oracle_choice, "oracle_dice": oracle,
                                    "oracle_only": True, "metric_resize_antialias": bool(spec["metric_resize_antialias"]),
                                    "metric_variant": spec["metric_variant"],
                                })
                        observed_setting_samples += len(names)
                        observed_tasks.add(reporting_task)
                require(observed_setting_samples == int(spec["expected_samples"]),
                        f"{setting} actual sample count changed during formal evaluation")
                require(len(observed_tasks) == int(spec["expected_tasks"]),
                        f"{setting} actual task count changed during formal evaluation")
                for percent in (0, 5, 10, 20):
                    require(coverage[(setting, percent)] == int(spec["expected_samples"]),
                            f"Incomplete {setting}/{percent}% formal output")
                logger.info("%s %s complete: %d samples", args.profile, setting, observed_setting_samples)
        os.replace(temporary_csv, output_dir / "per_sample.csv")
    except Exception:
        # Preserve an explicit partial file for inspection; it is never treated as an accepted result.
        raise

    require(source_files(code_root) == source_hashes_before, "Frozen isolated source changed during evaluation")
    require(source_non_head_digest(model) == model_info["source_non_head_state_sha256"],
            "Formal evaluation changed frozen non-head state")
    require(canonical_state_sha256(model.sam.mask_decoder.iou_prediction_head.state_dict()) == model_info["original_head_state_sha256"],
            "Formal evaluation unexpectedly changed original full-checkpoint quality head")

    per_task_rows: list[Dict[str, Any]] = []
    condition_rows: list[Dict[str, Any]] = []
    for setting in profile["settings"]:
        zero_task: Dict[str, Dict[str, Any]] = {}
        for percent in (0, 5, 10, 20):
            task_summaries: list[Dict[str, Any]] = []
            for key in sorted(task_stats):
                if key[:2] != (setting, percent):
                    continue
                task = key[2]
                summary = finalized_stats(task_stats[key])
                if percent == 0:
                    zero_task[task] = summary
                else:
                    require(task in zero_task and summary["samples"] == zero_task[task]["samples"],
                            f"Task membership differs from 0% prompt condition: {setting}/{task}")
                baseline = zero_task[task]
                per_task_rows.append({
                    "profile": args.profile, "model_family": profile["family"], "version": profile["version"],
                    "setting": setting, "box_perturbation_percent": percent, "task": task,
                    "metric_variant": TEST_SETTINGS[setting]["metric_variant"],
                    **summary,
                    "candidate_0_delta_vs_0_percent": summary["candidate_0_dice"] - baseline["candidate_0_dice"],
                    "tuned_delta_vs_0_percent": summary["tuned_quality_dice"] - baseline["tuned_quality_dice"],
                    "oracle_delta_vs_0_percent": summary["oracle_dice"] - baseline["oracle_dice"],
                })
                task_summaries.append(summary)
            require(len(task_summaries) == int(TEST_SETTINGS[setting]["expected_tasks"]), "Per-task result incomplete")
            micro = finalized_stats(condition_stats[(setting, percent)])
            macro = macro_summary(task_summaries)
            zero_micro = finalized_stats(condition_stats[(setting, 0)])
            zero_macro = macro_summary([finalized_stats(task_stats[(setting, 0, task)]) for task in zero_task])
            condition_rows.append({
                "profile": args.profile, "model_family": profile["family"], "version": profile["version"],
                "setting": setting, "protocol_scope": TEST_SETTINGS[setting]["protocol_scope"],
                "target_data_percent": 0, "box_perturbation_percent": percent,
                "metric_variant": TEST_SETTINGS[setting]["metric_variant"],
                "samples": micro["samples"], "tasks": macro["tasks"],
                "candidate_0_sample_mean_dice": micro["candidate_0_dice"],
                "tuned_sample_mean_dice": micro["tuned_quality_dice"],
                "oracle_sample_mean_dice": micro["oracle_dice"],
                "tuned_minus_candidate_0_sample_mean": micro["tuned_quality_dice"] - micro["candidate_0_dice"],
                "oracle_minus_tuned_sample_mean": micro["oracle_dice"] - micro["tuned_quality_dice"],
                "candidate_0_task_macro_dice": macro["candidate_0_dice"],
                "tuned_task_macro_dice": macro["tuned_quality_dice"],
                "oracle_task_macro_dice": macro["oracle_dice"],
                "tuned_minus_candidate_0_task_macro": macro["tuned_quality_dice"] - macro["candidate_0_dice"],
                "oracle_minus_tuned_task_macro": macro["oracle_dice"] - macro["tuned_quality_dice"],
                "tuned_sample_delta_vs_0_percent": micro["tuned_quality_dice"] - zero_micro["tuned_quality_dice"],
                "tuned_task_macro_delta_vs_0_percent": macro["tuned_quality_dice"] - zero_macro["tuned_quality_dice"],
                "candidate_mask_sha256": mask_digests[(setting, percent)].hexdigest(),
                "same_finalized_head_for_all_perturbations": True,
                "raw_selection_counts": micro["raw_selection_counts"],
                "tuned_selection_counts": micro["tuned_selection_counts"],
                "oracle_selection_counts": micro["oracle_selection_counts"],
            })
    require(per_task_rows and condition_rows, "Formal evaluation produced no rows")
    write_csv_atomic(output_dir / "per_task.csv", list(per_task_rows[0].keys()), per_task_rows)
    summary = {
        "schema": SCHEMA + "-profile-result",
        "status": "completed",
        "profile": {"key": args.profile, **profile},
        "model": model_info,
        "finalized_head": head_info,
        "source_files_sha256": source_hashes_before,
        "prompt_cache": {
            "manifest": str(prompt_cache_paths(workdir)["manifest"]),
            "manifest_sha256": sha256(prompt_cache_paths(workdir)["manifest"]),
            "prompt_cache_sha256": prompt_manifest["prompts"]["sha256"],
            "common_full_shared_prompts": True,
        },
        "migration_preflight": preflight_evidence,
        "selection_rules": {
            "fixed_candidate_0": "always returned candidate index 0",
            "tuned_quality": "argmax over finalized head output at raw IoU indices [1:4]",
            "oracle": "argmax over GT-scored Dice after the full forward; never fed into non-oracle selection",
            "same_three_candidates_per_forward": True,
        },
        "runtime": {
            "precision": "FP32",
            "batch_size": int(profile["evaluation_batch_size"]),
            "workers": int(args.workers),
            "device": str(device),
        },
        "metric_contract": {
            "threshold": "sigmoid(mask_logits) > 0.5",
            "empty_rule": "both empty -> 1; GT empty and prediction nonempty -> 0",
            "ID_resize": "bilinear align_corners=False antialias=True",
            "ODD_resize": "bilinear align_corners=False",
        },
        "conditions": condition_rows,
        "artifacts": {
            "per_sample": {"path": str(output_dir / "per_sample.csv"), "sha256": sha256(output_dir / "per_sample.csv")},
            "per_task": {"path": str(output_dir / "per_task.csv"), "sha256": sha256(output_dir / "per_task.csv")},
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    atomic_json(output_dir / "summary.json", summary)
    logger.info("formal evaluation completed: %s", output_dir / "summary.json")


def csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def profile_summary(workdir: Path, profile: str) -> Dict[str, Any]:
    path = workdir / "outputs" / profile / "summary.json"
    require(path.is_file(), f"Missing completed formal result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == SCHEMA + "-profile-result" and payload.get("status") == "completed",
            f"Invalid formal result manifest: {path}")
    require(payload.get("profile", {}).get("key") == profile, f"Profile mismatch in {path}")
    config = profile_config(profile)
    expected_conditions = len(config["settings"]) * len(PERTURBATION_LEVELS)
    require(len(payload.get("conditions", [])) == expected_conditions,
            f"Formal condition count mismatch in {path}")
    actual_condition_keys = {(row["setting"], int(row["box_perturbation_percent"])) for row in payload["conditions"]}
    expected_condition_keys = {(setting, int(round(level * 100)))
                               for setting in config["settings"] for level in PERTURBATION_LEVELS}
    require(actual_condition_keys == expected_condition_keys, f"Formal condition coverage mismatch in {path}")
    sample_csv = workdir / "outputs" / profile / "per_sample.csv"
    require(sample_csv.is_file(), f"Missing formal per-sample output: {sample_csv}")
    with sample_csv.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    expected_rows = sum(int(TEST_SETTINGS[setting]["expected_samples"]) for setting in config["settings"]) * len(PERTURBATION_LEVELS)
    require(row_count == expected_rows, f"Formal per-sample count mismatch in {sample_csv}: {row_count}")
    for condition in payload["conditions"]:
        require(condition["setting"] in config["settings"], "Profile emitted a setting outside its fixed protocol")
        require(condition["metric_variant"] == TEST_SETTINGS[condition["setting"]]["metric_variant"],
                "Formal condition uses the wrong metric resize variant")
        require(int(condition["samples"]) == int(TEST_SETTINGS[condition["setting"]]["expected_samples"]),
                "Formal condition sample count differs from the fixed test list")
    return payload


def paired_full_shared(workdir: Path, version: str) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    full_name, shared_name = f"full_{version}", f"shared_{version}"
    full_csv = workdir / "outputs" / full_name / "per_sample.csv"
    shared_csv = workdir / "outputs" / shared_name / "per_sample.csv"
    require(full_csv.is_file() and shared_csv.is_file(), f"Missing paired per-sample files for {version}")
    shared: Dict[tuple[str, str, str, str], tuple[float, float, float, str]] = {}
    for row in csv_rows(shared_csv):
        key = (row["setting"], row["box_perturbation_percent"], row["task"], row["sample_id"])
        require(key not in shared, f"Duplicate Shared sample key: {key}")
        shared[key] = (float(row["candidate_0_dice"]), float(row["tuned_selected_dice"]),
                       float(row["oracle_dice"]), row["prompt_sha256"])
    groups: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {"n": 0, "c0": 0.0, "tuned": 0.0, "oracle": 0.0})
    task_groups: Dict[tuple[str, str, str], Dict[str, Any]] = defaultdict(lambda: {"n": 0, "c0": 0.0, "tuned": 0.0, "oracle": 0.0})
    seen: set[tuple[str, str, str, str]] = set()
    for row in csv_rows(full_csv):
        key = (row["setting"], row["box_perturbation_percent"], row["task"], row["sample_id"])
        require(key in shared, f"Full/Shared sample membership differs: {key}")
        seen.add(key)
        s_c0, s_tuned, s_oracle, shared_prompt = shared[key]
        require(row["prompt_sha256"] == shared_prompt,
                f"Full/Shared prompt differs for a nominally paired sample: {key}")
        values = (float(row["candidate_0_dice"]) - s_c0,
                  float(row["tuned_selected_dice"]) - s_tuned,
                  float(row["oracle_dice"]) - s_oracle)
        for group in (groups[(key[0], key[1])], task_groups[(key[0], key[1], key[2])]):
            group["n"] += 1
            group["c0"] += values[0]
            group["tuned"] += values[1]
            group["oracle"] += values[2]
    require(seen == set(shared), f"Full/Shared paired membership differs for {version}")
    expected_pairs = sum(int(TEST_SETTINGS[setting]["expected_samples"]) for setting in PROFILES[f"full_{version}"]["settings"]) * len(PERTURBATION_LEVELS)
    require(len(seen) == expected_pairs, f"Full/Shared paired count mismatch for {version}")
    summary_rows = [
        {"full_profile": full_name, "shared_profile": shared_name, "version": version,
         "setting": setting, "box_perturbation_percent": int(percent), "samples": value["n"],
         "metric_variant": TEST_SETTINGS[setting]["metric_variant"],
         "full_minus_shared_candidate_0_dice": value["c0"] / value["n"],
         "full_minus_shared_tuned_quality_dice": value["tuned"] / value["n"],
         "full_minus_shared_oracle_dice": value["oracle"] / value["n"],
         "paired_membership_exact": True}
        for (setting, percent), value in sorted(groups.items())
    ]
    task_rows = [
        {"full_profile": full_name, "shared_profile": shared_name, "version": version,
         "setting": setting, "box_perturbation_percent": int(percent), "task": task, "samples": value["n"],
         "metric_variant": TEST_SETTINGS[setting]["metric_variant"],
         "full_minus_shared_candidate_0_dice": value["c0"] / value["n"],
         "full_minus_shared_tuned_quality_dice": value["tuned"] / value["n"],
         "full_minus_shared_oracle_dice": value["oracle"] / value["n"],
         "paired_membership_exact": True}
        for (setting, percent, task), value in sorted(task_groups.items())
    ]
    return summary_rows, task_rows


def aggregate_results(args: argparse.Namespace) -> None:
    workdir = args.workdir.resolve()
    aggregate_dir = workdir / "outputs" / "aggregate"
    require(not aggregate_dir.exists(), "Aggregate output already exists and is immutable")
    profiles = {name: profile_summary(workdir, name) for name in PROFILES}
    aggregate_dir.mkdir(parents=True)
    formal_rows: list[Dict[str, Any]] = []
    perturb_rows: list[Dict[str, Any]] = []
    for name, result in profiles.items():
        for condition in result["conditions"]:
            row = {"profile": name, **condition}
            formal_rows.append(row)
            perturb_rows.append({
                "profile": name, "model_family": condition["model_family"], "version": condition["version"],
                "setting": condition["setting"], "box_perturbation_percent": condition["box_perturbation_percent"],
                "metric_variant": condition["metric_variant"],
                "candidate_0_sample_mean_dice": condition["candidate_0_sample_mean_dice"],
                "tuned_sample_mean_dice": condition["tuned_sample_mean_dice"],
                "oracle_sample_mean_dice": condition["oracle_sample_mean_dice"],
                "tuned_sample_delta_vs_0_percent": condition["tuned_sample_delta_vs_0_percent"],
                "candidate_0_task_macro_dice": condition["candidate_0_task_macro_dice"],
                "tuned_task_macro_dice": condition["tuned_task_macro_dice"],
                "oracle_task_macro_dice": condition["oracle_task_macro_dice"],
                "tuned_task_macro_delta_vs_0_percent": condition["tuned_task_macro_delta_vs_0_percent"],
                "same_finalized_head_for_all_perturbations": True,
            })
    require(formal_rows, "No completed formal conditions to aggregate")
    require(len(formal_rows) == 32, f"Formal protocol requires exactly 32 conditions, got {len(formal_rows)}")
    total_profile_rows = sum(
        sum(1 for _ in csv_rows(workdir / "outputs" / name / "per_sample.csv"))
        for name in PROFILES
    )
    require(total_profile_rows == 331_616,
            f"Formal protocol requires 331616 per-sample rows, got {total_profile_rows}")
    write_csv_atomic(workdir / "outputs" / "formal_summary.csv", list(formal_rows[0].keys()), formal_rows)
    write_csv_atomic(workdir / "outputs" / "box_perturbation_summary.csv", list(perturb_rows[0].keys()), perturb_rows)
    paired_rows: list[Dict[str, Any]] = []
    paired_task_rows: list[Dict[str, Any]] = []
    for version in ("v1", "v2"):
        summary, per_task = paired_full_shared(workdir, version)
        paired_rows.extend(summary)
        paired_task_rows.extend(per_task)
    require(sum(int(row["samples"]) for row in paired_rows) == 165_808,
            "Formal protocol requires 165808 Full/Shared paired rows")
    write_csv_atomic(workdir / "outputs" / "full_shared_paired.csv", list(paired_rows[0].keys()), paired_rows)
    write_csv_atomic(aggregate_dir / "full_shared_paired_per_task.csv", list(paired_task_rows[0].keys()), paired_task_rows)
    lines = [
        "# InfraSegFM 质量头正式测试",
        "",
        "四个已选小头均按固定开发集选择结果迁移：Full v1 第7轮、Full v2 第11轮、Shared v1 第2轮、Shared v2 第6轮。正式测试未重新选择小头，也未训练或微调模型。",
        "",
        "- v1 使用 ID、CrossSite、RealWorld；v2 使用 CrossTask；全部为 0% 目标数据评测。",
        "- 每个条件使用同一小头测试 0%、5%、10%、20% 框扰动。",
        "- 每次完整前向仅产生一组固定顺序的三个候选；候选0、质量头选择和离线 Oracle 共用这组候选。",
        "- `outputs/formal_summary.csv` 给出单 seed 的正式汇总；`outputs/box_perturbation_summary.csv` 给出相对 0% 提示的下降；Full−Shared 只在版本、设置、提示和样本一致时配对比较。",
        "",
        "## 开发集收益在正式测试的延续情况",
        "",
        "| 模型 | 正式设置 | 0% 框扰动任务宏 Dice（候选0 → 质量头） | 样本平均 Dice（候选0 → 质量头） | 开发集收益方向 | 正式测试收益方向 |",
        "|---|---|---:|---:|---|---|",
    ]
    for name in sorted(PROFILES):
        dev = DEVELOPMENT_REFERENCE[name]
        dev_direction = (float(dev["tuned_macro"]) - float(dev["candidate_0_macro"]) > 0.0
                         and float(dev["tuned_micro"]) - float(dev["candidate_0_micro"]) > 0.0)
        for condition in sorted((row for row in formal_rows if row["profile"] == name and row["box_perturbation_percent"] == 0),
                                key=lambda row: str(row["setting"])):
            formal_direction = (float(condition["tuned_minus_candidate_0_task_macro"]) > 0.0
                                and float(condition["tuned_minus_candidate_0_sample_mean"]) > 0.0)
            lines.append(
                f"| {condition['model_family']} {condition['version']} | {condition['setting']} | "
                f"{condition['candidate_0_task_macro_dice']:.6f} → {condition['tuned_task_macro_dice']:.6f} | "
                f"{condition['candidate_0_sample_mean_dice']:.6f} → {condition['tuned_sample_mean_dice']:.6f} | "
                f"{'正向' if dev_direction else '非正向'} | {'正向' if formal_direction else '非正向'} |"
            )
    lines.extend(["", "## 任务变化", ""])
    for name in sorted(PROFILES):
        task_path = workdir / "outputs" / name / "per_task.csv"
        task_rows = [row for row in csv_rows(task_path) if int(row["box_perturbation_percent"]) == 0]
        for setting in PROFILES[name]["settings"]:
            subset = [row for row in task_rows if row["setting"] == setting]
            improved = sorted((row for row in subset if float(row["tuned_minus_candidate_0"]) > 0.0),
                              key=lambda row: float(row["tuned_minus_candidate_0"]), reverse=True)
            regressed = sorted((row for row in subset if float(row["tuned_minus_candidate_0"]) < 0.0),
                               key=lambda row: float(row["tuned_minus_candidate_0"]))
            def compact(rows: Sequence[Mapping[str, str]]) -> str:
                if not rows:
                    return "无"
                return "; ".join(f"{row['task']} ({float(row['tuned_minus_candidate_0']):+.4f})" for row in rows[:5])
            lines.extend([
                f"- {name}/{setting}：改善 {len(improved)} 个任务，退化 {len(regressed)} 个任务。",
                f"  - 改善幅度前五：{compact(improved)}。",
                f"  - 退化幅度前五：{compact(regressed)}。",
            ])
    lines.extend(["", "## Full 与 Shared 的配对差距", "", "| 版本 | 设置 | 0% 框扰动 Full−Shared 质量头 Dice |", "|---|---|---:|"])
    for row in sorted((row for row in paired_rows if int(row["box_perturbation_percent"]) == 0),
                      key=lambda row: (str(row["version"]), str(row["setting"]))):
        lines.append(f"| {row['version']} | {row['setting']} | {float(row['full_minus_shared_tuned_quality_dice']):+.6f} |")
    lines.extend([
        "",
        "逐样本和逐任务结果保留在各 profile 输出目录。Oracle 只用于离线评分，不参与非 Oracle 候选选择。",
    ])
    (workdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema": SCHEMA + "-aggregate",
        "status": "completed",
        "profiles": {name: {"summary": str(workdir / "outputs" / name / "summary.json"),
                             "summary_sha256": sha256(workdir / "outputs" / name / "summary.json")}
                     for name in PROFILES},
        "artifacts": {
            **{path.relative_to(workdir).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
               for path in sorted(aggregate_dir.iterdir()) if path.is_file()},
            **{path.relative_to(workdir).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
               for path in (workdir / "outputs" / "formal_summary.csv",
                            workdir / "outputs" / "box_perturbation_summary.csv",
                            workdir / "outputs" / "full_shared_paired.csv",
                            workdir / "REPORT.md")},
        },
        "full_shared_pairing": "only same version, setting, perturbation, task and sample_id; exact membership asserted",
        "single_seed_reporting": True,
    }
    atomic_json(aggregate_dir / "manifest.json", manifest)
    print(f"FORMAL_QUALITY_HEAD_AGGREGATE_COMPLETE: {aggregate_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, profile: bool, source: bool, eval_root: bool, device: bool) -> None:
        command.add_argument("--workdir", type=Path, required=True, help="isolated quality-head formal-test directory")
        if profile:
            command.add_argument("--profile", choices=sorted(PROFILES), required=True)
        if source or eval_root or profile:
            command.add_argument("--code-root", type=Path, required=True, help="isolated frozen source copy")
        if source:
            command.add_argument("--source-project-root", type=Path, required=True,
                                 help="read-only project root that contains source development splits")
        if eval_root:
            command.add_argument("--eval-root", type=Path, required=True,
                                 help="read-only playground/eval root containing ID and ODD final-test folders")
        if profile:
            command.add_argument("--sam-checkpoint", type=Path, required=True, help="existing SAM-B checkpoint")
        if device:
            command.add_argument("--device", default="cuda:0", help="one explicitly assigned visible CUDA device")

    prepare = sub.add_parser("prepare", help="write immutable final-test membership and prompt cache")
    common(prepare, profile=False, source=False, eval_root=True, device=False)
    prepare.add_argument("--prompt-seed", type=int, default=PROMPT_SEED, choices=(PROMPT_SEED,),
                         help="locked deterministic prompt seed")

    preflight = sub.add_parser("preflight", help="prove head-overlay migration on three source-development samples")
    common(preflight, profile=True, source=True, eval_root=False, device=True)

    evaluate = sub.add_parser("evaluate", help="evaluate one fixed profile on its formal-test settings")
    common(evaluate, profile=True, source=False, eval_root=True, device=True)
    evaluate.add_argument("--workers", type=int, default=2, choices=(2,), help="locked DataLoader worker count")

    aggregate = sub.add_parser("aggregate", help="make formal summary and paired Full--Shared comparison tables")
    aggregate.add_argument("--workdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_prompt_cache(args)
    elif args.command == "preflight":
        source_dev_preflight(args)
    elif args.command == "evaluate":
        evaluate_profile(args)
    elif args.command == "aggregate":
        aggregate_results(args)
    else:  # argparse makes this unreachable.
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QUALITY_HEAD_FORMAL_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
