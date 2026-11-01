# -*- coding: utf-8 -*-
"""Small runtime logging helpers for long training runs.

These helpers avoid buffered, fragile logs during overnight experiments:
JSONL records are appended and fsynced, and checkpoints are written to a
temporary file before being atomically moved into place.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict


def _json_default(value: Any) -> Any:
    try:
        return value.item()
    except Exception:
        pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=_json_default)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _atomic_write_json(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _record_heartbeat_error(out_dir: str, exc: Exception) -> None:
    try:
        os.makedirs(os.path.abspath(out_dir), exist_ok=True)
        path = os.path.join(out_dir, "heartbeat_errors.log")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {type(exc).__name__}: {exc}\n")
    except Exception:
        pass


def _cuda_snapshot(torch_module: Any) -> Dict[str, Any]:
    try:
        if not torch_module.cuda.is_available():
            return {"available": False}
        idx = torch_module.cuda.current_device()
        return {
            "available": True,
            "device": int(idx),
            "name": torch_module.cuda.get_device_name(idx),
            "memory_allocated_mb": round(torch_module.cuda.memory_allocated(idx) / (1024 ** 2), 2),
            "memory_reserved_mb": round(torch_module.cuda.memory_reserved(idx) / (1024 ** 2), 2),
            "max_memory_allocated_mb": round(torch_module.cuda.max_memory_allocated(idx) / (1024 ** 2), 2),
        }
    except Exception as exc:
        return {"available": None, "error": repr(exc)}


def write_heartbeat(out_dir: str, script: str, phase: str, torch_module: Any = None, **fields: Any) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    record: Dict[str, Any] = {
        "time": now,
        "time_unix": round(time.time(), 3),
        "pid": os.getpid(),
        "script": script,
        "phase": phase,
        "cwd": os.getcwd(),
    }
    record.update(fields)
    if torch_module is not None:
        record["cuda"] = _cuda_snapshot(torch_module)
    try:
        append_jsonl(os.path.join(out_dir, "heartbeat.jsonl"), record)
        _atomic_write_json(os.path.join(out_dir, "heartbeat_latest.json"), record)
    except Exception as exc:
        _record_heartbeat_error(out_dir, exc)


def atomic_torch_save(torch_module: Any, obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        torch_module.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
