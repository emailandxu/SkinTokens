"""Stress interactive session metadata without loading the model.

Run with:
    .venv/bin/python tests/stress_interactive_sessions.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
import tracemalloc
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interactive.asset_store import MeshAssetStore
from interactive.server import InteractiveHttpApi, InteractiveModelServer
from interactive.session import SessionRecord


MIB = 1024 * 1024


def rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if not statm.is_file():
        return 0
    resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def memory_sample() -> dict[str, float]:
    current, peak = tracemalloc.get_traced_memory()
    return {
        "python_current_mib": current / MIB,
        "python_peak_mib": peak / MIB,
        "rss_mib": rss_bytes() / MIB,
    }


def delta_mib(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return after[key] - before[key]


def run(args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="skintokens_session_stress_") as temporary:
        root = Path(temporary)
        obj_path = root / "mesh.obj"
        obj_path.write_text("v 0 0 0\n", encoding="utf-8")

        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.max_runtime_sessions = int(args.runtime_limit)
        service.max_sessions = int(args.session_limit)
        service.sessions = OrderedDict()
        service.runtime_cache = OrderedDict()
        api = InteractiveHttpApi(
            service=service,
            asset_store=MeshAssetStore(root / "assets"),
            result_dir=root / "results",
            max_upload_bytes=MIB,
            max_pending_gpu_requests=1,
        )

        tracemalloc.start()
        baseline = memory_sample()

        started_at = time.perf_counter()
        for index in range(args.sessions):
            session_id = f"session-{index:08d}"
            record = SessionRecord(
                session_id=session_id,
                owner_id=f"owner-{index % max(1, args.owners)}",
                asset_id="shared-asset",
                obj_path=obj_path,
                created_at=float(index),
                updated_at=float(index),
            )
            record.reserved_joint_names.update(
                f"bone_{bone}" for bone in range(args.reserved_names)
            )
            record.next_bone_id = args.reserved_names
            service.sessions[session_id] = record
            service._put_runtime(
                SimpleNamespace(
                    session_id=session_id,
                    payload=bytearray(args.runtime_bytes),
                )
            )
            service._enforce_session_limit()
        gc.collect()
        after_sessions = memory_sample()
        session_seconds = time.perf_counter() - started_at
        runtime_entries_before_reset = len(service.runtime_cache)
        retained_sessions = len(service.sessions)

        started_at = time.perf_counter()
        held_locks = []
        for session_id in list(service.sessions):
            held_locks.append(api._session_lock(session_id))
        for index in range(args.unknown_locks):
            held_locks.append(api._session_lock(f"unknown-{index:08d}"))
        gc.collect()
        after_locks = memory_sample()
        lock_seconds = time.perf_counter() - started_at
        lock_entries_before_reset = len(api.session_locks)
        del held_locks
        gc.collect()
        after_lock_release = memory_sample()

        started_at = time.perf_counter()
        for session_id in list(service.sessions):
            service.reset(
                {
                    "command": "reset",
                    "session_id": session_id,
                    "owner_id": service.sessions[session_id].owner_id,
                }
            )
        gc.collect()
        after_reset = memory_sample()
        reset_seconds = time.perf_counter() - started_at

        store = api.asset_store
        started_at = time.perf_counter()
        duplicate_content = b"v 0 0 0\n" + b"#" * max(0, args.asset_bytes - 8)
        for _ in range(args.assets):
            store.save_obj(duplicate_content)
        duplicate_files = len(list(store.root.glob("*.obj")))
        for index in range(args.assets):
            prefix = f"v {index} 0 0\n".encode("ascii")
            content = prefix + b"#" * max(0, args.asset_bytes - len(prefix))
            store.save_obj(content)
        asset_files = list(store.root.glob("*.obj"))
        asset_seconds = time.perf_counter() - started_at
        asset_bytes = sum(path.stat().st_size for path in asset_files)

        runtime_payload_mib = (
            min(retained_sessions, args.runtime_limit) * args.runtime_bytes / MIB
        )
        session_python_delta = delta_mib(
            after_sessions,
            baseline,
            "python_current_mib",
        )
        estimated_metadata_mib = max(0.0, session_python_delta - runtime_payload_mib)
        lock_python_delta = delta_mib(
            after_locks,
            after_sessions,
            "python_current_mib",
        )

        result = {
            "config": vars(args),
            "sessions": {
                "created": args.sessions,
                "remaining_before_reset": retained_sessions,
                "runtime_entries": runtime_entries_before_reset,
                "runtime_limit": args.runtime_limit,
                "creation_seconds": session_seconds,
                "estimated_metadata_mib": estimated_metadata_mib,
                "estimated_bytes_per_session": (
                    estimated_metadata_mib * MIB / max(1, retained_sessions)
                ),
                "memory": after_sessions,
            },
            "locks": {
                "entries": lock_entries_before_reset,
                "creation_seconds": lock_seconds,
                "python_delta_mib": lock_python_delta,
                "estimated_bytes_per_lock": (
                    lock_python_delta * MIB / max(1, lock_entries_before_reset)
                ),
                "memory": after_locks,
                "after_release": {
                    "entries": len(api.session_locks),
                    "memory": after_lock_release,
                },
            },
            "after_reset": {
                "sessions": len(service.sessions),
                "runtime_entries": len(service.runtime_cache),
                "lock_entries": len(api.session_locks),
                "reset_seconds": reset_seconds,
                "memory": after_reset,
            },
            "assets": {
                "duplicate_writes": args.assets,
                "files_after_duplicate_writes": duplicate_files,
                "unique_writes": args.assets,
                "total_files": len(asset_files),
                "total_bytes": asset_bytes,
                "write_seconds": asset_seconds,
            },
            "baseline": baseline,
        }
        tracemalloc.stop()
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--owners", type=int, default=1_000)
    parser.add_argument("--reserved-names", type=int, default=96)
    parser.add_argument("--runtime-limit", type=int, default=8)
    parser.add_argument("--session-limit", type=int, default=512)
    parser.add_argument("--runtime-bytes", type=int, default=MIB)
    parser.add_argument("--unknown-locks", type=int, default=20_000)
    parser.add_argument("--assets", type=int, default=1_000)
    parser.add_argument("--asset-bytes", type=int, default=1_024)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))
