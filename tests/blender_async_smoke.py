"""Run with: blender --background --python tests/blender_async_smoke.py"""

from __future__ import annotations

import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # type: ignore  # noqa: E402

from interactive.blender import addon  # noqa: E402


addon.register()
main_thread = threading.get_ident()
worker_started = threading.Event()
worker_release = threading.Event()
observed = {}


def work() -> dict:
    observed["worker_thread"] = threading.get_ident()
    worker_started.set()
    if not worker_release.wait(timeout=5):
        raise TimeoutError("async smoke worker was not released")
    return {"ok": True, "value": 7}


def apply(result: dict) -> dict:
    observed["apply_thread"] = threading.get_ident()
    observed["value"] = result["value"]
    return result


def success(_context, response: dict) -> None:
    observed["success_thread"] = threading.get_ident()
    observed["success_value"] = response["value"]


assert addon.submit_async(
    bpy.context,
    label="Async smoke",
    work=work,
    apply=apply,
    success=success,
)
assert worker_started.wait(timeout=5)
assert addon.async_busy()

# The main thread remains available while the worker is waiting.
bpy.context.scene.skintokens_status = "main-thread-responsive"
assert bpy.context.scene.skintokens_status == "main-thread-responsive"

worker_release.set()
addon._ASYNC_JOB["future"].result(timeout=5)
if bpy.app.timers.is_registered(addon._poll_async_job):
    bpy.app.timers.unregister(addon._poll_async_job)
assert addon._poll_async_job() is None

assert observed["worker_thread"] != main_thread
assert observed["apply_thread"] == main_thread
assert observed["success_thread"] == main_thread
assert observed["value"] == 7
assert observed["success_value"] == 7
assert not addon.async_busy()

addon.unregister()
print("SKINTOKENS_ASYNC_SMOKE_OK")
