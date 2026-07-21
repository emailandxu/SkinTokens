from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Callable

import bpy  # type: ignore
from bpy.app.handlers import persistent  # type: ignore

from .apply_skin import (
    parse_armature_object,
    promote_active_armature_bone_selection,
    selected_skin_bone_names,
)
from .core import (
    DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL,
    BlenderInteractiveCore,
    same_skeleton_context,
)
from .transport import DEFAULT_MODEL_URL, EXTENSION_VERSION


_CORE: BlenderInteractiveCore | None = None
_HISTORY_SYNCING = False
_HISTORY_SYNC_PENDING = False
_HISTORY_TIMER_REGISTERED = False
_HISTORY_SYNC_DEADLINE = 0.0
_VAE_LEVEL_SETTING = False
_LEGACY_PREVIEW_COLLECTION = "SkinTokens Interactive Preview"
_ASYNC_JOB: dict | None = None
_ASYNC_POLL_REGISTERED = False
_ARMATURE_WATCH_TIMER_REGISTERED = False
_ARMATURE_WATCH_SESSION_ID = ""
_ARMATURE_WATCH_SIGNATURE: tuple | None = None
_ARMATURE_WATCH_INTERVAL = 0.15
_DEBUG_LOG_PATH = Path(tempfile.gettempdir()) / "skintokens_interactive_debug.log"
_DEBUG_LOG_LOCK = threading.Lock()
_DEBUG_LOG_MAX_BYTES = 2 * 1024 * 1024


def _debug_log(event: str, **fields) -> None:
    record = {
        "time": time.time(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    try:
        line = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
        with _DEBUG_LOG_LOCK:
            mode = (
                "w"
                if _DEBUG_LOG_PATH.exists()
                and _DEBUG_LOG_PATH.stat().st_size >= _DEBUG_LOG_MAX_BYTES
                else "a"
            )
            with _DEBUG_LOG_PATH.open(mode, encoding="utf-8") as stream:
                stream.write(f"{line}\n")
    except OSError:
        pass


def _debug_bone_state(armature_obj) -> dict:
    state = {"mode": str(armature_obj.mode)}
    if armature_obj.mode != "EDIT":
        return state
    active = armature_obj.data.edit_bones.active
    if active is None:
        return {**state, "active": None}

    def endpoint(bone, attribute: str) -> list[float]:
        value = getattr(bone, attribute)
        return [float(value.x), float(value.y), float(value.z)]

    parent = active.parent
    state["active"] = {
        "name": str(active.name),
        "head": endpoint(active, "head"),
        "tail": endpoint(active, "tail"),
        "select": bool(active.select),
        "select_head": bool(active.select_head),
        "select_tail": bool(active.select_tail),
        "parent": (
            None
            if parent is None
            else {
                "name": str(parent.name),
                "head": endpoint(parent, "head"),
                "tail": endpoint(parent, "tail"),
            }
        ),
    }
    return state


class SKINTOKENS_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    server_url: bpy.props.StringProperty(
        name="Server URL",
        description="SkinTokens interactive model service",
        default=DEFAULT_MODEL_URL,
    )
    request_timeout: bpy.props.FloatProperty(
        name="Request Timeout",
        description="Maximum time to wait for one model request",
        default=300.0,
        min=10.0,
        max=3600.0,
        subtype="TIME",
    )

    def draw(self, _context) -> None:
        layout = self.layout
        layout.prop(self, "server_url")
        layout.prop(self, "request_timeout")
        layout.operator("skintokens_interactive.test_connection", icon="URL")
        layout.label(text=f"Extension {EXTENSION_VERSION}")


def get_addon_preferences(context):
    preferences = getattr(context, "preferences", None)
    addons = None if preferences is None else getattr(preferences, "addons", None)
    entry = None if addons is None else addons.get(__package__)
    return None if entry is None else entry.preferences


def get_core(context) -> BlenderInteractiveCore:
    global _CORE
    preferences = get_addon_preferences(context)
    if preferences is None:
        endpoint = context.scene.skintokens_model_socket.strip() or DEFAULT_MODEL_URL
        request_timeout = 300.0
    else:
        endpoint = preferences.server_url.strip() or DEFAULT_MODEL_URL
        request_timeout = float(preferences.request_timeout)
    owner_id = context.scene.skintokens_owner_id
    if not owner_id:
        owner_id = uuid.uuid4().hex
        context.scene.skintokens_owner_id = owner_id
    changed = (
        _CORE is not None
        and (
            _CORE.model_socket != endpoint
            or _CORE.owner_id != owner_id
            or _CORE.request_timeout != request_timeout
        )
    )
    if changed and _CORE.sessions:
        raise RuntimeError("Finish the active SkinTokens session before changing server settings")
    if _CORE is None or changed:
        _CORE = BlenderInteractiveCore(
            model_socket=endpoint,
            owner_id=owner_id,
            request_timeout=request_timeout,
        )
    return _CORE


def set_status(context, message: str) -> None:
    context.scene.skintokens_status = message
    print(f"[SkinTokens Interactive] {message}")


def online_access_enabled() -> bool:
    return bool(getattr(bpy.app, "online_access", True))


def online_access_forced_off() -> bool:
    return not online_access_enabled() and bool(
        getattr(bpy.app, "online_access_override", False)
    )


def async_busy() -> bool:
    return _ASYNC_JOB is not None


def _redraw_sidebar() -> None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _background_future(work: Callable[[], dict], *, name: str) -> Future:
    future = Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(work())
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=run, name=name, daemon=True).start()
    return future


def _poll_async_job() -> float | None:
    global _ASYNC_JOB, _ASYNC_POLL_REGISTERED
    job = _ASYNC_JOB
    if job is None:
        _ASYNC_POLL_REGISTERED = False
        return None
    future = job["future"]
    if not future.done():
        return 0.05

    context = bpy.context
    try:
        background_result = future.result()
        response = job["apply"](background_result)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "request failed"))
        job["success"](context, response)
        undo_label = job.get("undo_label")
        if undo_label:
            try:
                bpy.ops.ed.undo_push(message=str(undo_label))
            except Exception:
                traceback.print_exc()
    except Exception as exc:
        traceback.print_exc()
        set_status(context, f"{job['label']} failed: {exc}")
    finally:
        _ASYNC_JOB = None
        _ASYNC_POLL_REGISTERED = False
        _redraw_sidebar()
    return None


def submit_async(
    context,
    *,
    label: str,
    work: Callable[[], dict],
    apply: Callable[[dict], dict],
    success: Callable[[object, dict], None],
    undo_label: str | None = None,
) -> bool:
    global _ASYNC_JOB, _ASYNC_POLL_REGISTERED
    if async_busy():
        set_status(context, f"{_ASYNC_JOB['label']} is still running")
        return False
    _ASYNC_JOB = {
        "label": label,
        "future": _background_future(work, name="skintokens-http"),
        "apply": apply,
        "success": success,
        "undo_label": undo_label,
    }
    set_status(context, f"{label}...")
    if not _ASYNC_POLL_REGISTERED:
        _ASYNC_POLL_REGISTERED = True
        bpy.app.timers.register(_poll_async_job, first_interval=0.05)
    _redraw_sidebar()
    return True


def _apply_context_result(
    core: BlenderInteractiveCore,
    prepared: dict,
    remote_result: dict,
    apply_result: Callable[[dict, dict], dict],
) -> dict:
    response = core.apply_session_request(prepared["remote"], remote_result)
    if not response.get("ok"):
        return response
    if not core.prepared_context_is_current(prepared):
        _queue_history_sync()
        return {
            "ok": False,
            "code": "STALE_RESULT",
            "error": "Armature changed while the request was running; result discarded",
        }
    return apply_result(prepared, response)


def remove_legacy_preview_objects() -> None:
    collection = bpy.data.collections.get(_LEGACY_PREVIEW_COLLECTION)
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(collection)


def selected_vae_bone_name(context) -> str | None:
    if _CORE is None:
        return None
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    session = _CORE.sessions.get(session_id)
    if session is None or not session.mesh_object_name:
        return None
    try:
        names = selected_skin_bone_names(session.mesh_object_name)
    except Exception:
        return None
    return names[0] if len(names) == 1 else None


def has_vae_preview(context) -> bool:
    if _CORE is None:
        return False
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    session = _CORE.sessions.get(session_id)
    if session is None:
        return False
    return any(level > 0 for level in session.vae_levels.values())


def has_started_session(context) -> bool:
    if _CORE is None:
        return False
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    return bool(session_id and session_id in _CORE.sessions)


def has_session_marker(context) -> bool:
    return bool(getattr(context.scene, "skintokens_blender_session_id", ""))


def get_vae_reconstruction_level(_scene) -> int:
    context = bpy.context
    if _CORE is None:
        return 0
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    bone_name = selected_vae_bone_name(context)
    if not session_id or bone_name is None:
        return 0
    return _CORE.vae_reconstruction_level(session_id, bone_name)


def _report_vae_result(context, response: dict, bone_name: str) -> None:
    report = response.get("vae_reconstruction") or {}
    if response.get("generated"):
        detail = (
            f"generated levels 0-{response.get('max_level', 3)} "
            f"in {float(report.get('wall_sec', 0.0)):.2f}s"
        )
    else:
        detail = "cached"
    reset = (
        ", cache reset after weight edit"
        if response.get("cache_invalidated")
        else ""
    )
    set_status(
        context,
        f"VAE {response.get('bone_name', bone_name)} level "
        f"{response.get('level', 0)}: {detail}{reset}",
    )


def set_vae_reconstruction_level(_scene, value: int) -> None:
    global _VAE_LEVEL_SETTING
    if _VAE_LEVEL_SETTING:
        return
    _VAE_LEVEL_SETTING = True
    context = bpy.context
    try:
        if async_busy():
            raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
        session_id = context.scene.skintokens_blender_session_id
        if not session_id:
            raise RuntimeError("Start a session first")
        bone_name = selected_vae_bone_name(context)
        if bone_name is None:
            raise RuntimeError(
                "Activate one skin vertex group or select exactly one pose bone"
            )
        core = get_core(context)
        prepared = core.prepare_vae_reconstruction_level(
            session_id,
            int(value),
        )
        if not prepared.get("ok"):
            raise RuntimeError(prepared.get("error", "VAE reconstruction failed"))
        if not prepared["needs_remote"]:
            response = core._apply_vae_level(prepared)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "VAE reconstruction failed"))
            _report_vae_result(context, response, bone_name)
            return

        def apply_vae(result: dict) -> dict:
            response = core.apply_session_request(
                result["remote"],
                result["remote_result"],
            )
            if not response.get("ok"):
                return response
            if not core.prepared_vae_skin_is_current(prepared):
                session = core.sessions.get(session_id)
                if session is not None:
                    session.clear_vae_cache()
                _queue_history_sync()
                return {
                    "ok": False,
                    "code": "STALE_RESULT",
                    "error": (
                        "Armature or weights changed while VAE was running; "
                        "result discarded"
                    ),
                }
            return core._apply_vae_level(
                prepared,
                response=response,
                trajectory=result["trajectory"],
            )

        submit_async(
            context,
            label="VAE reconstruction",
            work=lambda: core.execute_vae_request(prepared),
            apply=apply_vae,
            success=lambda result_context, response: _report_vae_result(
                result_context,
                response,
                bone_name,
            ),
        )
    except Exception as exc:
        traceback.print_exc()
        set_status(context, f"VAE level failed: {exc}")
    finally:
        _VAE_LEVEL_SETTING = False


def selected_start_objects(context):
    active = context.object
    selected = list(context.selected_objects)
    if active is None:
        raise RuntimeError("Select a mesh, or select a mesh together with an armature")

    selected_meshes = [obj for obj in selected if obj.type == "MESH"]
    selected_armatures = [obj for obj in selected if obj.type == "ARMATURE"]
    if active.type == "MESH":
        mesh_obj = active
        if len(selected_armatures) > 1:
            raise RuntimeError("Select at most one armature for Start")
        armature_obj = selected_armatures[0] if selected_armatures else None
        return mesh_obj, armature_obj

    if active.type == "ARMATURE":
        if len(selected_meshes) != 1:
            raise RuntimeError(
                "Select exactly one target mesh together with the active armature"
            )
        return selected_meshes[0], active

    raise RuntimeError("Start requires a mesh and optionally one armature")


def export_selected_obj(context, obj) -> Path:
    path = Path(tempfile.gettempdir()) / f"skintokens_interactive_{obj.name}.obj"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    context.view_layer.objects.active = obj
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
    else:
        bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)
    return path


def source_obj_path(context, obj) -> Path:
    raw = obj.get("skintokens_source_obj")
    if raw:
        path = Path(str(raw)).expanduser().resolve()
        if path.exists():
            return path
    return export_selected_obj(context, obj)


def _flush_history_sync() -> float | None:
    global _HISTORY_SYNCING, _HISTORY_SYNC_PENDING, _HISTORY_TIMER_REGISTERED
    remaining = _HISTORY_SYNC_DEADLINE - time.monotonic()
    if remaining > 0.0:
        return remaining
    if async_busy():
        _debug_log("history.flush_waiting", busy_label=_ASYNC_JOB["label"])
        return 0.1
    _HISTORY_TIMER_REGISTERED = False
    if not _HISTORY_SYNC_PENDING:
        return None
    _HISTORY_SYNC_PENDING = False
    if _CORE is None or _HISTORY_SYNCING:
        return None
    scene = bpy.context.scene
    session_id = scene.skintokens_blender_session_id
    if not session_id:
        return None
    _HISTORY_SYNCING = True
    try:
        prepared = _CORE.prepare_history_sync(session_id)
        _debug_log(
            "history.prepared",
            session_id=session_id,
            ok=bool(prepared.get("ok")),
            needs_remote=bool(prepared.get("needs_remote")),
            joints=prepared.get("source_context", prepared.get("context", {})).get(
                "joints", []
            ),
        )
        if not prepared.get("ok"):
            set_status(
                bpy.context,
                f"Armature sync failed: {prepared.get('error')}",
            )
            return None
        if not prepared.get("needs_remote"):
            return None

        core = _CORE

        def apply_sync(result: dict) -> dict:
            response = core.apply_session_request(prepared["remote"], result)
            if not response.get("ok"):
                return response
            if not core.prepared_context_is_current(prepared):
                _queue_history_sync(0.1)
                return {
                    "ok": True,
                    "context": prepared["source_context"],
                    "synced": False,
                    "stale": True,
                }
            return core.apply_history_sync(prepared, response)

        def succeeded(result_context, response: dict) -> None:
            if response.get("stale"):
                _debug_log("history.stale", session_id=session_id)
                set_status(result_context, "Armature changed again; resync pending")
                return
            count = len(response.get("context", {}).get("joints", []))
            _debug_log(
                "history.succeeded",
                session_id=session_id,
                joint_count=count,
                joints=response.get("context", {}).get("joints", []),
            )
            set_status(result_context, f"Armature auto-synced: {count} joints")

        submit_async(
            bpy.context,
            label="Armature sync",
            work=lambda: core.execute_session_request(prepared["remote"]),
            apply=apply_sync,
            success=succeeded,
        )
    except Exception as exc:
        _debug_log("history.failed", session_id=session_id, error=str(exc))
        traceback.print_exc()
        set_status(bpy.context, f"Armature sync failed: {exc}")
    finally:
        _HISTORY_SYNCING = False
    return None


def _queue_history_sync(delay: float = 0.0) -> None:
    global _HISTORY_SYNC_PENDING, _HISTORY_TIMER_REGISTERED, _HISTORY_SYNC_DEADLINE
    if _CORE is None:
        return
    _HISTORY_SYNC_PENDING = True
    _HISTORY_SYNC_DEADLINE = time.monotonic() + max(0.0, float(delay))
    _debug_log(
        "history.queued",
        delay=float(delay),
        timer_registered=bool(_HISTORY_TIMER_REGISTERED),
    )
    if _HISTORY_TIMER_REGISTERED:
        return
    _HISTORY_TIMER_REGISTERED = True
    bpy.app.timers.register(
        _flush_history_sync,
        first_interval=max(0.0, float(delay)),
    )


def _armature_context_signature(context: dict) -> tuple:
    return (
        tuple(
            tuple(float(value) for value in joint)
            for joint in context.get("joints", [])
        ),
        tuple(int(parent) for parent in context.get("parents", [])),
        tuple(str(name) for name in context.get("joint_names", [])),
    )


def _watch_active_armature() -> float | None:
    global _ARMATURE_WATCH_TIMER_REGISTERED
    global _ARMATURE_WATCH_SESSION_ID, _ARMATURE_WATCH_SIGNATURE

    if _CORE is None:
        _ARMATURE_WATCH_TIMER_REGISTERED = False
        _ARMATURE_WATCH_SESSION_ID = ""
        _ARMATURE_WATCH_SIGNATURE = None
        return None
    scene = getattr(bpy.context, "scene", None)
    session_id = "" if scene is None else str(
        getattr(scene, "skintokens_blender_session_id", "")
    )
    session = _CORE.sessions.get(session_id)
    if session is None or not session.armature_object_name:
        _ARMATURE_WATCH_TIMER_REGISTERED = False
        _ARMATURE_WATCH_SESSION_ID = ""
        _ARMATURE_WATCH_SIGNATURE = None
        return None
    armature_obj = bpy.data.objects.get(session.armature_object_name)
    if armature_obj is None:
        return _ARMATURE_WATCH_INTERVAL

    try:
        promote_active_armature_bone_selection(session.armature_object_name)
        joints, parents, names = parse_armature_object(armature_obj)
    except (RuntimeError, ValueError):
        return _ARMATURE_WATCH_INTERVAL
    current = {
        "joints": joints,
        "parents": parents,
        "joint_names": names,
    }
    signature = _armature_context_signature(current)
    if session_id != _ARMATURE_WATCH_SESSION_ID:
        _ARMATURE_WATCH_SESSION_ID = session_id
        _ARMATURE_WATCH_SIGNATURE = signature
        _debug_log(
            "watch.session",
            session_id=session_id,
            joints=joints,
            bone=_debug_bone_state(armature_obj),
        )
        return _ARMATURE_WATCH_INTERVAL
    if signature != _ARMATURE_WATCH_SIGNATURE:
        _ARMATURE_WATCH_SIGNATURE = signature
        _debug_log(
            "watch.changed",
            session_id=session_id,
            joints=joints,
            bone=_debug_bone_state(armature_obj),
        )
        return _ARMATURE_WATCH_INTERVAL

    if (
        not same_skeleton_context(session.context, current)
        and not _HISTORY_SYNC_PENDING
        and not _HISTORY_SYNCING
    ):
        _debug_log(
            "watch.mismatch",
            session_id=session_id,
            session_joints=session.context.get("joints", []),
            armature_joints=joints,
            bone=_debug_bone_state(armature_obj),
        )
        _queue_history_sync()
    return _ARMATURE_WATCH_INTERVAL


def ensure_armature_watch() -> None:
    global _ARMATURE_WATCH_TIMER_REGISTERED
    if _ARMATURE_WATCH_TIMER_REGISTERED:
        return
    _ARMATURE_WATCH_TIMER_REGISTERED = True
    _debug_log("watch.started", interval=_ARMATURE_WATCH_INTERVAL)
    bpy.app.timers.register(
        _watch_active_armature,
        first_interval=_ARMATURE_WATCH_INTERVAL,
        persistent=True,
    )


def cancel_armature_watch() -> None:
    global _ARMATURE_WATCH_TIMER_REGISTERED
    global _ARMATURE_WATCH_SESSION_ID, _ARMATURE_WATCH_SIGNATURE
    if bpy.app.timers.is_registered(_watch_active_armature):
        bpy.app.timers.unregister(_watch_active_armature)
    _ARMATURE_WATCH_TIMER_REGISTERED = False
    _ARMATURE_WATCH_SESSION_ID = ""
    _ARMATURE_WATCH_SIGNATURE = None
    _debug_log("watch.stopped")


@persistent
def sync_model_after_history_change(_dummy=None) -> None:
    _queue_history_sync()


@persistent
def sync_model_after_armature_change(_scene, depsgraph) -> None:
    if _CORE is None or _HISTORY_SYNCING:
        return
    session_id = getattr(bpy.context.scene, "skintokens_blender_session_id", "")
    session = _CORE.sessions.get(session_id)
    if session is None or not session.armature_object_name:
        return
    armature_obj = bpy.data.objects.get(session.armature_object_name)
    if armature_obj is None:
        return
    watched = {armature_obj, armature_obj.data}
    updates = [
        {
            "type": type(update.id).__name__,
            "name": str(getattr(update.id, "name", "")),
            "geometry": bool(update.is_updated_geometry),
            "transform": bool(update.is_updated_transform),
        }
        for update in depsgraph.updates
    ]
    watched_update = any(
        update.id in watched or getattr(update.id, "original", None) in watched
        for update in depsgraph.updates
    )
    if not watched_update:
        return
    selection_promoted = promote_active_armature_bone_selection(
        session.armature_object_name
    )
    _debug_log(
        "depsgraph.armature",
        session_id=session_id,
        updates=updates,
        selection_promoted=bool(selection_promoted),
        bone=_debug_bone_state(armature_obj),
    )
    _queue_history_sync(0.25)


def cancel_history_sync() -> None:
    global _HISTORY_SYNC_PENDING, _HISTORY_TIMER_REGISTERED, _HISTORY_SYNC_DEADLINE
    _HISTORY_SYNC_PENDING = False
    _HISTORY_SYNC_DEADLINE = 0.0
    if bpy.app.timers.is_registered(_flush_history_sync):
        bpy.app.timers.unregister(_flush_history_sync)
    _HISTORY_TIMER_REGISTERED = False


def cancel_async_job() -> None:
    global _ASYNC_JOB, _ASYNC_POLL_REGISTERED
    if bpy.app.timers.is_registered(_poll_async_job):
        bpy.app.timers.unregister(_poll_async_job)
    if _ASYNC_JOB is not None:
        _ASYNC_JOB["future"].cancel()
    _ASYNC_JOB = None
    _ASYNC_POLL_REGISTERED = False


def _submit_server_cleanup(core: BlenderInteractiveCore, payload: dict) -> None:
    future = _background_future(
        lambda: core.model_request(payload),
        name="skintokens-reset",
    )

    def completed(done) -> None:
        try:
            response = done.result()
            if not response.get("ok"):
                print(
                    "[SkinTokens Interactive] Server cleanup failed: "
                    f"{response.get('error', 'reset failed')}"
                )
        except Exception:
            traceback.print_exc()

    future.add_done_callback(completed)


def _generation_options(scene, *, skin: bool = False) -> dict:
    options = {
        "top_k": scene.skintokens_top_k,
        "top_p": scene.skintokens_top_p,
        "temperature": scene.skintokens_temperature,
        "repetition_penalty": scene.skintokens_repetition_penalty,
        "num_beams": scene.skintokens_num_beams,
    }
    if skin:
        options["max_new_tokens"] = scene.skintokens_skin_max_new_tokens
    else:
        options["max_new_tokens"] = scene.skintokens_max_new_tokens
    return options


def _schedule_next(context, *, force_parent: bool) -> bool:
    if async_busy():
        raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("Start a session first")
    core = get_core(context)
    prepared = core.prepare_next(
        session_id,
        force_parent=force_parent,
        **_generation_options(context.scene),
    )
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "next failed"))

    def succeeded(result_context, response: dict) -> None:
        count = len(response.get("context", {}).get("joints", []))
        done = bool(response.get("context", {}).get("done", False))
        prefix = "Forced child. Joints" if force_parent else "Generated joints"
        set_status(result_context, f"{prefix}: {count}{' (done)' if done else ''}")

    return submit_async(
        context,
        label="Force Next" if force_parent else "Next",
        work=lambda: core.execute_session_request(prepared["remote"]),
        apply=lambda result: _apply_context_result(
            core,
            prepared,
            result,
            core.apply_next_request,
        ),
        success=succeeded,
        undo_label="SkinTokens Force Next" if force_parent else "SkinTokens Next",
    )


def _schedule_rig(context) -> bool:
    if async_busy():
        raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("Start a session first")
    core = get_core(context)
    options = _generation_options(context.scene)
    options.pop("max_new_tokens", None)
    prepared = core.prepare_rig(session_id, **options)
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "rig generation failed"))

    def succeeded(result_context, response: dict) -> None:
        result_skeleton = response.get("context", {})
        count = len(result_skeleton.get("joints", []))
        done = bool(result_skeleton.get("done", False))
        set_status(
            result_context,
            f"Rig continuation: {count} joints{' (complete)' if done else ''}",
        )

    return submit_async(
        context,
        label="Rig",
        work=lambda: core.execute_session_request(prepared["remote"]),
        apply=lambda result: _apply_context_result(
            core,
            prepared,
            result,
            core.apply_rig_request,
        ),
        success=succeeded,
        undo_label="SkinTokens Rig",
    )


def _schedule_skin(context) -> bool:
    if async_busy():
        raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("Start a session first")
    core = get_core(context)
    prepared = core.prepare_skin(
        session_id,
        **_generation_options(context.scene, skin=True),
    )
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "skin failed"))

    def apply_skin(result: dict) -> dict:
        response = core.apply_session_request(
            prepared["remote"],
            result["remote_result"],
        )
        if not response.get("ok"):
            return response
        if not core.prepared_context_is_current(prepared):
            _queue_history_sync()
            return {
                "ok": False,
                "code": "STALE_RESULT",
                "error": (
                    "Armature changed while Skin was running; result discarded"
                ),
            }
        return core._apply_skin_response(prepared, response, result["skin"])

    def succeeded(result_context, response: dict) -> None:
        ensemble = response.get("skin_ensemble") or {}
        ensemble_detail = ""
        if ensemble:
            ensemble_detail = (
                f", selected={ensemble.get('selected_candidate')}"
                f", candidates={ensemble.get('candidate_count')}"
            )
        set_status(
            result_context,
            "Skin complete "
            f"[midprocess={response.get('midprocess')}"
            f"{ensemble_detail}]",
        )

    return submit_async(
        context,
        label="Skin",
        work=lambda: core.execute_skin_request(prepared),
        apply=apply_skin,
        success=succeeded,
        undo_label="SkinTokens Skin",
    )


def _schedule_context_edit(
    context,
    *,
    label: str,
    prepared: dict,
    apply_result: Callable[[dict, dict], dict],
    success: Callable[[object, dict], None],
) -> bool:
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", f"{label} failed"))
    core = get_core(context)
    return submit_async(
        context,
        label=label,
        work=lambda: core.execute_session_request(prepared["remote"]),
        apply=lambda result: _apply_context_result(
            core,
            prepared,
            result,
            apply_result,
        ),
        success=success,
        undo_label=f"SkinTokens {label}",
    )


class SKINTOKENS_OT_test_connection(bpy.types.Operator):
    bl_idname = "skintokens_interactive.test_connection"
    bl_label = "Test Connection"
    bl_description = "Check the model server and protocol version"

    @classmethod
    def poll(cls, _context):
        if not online_access_enabled():
            cls.poll_message_set("Enable Online Access first")
            return False
        return True

    def execute(self, context):
        try:
            preferences = get_addon_preferences(context)
            endpoint = (
                preferences.server_url.strip()
                if preferences is not None
                else context.scene.skintokens_model_socket.strip()
            ) or DEFAULT_MODEL_URL
            timeout = (
                float(preferences.request_timeout)
                if preferences is not None
                else 300.0
            )
            core = BlenderInteractiveCore(
                model_socket=endpoint,
                owner_id=uuid.uuid4().hex,
                request_timeout=timeout,
            )
            response = core.server_status()
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "connection failed"))
            message = (
                f"Connected to SkinTokens server {response.get('server_version', 'unknown')}"
            )
            self.report({"INFO"}, message)
            set_status(context, message)
            return {"FINISHED"}
        except Exception as exc:
            message = f"Connection failed: {exc}"
            self.report({"ERROR"}, message)
            set_status(context, message)
            return {"CANCELLED"}


class SKINTOKENS_OT_enable_online_access(bpy.types.Operator):
    bl_idname = "skintokens_interactive.enable_online_access"
    bl_label = "Allow Online Access"
    bl_description = "Allow extensions such as SkinTokens to connect to network services"

    @classmethod
    def poll(cls, _context):
        if online_access_enabled():
            cls.poll_message_set("Online Access is already enabled")
            return False
        if online_access_forced_off():
            cls.poll_message_set(
                "Blender was launched with --offline-mode and must be restarted"
            )
            return False
        return True

    def execute(self, context):
        context.preferences.system.use_online_access = True
        if not online_access_enabled():
            self.report({"ERROR"}, "Blender did not enable Online Access")
            return {"CANCELLED"}
        set_status(context, "Online Access enabled")
        return {"FINISHED"}


class SKINTOKENS_OT_start(bpy.types.Operator):
    bl_idname = "skintokens_interactive.start"
    bl_label = "Start"
    bl_description = "Create a SkinTokens interactive session for the selected mesh"

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
            obj, armature_obj = selected_start_objects(context)
            obj_path = source_obj_path(context, obj)
            core = get_core(context)
            prepared = core.prepare_start(
                obj_path,
                import_mesh=False,
                mesh_object_name=obj.name,
                armature_object_name=(
                    None if armature_obj is None else armature_obj.name
                ),
                top_k=context.scene.skintokens_top_k,
                top_p=context.scene.skintokens_top_p,
                temperature=context.scene.skintokens_temperature,
                repetition_penalty=context.scene.skintokens_repetition_penalty,
                num_beams=context.scene.skintokens_num_beams,
            )
            if not prepared.get("ok"):
                raise RuntimeError(prepared.get("error", "start failed"))
            obj_name = str(obj.name)

            def start_succeeded(result_context, response: dict) -> None:
                mesh_obj = bpy.data.objects.get(obj_name)
                if mesh_obj is None:
                    raise RuntimeError("source mesh was removed while Start was running")
                result_context.scene.skintokens_blender_session_id = response[
                    "blender_session_id"
                ]
                mesh_obj["skintokens_blender_session_id"] = response[
                    "blender_session_id"
                ]
                mesh_obj["skintokens_model_session_id"] = response[
                    "model_session_id"
                ]
                mesh_obj["skintokens_source_obj"] = str(obj_path)
                if response.get("armature_object_name"):
                    mesh_obj["skintokens_source_armature"] = response[
                        "armature_object_name"
                    ]
                remove_legacy_preview_objects()
                joint_count = len(response.get("context", {}).get("joints", []))
                if response.get("context_source") == "scene-armature":
                    set_status(
                        result_context,
                        f"Started with scene armature: {joint_count} joints",
                    )
                else:
                    set_status(result_context, "Started with empty Armature")
                ensure_armature_watch()

            def apply_start(result: dict) -> dict:
                response = result["response"]
                started = result.get("start_response", {})

                def cleanup_started_session() -> None:
                    if started.get("session_id"):
                        _submit_server_cleanup(
                            core,
                            {
                                "command": "reset",
                                "session_id": str(started["session_id"]),
                                "end_reason": "start_cancelled",
                            },
                        )

                if response.get("ok") and not core.prepared_start_is_current(prepared):
                    cleanup_started_session()
                    return {
                        "ok": False,
                        "code": "STALE_RESULT",
                        "error": (
                            "Armature changed while Start was running; "
                            "result discarded"
                        ),
                    }
                try:
                    return core.apply_start_request(prepared, result)
                except Exception:
                    cleanup_started_session()
                    raise

            submit_async(
                context,
                label="Start",
                work=lambda: core.execute_start_request(prepared),
                apply=apply_start,
                success=start_succeeded,
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Start failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_finish(bpy.types.Operator):
    bl_idname = "skintokens_interactive.finish"
    bl_label = "Finish"
    bl_description = "End the interactive session and keep the generated asset"

    def execute(self, context):
        if async_busy():
            set_status(context, f"{_ASYNC_JOB['label']} is still running")
            return {"CANCELLED"}
        session_id = context.scene.skintokens_blender_session_id
        if not session_id:
            set_status(context, "No interactive session to finish")
            return {"CANCELLED"}

        cancel_history_sync()
        cancel_armature_watch()
        configured_output = context.scene.skintokens_output_path.strip()
        saved_path = None
        save_error = None
        reset_error = None
        session = None if _CORE is None else _CORE.sessions.get(session_id)
        try:
            if configured_output:
                destination = Path(configured_output).expanduser().resolve()
                if session is None:
                    save_error = "interactive session is unavailable"
                else:
                    export_response = _CORE.export_skin(session_id, destination)
                    if not export_response.get("ok"):
                        save_error = export_response.get("error", "TXT export failed")
                    else:
                        saved_path = Path(export_response["output_path"])
        except Exception as exc:
            traceback.print_exc()
            save_error = str(exc)

        try:
            if session is not None:
                reset_payload = _CORE.detach_session(
                    session_id,
                    end_reason="finish",
                )
                _submit_server_cleanup(_CORE, reset_payload)
        except Exception as exc:
            traceback.print_exc()
            reset_error = str(exc)
        finally:
            context.scene.skintokens_blender_session_id = ""
            for obj in bpy.data.objects:
                if obj.get("skintokens_blender_session_id") != session_id:
                    continue
                for key in (
                    "skintokens_blender_session_id",
                    "skintokens_model_session_id",
                ):
                    if key in obj:
                        del obj[key]

        errors = []
        if save_error:
            errors.append(f"TXT save failed: {save_error}")
        if reset_error:
            errors.append(f"server cleanup failed: {reset_error}")
        if errors:
            message = f"Finished locally; {'; '.join(errors)}"
            set_status(context, message)
            self.report({"WARNING"}, message)
        elif saved_path is not None:
            set_status(context, f"Finished and saved TXT: {saved_path}")
        else:
            set_status(context, "Finished interactive session; asset kept")
        return {"FINISHED"}


class SKINTOKENS_OT_next(bpy.types.Operator):
    bl_idname = "skintokens_interactive.next"
    bl_label = "Next"
    bl_description = "Generate the next natural DFS unit, branch, or skeleton EOS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_next(context, force_parent=False)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Next failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_force_next(bpy.types.Operator):
    bl_idname = "skintokens_interactive.force_next"
    bl_label = "Force Next"
    bl_description = "Generate one child using the active Armature bone as parent"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_next(context, force_parent=True)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Force Next failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_import_txt(bpy.types.Operator):
    bl_idname = "skintokens_interactive.import_txt"
    bl_label = "Import TXT"
    bl_description = "Import heter-skinning txt skeleton and skin into the current session"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            if not context.scene.skintokens_import_txt_path.strip():
                raise RuntimeError("Choose a heter-skinning txt first")
            txt_path = Path(context.scene.skintokens_import_txt_path).expanduser()
            response = get_core(context).import_txt(session_id, txt_path)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "import txt failed"))
            count = len(response.get("context", {}).get("joints", []))
            set_status(context, f"Imported TXT joints: {count}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Import TXT failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_rig(bpy.types.Operator):
    bl_idname = "skintokens_interactive.rig"
    bl_label = "Rig"
    bl_description = "Continue the current rig until skeleton EOS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_rig(context)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Rig failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_skin(bpy.types.Operator):
    bl_idname = "skintokens_interactive.skin"
    bl_label = "Skin"
    bl_description = "Generate skin for the current interactive skeleton"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_skin(context)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Skin failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_refresh(bpy.types.Operator):
    bl_idname = "skintokens_interactive.refresh"
    bl_label = "Refresh"
    bl_description = "Sync edited or deleted Armature bones without generating"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            core = get_core(context)
            prepared = core.prepare_refresh(session_id)
            _schedule_context_edit(
                context,
                label="Refresh",
                prepared=prepared,
                apply_result=core.apply_refresh_request,
                success=lambda result_context, response: set_status(
                    result_context,
                    f"Refreshed joints: {len(response.get('context', {}).get('joints', []))}",
                ),
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Refresh failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_vae_apply(bpy.types.Operator):
    bl_idname = "skintokens_interactive.vae_apply"
    bl_label = "Apply VAE Level"
    bl_description = "Commit the current VAE preview as the new editable level 0"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            response = get_core(context).apply_vae_reconstruction_levels(session_id)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "VAE apply failed"))
            applied = response.get("applied_levels", {})
            detail = ", ".join(
                f"{name}:{level}" for name, level in applied.items()
            )
            set_status(
                context,
                f"Applied VAE preview ({detail}); levels reset to 0",
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"VAE apply failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_split(bpy.types.Operator):
    bl_idname = "skintokens_interactive.split"
    bl_label = "Split"
    bl_description = "Insert a midpoint joint into the selected bone"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            core = get_core(context)
            prepared = core.prepare_split(session_id)
            _schedule_context_edit(
                context,
                label="Split",
                prepared=prepared,
                apply_result=core.apply_split_request,
                success=lambda result_context, response: set_status(
                    result_context,
                    "Split bone. Joints: "
                    f"{len(response.get('context', {}).get('joints', []))}",
                ),
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Split failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_delete(bpy.types.Operator):
    bl_idname = "skintokens_interactive.delete"
    bl_label = "Delete"
    bl_description = "Delete a leaf or dissolve the active bone's downstream joint"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"{_ASYNC_JOB['label']} is still running")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            core = get_core(context)
            prepared = core.prepare_delete(session_id)

            def succeeded(result_context, response: dict) -> None:
                count = len(response.get("context", {}).get("joints", []))
                set_status(
                    result_context,
                    f"Deleted {response.get('deleted_bone_name')} "
                    f"[{response.get('delete_operation')}]. Joints: {count}",
                )

            _schedule_context_edit(
                context,
                label="Delete",
                prepared=prepared,
                apply_result=core.apply_delete_request,
                success=succeeded,
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Delete failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_PT_interactive(bpy.types.Panel):
    bl_label = "SkinTokens Interactive"
    bl_idname = "SKINTOKENS_PT_interactive"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SkinTokens"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        url_row = layout.row()
        url_row.enabled = not async_busy() and not has_session_marker(context)
        preferences = get_addon_preferences(context)
        if preferences is None:
            url_row.prop(scene, "skintokens_model_socket")
        else:
            url_row.prop(preferences, "server_url", text="Server URL")
        connection_button = url_row.row(align=True)
        connection_button.enabled = online_access_enabled()
        connection_button.operator(
            "skintokens_interactive.test_connection",
            text="",
            icon="URL",
        )
        if not online_access_enabled():
            access_box = layout.box()
            access_box.alert = True
            if online_access_forced_off():
                access_box.label(
                    text="Restart Blender without --offline-mode",
                    icon="ERROR",
                )
            else:
                access_box.label(text="SkinTokens requires network access", icon="INFO")
                access_box.operator(
                    "skintokens_interactive.enable_online_access",
                    icon="CHECKMARK",
                )
        generate_row = layout.row(align=True)
        generate_row.enabled = has_started_session(context) and not async_busy()
        generate_row.operator("skintokens_interactive.rig", icon="OUTLINER_OB_ARMATURE")
        generate_row.operator("skintokens_interactive.skin", icon="MOD_ARMATURE")
        commands = layout.column()
        commands.enabled = has_started_session(context) and not async_busy()
        next_row = commands.row(align=True)
        next_row.operator("skintokens_interactive.next", icon="TRACKING_FORWARDS")
        next_row.operator("skintokens_interactive.force_next", icon="CON_CHILDOF")
        edit_row = commands.row(align=True)
        edit_row.operator("skintokens_interactive.split", icon="ADD")
        edit_row.operator("skintokens_interactive.delete", icon="REMOVE")
        vae_row = commands.split(factor=0.86, align=True)
        slider_row = vae_row.row(align=True)
        slider_row.enabled = selected_vae_bone_name(context) is not None
        slider_row.prop(
            scene,
            "skintokens_vae_reconstruction_level",
            text="VAE Level",
            slider=True,
        )
        apply_row = vae_row.row(align=True)
        apply_row.enabled = has_vae_preview(context)
        apply_row.operator(
            "skintokens_interactive.vae_apply",
            text="",
            icon="CHECKMARK",
        )
        layout.separator()
        layout.label(text=scene.skintokens_status)
        state_row = layout.row()
        state_row.enabled = not async_busy() and online_access_enabled()
        if has_session_marker(context):
            state_row.operator("skintokens_interactive.finish", icon="X")
        else:
            state_row.operator("skintokens_interactive.start", icon="PLAY")


classes = (
    SKINTOKENS_Preferences,
    SKINTOKENS_OT_test_connection,
    SKINTOKENS_OT_enable_online_access,
    SKINTOKENS_OT_start,
    SKINTOKENS_OT_finish,
    SKINTOKENS_OT_next,
    SKINTOKENS_OT_force_next,
    SKINTOKENS_OT_import_txt,
    SKINTOKENS_OT_rig,
    SKINTOKENS_OT_skin,
    SKINTOKENS_OT_refresh,
    SKINTOKENS_OT_vae_apply,
    SKINTOKENS_OT_split,
    SKINTOKENS_OT_delete,
    SKINTOKENS_PT_interactive,
)


def register():
    _debug_log("addon.register", extension_version=EXTENSION_VERSION)
    if hasattr(bpy.types.Scene, "skintokens_branch_parent"):
        delattr(bpy.types.Scene, "skintokens_branch_parent")
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.skintokens_model_socket = bpy.props.StringProperty(
        name="Server URL",
        default=DEFAULT_MODEL_URL,
    )
    bpy.types.Scene.skintokens_owner_id = bpy.props.StringProperty(default="", options={"HIDDEN"})
    bpy.types.Scene.skintokens_output_path = bpy.props.StringProperty(
        name="Output",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.skintokens_import_txt_path = bpy.props.StringProperty(
        name="Import TXT",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.skintokens_blender_session_id = bpy.props.StringProperty(default="")
    bpy.types.Scene.skintokens_status = bpy.props.StringProperty(default="Idle")
    bpy.types.Scene.skintokens_vae_reconstruction_level = bpy.props.IntProperty(
        name="VAE Level",
        description=(
            "Choose cached VAE reconstruction level 0-3 for the active skin bone"
        ),
        min=0,
        max=DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL,
        get=get_vae_reconstruction_level,
        set=set_vae_reconstruction_level,
    )
    bpy.types.Scene.skintokens_max_new_tokens = bpy.props.IntProperty(name="Next Tokens", default=16, min=1, max=64)
    bpy.types.Scene.skintokens_skin_max_new_tokens = bpy.props.IntProperty(name="Skin Tokens", default=2048, min=8, max=8192)
    bpy.types.Scene.skintokens_top_k = bpy.props.IntProperty(name="Top K", default=5, min=0, max=100)
    bpy.types.Scene.skintokens_top_p = bpy.props.FloatProperty(name="Top P", default=0.95, min=0.0, max=1.0)
    bpy.types.Scene.skintokens_temperature = bpy.props.FloatProperty(name="Temperature", default=1.5, min=0.01, max=5.0)
    bpy.types.Scene.skintokens_repetition_penalty = bpy.props.FloatProperty(name="Repetition Penalty", default=1.2, min=0.1, max=5.0)
    bpy.types.Scene.skintokens_num_beams = bpy.props.IntProperty(name="Beams", default=1, min=1, max=16)
    if sync_model_after_history_change not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(sync_model_after_history_change)
    if sync_model_after_history_change not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(sync_model_after_history_change)
    if sync_model_after_armature_change not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            sync_model_after_armature_change
        )


def unregister():
    global _CORE
    if sync_model_after_history_change in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(sync_model_after_history_change)
    if sync_model_after_history_change in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(sync_model_after_history_change)
    if sync_model_after_armature_change in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(
            sync_model_after_armature_change
        )
    cancel_history_sync()
    cancel_armature_watch()
    cancel_async_job()
    core = _CORE
    _CORE = None
    if core is not None:
        for session_id in list(core.sessions):
            try:
                _submit_server_cleanup(
                    core,
                    core.detach_session(
                        session_id,
                        end_reason="addon_unload",
                    ),
                )
            except Exception:
                traceback.print_exc()
    for scene in bpy.data.scenes:
        if hasattr(scene, "skintokens_blender_session_id"):
            scene.skintokens_blender_session_id = ""
    for obj in bpy.data.objects:
        for key in (
            "skintokens_blender_session_id",
            "skintokens_model_session_id",
        ):
            if key in obj:
                del obj[key]
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    for name in (
        "skintokens_model_socket",
        "skintokens_owner_id",
        "skintokens_output_path",
        "skintokens_import_txt_path",
        "skintokens_blender_session_id",
        "skintokens_branch_parent",
        "skintokens_status",
        "skintokens_vae_reconstruction_level",
        "skintokens_max_new_tokens",
        "skintokens_skin_max_new_tokens",
        "skintokens_top_k",
        "skintokens_top_p",
        "skintokens_temperature",
        "skintokens_repetition_penalty",
        "skintokens_num_beams",
    ):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)


if __name__ == "__main__":
    register()
