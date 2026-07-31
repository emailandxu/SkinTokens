from __future__ import annotations

import json
import os
import re
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
    promote_active_armature_bone_selection,
    selected_skin_bone_names,
)
from .core import (
    DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL,
    BlenderInteractiveCore,
)
from .rig_postprocess import (
    MIRROR_AXIS_X,
    MIRROR_AXIS_Y,
    MIRROR_AXIS_Z,
    MIRROR_DISTANCE_AVERAGE,
    MIRROR_DISTANCE_FARTHEST,
    MIRROR_DISTANCE_NEAREST,
    POSTPROCESS_READY_PROP,
    RigPostprocessError,
    mirror_align_bones,
    rename_body_regions,
    selected_edit_bone_names,
)
from .transport import (
    DEFAULT_MODEL_URL,
    EXTENSION_PACKAGE_ID,
    EXTENSION_VERSION,
    request as transport_request,
)


_CORE: BlenderInteractiveCore | None = None
_HISTORY_SYNCING = False
_HISTORY_SYNC_PENDING = False
_HISTORY_TIMER_REGISTERED = False
_HISTORY_SYNC_DEADLINE = 0.0
_VAE_LEVEL_SETTING = False
_LEGACY_PREVIEW_COLLECTION = "SkinTokens Interactive Preview"
_ASYNC_JOB: dict | None = None
_ASYNC_POLL_REGISTERED = False
_DEBUG_LOG_PATH = Path(tempfile.gettempdir()) / "skintokens_interactive_debug.log"
_DEBUG_LOG_LOCK = threading.Lock()
_DEBUG_LOG_MAX_BYTES = 2 * 1024 * 1024
_EXTENSION_PACKAGE_ID = EXTENSION_PACKAGE_ID
_SEMVER_PATTERN = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)
_UPDATE_INDEX_CACHE: dict | None = None
_UPDATE_SYNC_START_TIMER_REGISTERED = False
_UPDATE_SYNC_POLL_TIMER_REGISTERED = False
_EXTENSION_EVENT_JOB: dict | None = None
_EXTENSION_EVENT_START_TIMER_REGISTERED = False
_EXTENSION_EVENT_POLL_TIMER_REGISTERED = False


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
        name="服务器地址",
        description="SkinTokens 模型服务地址",
        default=DEFAULT_MODEL_URL,
    )
    request_timeout: bpy.props.FloatProperty(
        name="请求超时",
        description="单次模型请求的最长等待时间",
        default=300.0,
        min=10.0,
        max=3600.0,
        subtype="TIME",
    )
    installation_id: bpy.props.StringProperty(default="", options={"HIDDEN"})
    reported_extension_version: bpy.props.StringProperty(
        default="",
        options={"HIDDEN"},
    )

    def draw(self, _context) -> None:
        layout = self.layout
        server_row = layout.row()
        server_row.enabled = not async_busy()
        server_row.prop(self, "server_url")
        layout.prop(self, "request_timeout")
        layout.operator("skintokens_interactive.test_connection", icon="URL")
        layout.label(text=f"插件版本 {EXTENSION_VERSION}")


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
        raise RuntimeError("请先结束当前 SkinTokens 会话，再修改服务器设置")
    if _CORE is None or changed:
        _CORE = BlenderInteractiveCore(
            model_socket=endpoint,
            owner_id=owner_id,
            request_timeout=request_timeout,
        )
    return _CORE


def set_status(context, message: str) -> None:
    context.scene.skintokens_status = message
    print(f"[H3D Skintokens] {message}")


def online_access_enabled() -> bool:
    return bool(getattr(bpy.app, "online_access", True))


def online_access_forced_off() -> bool:
    return not online_access_enabled() and bool(
        getattr(bpy.app, "online_access_override", False)
    )


def _extension_repository(preferences):
    package_parts = __package__.split(".")
    if (
        len(package_parts) != 3
        or package_parts[0] != "bl_ext"
        or package_parts[2] != _EXTENSION_PACKAGE_ID
    ):
        return None
    repo_module = package_parts[1]
    for repo_index, repo in enumerate(preferences.extensions.repos):
        if repo.module == repo_module:
            return repo_index, repo
    return None


def _parse_semver(value: str):
    match = _SEMVER_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    prerelease = match.group(4)
    prerelease_parts = None
    if prerelease is not None:
        prerelease_parts = tuple(
            (0, int(part)) if part.isdigit() else (1, part)
            for part in prerelease.split(".")
        )
    return (
        (int(match.group(1)), int(match.group(2)), int(match.group(3))),
        prerelease_parts,
    )


def _version_is_newer(remote_version: str, installed_version: str) -> bool:
    remote = _parse_semver(remote_version)
    installed = _parse_semver(installed_version)
    if remote is None or installed is None:
        return False
    if remote[0] != installed[0]:
        return remote[0] > installed[0]
    remote_prerelease = remote[1]
    installed_prerelease = installed[1]
    if remote_prerelease is None:
        return installed_prerelease is not None
    if installed_prerelease is None:
        return False
    return remote_prerelease > installed_prerelease


def _extension_update_info(preferences):
    global _UPDATE_INDEX_CACHE
    repository = _extension_repository(preferences)
    if repository is None:
        return None
    repo_index, repo = repository
    if not repo.use_remote_url or not repo.remote_url:
        return None

    index_path = Path(repo.directory) / ".blender_ext" / "index.json"
    try:
        stat = index_path.stat()
    except OSError:
        return None
    cache_key = (
        str(index_path),
        stat.st_mtime_ns,
        stat.st_size,
        EXTENSION_VERSION,
    )
    if _UPDATE_INDEX_CACHE is not None and _UPDATE_INDEX_CACHE["key"] == cache_key:
        return _UPDATE_INDEX_CACHE["value"]

    update = None
    try:
        repository_index = json.loads(index_path.read_text(encoding="utf-8"))
        packages = (
            repository_index.get("data", [])
            if isinstance(repository_index, dict)
            else []
        )
        if not isinstance(packages, list):
            packages = []
        for package in packages:
            if not isinstance(package, dict):
                continue
            if package.get("id") != _EXTENSION_PACKAGE_ID:
                continue
            remote_version = str(package.get("version", ""))
            if _version_is_newer(remote_version, EXTENSION_VERSION):
                update = {
                    "repo_index": repo_index,
                    "version": remote_version,
                }
            break
    except (OSError, ValueError, TypeError):
        update = None
    _UPDATE_INDEX_CACHE = {"key": cache_key, "value": update}
    return update


def _finish_extension_repository_sync() -> None:
    global _UPDATE_INDEX_CACHE, _UPDATE_SYNC_POLL_TIMER_REGISTERED
    _UPDATE_INDEX_CACHE = None
    _UPDATE_SYNC_POLL_TIMER_REGISTERED = False
    _redraw_sidebar()


def _extension_repository_sync_running() -> bool:
    modal_operators = getattr(
        bpy.context.window_manager,
        "modal_operators",
        None,
    )
    return bool(
        modal_operators is not None
        and modal_operators.get("EXTENSIONS_OT_repo_sync") is not None
    )


def _poll_extension_repository_sync() -> float | None:
    if _extension_repository_sync_running():
        return 0.1
    _finish_extension_repository_sync()
    return None


def _start_extension_repository_sync() -> float | None:
    global _UPDATE_SYNC_START_TIMER_REGISTERED
    global _UPDATE_SYNC_POLL_TIMER_REGISTERED
    if _extension_repository_sync_running():
        return 0.25

    _UPDATE_SYNC_START_TIMER_REGISTERED = False
    repository = _extension_repository(bpy.context.preferences)
    if repository is None or not online_access_enabled():
        return None
    repo_index, repo = repository
    if not repo.use_remote_url or not repo.remote_url:
        return None

    try:
        result = bpy.ops.extensions.repo_sync(
            "INVOKE_DEFAULT",
            repo_index=repo_index,
        )
    except Exception as exc:
        _debug_log("extension.update_sync_failed", error=str(exc))
        _finish_extension_repository_sync()
        return None

    _debug_log(
        "extension.update_sync_started",
        repo_module=str(repo.module),
        result=sorted(result),
    )
    if "RUNNING_MODAL" not in result:
        _finish_extension_repository_sync()
        return None
    if not _UPDATE_SYNC_POLL_TIMER_REGISTERED:
        _UPDATE_SYNC_POLL_TIMER_REGISTERED = True
        bpy.app.timers.register(
            _poll_extension_repository_sync,
            first_interval=0.1,
        )
    return None


def _schedule_extension_repository_sync(delay: float = 2.0) -> None:
    global _UPDATE_SYNC_START_TIMER_REGISTERED
    if _UPDATE_SYNC_START_TIMER_REGISTERED:
        return
    repository = _extension_repository(bpy.context.preferences)
    if repository is None:
        return
    _repo_index, repo = repository
    if not repo.use_remote_url or not repo.remote_url:
        return
    repo.use_sync_on_startup = False
    if not online_access_enabled():
        return
    _UPDATE_SYNC_START_TIMER_REGISTERED = True
    bpy.app.timers.register(
        _start_extension_repository_sync,
        first_interval=max(0.0, float(delay)),
    )


def _cancel_extension_repository_sync() -> None:
    global _UPDATE_SYNC_START_TIMER_REGISTERED, _UPDATE_SYNC_POLL_TIMER_REGISTERED
    for callback in (
        _start_extension_repository_sync,
        _poll_extension_repository_sync,
    ):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    _UPDATE_SYNC_START_TIMER_REGISTERED = False
    _UPDATE_SYNC_POLL_TIMER_REGISTERED = False


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


def _extension_installation_source(preferences) -> str:
    repository = _extension_repository(preferences)
    if repository is None:
        return "disk"
    _repo_index, repo = repository
    return "repository" if repo.use_remote_url else "disk"


def _poll_extension_event_report() -> float | None:
    global _EXTENSION_EVENT_JOB, _EXTENSION_EVENT_POLL_TIMER_REGISTERED
    job = _EXTENSION_EVENT_JOB
    if job is None:
        _EXTENSION_EVENT_POLL_TIMER_REGISTERED = False
        return None
    future = job["future"]
    if not future.done():
        return 0.1
    try:
        result = future.result()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "扩展事件上报失败"))
        preferences = get_addon_preferences(bpy.context)
        if preferences is not None:
            preferences.reported_extension_version = str(job["version"])
        _debug_log(
            "extension.event_reported",
            action=str(job["action"]),
            version=str(job["version"]),
        )
    except Exception as exc:
        _debug_log("extension.event_report_failed", error=str(exc))
    finally:
        _EXTENSION_EVENT_JOB = None
        _EXTENSION_EVENT_POLL_TIMER_REGISTERED = False
    return None


def _start_extension_event_report() -> float | None:
    global _EXTENSION_EVENT_JOB, _EXTENSION_EVENT_START_TIMER_REGISTERED
    global _EXTENSION_EVENT_POLL_TIMER_REGISTERED
    _EXTENSION_EVENT_START_TIMER_REGISTERED = False
    if _EXTENSION_EVENT_JOB is not None or not online_access_enabled():
        return None
    preferences = get_addon_preferences(bpy.context)
    if preferences is None:
        return None
    previous_version = str(preferences.reported_extension_version).strip()
    if previous_version == EXTENSION_VERSION:
        return None
    installation_id = str(preferences.installation_id).strip()
    if not installation_id:
        installation_id = uuid.uuid4().hex
        preferences.installation_id = installation_id
    action = "install" if not previous_version else "update"
    endpoint = preferences.server_url.strip() or DEFAULT_MODEL_URL
    owner_id = installation_id
    payload = {
        "command": "extension_event",
        "owner_id": owner_id,
        "action": action,
        "package_id": _EXTENSION_PACKAGE_ID,
        "extension_version": EXTENSION_VERSION,
        "previous_version": previous_version,
        "blender_version": ".".join(str(value) for value in bpy.app.version),
        "installation_id": installation_id,
        "installation_source": _extension_installation_source(
            bpy.context.preferences
        ),
    }
    _EXTENSION_EVENT_JOB = {
        "action": action,
        "version": EXTENSION_VERSION,
        "future": _background_future(
            lambda: transport_request(
                endpoint,
                payload,
                timeout=min(15.0, float(preferences.request_timeout)),
            ),
            name="h3d-skintokens-extension-event",
        ),
    }
    if not _EXTENSION_EVENT_POLL_TIMER_REGISTERED:
        _EXTENSION_EVENT_POLL_TIMER_REGISTERED = True
        bpy.app.timers.register(
            _poll_extension_event_report,
            first_interval=0.1,
        )
    return None


def _schedule_extension_event_report(delay: float = 1.0) -> None:
    global _EXTENSION_EVENT_START_TIMER_REGISTERED
    if _EXTENSION_EVENT_START_TIMER_REGISTERED or _EXTENSION_EVENT_JOB is not None:
        return
    _EXTENSION_EVENT_START_TIMER_REGISTERED = True
    bpy.app.timers.register(
        _start_extension_event_report,
        first_interval=max(0.0, float(delay)),
    )


def _cancel_extension_event_report() -> None:
    global _EXTENSION_EVENT_JOB, _EXTENSION_EVENT_START_TIMER_REGISTERED
    global _EXTENSION_EVENT_POLL_TIMER_REGISTERED
    for callback in (
        _start_extension_event_report,
        _poll_extension_event_report,
    ):
        if bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    _EXTENSION_EVENT_JOB = None
    _EXTENSION_EVENT_START_TIMER_REGISTERED = False
    _EXTENSION_EVENT_POLL_TIMER_REGISTERED = False


def _poll_async_job() -> float | None:
    global _ASYNC_JOB, _ASYNC_POLL_REGISTERED
    job = _ASYNC_JOB
    if job is None:
        _ASYNC_POLL_REGISTERED = False
        return None
    future = job["future"]
    if not future.done():
        return 0.05
    if job.get("abort_on_transform") and _transform_modal_running():
        _debug_log(
            "async.deferred_transform",
            label=str(job.get("label", "")),
        )
        _ASYNC_JOB = None
        _ASYNC_POLL_REGISTERED = False
        _queue_history_sync(0.1)
        _redraw_sidebar()
        return None

    context = bpy.context
    try:
        background_result = future.result()
        response = job["apply"](background_result)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "请求失败"))
        job["success"](context, response)
        undo_label = job.get("undo_label")
        if undo_label:
            try:
                bpy.ops.ed.undo_push(message=str(undo_label))
            except Exception:
                traceback.print_exc()
    except Exception as exc:
        traceback.print_exc()
        set_status(context, f"{job['label']}失败：{exc}")
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
    abort_on_transform: bool = False,
) -> bool:
    global _ASYNC_JOB, _ASYNC_POLL_REGISTERED
    if async_busy():
        set_status(context, f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
        return False
    _ASYNC_JOB = {
        "label": label,
        "future": _background_future(work, name="skintokens-http"),
        "apply": apply,
        "success": success,
        "undo_label": undo_label,
        "abort_on_transform": bool(abort_on_transform),
    }
    set_status(context, f"{label}中...")
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
            "error": "请求执行期间骨架发生变化，结果已丢弃",
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


def has_selected_vae_cache(context) -> bool:
    if _CORE is None:
        return False
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    session = _CORE.sessions.get(session_id)
    if session is None:
        return False
    bone_name = selected_vae_bone_name(context)
    return bool(
        bone_name is not None
        and bone_name in session.vae_weight_fields
    )


def has_started_session(context) -> bool:
    if _CORE is None:
        return False
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    return bool(session_id and session_id in _CORE.sessions)


def session_armature(context):
    if _CORE is None:
        return None
    session_id = getattr(context.scene, "skintokens_blender_session_id", "")
    session = _CORE.sessions.get(session_id)
    if session is None or not session.armature_object_name:
        return None
    armature = bpy.data.objects.get(session.armature_object_name)
    if armature is None or armature.type != "ARMATURE":
        return None
    return armature


def selected_session_edit_bone_names(context) -> tuple[str, ...]:
    armature = session_armature(context)
    if armature is None:
        return ()
    return selected_edit_bone_names(context, armature)


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
            f"已生成 0-{response.get('max_level', 3)} 级，"
            f"耗时 {float(report.get('wall_sec', 0.0)):.2f} 秒"
        )
    else:
        detail = "使用缓存"
    reset = (
        "，权重编辑后缓存已重置"
        if response.get("cache_invalidated")
        else ""
    )
    set_status(
        context,
        f"骨骼 {response.get('bone_name', bone_name)} 的权重场递归重建级别 "
        f"{response.get('level', 0)}：{detail}{reset}",
    )


def set_vae_reconstruction_level(_scene, value: int) -> None:
    global _VAE_LEVEL_SETTING
    if _VAE_LEVEL_SETTING:
        return
    _VAE_LEVEL_SETTING = True
    context = bpy.context
    try:
        if async_busy():
            raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
        session_id = context.scene.skintokens_blender_session_id
        if not session_id:
            raise RuntimeError("请先开始会话")
        bone_name = selected_vae_bone_name(context)
        if bone_name is None:
            raise RuntimeError(
                "请激活一个蒙皮顶点组，或只选择一根骨骼"
            )
        response = get_core(context).set_cached_vae_reconstruction_level(
            session_id,
            int(value),
        )
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "缓存级别切换失败"))
        _report_vae_result(context, response, bone_name)
    except Exception as exc:
        traceback.print_exc()
        set_status(context, f"权重场递归重建切换失败：{exc}")
    finally:
        _VAE_LEVEL_SETTING = False


def _schedule_vae_cache_generation(context) -> bool:
    if async_busy():
        raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("请先开始会话")
    bone_name = selected_vae_bone_name(context)
    if bone_name is None:
        raise RuntimeError("请激活一个蒙皮顶点组，或只选择一根骨骼")
    if has_selected_vae_cache(context):
        raise RuntimeError("当前骨骼已经生成重建缓存")
    core = get_core(context)
    prepared = core.prepare_vae_reconstruction_level(session_id, 1)
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "VAE 重建失败"))

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
                "error": "VAE 重建期间骨架或权重发生变化，结果已丢弃",
            }
        return core._apply_vae_level(
            prepared,
            response=response,
            trajectory=result["trajectory"],
        )

    return submit_async(
        context,
        label="权重场递归重建",
        work=lambda: core.execute_vae_request(prepared),
        apply=apply_vae,
        success=lambda result_context, response: _report_vae_result(
            result_context,
            response,
            bone_name,
        ),
    )


def selected_start_objects(context):
    active = context.object
    selected = list(context.selected_objects)
    if active is None:
        raise RuntimeError("请选择一个网格，或同时选择网格和骨架")

    selected_meshes = [obj for obj in selected if obj.type == "MESH"]
    selected_armatures = [obj for obj in selected if obj.type == "ARMATURE"]
    if active.type == "MESH":
        mesh_obj = active
        if len(selected_armatures) > 1:
            raise RuntimeError("开始时最多只能选择一个骨架")
        armature_obj = selected_armatures[0] if selected_armatures else None
        return mesh_obj, armature_obj

    if active.type == "ARMATURE":
        if len(selected_meshes) != 1:
            raise RuntimeError(
                "请为当前骨架同时选择一个目标网格"
            )
        return selected_meshes[0], active

    raise RuntimeError("开始会话需要一个网格，也可以同时选择一个骨架")


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


def _clear_stale_sessions_before_start(
    context,
    core: BlenderInteractiveCore | None,
) -> None:
    stale_ids = set() if core is None else set(core.sessions)
    for obj in bpy.data.objects:
        marker = obj.get("skintokens_blender_session_id")
        if marker:
            stale_ids.add(str(marker))
    if not stale_ids:
        return
    cancel_history_sync()
    finished_armatures = set()
    if core is not None:
        for session_id in stale_ids:
            session = core.sessions.get(session_id)
            if session is not None:
                finished_armatures.add(str(session.armature_object_name))
    for obj in bpy.data.objects:
        if obj.get("skintokens_blender_session_id") not in stale_ids:
            continue
        source_armature = str(obj.get("skintokens_source_armature", ""))
        if source_armature:
            finished_armatures.add(source_armature)
    for session_id in sorted(stale_ids):
        if core is not None and session_id in core.sessions:
            try:
                payload = core.detach_session(
                    session_id,
                    end_reason="start_replaced",
                )
                _submit_server_cleanup(core, payload)
            except Exception:
                traceback.print_exc()
    for armature_name in finished_armatures:
        armature = bpy.data.objects.get(armature_name)
        if armature is not None and armature.type == "ARMATURE":
            armature[POSTPROCESS_READY_PROP] = True
    context.scene.skintokens_blender_session_id = ""
    for obj in bpy.data.objects:
        if obj.get("skintokens_blender_session_id") not in stale_ids:
            continue
        for key in (
            "skintokens_blender_session_id",
            "skintokens_model_session_id",
        ):
            if key in obj:
                del obj[key]


def _restart_core_for_start(context) -> BlenderInteractiveCore:
    _clear_stale_sessions_before_start(context, _CORE)
    return get_core(context)


def _flush_history_sync() -> float | None:
    global _HISTORY_SYNCING, _HISTORY_SYNC_PENDING, _HISTORY_TIMER_REGISTERED
    remaining = _HISTORY_SYNC_DEADLINE - time.monotonic()
    if remaining > 0.0:
        return remaining
    if async_busy():
        _debug_log("history.flush_waiting", busy_label=_ASYNC_JOB["label"])
        return 0.1
    if not _HISTORY_SYNC_PENDING:
        _HISTORY_TIMER_REGISTERED = False
        return None
    if _transform_modal_running():
        _debug_log("history.deferred_transform")
        return 0.1
    _HISTORY_TIMER_REGISTERED = False
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
                f"骨架同步失败：{prepared.get('error')}",
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
                set_status(result_context, "骨架再次发生变化，等待重新同步")
                return
            count = len(response.get("context", {}).get("joints", []))
            _debug_log(
                "history.succeeded",
                session_id=session_id,
                joint_count=count,
                joints=response.get("context", {}).get("joints", []),
            )
            set_status(result_context, f"骨架已自动同步：{count} 根骨骼")

        submit_async(
            bpy.context,
            label="骨架同步",
            work=lambda: core.execute_session_request(prepared["remote"]),
            apply=apply_sync,
            success=succeeded,
            abort_on_transform=True,
        )
    except Exception as exc:
        _debug_log("history.failed", session_id=session_id, error=str(exc))
        traceback.print_exc()
        set_status(bpy.context, f"骨架同步失败：{exc}")
    finally:
        _HISTORY_SYNCING = False
    return None


def _transform_modal_running() -> bool:
    modal_operators = getattr(
        bpy.context.window_manager,
        "modal_operators",
        None,
    )
    if modal_operators is None:
        return False
    try:
        items = list(modal_operators)
    except TypeError:
        items = []
    for operator in items:
        bl_idname = str(
            getattr(
                getattr(operator, "bl_rna", None),
                "identifier",
                "",
            )
        )
        if bl_idname.startswith("TRANSFORM_OT_"):
            return True
    getter = getattr(modal_operators, "get", None)
    return bool(
        callable(getter)
        and any(
            getter(identifier) is not None
            for identifier in (
                "TRANSFORM_OT_translate",
                "TRANSFORM_OT_rotate",
                "TRANSFORM_OT_resize",
                "TRANSFORM_OT_transform",
            )
        )
    )


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
                    "[H3D Skintokens] 服务端清理失败："
                    f"{response.get('error', '重置失败')}"
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
        raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("请先开始会话")
    core = get_core(context)
    prepared = core.prepare_next(
        session_id,
        force_parent=force_parent,
        **_generation_options(context.scene),
    )
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "衍生骨骼失败"))

    def succeeded(result_context, response: dict) -> None:
        count = len(response.get("context", {}).get("joints", []))
        done = bool(response.get("context", {}).get("done", False))
        prefix = "已强制衍生子骨骼" if force_parent else "已衍生骨骼"
        set_status(result_context, f"{prefix}：共 {count} 根骨骼{'，已完成' if done else ''}")

    return submit_async(
        context,
        label="强制衍生子骨骼" if force_parent else "衍生骨骼",
        work=lambda: core.execute_session_request(prepared["remote"]),
        apply=lambda result: _apply_context_result(
            core,
            prepared,
            result,
            core.apply_next_request,
        ),
        success=succeeded,
        undo_label="SkinTokens 强制衍生子骨骼" if force_parent else "SkinTokens 衍生骨骼",
    )


def _schedule_rig(context) -> bool:
    if async_busy():
        raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("请先开始会话")
    core = get_core(context)
    options = _generation_options(context.scene)
    options.pop("max_new_tokens", None)
    prepared = core.prepare_rig(session_id, **options)
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "生成骨骼树失败"))

    def succeeded(result_context, response: dict) -> None:
        result_skeleton = response.get("context", {})
        count = len(result_skeleton.get("joints", []))
        done = bool(result_skeleton.get("done", False))
        set_status(
            result_context,
            f"骨架生成完成：共 {count} 根骨骼{'，已结束' if done else ''}",
        )

    return submit_async(
        context,
        label="生成骨骼树",
        work=lambda: core.execute_session_request(prepared["remote"]),
        apply=lambda result: _apply_context_result(
            core,
            prepared,
            result,
            core.apply_rig_request,
        ),
        success=succeeded,
        undo_label="SkinTokens 生成骨骼树",
    )


def _schedule_skin(context) -> bool:
    if async_busy():
        raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        raise RuntimeError("请先开始会话")
    core = get_core(context)
    prepared = core.prepare_skin(
        session_id,
        **_generation_options(context.scene, skin=True),
    )
    if not prepared.get("ok"):
        raise RuntimeError(prepared.get("error", "生成蒙皮失败"))

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
                    "蒙皮生成期间骨架发生变化，结果已丢弃"
                ),
            }
        return core._apply_skin_response(prepared, response, result["skin"])

    def succeeded(result_context, response: dict) -> None:
        ensemble = response.get("skin_ensemble") or {}
        if ensemble:
            message = (
                f"蒙皮生成完成：已从 {ensemble.get('candidate_count')} 个候选中"
                f"选择第 {ensemble.get('selected_candidate')} 个"
            )
        else:
            message = "蒙皮生成完成"
        set_status(result_context, message)

    return submit_async(
        context,
        label="生成蒙皮",
        work=lambda: core.execute_skin_request(prepared),
        apply=apply_skin,
        success=succeeded,
        undo_label="SkinTokens 生成蒙皮",
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
        raise RuntimeError(prepared.get("error", f"{label}失败"))
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
    bl_label = "测试连接"
    bl_description = "检查模型服务器及协议版本"

    @classmethod
    def poll(cls, _context):
        if not online_access_enabled():
            cls.poll_message_set("请先允许 Blender 联网")
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
                raise RuntimeError(response.get("error", "连接失败"))
            message = (
                f"已连接 SkinTokens 服务器，版本 "
                f"{response.get('server_version', '未知')}"
            )
            self.report({"INFO"}, message)
            set_status(context, message)
            return {"FINISHED"}
        except Exception as exc:
            message = f"连接失败：{exc}"
            self.report({"ERROR"}, message)
            set_status(context, message)
            return {"CANCELLED"}


class SKINTOKENS_OT_enable_online_access(bpy.types.Operator):
    bl_idname = "skintokens_interactive.enable_online_access"
    bl_label = "允许联网"
    bl_description = "允许 SkinTokens 等扩展连接网络服务"

    @classmethod
    def poll(cls, _context):
        if online_access_enabled():
            cls.poll_message_set("Blender 已允许联网")
            return False
        if online_access_forced_off():
            cls.poll_message_set(
                "Blender 以 --offline-mode 启动，需要重启"
            )
            return False
        return True

    def execute(self, context):
        context.preferences.system.use_online_access = True
        if not online_access_enabled():
            self.report({"ERROR"}, "无法启用 Blender 联网权限")
            return {"CANCELLED"}
        set_status(context, "已允许 Blender 联网")
        _schedule_extension_repository_sync(delay=0.0)
        _schedule_extension_event_report(delay=0.0)
        return {"FINISHED"}


class SKINTOKENS_OT_start(bpy.types.Operator):
    bl_idname = "skintokens_interactive.start"
    bl_label = "开始"
    bl_description = "为选中的网格创建 SkinTokens 交互会话"

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            obj, armature_obj = selected_start_objects(context)
            obj_path = source_obj_path(context, obj)
            core = _restart_core_for_start(context)
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
                raise RuntimeError(prepared.get("error", "开始会话失败"))
            obj_name = str(obj.name)

            def start_succeeded(result_context, response: dict) -> None:
                mesh_obj = bpy.data.objects.get(obj_name)
                if mesh_obj is None:
                    raise RuntimeError("开始会话期间源网格已被删除")
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
                    armature_name = response["armature_object_name"]
                    mesh_obj["skintokens_source_armature"] = armature_name
                    armature = bpy.data.objects.get(armature_name)
                    if armature is not None and armature.type == "ARMATURE":
                        armature[POSTPROCESS_READY_PROP] = False
                remove_legacy_preview_objects()
                joint_count = len(response.get("context", {}).get("joints", []))
                if response.get("context_source") == "scene-armature":
                    set_status(
                        result_context,
                        f"会话已开始，已载入场景骨架：{joint_count} 根骨骼",
                    )
                else:
                    set_status(result_context, "会话已开始，已创建空骨架")

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
                            "开始会话期间骨架发生变化，结果已丢弃"
                        ),
                    }
                try:
                    return core.apply_start_request(prepared, result)
                except Exception:
                    cleanup_started_session()
                    raise

            submit_async(
                context,
                label="开始会话",
                work=lambda: core.execute_start_request(prepared),
                apply=apply_start,
                success=start_succeeded,
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"开始失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_finish(bpy.types.Operator):
    bl_idname = "skintokens_interactive.finish"
    bl_label = "结束"
    bl_description = "结束交互会话并保留生成结果"

    def execute(self, context):
        if async_busy():
            set_status(context, f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            return {"CANCELLED"}
        session_id = context.scene.skintokens_blender_session_id
        if not session_id:
            set_status(context, "当前没有可结束的会话")
            return {"CANCELLED"}

        cancel_history_sync()
        configured_output = context.scene.skintokens_output_path.strip()
        saved_path = None
        save_error = None
        reset_error = None
        session = None if _CORE is None else _CORE.sessions.get(session_id)
        finished_armature = None
        if session is not None:
            finished_armature = bpy.data.objects.get(session.armature_object_name)
        if finished_armature is None:
            for obj in bpy.data.objects:
                if obj.get("skintokens_blender_session_id") != session_id:
                    continue
                source_name = str(obj.get("skintokens_source_armature", ""))
                candidate = bpy.data.objects.get(source_name)
                if candidate is not None and candidate.type == "ARMATURE":
                    finished_armature = candidate
                    break
        try:
            if configured_output:
                destination = Path(configured_output).expanduser().resolve()
                if session is None:
                    save_error = "交互会话不可用"
                else:
                    export_response = _CORE.export_skin(session_id, destination)
                    if not export_response.get("ok"):
                        save_error = export_response.get("error", "TXT 导出失败")
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
            if finished_armature is not None:
                finished_armature[POSTPROCESS_READY_PROP] = True
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
            errors.append(f"TXT 保存失败：{save_error}")
        if reset_error:
            errors.append(f"服务端清理失败：{reset_error}")
        if errors:
            message = f"本地会话已结束；{'；'.join(errors)}"
            set_status(context, message)
            self.report({"WARNING"}, message)
        elif saved_path is not None:
            set_status(context, f"会话已结束，TXT 已保存到：{saved_path}")
        else:
            set_status(context, "会话已结束，模型和蒙皮结果已保留")
        return {"FINISHED"}


class SKINTOKENS_OT_semantic_rename(bpy.types.Operator):
    bl_idname = "skintokens_interactive.semantic_rename"
    bl_label = "身体分区命名"
    bl_description = "按二足或四足模板识别身体分区并重命名通用骨骼"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if async_busy():
            cls.poll_message_set("请等待当前操作完成")
            return False
        armature = session_armature(context)
        if armature is None:
            cls.poll_message_set("请先开始 SkinTokens 会话")
            return False
        return True

    def execute(self, context):
        try:
            armature = session_armature(context)
            if armature is None:
                raise RigPostprocessError("请先开始 SkinTokens 会话")
            result = rename_body_regions(
                context,
                armature,
                context.scene.skintokens_rig_template,
            )
            _queue_history_sync(0.0)
            message = (
                f"身体分区命名完成：识别 {result.classified_bones} 根，"
                f"重命名 {result.renamed_bones} 根，"
                f"建立 {result.mirror_pairs} 组镜像配对"
            )
            if result.warnings:
                message = f"{message}；{'；'.join(result.warnings)}"
            set_status(context, message)
            self.report({"INFO"}, message)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            message = f"身体分区命名失败：{exc}"
            set_status(context, message)
            self.report({"ERROR"}, message)
            return {"CANCELLED"}


class SKINTOKENS_OT_mirror_align(bpy.types.Operator):
    bl_idname = "skintokens_interactive.mirror_align"
    bl_label = "镜像对齐"
    bl_description = (
        "骨架编辑模式下高亮至少两根时对齐高亮骨骼，高亮零根或一根时对齐全部骨骼；"
        "其他模式下也对齐全部骨骼；"
        "不依赖骨骼名称，不修改权重和 Bone Roll，并保持连接关节相连"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        if async_busy():
            cls.poll_message_set("请等待当前操作完成")
            return False
        armature = session_armature(context)
        if armature is None:
            cls.poll_message_set("请先开始 SkinTokens 会话")
            return False
        is_session_edit = (
            context.object is armature and context.mode == "EDIT_ARMATURE"
        )
        selected_count = (
            len(selected_edit_bone_names(context, armature))
            if is_session_edit
            else 0
        )
        scope_count = (
            selected_count
            if selected_count >= 2
            else len(armature.data.bones)
        )
        if scope_count < 2:
            cls.poll_message_set("当前会话骨架没有足够的骨骼")
            return False
        return True

    def execute(self, context):
        try:
            armature = session_armature(context)
            if armature is None:
                raise RigPostprocessError("请先开始 SkinTokens 会话")
            axis = context.scene.skintokens_mirror_axis
            center = context.scene.skintokens_mirror_center
            distance_mode = context.scene.skintokens_mirror_distance_mode
            selected_subset = (
                context.object is armature
                and context.mode == "EDIT_ARMATURE"
                and len(selected_edit_bone_names(context, armature)) >= 2
            )
            result = mirror_align_bones(
                context,
                armature,
                axis=axis,
                center=center,
                distance_mode=distance_mode,
            )
            _queue_history_sync(0.0)
            distance_label = {
                MIRROR_DISTANCE_FARTHEST: "最远",
                MIRROR_DISTANCE_NEAREST: "最近",
                MIRROR_DISTANCE_AVERAGE: "平均",
            }[distance_mode]
            scope_label = "选中骨骼" if selected_subset else "全部骨骼"
            message = (
                f"镜像对齐完成：已按 {axis}={center:g}、{distance_label}距离"
                f"在{scope_label}中对齐 {result.mirror_pairs} 组骨骼"
            )
            if result.skipped_bones:
                names = ", ".join(result.skipped_bones[:4])
                if len(result.skipped_bones) > 4:
                    names = f"{names} 等"
                message = (
                    f"{message}；跳过 {len(result.skipped_bones)} 根未可靠配对骨骼："
                    f"{names}"
                )
            set_status(context, message)
            self.report({"INFO"}, message)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            message = f"镜像对齐失败：{exc}"
            set_status(context, message)
            self.report({"ERROR"}, message)
            return {"CANCELLED"}


class SKINTOKENS_OT_next(bpy.types.Operator):
    bl_idname = "skintokens_interactive.next"
    bl_label = "衍生骨骼"
    bl_description = "按模型预测继续生成下一段、其他分支或结束骨架"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_next(context, force_parent=False)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"衍生骨骼失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_force_next(bpy.types.Operator):
    bl_idname = "skintokens_interactive.force_next"
    bl_label = "强制衍生子骨骼"
    bl_description = "以当前选中的骨骼为父级生成一根子骨骼"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_next(context, force_parent=True)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"强制衍生子骨骼失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_import_txt(bpy.types.Operator):
    bl_idname = "skintokens_interactive.import_txt"
    bl_label = "导入 TXT"
    bl_description = "将 TXT 骨架和蒙皮导入当前会话"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("请先开始会话")
            if not context.scene.skintokens_import_txt_path.strip():
                raise RuntimeError("请先选择 TXT 文件")
            txt_path = Path(context.scene.skintokens_import_txt_path).expanduser()
            response = get_core(context).import_txt(session_id, txt_path)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "导入 TXT 失败"))
            count = len(response.get("context", {}).get("joints", []))
            set_status(context, f"TXT 已导入：{count} 根骨骼")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"导入 TXT 失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_rig(bpy.types.Operator):
    bl_idname = "skintokens_interactive.rig"
    bl_label = "生成骨骼树"
    bl_description = "从当前骨架继续生成，直到骨架结束"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_rig(context)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"生成骨骼树失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_skin(bpy.types.Operator):
    bl_idname = "skintokens_interactive.skin"
    bl_label = "生成蒙皮"
    bl_description = "为当前骨架生成蒙皮权重"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            _schedule_skin(context)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"生成蒙皮失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_refresh(bpy.types.Operator):
    bl_idname = "skintokens_interactive.refresh"
    bl_label = "刷新"
    bl_description = "同步编辑或删除后的骨架，不执行生成"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("请先开始会话")
            core = get_core(context)
            prepared = core.prepare_refresh(session_id)
            _schedule_context_edit(
                context,
                label="刷新骨架",
                prepared=prepared,
                apply_result=core.apply_refresh_request,
                success=lambda result_context, response: set_status(
                    result_context,
                    f"骨架已刷新：{len(response.get('context', {}).get('joints', []))} 根骨骼",
                ),
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"刷新失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_vae_generate(bpy.types.Operator):
    bl_idname = "skintokens_interactive.vae_generate"
    bl_label = "权重场递归重建"
    bl_description = "为当前骨骼生成 0-3 级递归重建缓存，并预览 1 级"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            _schedule_vae_cache_generation(context)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"权重场递归重建失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_vae_apply(bpy.types.Operator):
    bl_idname = "skintokens_interactive.vae_apply"
    bl_label = "应用权重场递归重建"
    bl_description = "将当前权重场递归重建预览应用为新的可编辑 0 级"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("请先开始会话")
            response = get_core(context).apply_vae_reconstruction_levels(session_id)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "应用权重场递归重建失败"))
            applied = response.get("applied_levels", {})
            detail = ", ".join(
                f"{name}:{level}" for name, level in applied.items()
            )
            set_status(
                context,
                f"权重场递归重建预览已应用（{detail}），级别已重置为 0",
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"应用权重场递归重建失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_split(bpy.types.Operator):
    bl_idname = "skintokens_interactive.split"
    bl_label = "拆分骨骼"
    bl_description = "在选中骨骼中插入一个中点骨骼"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("请先开始会话")
            core = get_core(context)
            prepared = core.prepare_split(session_id)
            _schedule_context_edit(
                context,
                label="拆分骨骼",
                prepared=prepared,
                apply_result=core.apply_split_request,
                success=lambda result_context, response: set_status(
                    result_context,
                    "骨骼已拆分：共 "
                    f"{len(response.get('context', {}).get('joints', []))} 根骨骼",
                ),
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"拆分失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_delete(bpy.types.Operator):
    bl_idname = "skintokens_interactive.delete"
    bl_label = "收合子骨骼"
    bl_description = "收合当前骨骼的下一段，或移除末端骨骼"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            if async_busy():
                raise RuntimeError(f"正在执行“{_ASYNC_JOB['label']}”，请稍候")
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("请先开始会话")
            core = get_core(context)
            prepared = core.prepare_delete(session_id)

            def succeeded(result_context, response: dict) -> None:
                count = len(response.get("context", {}).get("joints", []))
                suffix = (
                    "；骨架已变化，请重新生成蒙皮"
                    if response.get("skin_invalidated")
                    else ""
                )
                set_status(
                    result_context,
                    f"已收合 {response.get('deleted_bone_name')}：共 {count} 根骨骼"
                    f"{suffix}",
                )

            _schedule_context_edit(
                context,
                label="收合子骨骼",
                prepared=prepared,
                apply_result=core.apply_delete_request,
                success=succeeded,
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"收合失败：{exc}")
            return {"CANCELLED"}


class SKINTOKENS_PT_interactive(bpy.types.Panel):
    bl_label = "H3D Skintokens"
    bl_idname = "SKINTOKENS_PT_interactive"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "H3D Skintokens"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        url_row = layout.row(align=True)
        url_value = url_row.row(align=True)
        url_value.enabled = not async_busy()
        preferences = get_addon_preferences(context)
        if preferences is None:
            url_value.prop(scene, "skintokens_model_socket")
        else:
            url_value.prop(preferences, "server_url", text="服务器地址")
        connection_button = url_row.row(align=True)
        connection_button.enabled = online_access_enabled() and not async_busy()
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
                    text="请关闭离线模式并重启 Blender",
                    icon="ERROR",
                )
            else:
                access_box.label(text="SkinTokens 需要联网权限", icon="INFO")
                access_box.operator(
                    "skintokens_interactive.enable_online_access",
                    icon="CHECKMARK",
                )
        global_box = layout.box()
        global_box.label(text="全局生成")
        generate_row = global_box.row(align=True)
        generate_row.enabled = has_started_session(context) and not async_busy()
        generate_row.operator("skintokens_interactive.rig", icon="OUTLINER_OB_ARMATURE")
        generate_row.operator("skintokens_interactive.skin", icon="MOD_ARMATURE")
        bone_box = layout.box()
        bone_box.label(text="逐骨骼操作")
        commands = bone_box.column()
        commands.enabled = has_started_session(context) and not async_busy()
        next_row = commands.row(align=True)
        next_row.operator("skintokens_interactive.next", icon="TRACKING_FORWARDS")
        next_row.operator("skintokens_interactive.force_next", icon="CON_CHILDOF")
        edit_row = commands.row(align=True)
        edit_row.operator("skintokens_interactive.split", icon="ADD")
        edit_row.operator("skintokens_interactive.delete", icon="REMOVE")
        vae_row = commands.split(factor=0.5, align=True)
        generate_vae_row = vae_row.row(align=True)
        generate_vae_row.enabled = (
            selected_vae_bone_name(context) is not None
            and not has_selected_vae_cache(context)
        )
        generate_vae_row.operator(
            "skintokens_interactive.vae_generate",
            text="权重递归重建",
            icon="FILE_REFRESH",
        )
        vae_controls = vae_row.split(factor=0.82, align=True)
        slider_row = vae_controls.row(align=True)
        slider_row.enabled = has_selected_vae_cache(context)
        slider_row.prop(
            scene,
            "skintokens_vae_reconstruction_level",
            text="重建级别",
            slider=True,
        )
        apply_row = vae_controls.row(align=True)
        apply_row.enabled = has_selected_vae_cache(context)
        apply_row.operator(
            "skintokens_interactive.vae_apply",
            text="",
            icon="CHECKMARK",
        )
        rig_tools_box = layout.box()
        rig_tools_box.label(text="骨架整理")
        armature = session_armature(context)
        rig_tools = rig_tools_box.column()
        rig_tools.enabled = (
            has_started_session(context)
            and not async_busy()
            and armature is not None
        )
        naming_row = rig_tools.row(align=True)
        naming_row.prop(scene, "skintokens_rig_template", text="")
        naming_row.operator(
            "skintokens_interactive.semantic_rename",
            icon="OUTLINER_DATA_ARMATURE",
        )
        plane_row = rig_tools.row(align=True)
        plane_row.label(text="对称面")
        plane_row.prop(scene, "skintokens_mirror_axis", text="")
        plane_row.label(text="=")
        plane_row.prop(scene, "skintokens_mirror_center", text="")
        mirror_row = rig_tools.row(align=True)
        mirror_row.prop(
            scene,
            "skintokens_mirror_distance_mode",
            text="",
        )
        mirror_button = mirror_row.row(align=True)
        is_session_edit = (
            armature is not None
            and context.object is armature
            and context.mode == "EDIT_ARMATURE"
        )
        selected_count = (
            len(selected_session_edit_bone_names(context))
            if is_session_edit
            else 0
        )
        scope_count = (
            (
                selected_count
                if selected_count >= 2
                else len(armature.data.bones)
            )
            if armature is not None
            else 0
        )
        mirror_button.enabled = (
            armature is not None
            and scope_count >= 2
        )
        mirror_button.operator(
            "skintokens_interactive.mirror_align",
            icon="MOD_MIRROR",
        )
        layout.separator()
        layout.label(text=scene.skintokens_status)
        state_row = layout.row()
        state_row.enabled = not async_busy() and online_access_enabled()
        if has_session_marker(context):
            state_row.operator("skintokens_interactive.finish", icon="X")
        else:
            update = (
                None
                if preferences is None
                else _extension_update_info(context.preferences)
            )
            if update is None:
                state_row.operator("skintokens_interactive.start", icon="PLAY")
            else:
                props = state_row.operator(
                    "extensions.package_install",
                    text=f"更新到 {update['version']}",
                    icon="FILE_REFRESH",
                )
                props.repo_index = update["repo_index"]
                props.pkg_id = _EXTENSION_PACKAGE_ID
                props.enable_on_install = True


classes = (
    SKINTOKENS_Preferences,
    SKINTOKENS_OT_test_connection,
    SKINTOKENS_OT_enable_online_access,
    SKINTOKENS_OT_start,
    SKINTOKENS_OT_finish,
    SKINTOKENS_OT_semantic_rename,
    SKINTOKENS_OT_mirror_align,
    SKINTOKENS_OT_next,
    SKINTOKENS_OT_force_next,
    SKINTOKENS_OT_import_txt,
    SKINTOKENS_OT_rig,
    SKINTOKENS_OT_skin,
    SKINTOKENS_OT_refresh,
    SKINTOKENS_OT_vae_generate,
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
        name="服务器地址",
        default=DEFAULT_MODEL_URL,
    )
    bpy.types.Scene.skintokens_owner_id = bpy.props.StringProperty(default="", options={"HIDDEN"})
    bpy.types.Scene.skintokens_output_path = bpy.props.StringProperty(
        name="输出路径",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.skintokens_import_txt_path = bpy.props.StringProperty(
        name="导入 TXT",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.skintokens_blender_session_id = bpy.props.StringProperty(default="")
    bpy.types.Scene.skintokens_status = bpy.props.StringProperty(default="未开始")
    bpy.types.Scene.skintokens_rig_template = bpy.props.EnumProperty(
        name="骨架模板",
        description="身体分区命名使用的拓扑模板",
        items=(
            (
                "BIPED",
                "二足",
                "识别脊柱、头部、双臂、双腿和可选尾部",
            ),
            (
                "QUADRUPED",
                "四足",
                "识别脊柱、头部、前后腿和尾部",
            ),
        ),
        default="QUADRUPED",
    )
    bpy.types.Scene.skintokens_mirror_axis = bpy.props.EnumProperty(
        name="对称轴",
        description="选择 Armature 局部坐标中的轴；对称面不允许旋转",
        items=(
            (MIRROR_AXIS_X, "X", "使用 Armature 局部坐标平面 X=指定值"),
            (MIRROR_AXIS_Y, "Y", "使用 Armature 局部坐标平面 Y=指定值"),
            (MIRROR_AXIS_Z, "Z", "使用 Armature 局部坐标平面 Z=指定值"),
        ),
        default=MIRROR_AXIS_X,
    )
    bpy.types.Scene.skintokens_mirror_center = bpy.props.FloatProperty(
        name="对称面位置",
        description="对称面在 Armature 局部坐标中的轴向位置",
        default=0.0,
        unit="LENGTH",
    )
    bpy.types.Scene.skintokens_mirror_distance_mode = bpy.props.EnumProperty(
        name="对齐距离",
        description="左右点对齐后到对称面的目标距离",
        items=(
            (
                MIRROR_DISTANCE_FARTHEST,
                "最远",
                "两侧都采用原来离对称面较远一侧的距离",
            ),
            (
                MIRROR_DISTANCE_NEAREST,
                "最近",
                "两侧都采用原来离对称面较近一侧的距离",
            ),
            (
                MIRROR_DISTANCE_AVERAGE,
                "平均",
                "两侧都采用原有两个距离的平均值",
            ),
        ),
        default=MIRROR_DISTANCE_AVERAGE,
    )
    bpy.types.Scene.skintokens_vae_reconstruction_level = bpy.props.IntProperty(
        name="权重场递归重建",
        description=(
            "为当前蒙皮骨骼选择缓存的 VAE 重建级别 0-3"
        ),
        min=0,
        max=DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL,
        get=get_vae_reconstruction_level,
        set=set_vae_reconstruction_level,
    )
    bpy.types.Scene.skintokens_max_new_tokens = bpy.props.IntProperty(name="下一步 Token 数", default=16, min=1, max=64)
    bpy.types.Scene.skintokens_skin_max_new_tokens = bpy.props.IntProperty(name="蒙皮 Token 数", default=2048, min=8, max=8192)
    bpy.types.Scene.skintokens_top_k = bpy.props.IntProperty(name="Top K", default=5, min=0, max=100)
    bpy.types.Scene.skintokens_top_p = bpy.props.FloatProperty(name="Top P", default=0.95, min=0.0, max=1.0)
    bpy.types.Scene.skintokens_temperature = bpy.props.FloatProperty(name="温度", default=1.5, min=0.01, max=5.0)
    bpy.types.Scene.skintokens_repetition_penalty = bpy.props.FloatProperty(name="重复惩罚", default=1.2, min=0.1, max=5.0)
    bpy.types.Scene.skintokens_num_beams = bpy.props.IntProperty(name="束搜索数", default=1, min=1, max=16)
    if sync_model_after_history_change not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(sync_model_after_history_change)
    if sync_model_after_history_change not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(sync_model_after_history_change)
    if sync_model_after_armature_change not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            sync_model_after_armature_change
        )
    _schedule_extension_repository_sync()
    _schedule_extension_event_report()


def unregister():
    global _CORE, _UPDATE_INDEX_CACHE
    _UPDATE_INDEX_CACHE = None
    if sync_model_after_history_change in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(sync_model_after_history_change)
    if sync_model_after_history_change in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(sync_model_after_history_change)
    if sync_model_after_armature_change in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(
            sync_model_after_armature_change
        )
    cancel_history_sync()
    cancel_async_job()
    _cancel_extension_repository_sync()
    _cancel_extension_event_report()
    core = _CORE
    _CORE = None
    if core is not None:
        for session_id in list(core.sessions):
            try:
                session = core.sessions.get(session_id)
                if session is not None:
                    armature = bpy.data.objects.get(session.armature_object_name)
                    if armature is not None and armature.type == "ARMATURE":
                        armature[POSTPROCESS_READY_PROP] = True
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
        "skintokens_rig_template",
        "skintokens_mirror_axis",
        "skintokens_mirror_center",
        "skintokens_mirror_distance_mode",
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
