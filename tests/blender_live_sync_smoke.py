"""Run with Blender's UI event loop; exits automatically on success or failure."""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # type: ignore  # noqa: E402

extension_packages = [
    name
    for name in bpy.context.preferences.addons.keys()
    if name.endswith(".h3d_skintokens")
]
if extension_packages:
    assert len(extension_packages) == 1, extension_packages
    package_name = extension_packages[0]
    addon = importlib.import_module(f"{package_name}.addon")
    apply_skin = importlib.import_module(f"{package_name}.apply_skin")
    core_module = importlib.import_module(f"{package_name}.core")
    installed_extension = True
else:
    from interactive.blender import addon  # type: ignore  # noqa: E402
    from interactive.blender import apply_skin  # type: ignore  # noqa: E402
    from interactive.blender import core as core_module  # type: ignore  # noqa: E402

    installed_extension = False

apply_armature_context = apply_skin.apply_armature_context
ensure_mesh_armature = apply_skin.ensure_mesh_armature
BlenderInteractiveCore = core_module.BlenderInteractiveCore
BlenderInteractiveSession = core_module.BlenderInteractiveSession


context = {
    "joints": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    "parents": [-1, 0],
    "joint_names": ["root", "child"],
    "done": False,
}
mesh_data = bpy.data.meshes.new("LiveSyncMesh")
mesh_data.from_pydata(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    [],
    [(0, 1, 2)],
)
mesh_object = bpy.data.objects.new("LiveSyncMesh", mesh_data)
bpy.context.scene.collection.objects.link(mesh_object)
armature_object = ensure_mesh_armature(mesh_object.name)
apply_armature_context(
    armature_object.name,
    context["joints"],
    context["parents"],
    context["joint_names"],
    select_name="child",
    mode_after="EDIT",
)

if not installed_extension:
    addon.register()
scene = bpy.context.scene
scene.skintokens_owner_id = "live-sync-smoke"
scene.skintokens_blender_session_id = "blender-session"
core = BlenderInteractiveCore(
    scene.skintokens_model_socket,
    owner_id=scene.skintokens_owner_id,
)
session = BlenderInteractiveSession(
    blender_session_id="blender-session",
    model_session_id="model-session",
    obj_path=ROOT / "tests" / "live-sync-smoke.obj",
    context=context,
    mesh_object_name=mesh_object.name,
    armature_object_name=armature_object.name,
)
core.sessions[session.blender_session_id] = session


def echo_model_request(payload):
    return {
        "ok": True,
        "context": {
            "joints": payload.get("joints", []),
            "parents": payload.get("parents", []),
            "joint_names": payload.get("joint_names", []),
            "done": bool(payload.get("done", False)),
        },
    }


core.model_request = echo_model_request
addon._CORE = core
deadline = time.monotonic() + 5.0


def move_bone() -> None:
    child = armature_object.data.edit_bones["child"]
    child.head.x += 0.5
    child.tail.x += 0.5
    bpy.context.view_layer.update()
    return None


def check_sync() -> float | None:
    position_synced = abs(float(session.context["joints"][1][0]) - 0.5) < 1e-6
    parent_tail_refreshed = (
        armature_object.mode == "EDIT"
        and abs(
            float(armature_object.data.edit_bones["root"].tail.x) - 0.5
        )
        < 1e-6
    )
    request_finished = (
        not addon.async_busy()
        and scene.skintokens_status == "骨架已自动同步：2 根骨骼"
    )
    if position_synced and parent_tail_refreshed and request_finished:
        print("SKINTOKENS_LIVE_SYNC_OK")
        bpy.ops.wm.quit_blender()
        return None
    if time.monotonic() >= deadline:
        print(
            "SKINTOKENS_LIVE_SYNC_FAILED",
            session.context,
            scene.skintokens_status,
            "parent_tail_x=",
            float(armature_object.data.edit_bones["root"].tail.x),
        )
        os._exit(2)
    return 0.05


bpy.app.timers.register(move_bone, first_interval=0.25)
bpy.app.timers.register(check_sync, first_interval=0.3)
