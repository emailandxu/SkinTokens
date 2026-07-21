"""Run with: blender --background --python tests/blender_undo_smoke.py"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bpy  # type: ignore  # noqa: E402

from interactive.blender import addon  # noqa: E402
from interactive.blender.apply_skin import (  # noqa: E402
    apply_armature_context,
    ensure_mesh_armature,
    parse_armature_object,
)
from interactive.blender.core import (  # noqa: E402
    BlenderInteractiveCore,
    BlenderInteractiveSession,
)


CONTEXT = {
    "joints": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    "parents": [-1, 0, 1, 0],
    "joint_names": ["root", "branch", "leaf", "sibling"],
    "done": False,
}


def flush_timer(callback) -> None:
    if bpy.app.timers.is_registered(callback):
        bpy.app.timers.unregister(callback)
    callback()
    if addon.async_busy():
        flush_async_job()


def flush_async_job() -> None:
    assert addon._ASYNC_JOB is not None
    addon._ASYNC_JOB["future"].result(timeout=5)
    if bpy.app.timers.is_registered(addon._poll_async_job):
        bpy.app.timers.unregister(addon._poll_async_job)
    assert addon._poll_async_job() is None
    assert not addon.async_busy()


mesh_data = bpy.data.meshes.new("UndoSmokeMesh")
mesh_data.from_pydata(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    [],
    [(0, 1, 2)],
)
mesh_object = bpy.data.objects.new("UndoSmokeMesh", mesh_data)
bpy.context.scene.collection.objects.link(mesh_object)
armature_object = ensure_mesh_armature(mesh_object.name)
apply_armature_context(
    armature_object.name,
    CONTEXT["joints"],
    CONTEXT["parents"],
    CONTEXT["joint_names"],
    select_name="sibling",
    mode_after="EDIT",
)

addon.register()
scene = bpy.context.scene
assert bpy.types.Scene.bl_rna.properties[
    "skintokens_vae_reconstruction_level"
].hard_max == 3
scene.skintokens_owner_id = "undo-smoke"
scene.skintokens_blender_session_id = "blender-session"
assert not addon.has_started_session(bpy.context)

core = BlenderInteractiveCore(
    scene.skintokens_model_socket,
    owner_id=scene.skintokens_owner_id,
)
session = BlenderInteractiveSession(
    blender_session_id="blender-session",
    model_session_id="model-session",
    obj_path=ROOT / "tests" / "undo-smoke.obj",
    context=dict(CONTEXT),
    mesh_object_name=mesh_object.name,
    armature_object_name=armature_object.name,
)
core.sessions[session.blender_session_id] = session


def echo_session_request(current_session, payload):
    return {
        "ok": True,
        "context": {
            **current_session.context,
            "joints": payload.get("joints", []),
            "parents": payload.get("parents", []),
            "joint_names": payload.get("joint_names", []),
            "done": False,
        },
    }


core.session_request = echo_session_request


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
assert addon.has_started_session(bpy.context)
addon.ensure_armature_watch()
assert addon._watch_active_armature() == addon._ARMATURE_WATCH_INTERVAL

bpy.ops.ed.undo_push(message="SkinTokens initial armature")
for edit_bone in armature_object.data.edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
sibling = armature_object.data.edit_bones["sibling"]
sibling.select_head = True
armature_object.data.edit_bones.active = sibling
bpy.context.view_layer.update()
assert sibling.select and sibling.select_head and sibling.select_tail
addon.cancel_history_sync()
bpy.app.handlers.depsgraph_update_post.remove(
    addon.sync_model_after_armature_change
)
sibling.head.x += 0.5
sibling.tail.x += 0.5
bpy.context.view_layer.update()
bpy.ops.ed.undo_push(message="SkinTokens moved sibling")
assert not addon._HISTORY_SYNC_PENDING
assert addon._watch_active_armature() == addon._ARMATURE_WATCH_INTERVAL
assert not addon._HISTORY_SYNC_PENDING
assert addon._watch_active_armature() == addon._ARMATURE_WATCH_INTERVAL
assert addon._HISTORY_SYNC_PENDING
bpy.app.handlers.depsgraph_update_post.append(
    addon.sync_model_after_armature_change
)
addon._HISTORY_SYNC_DEADLINE = 0.0
flush_timer(addon._flush_history_sync)
assert abs(session.context["joints"][3][0] - 0.5) < 1e-6

bpy.ops.ed.undo()
addon._HISTORY_SYNC_DEADLINE = 0.0
flush_timer(addon._flush_history_sync)
assert abs(session.context["joints"][3][0]) < 1e-6

bpy.ops.ed.redo()
addon._HISTORY_SYNC_DEADLINE = 0.0
flush_timer(addon._flush_history_sync)
assert abs(session.context["joints"][3][0] - 0.5) < 1e-6

# Split operates on the active bone's visible downstream segment. Selecting the
# same upstream bone and dissolving removes the inserted joint again.
bpy.ops.object.mode_set(mode="OBJECT")
session.context = dict(CONTEXT)
apply_armature_context(
    armature_object.name,
    CONTEXT["joints"],
    CONTEXT["parents"],
    CONTEXT["joint_names"],
    select_name="root",
    remove_missing=True,
    mode_after="EDIT",
)
split = core.split_selected_bone(session.blender_session_id)
assert split["ok"]
assert split["context"]["joint_names"][:3] == ["root", "root_split", "branch"]
for edit_bone in armature_object.data.edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
root = armature_object.data.edit_bones["root"]
root.select = True
root.select_head = True
root.select_tail = True
armature_object.data.edit_bones.active = root
dissolved = core.delete_selected_bone(session.blender_session_id)
assert dissolved["ok"]
assert dissolved["deleted_bone_name"] == "root_split"
assert dissolved["delete_operation"] == "dissolve"
assert dissolved["context"]["joint_names"] == CONTEXT["joint_names"]

for edit_bone in armature_object.data.edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
leaf = armature_object.data.edit_bones["leaf"]
leaf.select = True
leaf.select_head = True
leaf.select_tail = True
armature_object.data.edit_bones.active = leaf
deleted_leaf = core.delete_selected_bone(session.blender_session_id)
assert deleted_leaf["ok"]
assert deleted_leaf["deleted_bone_name"] == "leaf"
assert deleted_leaf["delete_operation"] == "delete"
assert "leaf" not in deleted_leaf["context"]["joint_names"]

bpy.ops.object.mode_set(mode="OBJECT")
session.context = dict(CONTEXT)
apply_armature_context(
    armature_object.name,
    CONTEXT["joints"],
    CONTEXT["parents"],
    CONTEXT["joint_names"],
    select_name="branch",
    remove_missing=True,
    mode_after="EDIT",
)

# Native Blender deletion reparents children. Refresh accepts that exact native
# hierarchy instead of applying the old preview's subtree-deletion rule.
branch = armature_object.data.edit_bones["branch"]
armature_object.data.edit_bones.remove(branch)
assert bpy.ops.skintokens_interactive.refresh() == {"FINISHED"}
flush_async_job()
_joints, parents, names = parse_armature_object(armature_object)
assert parents == [-1, 0, 0], parents
assert names == ["root", "leaf", "sibling"], names

# Deleting a root from a branched rig leaves several Blender roots. Refresh
# rejects that invalid model context but must not replace the remaining rig with
# an empty Armature.
bpy.ops.object.mode_set(mode="OBJECT")
session.context = dict(CONTEXT)
apply_armature_context(
    armature_object.name,
    CONTEXT["joints"],
    CONTEXT["parents"],
    CONTEXT["joint_names"],
    select_name="root",
    remove_missing=True,
    mode_after="EDIT",
)
armature_object.data.edit_bones.remove(armature_object.data.edit_bones["root"])
assert bpy.ops.skintokens_interactive.refresh() == {"CANCELLED"}
assert len(armature_object.data.edit_bones) == 3

bpy.ops.object.mode_set(mode="OBJECT")
session.context = dict(CONTEXT)
apply_armature_context(
    armature_object.name,
    CONTEXT["joints"],
    CONTEXT["parents"],
    CONTEXT["joint_names"],
    select_name="sibling",
    remove_missing=True,
    mode_after="EDIT",
)

bpy.ops.object.mode_set(mode="POSE")
bpy.ops.ed.undo_push(message="Pose before rotation")
armature_object.pose.bones["sibling"].rotation_mode = "XYZ"
armature_object.pose.bones["sibling"].rotation_euler.z = 0.75
bpy.context.view_layer.update()
bpy.ops.ed.undo_push(message="Pose after rotation")
bpy.ops.ed.undo()
addon._HISTORY_SYNC_DEADLINE = 0.0
flush_timer(addon._flush_history_sync)
assert bpy.context.mode == "POSE", bpy.context.mode

bpy.ops.object.mode_set(mode="OBJECT")
finished_scene = bpy.context.scene
finished_armature = bpy.data.objects.get(session.armature_object_name)
assert finished_armature is not None
mesh_object["skintokens_blender_session_id"] = session.blender_session_id
mesh_object["skintokens_model_session_id"] = session.model_session_id
mesh_object["skintokens_source_obj"] = str(session.obj_path)
mesh_object["skintokens_source_armature"] = finished_armature.name
root_group = mesh_object.vertex_groups.get("root")
if root_group is None:
    root_group = mesh_object.vertex_groups.new(name="root")
root_group.add([0, 1, 2], 1.0, "REPLACE")
session.skin_generated = True
session.blender_to_obj_text = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=float,
)
saved_output = Path(tempfile.gettempdir()) / "skintokens_finish_current_skin.txt"
saved_output.unlink(missing_ok=True)
finished_scene.skintokens_output_path = str(saved_output)
reset_requests = []


def reset_model_request(payload):
    reset_requests.append(payload)
    return {"ok": True}


core.model_request = reset_model_request
addon._queue_history_sync(60.0)
assert bpy.app.timers.is_registered(addon._flush_history_sync)
assert bpy.ops.skintokens_interactive.finish() == {"FINISHED"}
deadline = time.monotonic() + 5.0
while not reset_requests and time.monotonic() < deadline:
    time.sleep(0.01)
assert finished_scene.skintokens_blender_session_id == ""
assert session.blender_session_id not in core.sessions
assert reset_requests == [
    {
        "command": "reset",
        "session_id": "model-session",
        "end_reason": "finish",
    }
]
saved_content = saved_output.read_text(encoding="utf-8")
assert "\nroot root\n" in saved_content
assert "skin 0 root 1.000000" in saved_content
assert "skin 1 root 1.000000" in saved_content
assert "skintokens_blender_session_id" not in mesh_object
assert "skintokens_model_session_id" not in mesh_object
assert mesh_object["skintokens_source_obj"] == str(session.obj_path)
assert mesh_object["skintokens_source_armature"] == finished_armature.name
assert bpy.data.objects.get(mesh_object.name) is mesh_object
assert bpy.data.objects.get(finished_armature.name) is finished_armature
assert not bpy.app.timers.is_registered(addon._flush_history_sync)
assert not addon.has_started_session(bpy.context)
saved_output.unlink()
addon.unregister()
print("SKINTOKENS_UNDO_SMOKE_OK")
