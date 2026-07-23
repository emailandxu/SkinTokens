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
    JOINT_ID_PROP,
    apply_armature_context,
    capture_armature_bone_states,
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
assert addon.SKINTOKENS_OT_next.bl_label == "衍生骨骼"
assert addon.SKINTOKENS_OT_force_next.bl_label == "强制衍生子骨骼"
assert addon.SKINTOKENS_OT_rig.bl_label == "生成骨骼树"
assert addon.SKINTOKENS_OT_split.bl_label == "拆分骨骼"
assert addon.SKINTOKENS_OT_delete.bl_label == "收合子骨骼"
assert addon.SKINTOKENS_OT_vae_generate.bl_label == "权重场递归重建"
assert bpy.types.Scene.bl_rna.properties[
    "skintokens_vae_reconstruction_level"
].name == "权重场递归重建"
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
# Keep deliberately authored pre-session Tails. Context application must not
# rewrite the non-leaf Tail, and a leaf changes only when it gains a child.
armature_object.data.edit_bones["root"].tail = (0.4, 0.6, 0.2)
armature_object.data.edit_bones["sibling"].tail = (0.0, 1.0, 2.0)
initial_bone_snapshot = capture_armature_bone_states(armature_object)
session = BlenderInteractiveSession(
    blender_session_id="blender-session",
    model_session_id="model-session",
    obj_path=ROOT / "tests" / "undo-smoke.obj",
    context=dict(CONTEXT),
    mesh_object_name=mesh_object.name,
    armature_object_name=armature_object.name,
    original_bone_ids=set(initial_bone_snapshot),
    bone_edit_snapshot=initial_bone_snapshot,
)
core.sessions[session.blender_session_id] = session

# A model response may derive the Tail of an original leaf, but it must not
# move that bone's Head.
original_sibling_head = armature_object.data.edit_bones["sibling"].head.copy()
original_sibling_tail = armature_object.data.edit_bones["sibling"].tail.copy()
original_root_tail = armature_object.data.edit_bones["root"].tail.copy()
extended_context = {
    **CONTEXT,
    "joints": [
        *CONTEXT["joints"][:3],
        [99.0, 99.0, 99.0],
        [0.0, 2.0, 0.0],
    ],
    "parents": [-1, 0, 1, 0, 3],
    "joint_names": [*CONTEXT["joint_names"], "sibling_child"],
}
core._apply_context_to_armature(
    session,
    extended_context,
    select_name="sibling_child",
    remove_missing=True,
    mode_after="EDIT",
)
sibling = armature_object.data.edit_bones["sibling"]
sibling_child = armature_object.data.edit_bones["sibling_child"]
assert (sibling.head - original_sibling_head).length < 1e-6
assert (sibling.tail - sibling_child.head).length < 1e-6
assert not sibling_child.use_connect
assert str(sibling.get(JOINT_ID_PROP, "")) in session.tail_follow_bone_ids
assert (
    armature_object.data.edit_bones["root"].tail - original_root_tail
).length < 1e-6

# User edits do not change a session-created bone into an original bone. The
# next synchronized model context may therefore continue to update its Head.
sibling_child.tail.z += 0.75
bpy.context.view_layer.update()
core.sync_context_from_armature(session)
edited_child_id = str(sibling_child.get(JOINT_ID_PROP, ""))
assert edited_child_id not in session.original_bone_ids
attempted_move = {
    **extended_context,
    "joints": [*extended_context["joints"][:-1], [0.0, 3.0, 0.0]],
}
core._apply_context_to_armature(
    session,
    attempted_move,
    select_name="sibling_child",
    remove_missing=True,
    mode_after="EDIT",
)
sibling_child = armature_object.data.edit_bones["sibling_child"]
assert (sibling_child.head - sibling_child.parent.head).y > 1.9
assert (
    armature_object.data.edit_bones["sibling"].tail - sibling_child.head
).length < 1e-6
last_followed_tail = sibling_child.head.copy()
restored_context = dict(CONTEXT)
core._apply_context_to_armature(
    session,
    restored_context,
    select_name="sibling",
    remove_missing=True,
    mode_after="EDIT",
)
assert "sibling_child" not in armature_object.data.edit_bones
restored_sibling = armature_object.data.edit_bones["sibling"]
assert (restored_sibling.tail - last_followed_tail).length < 1e-6
assert (restored_sibling.tail - original_sibling_tail).length > 0.1
session.context = restored_context


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

# Transform modal operations retain the dirty state until the transform ends.
original_transform_check = addon._transform_modal_running
original_prepare_history_sync = core.prepare_history_sync
prepare_called = False


def unexpected_prepare(_session_id):
    global prepare_called
    prepare_called = True
    raise AssertionError("history sync prepared during Transform modal")


addon._transform_modal_running = lambda: True
core.prepare_history_sync = unexpected_prepare
addon._queue_history_sync()
if bpy.app.timers.is_registered(addon._flush_history_sync):
    bpy.app.timers.unregister(addon._flush_history_sync)
assert addon._flush_history_sync() == 0.1
assert addon._HISTORY_SYNC_PENDING
assert not prepare_called
core.prepare_history_sync = original_prepare_history_sync
addon._transform_modal_running = lambda: False
addon._HISTORY_SYNC_DEADLINE = 0.0
flush_timer(addon._flush_history_sync)
assert not addon._HISTORY_SYNC_PENDING

async_applied = False


def unexpected_apply(_result):
    global async_applied
    async_applied = True
    raise AssertionError("async result applied during Transform modal")


addon.submit_async(
    bpy.context,
    label="骨架同步",
    work=lambda: {"ok": True},
    apply=unexpected_apply,
    success=lambda _context, _response: None,
    abort_on_transform=True,
)
addon._transform_modal_running = lambda: True
addon._ASYNC_JOB["future"].result(timeout=5)
if bpy.app.timers.is_registered(addon._poll_async_job):
    bpy.app.timers.unregister(addon._poll_async_job)
assert addon._poll_async_job() is None
assert not async_applied
assert not addon.async_busy()
assert addon._HISTORY_SYNC_PENDING
addon.cancel_history_sync()
addon._transform_modal_running = original_transform_check

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
sibling.head.x += 0.5
sibling.tail.x += 0.5
bpy.context.view_layer.update()
bpy.ops.ed.undo_push(message="SkinTokens moved sibling")
assert addon._HISTORY_SYNC_PENDING
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
assert split["context"]["parents"] == [-1, 0, 1, 2, 1]
assert armature_object.data.edit_bones["branch"].parent.name == "root_split"
assert armature_object.data.edit_bones["sibling"].parent.name == "root_split"
assert armature_object.data.edit_bones.active.name == "root"
dissolved = core.delete_selected_bone(session.blender_session_id)
assert dissolved["ok"]
assert dissolved["deleted_bone_name"] == "root_split"
assert dissolved["delete_operation"] == "dissolve"
assert dissolved["context"]["joint_names"] == CONTEXT["joint_names"]
assert dissolved["context"]["parents"] == CONTEXT["parents"]
assert armature_object.data.edit_bones["branch"].parent.name == "root"
assert armature_object.data.edit_bones["sibling"].parent.name == "root"

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
finished_armature.show_in_front = True
session.armature_show_in_front_before = False
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
core.sessions.pop(session.blender_session_id, None)
stale_session = BlenderInteractiveSession(
    blender_session_id="stale-blender-session",
    model_session_id="stale-model-session",
    obj_path=ROOT / "tests" / "stale.obj",
    context=dict(CONTEXT),
    mesh_object_name=mesh_object.name,
    armature_object_name=finished_armature.name,
    armature_show_in_front_before=False,
)
core.sessions[stale_session.blender_session_id] = stale_session
mesh_object["skintokens_blender_session_id"] = stale_session.blender_session_id
mesh_object["skintokens_model_session_id"] = stale_session.model_session_id
finished_armature.show_in_front = True
addon._clear_stale_sessions_before_start(bpy.context, core)
deadline = time.monotonic() + 5.0
while not reset_requests and time.monotonic() < deadline:
    time.sleep(0.01)
assert stale_session.blender_session_id not in core.sessions
assert not finished_armature.show_in_front
assert reset_requests == [
    {
        "command": "reset",
        "session_id": "stale-model-session",
        "end_reason": "start_replaced",
    }
]
assert "skintokens_blender_session_id" not in mesh_object
assert "skintokens_model_session_id" not in mesh_object

# Start must dispose the old Core before reading changed server settings. This
# covers Undo restoring Scene properties while the Python Core keeps a session.
reset_requests.clear()
core.sessions[stale_session.blender_session_id] = stale_session
mesh_object["skintokens_blender_session_id"] = stale_session.blender_session_id
mesh_object["skintokens_model_session_id"] = stale_session.model_session_id
preferences = addon.get_addon_preferences(bpy.context)
if preferences is None:
    original_server_url = finished_scene.skintokens_model_socket
    changed_server_url = "http://127.0.0.1:18765"
    finished_scene.skintokens_model_socket = changed_server_url
else:
    original_server_url = preferences.server_url
    changed_server_url = "http://127.0.0.1:18765"
    preferences.server_url = changed_server_url
addon._CORE = core
replacement_core = addon._restart_core_for_start(bpy.context)
deadline = time.monotonic() + 5.0
while not reset_requests and time.monotonic() < deadline:
    time.sleep(0.01)
assert replacement_core is addon._CORE
assert replacement_core is not core
assert replacement_core.model_socket == changed_server_url
assert reset_requests == [
    {
        "command": "reset",
        "session_id": "stale-model-session",
        "end_reason": "start_replaced",
    }
]
if preferences is None:
    finished_scene.skintokens_model_socket = original_server_url
else:
    preferences.server_url = original_server_url
addon._CORE = core
reset_requests.clear()
core.sessions[session.blender_session_id] = session
finished_scene.skintokens_blender_session_id = session.blender_session_id
mesh_object["skintokens_blender_session_id"] = session.blender_session_id
mesh_object["skintokens_model_session_id"] = session.model_session_id
finished_armature.show_in_front = True
addon._queue_history_sync(60.0)
assert bpy.app.timers.is_registered(addon._flush_history_sync)
assert bpy.ops.skintokens_interactive.finish() == {"FINISHED"}
deadline = time.monotonic() + 5.0
while not reset_requests and time.monotonic() < deadline:
    time.sleep(0.01)
assert finished_scene.skintokens_blender_session_id == ""
assert session.blender_session_id not in core.sessions
assert not finished_armature.show_in_front
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
