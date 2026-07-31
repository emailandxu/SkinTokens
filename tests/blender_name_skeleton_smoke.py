"""Run with Blender after opening examples/name_skeleton.blend."""

from __future__ import annotations

import sys
from pathlib import Path

import bpy  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interactive.blender import rig_postprocess  # noqa: E402


armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
assert len(armatures) == 1, [obj.name for obj in armatures]
armature = armatures[0]
armature[rig_postprocess.POSTPROCESS_READY_PROP] = True

snapshots = rig_postprocess.snapshot_armature(armature)
plan = rig_postprocess.build_naming_plan(
    snapshots,
    rig_postprocess.TEMPLATE_QUADRUPED,
)
semantic_by_old_name = {
    assignment.old_name: assignment.semantic_name
    for assignment in plan.assignments
}
expected_limb_roots = {
    "bone_34": "leg_hind_l_1",
    "bone_5": "leg_hind_r_1",
    "bone_27": "leg_front_l_1",
    "bone_21": "leg_front_r_1",
}
expected_tail = {
    "bone_1": "tail_1",
    "bone_2": "tail_2",
    "bone_3": "tail_3",
    "bone_4": "tail_4",
    "bone_40": "tail_5",
}
assert {
    name: semantic_by_old_name.get(name)
    for name in expected_limb_roots
} == expected_limb_roots
assert {
    name: semantic_by_old_name.get(name)
    for name in expected_tail
} == expected_tail
assert not semantic_by_old_name["bone_18"].startswith("leg_")
assert not semantic_by_old_name["bone_15"].startswith("leg_")

for obj in bpy.context.selected_objects:
    obj.select_set(False)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
result = rig_postprocess.rename_body_regions(
    bpy.context,
    armature,
    rig_postprocess.TEMPLATE_QUADRUPED,
)
assert result.mirror_pairs == 11, result
assert {bone.name for bone in armature.data.bones}.issuperset(
    {*expected_limb_roots.values(), *expected_tail.values()}
)
print("SKINTOKENS_NAME_SKELETON_SMOKE_OK")
