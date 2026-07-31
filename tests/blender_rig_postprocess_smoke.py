"""Run with Blender: --background --factory-startup --python this file."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy  # type: ignore
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interactive.blender import rig_postprocess  # noqa: E402
from interactive.blender.apply_skin import (  # noqa: E402
    managed_vertex_group_names,
)


def transform(point):
    angle = math.radians(29.0)
    x, y, z = point
    return (
        math.cos(angle) * x - math.sin(angle) * y + 3.5,
        math.sin(angle) * x + math.cos(angle) * y - 2.0,
        z + 1.25,
    )


def inverse_transform(point):
    angle = math.radians(29.0)
    x = float(point[0]) - 3.5
    y = float(point[1]) + 2.0
    return (
        math.cos(angle) * x + math.sin(angle) * y,
        -math.sin(angle) * x + math.cos(angle) * y,
        float(point[2]) - 1.25,
    )


rows = [
    (-1, (0.0, 0.0, 1.2), (0.0, 0.8, 1.25)),
    (0, (0.0, 0.8, 1.25), (0.0, 1.7, 1.3)),
    (1, (0.0, 1.7, 1.3), (0.0, 2.5, 1.45)),
    (2, (0.0, 2.5, 1.45), (0.0, 3.15, 1.7)),
    (3, (0.0, 3.15, 1.7), (0.0, 3.65, 1.8)),
    (0, (0.65, 0.05, 1.15), (0.72, 0.05, 0.45)),
    (5, (0.72, 0.05, 0.45), (0.78, 0.15, -0.2)),
    (0, (-0.65, 0.05, 1.15), (-0.72, 0.05, 0.45)),
    (7, (-0.72, 0.05, 0.45), (-0.78, 0.15, -0.2)),
    (2, (0.62, 2.45, 1.35), (0.7, 2.45, 0.55)),
    (9, (0.7, 2.45, 0.55), (0.77, 2.55, -0.2)),
    (2, (-0.62, 2.45, 1.35), (-0.7, 2.45, 0.55)),
    (11, (-0.7, 2.45, 0.55), (-0.77, 2.55, -0.2)),
    (0, (0.0, -0.45, 1.2), (0.0, -1.1, 1.15)),
    (13, (0.0, -1.1, 1.15), (0.0, -1.75, 1.05)),
]

armature_data = bpy.data.armatures.new("RigPostprocessSmokeArmature")
armature = bpy.data.objects.new(armature_data.name, armature_data)
armature[rig_postprocess.POSTPROCESS_READY_PROP] = False
bpy.context.scene.collection.objects.link(armature)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
edit_bones = []
for index, (parent, head, tail) in enumerate(rows):
    bone = armature_data.edit_bones.new(f"bone_{index}")
    bone.head = transform(head)
    bone.tail = transform(tail)
    if parent >= 0:
        bone.parent = edit_bones[parent]
        if (bone.head - bone.parent.tail).length < 1e-6:
            bone.use_connect = True
    edit_bones.append(bone)
bpy.ops.object.mode_set(mode="OBJECT")

vertices = [transform((0.0, index * 0.03, 0.0)) for index in range(len(rows))]
mesh_data = bpy.data.meshes.new("RigPostprocessSmokeMesh")
mesh_data.from_pydata(vertices, [], [])
mesh = bpy.data.objects.new(mesh_data.name, mesh_data)
bpy.context.scene.collection.objects.link(mesh)
modifier = mesh.modifiers.new(name="Armature", type="ARMATURE")
modifier.object = armature
mask_modifier = mesh.modifiers.new(name="RegionMask", type="MASK")
nodes_modifier = mesh.modifiers.new(name="RigAttributes", type="NODES")
nodes_group = bpy.data.node_groups.new(
    "RigPostprocessGeometryNodes",
    "GeometryNodeTree",
)
nodes_socket = nodes_group.interface.new_socket(
    name="Weight Attribute",
    in_out="INPUT",
    socket_type="NodeSocketFloat",
)
plain_string_socket = nodes_group.interface.new_socket(
    name="Unrelated Label",
    in_out="INPUT",
    socket_type="NodeSocketString",
)
nodes_modifier.node_group = nodes_group
nodes_input = getattr(nodes_modifier.properties.inputs, nodes_socket.identifier)
nodes_input.type = "ATTRIBUTE"
nodes_input.attribute_name = "bone_6"
plain_string_input = getattr(
    nodes_modifier.properties.inputs,
    plain_string_socket.identifier,
)
plain_string_input.value = "bone_5"

bone_attachment = bpy.data.objects.new("RigPostprocessBoneAttachment", None)
bpy.context.scene.collection.objects.link(bone_attachment)
bone_attachment.parent = armature
bone_attachment.parent_type = "BONE"
bone_attachment.parent_bone = "bone_5"
copy_location = bone_attachment.constraints.new(type="COPY_LOCATION")
copy_location.target = armature
copy_location.subtarget = "bone_9"

for index in range(len(rows)):
    group = mesh.vertex_groups.new(name=f"bone_{index}")
    group.add([index], 0.25 + index * 0.01, "REPLACE")
mask_modifier.vertex_group = "bone_5"

armature.select_set(False)
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh
assert rig_postprocess.resolve_postprocess_armature(bpy.context) is armature

# Renaming shared Armature data is rejected, and the IDs assigned while taking
# the snapshot are rolled back with the failed transaction.
shared_armature = bpy.data.objects.new("SharedRigPostprocessArmature", armature_data)
bpy.context.scene.collection.objects.link(shared_armature)
assert not any(
    rig_postprocess.JOINT_ID_PROP in bone for bone in armature.data.bones
)
try:
    rig_postprocess.rename_body_regions(
        bpy.context,
        armature,
        rig_postprocess.TEMPLATE_QUADRUPED,
    )
except rig_postprocess.RigPostprocessError as error:
    assert "共享" in str(error), error
else:
    raise AssertionError("shared Armature data should be rejected")
assert not any(
    rig_postprocess.JOINT_ID_PROP in bone for bone in armature.data.bones
)
bpy.data.objects.remove(shared_armature, do_unlink=True)

# An unmanaged artist group remains protected even when its name is a semantic
# target. Only groups recorded as SkinTokens-owned may be replaced.
artist_spine = mesh.vertex_groups.new(name="spine_1")
artist_spine.add([0], 0.95, "REPLACE")
try:
    rig_postprocess.rename_body_regions(
        bpy.context,
        armature,
        rig_postprocess.TEMPLATE_QUADRUPED,
    )
except rig_postprocess.RigPostprocessError as error:
    assert "已存在顶点组 spine_1" in str(error), error
else:
    raise AssertionError("unmanaged semantic target should remain protected")
assert abs(mesh.vertex_groups["spine_1"].weight(0) - 0.95) < 1e-6
assert armature.data.bones.get("bone_0") is not None
mesh.vertex_groups.remove(artist_spine)

plan = rig_postprocess.build_naming_plan(
    rig_postprocess.snapshot_armature(armature),
    rig_postprocess.TEMPLATE_QUADRUPED,
)
mapping = {
    assignment.old_name: assignment.target_name
    for assignment in plan.assignments
}
assert any(
    owner.as_pointer() == mask_modifier.as_pointer()
    and attribute == "vertex_group"
    and old_name == "bone_5"
    for owner, attribute, old_name in rig_postprocess._vertex_group_references(
        [mesh], mapping
    )
)
expected_weights = {
    mapping[f"bone_{index}"]: 0.25 + index * 0.01
    for index in range(len(rows))
    if f"bone_{index}" in mapping
}

result = rig_postprocess.rename_body_regions(
    bpy.context,
    armature,
    rig_postprocess.TEMPLATE_QUADRUPED,
)
assert result.renamed_bones == len(mapping), (result, mapping)
assert result.mirror_pairs == 4, result
assert rig_postprocess.semantic_mirror_pair_count(armature) == 4
assert not any(bone.name.startswith("bone_") for bone in armature.data.bones)

bone_names = {bone.name for bone in armature.data.bones}
expected_names = {
    f"leg_{group}_{side}_{segment}"
    for group in ("front", "hind")
    for side in ("l", "r")
    for segment in (1, 2)
}
assert expected_names.issubset(bone_names), (expected_names - bone_names, bone_names)
for target_name, expected_weight in expected_weights.items():
    group = mesh.vertex_groups.get(target_name)
    assert group is not None, target_name
    source_index = next(
        index
        for index in range(len(rows))
        if mapping.get(f"bone_{index}") == target_name
    )
    assert abs(group.weight(source_index) - expected_weight) < 1e-6
assert not any(group.name.startswith("bone_") for group in mesh.vertex_groups)
assert managed_vertex_group_names(mesh, armature) == set(mapping.values())
assert mask_modifier.vertex_group == mapping["bone_5"], (
    mask_modifier.vertex_group,
    mapping["bone_5"],
)
assert nodes_input.attribute_name == mapping["bone_6"]
assert plain_string_input.value == "bone_5"
assert bone_attachment.parent_bone == mapping["bone_5"]
assert copy_location.subtarget == mapping["bone_9"]
repeat_result = rig_postprocess.rename_body_regions(
    bpy.context,
    armature,
    rig_postprocess.TEMPLATE_QUADRUPED,
)
assert repeat_result.renamed_bones == 0, repeat_result
assert bpy.context.view_layer.objects.active is mesh

# Remove the naming pass' pair metadata to prove that selected-bone mirror
# alignment depends only on Edit Mode geometry and hierarchy.
for data_bone in armature.data.bones:
    for key in (
        rig_postprocess.SEMANTIC_MIRROR_ID_PROP,
        rig_postprocess.SEMANTIC_SIDE_PROP,
    ):
        if key in data_bone:
            del data_bone[key]
armature.data[rig_postprocess.SEMANTIC_VERSION_PROP] = 0
assert rig_postprocess.semantic_mirror_pair_count(armature) == 0

# Introduce a visible asymmetric edit before exercising the axis-aligned,
# selection-driven API. The complete connected front-limb chains are selected.
mirror_axis = rig_postprocess.MIRROR_AXIS_X
mirror_axis_index = 0
mirror_center = 0.0
mirror_distance_mode = rig_postprocess.MIRROR_DISTANCE_AVERAGE
axis_vector = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
hierarchy_before = {
    str(bone.name): (
        "" if bone.parent is None else str(bone.parent.name),
        bool(bone.use_connect),
    )
    for bone in armature.data.bones
}
mesh.select_set(False)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
edit_bones = armature.data.edit_bones
for edit_bone in edit_bones:
    edit_bone.head = inverse_transform(edit_bone.head)
    edit_bone.tail = inverse_transform(edit_bone.tail)
for edit_bone in edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
selected_names = {
    "leg_front_l_1",
    "leg_front_l_2",
    "leg_front_r_1",
    "leg_front_r_2",
}
for name in selected_names:
    edit_bones[name].select = True

positive_root = edit_bones["leg_front_l_1"]
negative_child = edit_bones["leg_front_r_2"]
positive_root.head = tuple(
    np.asarray(positive_root.head)
    + axis_vector * 0.35
    + np.asarray((0.0, 0.0, 0.2))
)
negative_child.tail = tuple(
    np.asarray(negative_child.tail)
    - axis_vector * 0.15
    + np.asarray((0.0, 0.0, -0.3))
)
selection_before = {
    str(edit_bone.name) for edit_bone in edit_bones if edit_bone.select
}
assert selection_before == selected_names

mirror_result = rig_postprocess.mirror_align_selected_edit_bones(
    bpy.context,
    armature,
    axis=mirror_axis,
    center=mirror_center,
    distance_mode=mirror_distance_mode,
)
assert mirror_result.mirror_pairs == 2, mirror_result
assert abs(mirror_result.center - mirror_center) < 1e-7
assert mirror_result.skipped_bones == (), mirror_result
assert bpy.context.object is armature
assert bpy.context.mode == "EDIT_ARMATURE"
assert armature.mode == "EDIT"
assert {
    str(edit_bone.name) for edit_bone in edit_bones if edit_bone.select
} == selection_before

expected_pairs = {
    frozenset(("leg_front_l_1", "leg_front_r_1")),
    frozenset(("leg_front_l_2", "leg_front_r_2")),
}
assert {
    frozenset(pair) for pair in mirror_result.paired_bones
} == expected_pairs, mirror_result
for positive_name, negative_name in mirror_result.paired_bones:
    positive = edit_bones[positive_name]
    negative = edit_bones[negative_name]
    for positive_point, negative_point in (
        (np.asarray(positive.head), np.asarray(negative.head)),
        (np.asarray(positive.tail), np.asarray(negative.tail)),
    ):
        positive_coordinate = float(positive_point[mirror_axis_index])
        negative_coordinate = float(negative_point[mirror_axis_index])
        assert abs(
            (positive_coordinate + negative_coordinate) * 0.5 - mirror_center
        ) < 1e-5
        positive_tangent = np.delete(positive_point, mirror_axis_index)
        negative_tangent = np.delete(negative_point, mirror_axis_index)
        np.testing.assert_allclose(
            positive_tangent,
            negative_tangent,
            atol=1e-5,
        )

hierarchy_after = {
    str(edit_bone.name): (
        "" if edit_bone.parent is None else str(edit_bone.parent.name),
        bool(edit_bone.use_connect),
    )
    for edit_bone in edit_bones
}
assert hierarchy_after == hierarchy_before
for edit_bone in edit_bones:
    if not edit_bone.use_connect:
        continue
    assert edit_bone.parent is not None
    assert (edit_bone.head - edit_bone.parent.tail).length < 1e-6

# Generated rigs often keep contiguous chain joints at identical coordinates
# while leaving use_connect disabled. Aligning only the roots must still move
# the unselected child Heads with those logical joints.
for edit_bone in edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
virtual_chains = (
    ("leg_front_l_1", "leg_front_l_2"),
    ("leg_front_r_1", "leg_front_r_2"),
)
for root_name, child_name in virtual_chains:
    root = edit_bones[root_name]
    child = edit_bones[child_name]
    child.use_connect = False
    child.head = root.tail

positive_root = edit_bones["leg_front_l_1"]
positive_child = edit_bones["leg_front_l_2"]
positive_root.tail = tuple(
    np.asarray(positive_root.tail) + np.asarray((0.08, 0.04, 0.03))
)
positive_child.head = positive_root.tail
child_tails_before = {
    child_name: np.asarray(edit_bones[child_name].tail).copy()
    for _root_name, child_name in virtual_chains
}
root_names = {root_name for root_name, _child_name in virtual_chains}
for root_name in root_names:
    edit_bones[root_name].select = True

virtual_result = rig_postprocess.mirror_align_selected_edit_bones(
    bpy.context,
    armature,
    axis=mirror_axis,
    center=mirror_center,
    distance_mode=mirror_distance_mode,
)
assert virtual_result.mirror_pairs == 1, virtual_result
assert virtual_result.skipped_bones == (), virtual_result
assert {
    str(edit_bone.name) for edit_bone in edit_bones if edit_bone.select
} == root_names
for root_name, child_name in virtual_chains:
    root = edit_bones[root_name]
    child = edit_bones[child_name]
    assert not child.use_connect
    assert (root.tail - child.head).length < 1e-6
    np.testing.assert_allclose(
        child.tail,
        child_tails_before[child_name],
        atol=1e-7,
    )

# A single highlighted bone also means the whole armature, while preserving
# that visible selection after alignment.
for edit_bone in edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
single_selected_name = "leg_front_l_1"
edit_bones[single_selected_name].select = True
single_selection_result = rig_postprocess.mirror_align_bones(
    bpy.context,
    armature,
    axis=mirror_axis,
    center=mirror_center,
    distance_mode=mirror_distance_mode,
)
assert {
    frozenset(pair) for pair in single_selection_result.paired_bones
} == expected_pairs, single_selection_result
assert {
    str(edit_bone.name) for edit_bone in edit_bones if edit_bone.select
} == {single_selected_name}

# An empty Edit Mode selection means the whole armature, while leaving the
# visible selection empty after the operation.
for edit_bone in edit_bones:
    edit_bone.select = False
    edit_bone.select_head = False
    edit_bone.select_tail = False
empty_selection_result = rig_postprocess.mirror_align_bones(
    bpy.context,
    armature,
    axis=mirror_axis,
    center=mirror_center,
    distance_mode=mirror_distance_mode,
)
assert {
    frozenset(pair) for pair in empty_selection_result.paired_bones
} == expected_pairs, empty_selection_result
assert bpy.context.mode == "EDIT_ARMATURE"
assert not any(edit_bone.select for edit_bone in edit_bones)

# Outside Edit Mode, the public API treats the entire session armature as the
# selection and restores the user's active object, object selection, and mode.
bpy.ops.object.mode_set(mode="OBJECT")
for selected_object in list(bpy.context.selected_objects):
    selected_object.select_set(False)
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh
object_selection_before = {
    selected_object.name for selected_object in bpy.context.selected_objects
}
all_result = rig_postprocess.mirror_align_bones(
    bpy.context,
    armature,
    axis=mirror_axis,
    center=mirror_center,
    distance_mode=mirror_distance_mode,
)
assert {
    frozenset(pair) for pair in all_result.paired_bones
} == expected_pairs, all_result
assert bpy.context.object is mesh
assert bpy.context.mode == "OBJECT"
assert {
    selected_object.name for selected_object in bpy.context.selected_objects
} == object_selection_before
try:
    rig_postprocess.mirror_align_bones(
        bpy.context,
        armature,
        axis=mirror_axis,
        center=100.0,
        distance_mode=mirror_distance_mode,
    )
except rig_postprocess.RigPostprocessError:
    pass
else:
    raise AssertionError("invalid full-armature alignment should fail")
assert bpy.context.object is mesh
assert bpy.context.mode == "OBJECT"
assert {
    selected_object.name for selected_object in bpy.context.selected_objects
} == object_selection_before

mesh.select_set(False)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="POSE")
for pose_bone in armature.pose.bones:
    pose_bone.select = False
pose_selection_name = "leg_front_l_1"
armature.pose.bones[pose_selection_name].select = True
armature.data.bones.active = armature.data.bones[pose_selection_name]
pose_result = rig_postprocess.mirror_align_bones(
    bpy.context,
    armature,
    axis=mirror_axis,
    center=mirror_center,
    distance_mode=mirror_distance_mode,
)
assert pose_result.mirror_pairs == all_result.mirror_pairs, pose_result
assert bpy.context.object is armature
assert bpy.context.mode == "POSE"
assert {
    pose_bone.name for pose_bone in armature.pose.bones if pose_bone.select
} == {pose_selection_name}
assert armature.data.bones.active.name == pose_selection_name

print("SKINTOKENS_RIG_POSTPROCESS_SMOKE_OK")
