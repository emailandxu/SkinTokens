"""Run with Blender after opening name_skeleton_exist_vertex_group.blend."""

from __future__ import annotations

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


def weights(mesh, group_name: str) -> np.ndarray:
    group = mesh.vertex_groups.get(group_name)
    assert group is not None, group_name
    result = np.zeros(len(mesh.data.vertices), dtype=np.float64)
    for vertex in mesh.data.vertices:
        try:
            result[vertex.index] = float(group.weight(vertex.index))
        except RuntimeError:
            pass
    return result


armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
assert len(armatures) == 1, [obj.name for obj in armatures]
assert len(meshes) == 1, [obj.name for obj in meshes]
armature = armatures[0]
mesh = meshes[0]

current_spine_weights = weights(mesh, "bone_41")
stale_spine_weights = weights(mesh, "spine_1")
assert not np.allclose(current_spine_weights, stale_spine_weights)
assert "head_7" in mesh.vertex_groups
assert "head_8" in mesh.vertex_groups

for obj in bpy.context.selected_objects:
    obj.select_set(False)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
result = rig_postprocess.rename_body_regions(
    bpy.context,
    armature,
    rig_postprocess.TEMPLATE_QUADRUPED,
)

assert result.renamed_bones == 39, result
np.testing.assert_allclose(
    weights(mesh, "spine_1"),
    current_spine_weights,
    atol=1e-7,
)
assert mesh.vertex_groups.get("head_7") is None
assert mesh.vertex_groups.get("head_8") is None
assert not any(group.name.startswith("bone_") for group in mesh.vertex_groups)
assert not any(group.name.endswith(".001") for group in mesh.vertex_groups)
managed = managed_vertex_group_names(mesh, armature)
assert "spine_1" in managed
assert "head_7" not in managed
assert managed == {
    group.name
    for group in mesh.vertex_groups
    if armature.data.bones.get(group.name) is not None
}
print("SKINTOKENS_EXISTING_VERTEX_GROUPS_SMOKE_OK")
