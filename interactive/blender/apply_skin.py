from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _bpy():
    import bpy  # type: ignore

    return bpy


def _mathutils():
    import mathutils  # type: ignore

    return mathutils


def parse_skin_txt(path: Path) -> dict[int, list[tuple[str, float]]]:
    rows: dict[int, list[tuple[str, float]]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "skin":
            continue
        vertex_id = int(parts[1])
        pairs = []
        for i in range(2, len(parts), 2):
            if i + 1 >= len(parts):
                break
            pairs.append((parts[i], float(parts[i + 1])))
        rows[vertex_id] = pairs
    return rows


def obj_to_blender_point(x: float, y: float, z: float) -> list[float]:
    return [x, -z, y]


def parse_rig_txt(path: str | Path) -> tuple[list[list[float]], list[int], list[str]]:
    resolved = Path(path).expanduser().resolve()
    joints_by_name: dict[str, list[float]] = {}
    parents_by_child: dict[str, str] = {}
    root_name: str | None = None

    for line in resolved.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "joints":
            if len(parts) < 5:
                raise ValueError(f"invalid joints line: {line}")
            joints_by_name[parts[1]] = obj_to_blender_point(
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        elif parts[0] == "root":
            if len(parts) < 2:
                raise ValueError(f"invalid root line: {line}")
            root_name = parts[1]
        elif parts[0] == "hier":
            if len(parts) < 3:
                raise ValueError(f"invalid hier line: {line}")
            parents_by_child[parts[2]] = parts[1]

    if not joints_by_name:
        raise ValueError(f"rig txt has no joints: {resolved}")
    if root_name is None:
        root_candidates = set(joints_by_name) - set(parents_by_child)
        if len(root_candidates) != 1:
            raise ValueError(f"rig txt must provide one root: {resolved}")
        root_name = next(iter(root_candidates))
    if root_name not in joints_by_name:
        raise ValueError(f"root joint {root_name!r} is not declared in {resolved}")

    children_by_parent: dict[str, list[str]] = {name: [] for name in joints_by_name}
    for child, parent in parents_by_child.items():
        if child in joints_by_name and parent in joints_by_name:
            children_by_parent[parent].append(child)

    names: list[str] = []
    parents: list[int] = []

    def visit(name: str, parent_id: int) -> None:
        current_id = len(names)
        names.append(name)
        parents.append(parent_id)
        for child in children_by_parent.get(name, []):
            visit(child, current_id)

    visit(root_name, -1)
    if len(names) != len(joints_by_name):
        missing = sorted(set(joints_by_name) - set(names))
        raise ValueError(f"rig txt has joints disconnected from root {root_name!r}: {missing}")

    joints = [joints_by_name[name] for name in names]
    return joints, parents, names


def _tail_for_joint(joints, parents, joint_id: int):
    children = [idx for idx, parent in enumerate(parents) if int(parent) == joint_id]
    head = joints[joint_id]
    if children:
        return joints[children[0]]
    parent = int(parents[joint_id])
    if parent >= 0:
        p = joints[parent]
        direction = [head[i] - p[i] for i in range(3)]
        length = sum(v * v for v in direction) ** 0.5
        if length > 1e-8:
            return [head[i] + direction[i] * 0.35 for i in range(3)]
    return [head[0], head[1], head[2] + 0.1]


def armature_for_mesh(mesh_obj):
    if mesh_obj is None or mesh_obj.type != "MESH":
        return None
    for modifier in mesh_obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object is not None:
            return modifier.object
    parent = mesh_obj.parent
    if parent is not None and parent.type == "ARMATURE":
        return parent
    return None


def parse_armature_object(
    armature_obj,
    keep_bone_names: set[str] | None = None,
) -> tuple[list[list[float]], list[int], list[str]]:
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise ValueError("mesh has no armature")

    roots = [bone for bone in armature_obj.data.bones if bone.parent is None]
    if len(roots) != 1:
        raise ValueError(f"expected one FBX armature root, found {len(roots)}")

    ordered_bones = []

    def visit(bone) -> None:
        ordered_bones.append(bone)
        for child in bone.children:
            visit(child)

    visit(roots[0])
    if len(ordered_bones) != len(armature_obj.data.bones):
        raise ValueError("FBX armature has bones disconnected from root")

    if keep_bone_names is not None:
        keep_bone_names = {str(name) for name in keep_bone_names}
        root_name = str(roots[0].name)
        if root_name not in keep_bone_names:
            keep_bone_names.add(root_name)
        ordered_bones = [bone for bone in ordered_bones if bone.name in keep_bone_names]
        if not ordered_bones:
            raise ValueError("FBX armature has no bones matching mesh vertex groups")

    old_to_new = {bone.name: idx for idx, bone in enumerate(ordered_bones)}
    joints = []
    parents = []
    names = []
    for bone in ordered_bones:
        head = armature_obj.matrix_world @ bone.head_local
        joints.append([float(head.x), float(head.y), float(head.z)])
        parent = bone.parent
        while parent is not None and parent.name not in old_to_new:
            parent = parent.parent
        parents.append(-1 if parent is None else old_to_new[parent.name])
        names.append(str(bone.name))
    return joints, parents, names


def parse_mesh_armature(mesh_obj) -> tuple[list[list[float]], list[int], list[str]]:
    armature_obj = armature_for_mesh(mesh_obj)
    keep_names = {
        str(group.name)
        for group in getattr(mesh_obj, "vertex_groups", [])
        if armature_obj is not None and group.name in armature_obj.data.bones
    }
    return parse_armature_object(armature_obj, keep_bone_names=keep_names or None)


def create_armature(
    mesh_obj,
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
):
    bpy = _bpy()
    mathutils = _mathutils()
    if not joints:
        return None

    mesh_world = mesh_obj.matrix_world.copy()
    old_skintokens_armatures = []
    old_foreign_armatures = []
    for modifier in list(mesh_obj.modifiers):
        if modifier.type != "ARMATURE":
            continue
        if modifier.object is not None and "SkinTokensArmature" in modifier.object.name:
            old_skintokens_armatures.append(modifier.object)
        elif modifier.object is not None:
            old_foreign_armatures.append(modifier.object)
        mesh_obj.modifiers.remove(modifier)
    if mesh_obj.parent is not None and mesh_obj.parent.type == "ARMATURE":
        old_foreign_armatures.append(mesh_obj.parent)
        mesh_obj.parent = None
        mesh_obj.matrix_world = mesh_world
    for old_armature in old_skintokens_armatures:
        if old_armature.name in bpy.data.objects:
            bpy.data.objects.remove(old_armature, do_unlink=True)
    for old_armature in old_foreign_armatures:
        if old_armature.name in bpy.data.objects:
            old_armature.hide_set(True)
            old_armature.hide_viewport = True

    armature_data = bpy.data.armatures.new(f"{mesh_obj.name}_SkinTokensArmature")
    armature_obj = bpy.data.objects.new(armature_data.name, armature_data)
    bpy.context.scene.collection.objects.link(armature_obj)
    armature_obj.matrix_world = mesh_world
    armature_obj.show_in_front = True
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.ops.object.mode_set.poll() else None
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_data.edit_bones
    world_to_armature = armature_obj.matrix_world.inverted()
    for idx, joint in enumerate(joints):
        name = joint_names[idx] if idx < len(joint_names) else f"bone_{idx}"
        bone = edit_bones.new(name)
        head = world_to_armature @ mathutils.Vector(tuple(float(v) for v in joint))
        tail = world_to_armature @ mathutils.Vector(
            tuple(float(v) for v in _tail_for_joint(joints, parents, idx))
        )
        bone.head = head
        bone.tail = tail
        if sum((bone.tail[i] - bone.head[i]) ** 2 for i in range(3)) < 1e-10:
            bone.tail = (bone.head.x, bone.head.y, bone.head.z + 0.1)

    for idx, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            continue
        name = joint_names[idx] if idx < len(joint_names) else f"bone_{idx}"
        parent_name = joint_names[parent] if parent < len(joint_names) else f"bone_{parent}"
        if name in edit_bones and parent_name in edit_bones:
            edit_bones[name].parent = edit_bones[parent_name]
            edit_bones[name].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    modifier = mesh_obj.modifiers.get("SkinTokensArmature")
    if modifier is None:
        modifier = mesh_obj.modifiers.new("SkinTokensArmature", "ARMATURE")
    modifier.object = armature_obj
    mesh_obj.select_set(True)
    return armature_obj


def import_skin_output(
    path: str | Path,
    mesh_object_name: str | None = None,
    joints: Sequence[Sequence[float]] | None = None,
    parents: Sequence[int] | None = None,
    joint_names: Sequence[str] | None = None,
) -> None:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    if mesh_object_name is None:
        return

    bpy = _bpy()
    obj = bpy.data.objects.get(mesh_object_name)
    if obj is None:
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")
    if obj.type != "MESH":
        raise RuntimeError(f"object is not a mesh: {mesh_object_name}")

    rows = parse_skin_txt(resolved)
    skin_names = sorted({name for weights in rows.values() for name, _ in weights})
    for group in list(obj.vertex_groups):
        if group.name in skin_names or group.name.startswith("bone_"):
            obj.vertex_groups.remove(group)
    group_by_name = {group.name: group for group in obj.vertex_groups}
    for vertex_id, weights in rows.items():
        if vertex_id >= len(obj.data.vertices):
            continue
        for name, weight in weights:
            group = group_by_name.get(name)
            if group is None:
                group = obj.vertex_groups.new(name=name)
                group_by_name[name] = group
            group.add([vertex_id], float(weight), "REPLACE")

    if joints is not None and parents is not None:
        if joint_names is None:
            joint_names = [f"bone_{i}" for i in range(len(joints))]
        create_armature(obj, joints, parents, joint_names)
