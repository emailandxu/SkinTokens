from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


JOINT_ID_PROP = "skintokens_joint_id"
PARENT_ID_PROP = "skintokens_parent_id"
DFS_ORDER_PROP = "skintokens_dfs_order"
MANAGED_BONE_PROP = "skintokens_managed_bone"
CREATED_SESSION_PROP = "skintokens_created_session"
MANAGED_VERTEX_GROUPS_PROP = "skintokens_managed_vertex_groups"

_LEGACY_SEMANTIC_GROUP_PATTERN = re.compile(
    r"^(?:spine|head|tail)_\d+$"
    r"|^(?:arm|leg)_[lr]_\d+$"
    r"|^leg_(?:front|hind)_[lr]_\d+$",
    re.IGNORECASE,
)
_LEGACY_GENERIC_GROUP_PATTERN = re.compile(r"^bone_\d+$", re.IGNORECASE)
_CURRENT_VERTEX_GROUP_WEIGHTS = object()


@dataclass(frozen=True)
class VertexGroupWeightSnapshot:
    weights: tuple[float, ...]
    managed: bool


def _bpy():
    import bpy  # type: ignore

    return bpy


def _mathutils():
    import mathutils  # type: ignore

    return mathutils


def managed_vertex_group_names(mesh_obj, armature_obj=None) -> set[str]:
    if mesh_obj is None or getattr(mesh_obj, "type", "") != "MESH":
        return set()
    if MANAGED_VERTEX_GROUPS_PROP in mesh_obj:
        raw = mesh_obj.get(MANAGED_VERTEX_GROUPS_PROP, "[]")
        try:
            decoded = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return set()
        if not isinstance(decoded, list):
            return set()
        return {
            str(name)
            for name in decoded
            if isinstance(name, str) and name
        }

    if armature_obj is None:
        armature_obj = armature_for_mesh(mesh_obj)
    if armature_obj is None or getattr(armature_obj, "type", "") != "ARMATURE":
        return set()

    group_names = {str(group.name) for group in mesh_obj.vertex_groups}
    bone_names = {str(bone.name) for bone in armature_obj.data.bones}
    managed_bone_names = {
        str(bone.name)
        for bone in armature_obj.data.bones
        if bool(bone.get(MANAGED_BONE_PROP, False))
    }
    managed = {
        name
        for name in group_names
        if name in managed_bone_names
        or (
            bool(managed_bone_names)
            and _LEGACY_GENERIC_GROUP_PATTERN.fullmatch(name)
        )
    }
    if int(armature_obj.data.get("skintokens_semantic_version", 0)) > 0:
        managed.update(
            name
            for name in group_names
            if _LEGACY_SEMANTIC_GROUP_PATTERN.fullmatch(name)
            and (name in bone_names or bool(managed_bone_names))
        )
    return managed


def set_managed_vertex_group_names(mesh_obj, names: Sequence[str]) -> None:
    unique = sorted({str(name) for name in names if str(name)})
    mesh_obj[MANAGED_VERTEX_GROUPS_PROP] = json.dumps(
        unique,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _prepare_managed_vertex_groups(mesh_obj, names: Sequence[str]) -> dict[str, object]:
    current_names = tuple(dict.fromkeys(str(name) for name in names))
    current = set(current_names)
    previous = managed_vertex_group_names(mesh_obj)
    for group in list(mesh_obj.vertex_groups):
        if str(group.name) in previous - current:
            mesh_obj.vertex_groups.remove(group)

    vertex_ids = list(range(len(mesh_obj.data.vertices)))
    groups = {}
    for name in current_names:
        group = mesh_obj.vertex_groups.get(name)
        if group is None:
            group = mesh_obj.vertex_groups.new(name=name)
        if vertex_ids:
            group.remove(vertex_ids)
        groups[name] = group
    return groups


def capture_vertex_group_weights(
    mesh_object_name: str,
    group_name: str,
) -> VertexGroupWeightSnapshot | None:
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")
    group_name = str(group_name)
    group = mesh_obj.vertex_groups.get(group_name)
    if group is None:
        return None
    weights = []
    for vertex in mesh_obj.data.vertices:
        try:
            weight = max(0.0, float(group.weight(int(vertex.index))))
        except RuntimeError:
            weight = 0.0
        weights.append(weight)
    return VertexGroupWeightSnapshot(
        weights=tuple(weights),
        managed=group_name in managed_vertex_group_names(mesh_obj),
    )


def transfer_managed_vertex_group_weights(
    mesh_object_name: str,
    source_name: str,
    target_name: str | None,
    *,
    snapshot: VertexGroupWeightSnapshot | None | object = (
        _CURRENT_VERTEX_GROUP_WEIGHTS
    ),
) -> bool:
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")

    source_name = str(source_name)
    target_name = None if target_name is None else str(target_name)
    managed = managed_vertex_group_names(mesh_obj)
    if target_name == source_name:
        return False

    source = mesh_obj.vertex_groups.get(source_name)
    if snapshot is _CURRENT_VERTEX_GROUP_WEIGHTS:
        snapshot = capture_vertex_group_weights(mesh_object_name, source_name)
    if snapshot is None:
        return False
    if not isinstance(snapshot, VertexGroupWeightSnapshot):
        raise TypeError("invalid vertex-group weight snapshot")
    if len(snapshot.weights) != len(mesh_obj.data.vertices):
        raise RuntimeError(
            f"mesh vertex count changed while deleting {source_name}: "
            f"{len(snapshot.weights)} -> {len(mesh_obj.data.vertices)}"
        )
    source_was_managed = bool(snapshot.managed)

    if target_name is not None:
        target = mesh_obj.vertex_groups.get(target_name)
        if target is None:
            target = mesh_obj.vertex_groups.new(name=target_name)

        for vertex, source_weight in zip(mesh_obj.data.vertices, snapshot.weights):
            try:
                target_weight = max(
                    0.0,
                    float(target.weight(int(vertex.index))),
                )
            except RuntimeError:
                target_weight = 0.0
            if source_weight > 0.0:
                target.add(
                    [int(vertex.index)],
                    min(1.0, source_weight + target_weight),
                    "REPLACE",
                )

    if source is not None:
        mesh_obj.vertex_groups.remove(source)
    managed.discard(source_name)
    if target_name is not None and source_was_managed:
        managed.add(target_name)
    set_managed_vertex_group_names(mesh_obj, managed)
    mesh_obj.data.update()
    return target_name is not None


def mesh_object_names_for_armature(armature_object_name: str | None) -> tuple[str, ...]:
    armature_obj = armature_by_name(armature_object_name)
    if armature_obj is None:
        return ()
    result = []
    for obj in _bpy().data.objects:
        if obj.type != "MESH":
            continue
        uses_modifier = any(
            modifier.type == "ARMATURE" and modifier.object is armature_obj
            for modifier in obj.modifiers
        )
        uses_parent = (
            obj.parent is armature_obj
            and str(getattr(obj, "parent_type", "")) == "ARMATURE"
        )
        if uses_modifier or uses_parent:
            result.append(str(obj.name))
    return tuple(result)


def validate_vertex_group_transfer(
    mesh_object_names: Sequence[str],
    source_name: str,
) -> None:
    bpy = _bpy()
    for mesh_object_name in mesh_object_names:
        mesh_obj = bpy.data.objects.get(str(mesh_object_name))
        if mesh_obj is None or mesh_obj.type != "MESH":
            raise RuntimeError(f"mesh object not found: {mesh_object_name}")
        if mesh_obj.vertex_groups.get(str(source_name)) is None:
            continue
        if not bool(getattr(mesh_obj, "is_editable", True)) or not bool(
            getattr(mesh_obj.data, "is_editable", True)
        ):
            raise RuntimeError(
                f"mesh {mesh_object_name} is linked or read-only; "
                f"cannot transfer weights for {source_name}"
            )


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


def armature_by_name(name: str | None):
    if not name:
        return None
    armature_obj = _bpy().data.objects.get(name)
    if armature_obj is None or armature_obj.type != "ARMATURE":
        return None
    return armature_obj


def _bone_by_name(collection, bone_name: str):
    get = getattr(collection, "get", None)
    if get is not None:
        return get(bone_name)
    return next(
        (bone for bone in collection if str(bone.name) == bone_name),
        None,
    )


def _pose_bone_selected(armature_obj, bone_name: str) -> bool:
    data_bone = _bone_by_name(armature_obj.data.bones, bone_name)
    if data_bone is not None and hasattr(data_bone, "select"):
        return bool(data_bone.select)
    pose = getattr(armature_obj, "pose", None)
    pose_bone = None if pose is None else _bone_by_name(pose.bones, bone_name)
    return bool(pose_bone is not None and getattr(pose_bone, "select", False))


def _set_pose_bone_selected(armature_obj, bone_name: str, selected: bool) -> None:
    data_bone = _bone_by_name(armature_obj.data.bones, bone_name)
    if data_bone is not None and hasattr(data_bone, "select"):
        data_bone.select = bool(selected)
        return
    pose = getattr(armature_obj, "pose", None)
    pose_bone = None if pose is None else _bone_by_name(pose.bones, bone_name)
    if pose_bone is not None and hasattr(pose_bone, "select"):
        pose_bone.select = bool(selected)


def _bone_point(bone, attribute: str) -> tuple[float, float, float]:
    value = getattr(bone, attribute)
    return (float(value.x), float(value.y), float(value.z))


def capture_armature_bone_states(
    armature_obj,
) -> dict[str, tuple[str, str, tuple[float, float, float], tuple[float, float, float]]]:
    """Capture edit-relevant state keyed by a persistent joint id."""
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise ValueError("mesh has no armature")
    edit_mode = armature_obj.mode == "EDIT"
    source_bones = (
        list(armature_obj.data.edit_bones)
        if edit_mode
        else list(armature_obj.data.bones)
    )
    result = {}
    for bone in source_bones:
        joint_id = str(bone.get(JOINT_ID_PROP, "")) or uuid.uuid4().hex
        bone[JOINT_ID_PROP] = joint_id
        parent = bone.parent
        parent_id = ""
        if parent is not None:
            parent_id = str(parent.get(JOINT_ID_PROP, "")) or uuid.uuid4().hex
            parent[JOINT_ID_PROP] = parent_id
        result[joint_id] = (
            str(bone.name),
            parent_id,
            _bone_point(bone, "head" if edit_mode else "head_local"),
            _bone_point(bone, "tail" if edit_mode else "tail_local"),
        )
    return result


def parse_armature_object(armature_obj) -> tuple[list[list[float]], list[int], list[str]]:
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise ValueError("mesh has no armature")

    source_bones = (
        list(armature_obj.data.edit_bones)
        if armature_obj.mode == "EDIT"
        else list(armature_obj.data.bones)
    )
    if not source_bones:
        return [], [], []

    fallback_order = {bone.name: index for index, bone in enumerate(source_bones)}

    def bone_order(bone) -> int:
        try:
            return int(bone.get(DFS_ORDER_PROP, fallback_order[bone.name]))
        except (AttributeError, TypeError, ValueError):
            return fallback_order[bone.name]

    roots = [bone for bone in source_bones if bone.parent is None]
    if len(roots) != 1:
        raise ValueError(f"expected one FBX armature root, found {len(roots)}")

    ordered_bones = []
    source_names = {bone.name for bone in source_bones}

    def visit(bone) -> None:
        ordered_bones.append(bone)
        children = [child for child in bone.children if child.name in source_names]
        for child in sorted(children, key=bone_order):
            visit(child)

    visit(roots[0])
    if len(ordered_bones) != len(source_bones):
        raise ValueError("FBX armature has bones disconnected from root")

    old_to_new = {bone.name: idx for idx, bone in enumerate(ordered_bones)}
    joints = []
    parents = []
    names = []
    for bone in ordered_bones:
        local_head = bone.head if armature_obj.mode == "EDIT" else bone.head_local
        head = armature_obj.matrix_world @ local_head
        joints.append([float(head.x), float(head.y), float(head.z)])
        parent = bone.parent
        while parent is not None and parent.name not in old_to_new:
            parent = parent.parent
        parents.append(-1 if parent is None else old_to_new[parent.name])
        names.append(str(bone.name))
    return joints, parents, names


def ensure_mesh_armature(
    mesh_object_name: str,
    armature_object_name: str | None = None,
):
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")

    armature_obj = armature_by_name(armature_object_name)
    if armature_object_name and armature_obj is None:
        raise RuntimeError(f"armature object not found: {armature_object_name}")
    if armature_obj is None:
        armature_obj = armature_for_mesh(mesh_obj)
    if armature_obj is None:
        armature_data = bpy.data.armatures.new(
            f"{mesh_obj.name}_SkinTokensArmature"
        )
        armature_obj = bpy.data.objects.new(armature_data.name, armature_data)
        bpy.context.scene.collection.objects.link(armature_obj)
        armature_obj.matrix_world = mesh_obj.matrix_world.copy()
        armature_obj.show_in_front = True

    modifier = next(
        (
            item
            for item in mesh_obj.modifiers
            if item.type == "ARMATURE" and item.object is armature_obj
        ),
        None,
    )
    if modifier is None:
        modifier = mesh_obj.modifiers.get("SkinTokensArmature")
        if modifier is None or modifier.type != "ARMATURE":
            modifier = mesh_obj.modifiers.new("SkinTokensArmature", "ARMATURE")
        modifier.object = armature_obj
    return armature_obj


def active_armature_bone_name(armature_object_name: str | None) -> str | None:
    armature_obj = armature_by_name(armature_object_name)
    if armature_obj is None:
        return None
    bpy = _bpy()
    if bpy.context.object is not armature_obj:
        return None
    if armature_obj.mode == "EDIT":
        active = armature_obj.data.edit_bones.active
        if active is not None and (
            active.select or active.select_head or active.select_tail
        ):
            active.select = True
            active.select_head = True
            active.select_tail = True
    elif armature_obj.mode == "POSE":
        active = armature_obj.data.bones.active
    else:
        return None
    if active is None:
        return None
    if armature_obj.mode == "POSE" and not _pose_bone_selected(
        armature_obj,
        str(active.name),
    ):
        return None
    return str(active.name)


def promote_active_armature_bone_selection(
    armature_object_name: str | None,
) -> bool:
    armature_obj = armature_by_name(armature_object_name)
    if armature_obj is None or armature_obj.mode != "EDIT":
        return False
    bpy = _bpy()
    if bpy.context.object is not armature_obj:
        return False
    active = armature_obj.data.edit_bones.active
    if active is None or not (
        active.select or active.select_head or active.select_tail
    ):
        return False
    changed = not (active.select and active.select_head and active.select_tail)
    if not changed:
        return False
    active.select = True
    active.select_head = True
    active.select_tail = True
    return True


def apply_armature_context(
    armature_object_name: str,
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
    *,
    select_name: str | None = None,
    remove_missing: bool = False,
    mode_after: str | None = None,
    session_id: str | None = None,
    original_bone_ids: set[str] | None = None,
    tail_follow_bone_ids: set[str] | None = None,
    allow_remove_names: set[str] | None = None,
    allow_reparent_names: set[str] | None = None,
) -> None:
    bpy = _bpy()
    mathutils = _mathutils()
    armature_obj = armature_by_name(armature_object_name)
    if armature_obj is None:
        raise RuntimeError(f"armature object not found: {armature_object_name}")

    count = len(joints)
    names = [
        str(joint_names[index]) if index < len(joint_names) else f"bone_{index}"
        for index in range(count)
    ]
    if len(parents) != count or len(set(names)) != count:
        raise ValueError("armature context has invalid parent or bone-name counts")
    for index, parent in enumerate(parents):
        if int(parent) < -1 or int(parent) >= index:
            raise ValueError(f"bone {index} has invalid parent {int(parent)}")

    was_active = bpy.context.object is armature_obj
    previous_mode = armature_obj.mode if was_active else "OBJECT"
    active_before = None
    if was_active and previous_mode == "EDIT":
        active = armature_obj.data.edit_bones.active
        active_before = None if active is None else str(active.name)
    elif was_active and previous_mode == "POSE":
        active = armature_obj.data.bones.active
        active_before = None if active is None else str(active.name)
    selected_name = select_name or active_before
    final_mode = mode_after or previous_mode
    if final_mode not in {"OBJECT", "EDIT", "POSE"}:
        final_mode = "OBJECT"

    original_bone_ids = set(original_bone_ids or ())
    if tail_follow_bone_ids is None:
        tail_follow_bone_ids = set()
    allow_remove_names = set(allow_remove_names or ())
    allow_reparent_names = set(allow_reparent_names or ())
    if session_id is None:
        managed_before = {
            bone.name
            for bone in armature_obj.data.bones
            if bool(bone.get(MANAGED_BONE_PROP, False))
        }
    else:
        managed_before = {
            str(bone.name)
            for bone in armature_obj.data.bones
            if str(bone.get(JOINT_ID_PROP, "")) not in original_bone_ids
        }
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones = armature_obj.data.edit_bones
    existing_names = {str(bone.name) for bone in edit_bones}
    children_before = {
        str(bone.name): {str(child.name) for child in bone.children}
        for bone in edit_bones
    }
    original_names_before = {
        str(bone.name)
        for bone in edit_bones
        if str(bone.get(JOINT_ID_PROP, "")) in original_bone_ids
    }
    if session_id is None:
        non_model_managed_names = existing_names - managed_before
    else:
        non_model_managed_names = {
            str(bone.name)
            for bone in edit_bones
            if str(bone.get(JOINT_ID_PROP, "")) in original_bone_ids
        }
    if remove_missing:
        keep = set(names)
        blocked = sorted(
            non_model_managed_names - keep - allow_remove_names
        )
        if blocked:
            raise ValueError(
                "服务端结果不能删除会话开始前已有的骨骼："
                + ", ".join(blocked)
            )
        for bone in list(edit_bones):
            if bone.name not in keep:
                # Disconnect before removal so Blender does not move the
                # surviving parent's Tail as a side effect.
                bone.use_connect = False
                edit_bones.remove(bone)

    world_to_armature = armature_obj.matrix_world.inverted()
    created_names: set[str] = set()
    for index, (name, joint) in enumerate(zip(names, joints)):
        bone = edit_bones.get(name)
        if bone is None:
            bone = edit_bones.new(name)
            if bone.name != name:
                raise ValueError(f"generated bone name is not unique: {name}")
            created_names.add(name)
            bone[JOINT_ID_PROP] = uuid.uuid4().hex
            if session_id is not None:
                bone[CREATED_SESSION_PROP] = session_id
        if name in created_names or name in managed_before:
            # A connected child moves its parent's Tail when its Head changes.
            bone.use_connect = False
            head = world_to_armature @ mathutils.Vector(
                tuple(float(value) for value in joint)
            )
            bone.head = head
            if (bone.tail - bone.head).length_squared < 1e-10:
                bone.tail = bone.head + mathutils.Vector((0.0, 0.0, 0.1))

    for index, name in enumerate(names):
        bone = edit_bones[name]
        parent_index = int(parents[index])
        new_parent = None if parent_index < 0 else edit_bones[names[parent_index]]
        current_parent_name = None if bone.parent is None else str(bone.parent.name)
        new_parent_name = None if new_parent is None else str(new_parent.name)
        parent_changed = current_parent_name != new_parent_name
        can_reparent = (
            name in created_names
            or name in managed_before
            or name in allow_reparent_names
        )
        if parent_changed and not can_reparent:
            raise ValueError(
                f"服务端结果不能修改会话开始前已有的骨骼层级：{name}"
            )
        if can_reparent:
            if parent_changed:
                # A connected endpoint can move while its parent is replaced.
                bone.use_connect = False
            bone.parent = new_parent
        bone.select = False
        bone.select_head = False
        bone.select_tail = False

    children_by_parent = [[] for _ in names]
    for child_index, parent_index in enumerate(parents):
        parent_index = int(parent_index)
        if parent_index >= 0:
            children_by_parent[parent_index].append(child_index)

    # Session-created bones remain model-managed. An original leaf starts
    # following its first session child and keeps following that child's Head
    # for the rest of the session. Removing the child leaves the last Tail in
    # place rather than restoring a pre-session value.
    for index, name in enumerate(names):
        bone = edit_bones[name]
        bone_id = str(bone.get(JOINT_ID_PROP, ""))
        children = children_by_parent[index]
        child_names = {names[child_index] for child_index in children}
        was_original_leaf = (
            name in original_names_before and not children_before.get(name)
        )
        gained_first_child = bool(
            was_original_leaf
            and child_names.difference(children_before.get(name, set()))
        )
        if gained_first_child and bone_id:
            tail_follow_bone_ids.add(bone_id)
        follows_session_child = bool(
            children and bone_id in tail_follow_bone_ids
        )
        if (
            name not in created_names
            and name not in managed_before
            and not follows_session_child
        ):
            continue
        if children:
            tail = edit_bones[names[children[0]]].head.copy()
        else:
            parent_index = int(parents[index])
            if parent_index >= 0:
                parent_head = edit_bones[names[parent_index]].head
                direction = bone.head - parent_head
                if direction.length_squared > 1e-10:
                    tail = bone.head + direction * 0.35
                else:
                    tail = bone.head + mathutils.Vector((0.0, 0.0, 0.1))
            else:
                tail = bone.head + mathutils.Vector((0.0, 0.0, 0.1))
        bone.tail = tail
        if (bone.tail - bone.head).length_squared < 1e-10:
            bone.tail = bone.head + mathutils.Vector((0.0, 0.0, 0.1))
    if selected_name is not None and edit_bones.get(selected_name) is not None:
        selected = edit_bones[selected_name]
        selected.select = True
        selected.select_head = True
        selected.select_tail = True
        edit_bones.active = selected

    bpy.ops.object.mode_set(mode="OBJECT")
    for index, name in enumerate(names):
        bone = armature_obj.data.bones[name]
        joint_id = str(bone.get(JOINT_ID_PROP, "")) or uuid.uuid4().hex
        bone[JOINT_ID_PROP] = joint_id
        parent_index = int(parents[index])
        bone[PARENT_ID_PROP] = (
            ""
            if parent_index < 0
            else str(
                armature_obj.data.bones[names[parent_index]].get(
                    JOINT_ID_PROP,
                    names[parent_index],
                )
            )
        )
        bone[DFS_ORDER_PROP] = index
        if name in created_names:
            bone[MANAGED_BONE_PROP] = True
            if session_id is not None:
                bone[CREATED_SESSION_PROP] = session_id
        _set_pose_bone_selected(armature_obj, name, name == selected_name)
    if (
        selected_name is not None
        and armature_obj.data.bones.get(selected_name) is not None
    ):
        armature_obj.data.bones.active = armature_obj.data.bones[selected_name]

    if final_mode != "OBJECT":
        bpy.ops.object.mode_set(mode=final_mode)
        if (
            final_mode == "EDIT"
            and selected_name is not None
            and armature_obj.data.edit_bones.get(selected_name) is not None
        ):
            selected = armature_obj.data.edit_bones[selected_name]
            selected.select = True
            selected.select_head = True
            selected.select_tail = True
            armature_obj.data.edit_bones.active = selected
        elif (
            final_mode == "POSE"
            and selected_name is not None
            and armature_obj.data.bones.get(selected_name) is not None
        ):
            _set_pose_bone_selected(armature_obj, selected_name, True)
            armature_obj.data.bones.active = armature_obj.data.bones[selected_name]


def parse_mesh_armature(mesh_obj) -> tuple[list[list[float]], list[int], list[str]]:
    armature_obj = armature_for_mesh(mesh_obj)
    return parse_armature_object(armature_obj)


def armature_matches_context(
    mesh_obj,
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
    *,
    tolerance: float = 1e-4,
    armature_obj=None,
) -> bool:
    if armature_obj is None:
        armature_obj = armature_for_mesh(mesh_obj)
    if armature_obj is None:
        return False
    existing_joints, existing_parents, existing_names = parse_armature_object(armature_obj)
    if existing_names != [str(name) for name in joint_names]:
        return False
    if existing_parents != [int(parent) for parent in parents]:
        return False
    if len(existing_joints) != len(joints):
        return False
    tolerance_sq = float(tolerance) ** 2
    return all(
        sum(
            (float(existing[axis]) - float(expected[axis])) ** 2
            for axis in range(3)
        )
        <= tolerance_sq
        for existing, expected in zip(existing_joints, joints)
    )


def selected_skin_bone_names(mesh_object_name: str) -> list[str]:
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")

    active_obj = bpy.context.object
    if active_obj is mesh_obj:
        active_group = mesh_obj.vertex_groups.active
        return [] if active_group is None else [str(active_group.name)]

    armature_obj = armature_for_mesh(mesh_obj)
    if armature_obj is None:
        raise RuntimeError("the session mesh has no armature")
    if active_obj is not armature_obj:
        return []

    if armature_obj.mode == "EDIT":
        bones = armature_obj.data.edit_bones
        return [bone.name for bone in bones if bone.select]
    elif armature_obj.mode == "POSE":
        return [
            bone.name
            for bone in armature_obj.data.bones
            if _pose_bone_selected(armature_obj, bone.name)
        ]
    else:
        return []


def read_mesh_skin_weights(
    mesh_object_name: str,
    joint_names: Sequence[str],
) -> np.ndarray:
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")

    weights = np.zeros(
        (len(mesh_obj.data.vertices), len(joint_names)),
        dtype=np.float32,
    )
    group_to_bone = {}
    for bone_index, name in enumerate(joint_names):
        group = mesh_obj.vertex_groups.get(str(name))
        if group is not None:
            group_to_bone[group.index] = bone_index
    for vertex in mesh_obj.data.vertices:
        for assignment in vertex.groups:
            bone_index = group_to_bone.get(assignment.group)
            if bone_index is not None:
                weights[vertex.index, bone_index] = max(0.0, float(assignment.weight))
    return weights


def apply_mesh_skin_weights(
    mesh_object_name: str,
    joint_names: Sequence[str],
    skin: np.ndarray,
    *,
    eps: float = 1e-8,
) -> None:
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise RuntimeError(f"mesh object not found: {mesh_object_name}")

    weights = np.asarray(skin, dtype=np.float32)
    expected_shape = (len(mesh_obj.data.vertices), len(joint_names))
    if weights.shape != expected_shape:
        raise ValueError(f"skin shape {weights.shape} does not match {expected_shape}")

    names = [str(name) for name in joint_names]
    group_by_name = _prepare_managed_vertex_groups(mesh_obj, names)
    groups = [group_by_name[name] for name in names]

    for bone_index, group in enumerate(groups):
        nonzero = np.flatnonzero(weights[:, bone_index] > float(eps))
        for vertex_index in nonzero:
            group.add(
                [int(vertex_index)],
                float(weights[vertex_index, bone_index]),
                "REPLACE",
            )
    set_managed_vertex_group_names(mesh_obj, names)
    mesh_obj.data.update()


def _skin_items(
    row: np.ndarray,
    names: Sequence[str],
    *,
    topk: int,
    eps: float,
) -> list[tuple[str, float]]:
    clean = np.nan_to_num(
        np.asarray(row, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    clean[clean < 0.0] = 0.0
    if clean.sum() <= eps:
        clean[int(np.argmax(clean))] = 1.0
    count = min(int(topk), clean.shape[0]) if topk > 0 else clean.shape[0]
    indices = np.argsort(-clean)[:count]
    indices = indices[clean[indices] > eps]
    if indices.shape[0] == 0:
        indices = np.asarray([int(np.argmax(clean))])
    weights = clean[indices]
    weights /= max(float(weights.sum()), float(eps))
    return [(str(names[int(index)]), float(weight)) for index, weight in zip(indices, weights)]


def write_mesh_skin_txt(
    output_path: str | Path,
    *,
    mesh_object_name: str,
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
    blender_to_obj_text: np.ndarray,
    topk: int = 4,
    eps: float = 1e-8,
) -> Path:
    names = [str(name) for name in joint_names]
    joint_array = np.asarray(joints, dtype=np.float64)
    parent_array = np.asarray(parents, dtype=np.int64)
    matrix = np.asarray(blender_to_obj_text, dtype=np.float64)
    if joint_array.shape != (len(names), 3):
        raise ValueError("joint positions and names do not match")
    if parent_array.shape != (len(names),):
        raise ValueError("joint parents and names do not match")
    if matrix.shape != (4, 3):
        raise ValueError(f"Blender-to-OBJ transform must be 4x3, got {matrix.shape}")
    roots = np.flatnonzero(parent_array == -1)
    if roots.shape[0] != 1:
        raise ValueError(f"expected one skeleton root, found {roots.shape[0]}")
    if not names:
        raise ValueError("cannot export skin for an empty skeleton")

    homogeneous = np.concatenate(
        [joint_array, np.ones((joint_array.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    obj_joints = homogeneous @ matrix
    skin = read_mesh_skin_weights(mesh_object_name, names)

    lines = []
    for name, xyz in zip(names, obj_joints):
        lines.append(f"joints {name} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
    lines.append(f"root {names[int(roots[0])]}\n")
    for vertex_id, row in enumerate(skin):
        parts = [f"skin {vertex_id}"]
        for name, weight in _skin_items(row, names, topk=topk, eps=eps):
            parts.extend([name, f"{weight:.6f}"])
        lines.append(" ".join(parts) + "\n")
    for child, parent in enumerate(parent_array.tolist()):
        if parent != -1:
            lines.append(f"hier {names[int(parent)]} {names[child]}\n")
    lines.extend(
        [
            "info Scale 1.000000\n",
            "info Pivot 0.000000 0.000000 0.000000\n",
        ]
    )

    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("".join(lines), encoding="utf-8")
    return resolved


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
    for modifier in list(mesh_obj.modifiers):
        if modifier.type != "ARMATURE":
            continue
        if modifier.object is not None and "SkinTokensArmature" in modifier.object.name:
            old_skintokens_armatures.append(modifier.object)
        mesh_obj.modifiers.remove(modifier)
    if mesh_obj.parent is not None and mesh_obj.parent.type == "ARMATURE":
        mesh_obj.parent = None
        mesh_obj.matrix_world = mesh_world
    for old_armature in old_skintokens_armatures:
        if old_armature.name in bpy.data.objects:
            bpy.data.objects.remove(old_armature, do_unlink=True)
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
    armature_object_name: str | None = None,
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
    group_by_name = _prepare_managed_vertex_groups(obj, skin_names)
    for vertex_id, weights in rows.items():
        if vertex_id >= len(obj.data.vertices):
            continue
        for name, weight in weights:
            group = group_by_name.get(name)
            if group is None:
                group = obj.vertex_groups.new(name=name)
                group_by_name[name] = group
            group.add([vertex_id], float(weight), "REPLACE")
    set_managed_vertex_group_names(obj, skin_names)

    if joints is not None and parents is not None:
        if joint_names is None:
            joint_names = [f"bone_{i}" for i in range(len(joints))]
        source_armature = armature_by_name(armature_object_name)
        matches = armature_matches_context(
            obj,
            joints,
            parents,
            joint_names,
            armature_obj=source_armature,
        )
        if source_armature is not None:
            if not matches:
                raise RuntimeError(
                    "working armature no longer matches the skin skeleton"
                )
            if armature_for_mesh(obj) is not source_armature and source_armature is not None:
                modifier = obj.modifiers.new("SkinTokensArmature", "ARMATURE")
                modifier.object = source_armature
        elif matches:
            pass
        else:
            create_armature(obj, joints, parents, joint_names)
