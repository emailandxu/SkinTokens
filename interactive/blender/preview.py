from __future__ import annotations

import math
import traceback
from typing import Sequence


PREVIEW_COLLECTION = "SkinTokens Interactive Preview"
JOINT_INDEX_PROP = "skintokens_joint_index"
JOINT_NAME_PROP = "skintokens_joint_name"
JOINT_ID_PROP = "skintokens_joint_id"
PARENT_ID_PROP = "skintokens_parent_id"
DFS_ORDER_PROP = "skintokens_dfs_order"
DEFAULT_JOINT_DISPLAY_SIZE = 0.025
DEFAULT_BONE_BEVEL_DEPTH = 0.004
JOINT_SIZE_FACTOR = 0.02
BONE_WIDTH_FACTOR = 0.0032


_PENDING_DRAW: tuple[
    list[list[float]],
    list[int],
    list[str],
    str,
    float | None,
] | None = None
_DRAW_TIMER_REGISTERED = False


def _bpy():
    import bpy  # type: ignore

    return bpy


def ensure_collection(name: str = PREVIEW_COLLECTION):
    bpy = _bpy()
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def clear_collection(name: str = PREVIEW_COLLECTION) -> None:
    bpy = _bpy()
    collection = bpy.data.collections.get(name)
    if collection is None:
        return
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def preview_style(
    joints: Sequence[Sequence[float]],
    *,
    reference_diagonal: float | None = None,
) -> tuple[float, float]:
    diagonal = 0.0 if reference_diagonal is None else float(reference_diagonal)
    if diagonal <= 1e-8 and len(joints) >= 2:
        minimum = [min(float(joint[axis]) for joint in joints) for axis in range(3)]
        maximum = [max(float(joint[axis]) for joint in joints) for axis in range(3)]
        diagonal = math.sqrt(
            sum((maximum[axis] - minimum[axis]) ** 2 for axis in range(3))
        )
    if diagonal <= 1e-8:
        return DEFAULT_JOINT_DISPLAY_SIZE, DEFAULT_BONE_BEVEL_DEPTH
    return diagonal * JOINT_SIZE_FACTOR, diagonal * BONE_WIDTH_FACTOR


def mesh_preview_diagonal(mesh_object_name: str | None) -> float | None:
    if not mesh_object_name:
        return None
    bpy = _bpy()
    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        return None
    dimensions = mesh_obj.dimensions
    diagonal = math.sqrt(sum(float(dimensions[axis]) ** 2 for axis in range(3)))
    return diagonal if diagonal > 1e-8 else None


def draw_skeleton(
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
    *,
    collection_name: str = PREVIEW_COLLECTION,
    reference_diagonal: float | None = None,
) -> None:
    bpy = _bpy()
    collection = ensure_collection(collection_name)
    clear_collection(collection_name)
    joint_display_size, bone_bevel_depth = preview_style(
        joints,
        reference_diagonal=reference_diagonal,
    )
    names = [
        str(joint_names[idx]) if idx < len(joint_names) else f"bone_{idx}"
        for idx in range(len(joints))
    ]
    name_counts = {name: names.count(name) for name in names}
    joint_ids = [
        name if name_counts[name] == 1 else f"{idx}:{name}"
        for idx, name in enumerate(names)
    ]

    for idx, xyz in enumerate(joints):
        name = names[idx]
        parent = int(parents[idx])
        empty = bpy.data.objects.new(
            f"SKT_JOINT_{idx:03d}_{name}",
            None,
        )
        empty[JOINT_INDEX_PROP] = idx
        empty[JOINT_NAME_PROP] = name
        empty[JOINT_ID_PROP] = joint_ids[idx]
        empty[PARENT_ID_PROP] = "" if parent < 0 else joint_ids[parent]
        empty[DFS_ORDER_PROP] = idx
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = joint_display_size
        empty.show_in_front = True
        empty.location = tuple(float(v) for v in xyz)
        collection.objects.link(empty)

    for child, parent in enumerate(parents):
        if int(parent) < 0:
            continue
        curve = bpy.data.curves.new(f"bone_{int(parent)}_{child}", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = bone_bevel_depth
        polyline = curve.splines.new("POLY")
        polyline.points.add(1)
        a = joints[int(parent)]
        b = joints[child]
        polyline.points[0].co = (float(a[0]), float(a[1]), float(a[2]), 1.0)
        polyline.points[1].co = (float(b[0]), float(b[1]), float(b[2]), 1.0)
        obj = bpy.data.objects.new(curve.name, curve)
        obj.show_in_front = True
        collection.objects.link(obj)


def _flush_scheduled_draw() -> float | None:
    global _PENDING_DRAW, _DRAW_TIMER_REGISTERED
    bpy = _bpy()
    if bpy.context.mode != "OBJECT":
        return 0.1
    payload = _PENDING_DRAW
    _PENDING_DRAW = None
    _DRAW_TIMER_REGISTERED = False
    if payload is None:
        return None
    try:
        joints, parents, joint_names, collection_name, reference_diagonal = payload
        draw_skeleton(
            joints,
            parents,
            joint_names,
            collection_name=collection_name,
            reference_diagonal=reference_diagonal,
        )
    except Exception:
        traceback.print_exc()
    return None


def schedule_draw_skeleton(
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
    *,
    collection_name: str = PREVIEW_COLLECTION,
    reference_diagonal: float | None = None,
) -> None:
    """Rebuild the preview after Blender finishes the current undo transaction."""
    global _PENDING_DRAW, _DRAW_TIMER_REGISTERED
    bpy = _bpy()
    _PENDING_DRAW = (
        [[float(value) for value in joint] for joint in joints],
        [int(parent) for parent in parents],
        [str(name) for name in joint_names],
        str(collection_name),
        None if reference_diagonal is None else float(reference_diagonal),
    )
    if _DRAW_TIMER_REGISTERED:
        return
    _DRAW_TIMER_REGISTERED = True
    bpy.app.timers.register(_flush_scheduled_draw, first_interval=0.0)


def cancel_scheduled_draw() -> None:
    global _PENDING_DRAW, _DRAW_TIMER_REGISTERED
    bpy = _bpy()
    _PENDING_DRAW = None
    if bpy.app.timers.is_registered(_flush_scheduled_draw):
        bpy.app.timers.unregister(_flush_scheduled_draw)
    _DRAW_TIMER_REGISTERED = False


def _context_without_deleted_joints(context: dict, joints: list[list[float] | None]) -> dict:
    parents = [int(parent) for parent in context.get("parents", [])]
    names = list(context.get("joint_names", []))
    deleted = {idx for idx, joint in enumerate(joints) if joint is None}
    changed = True
    while changed:
        changed = False
        for idx, parent in enumerate(parents):
            if idx not in deleted and parent in deleted:
                deleted.add(idx)
                changed = True
    keep = [idx for idx in range(len(joints)) if idx not in deleted]
    if not keep:
        return {
            **context,
            "joints": [],
            "parents": [],
            "joint_names": [],
            "done": False,
        }

    old_to_new = {old: new for new, old in enumerate(keep)}
    compact_parents = [
        -1 if parents[old] == -1 else old_to_new[parents[old]]
        for old in keep
    ]

    root_count = sum(1 for parent in compact_parents if parent == -1)
    if root_count != 1:
        raise ValueError(
            "preview deletion must leave exactly one root"
        )

    return {
        **context,
        "joints": [joints[old] for old in keep],
        "parents": compact_parents,
        "joint_names": [
            names[old] if old < len(names) else f"bone_{old}"
            for old in keep
        ],
        "done": False,
    }


def _read_stable_preview_context(context: dict, joint_objects: Sequence) -> dict:
    records = {}
    for obj in joint_objects:
        joint_id = str(obj[JOINT_ID_PROP])
        if joint_id in records:
            raise ValueError(f"duplicate preview joint id: {joint_id}")
        location = obj.matrix_world.translation
        records[joint_id] = {
            "name": str(obj[JOINT_NAME_PROP]),
            "parent_id": str(obj[PARENT_ID_PROP]),
            "order": int(obj[DFS_ORDER_PROP]),
            "joint": [float(location.x), float(location.y), float(location.z)],
        }

    if not records:
        return {
            **context,
            "joints": [],
            "parents": [],
            "joint_names": [],
            "done": False,
        }

    deleted = {
        joint_id
        for joint_id, record in records.items()
        if record["parent_id"] and record["parent_id"] not in records
    }
    changed = True
    while changed:
        changed = False
        for joint_id, record in records.items():
            if joint_id not in deleted and record["parent_id"] in deleted:
                deleted.add(joint_id)
                changed = True
    for joint_id in deleted:
        records.pop(joint_id, None)
    if not records:
        return {
            **context,
            "joints": [],
            "parents": [],
            "joint_names": [],
            "done": False,
        }

    roots = [
        joint_id
        for joint_id, record in records.items()
        if not record["parent_id"]
    ]
    if len(roots) != 1:
        raise ValueError(f"preview must contain one root, found {len(roots)}")

    children = {joint_id: [] for joint_id in records}
    for joint_id, record in records.items():
        parent_id = record["parent_id"]
        if parent_id:
            if parent_id not in records:
                raise ValueError(f"preview joint {joint_id} has missing parent {parent_id}")
            children[parent_id].append(joint_id)
    for child_ids in children.values():
        child_ids.sort(key=lambda child_id: records[child_id]["order"])

    order = []

    def visit(joint_id: str) -> None:
        order.append(joint_id)
        for child_id in children[joint_id]:
            visit(child_id)

    visit(roots[0])
    if len(order) != len(records):
        raise ValueError("preview hierarchy is cyclic or disconnected")
    id_to_index = {joint_id: idx for idx, joint_id in enumerate(order)}
    return {
        **context,
        "joints": [records[joint_id]["joint"] for joint_id in order],
        "parents": [
            -1
            if not records[joint_id]["parent_id"]
            else id_to_index[records[joint_id]["parent_id"]]
            for joint_id in order
        ],
        "joint_names": [records[joint_id]["name"] for joint_id in order],
        "done": False,
    }


def read_preview_context(context: dict, collection_name: str = PREVIEW_COLLECTION):
    """Rebuild context from the current Blender preview snapshot."""
    bpy = _bpy()
    collection = bpy.data.collections.get(collection_name)
    expected_count = len(context.get("joints", []))
    if collection is None:
        return None

    joint_objects = [
        obj for obj in collection.objects if JOINT_INDEX_PROP in obj
    ]
    has_stable_metadata = all(
        JOINT_ID_PROP in obj and PARENT_ID_PROP in obj and DFS_ORDER_PROP in obj
        for obj in joint_objects
    )
    if has_stable_metadata:
        return _read_stable_preview_context(context, joint_objects)

    if expected_count == 0:
        return None
    joints = [None] * expected_count
    for obj in joint_objects:
        idx = int(obj[JOINT_INDEX_PROP])
        if 0 <= idx < expected_count:
            loc = obj.matrix_world.translation
            joints[idx] = [float(loc.x), float(loc.y), float(loc.z)]

    if any(joint is None for joint in joints):
        return _context_without_deleted_joints(context, joints)
    return {
        **context,
        "joints": joints,
        "done": False,
    }


def selected_preview_joint_index():
    bpy = _bpy()
    obj = bpy.context.object
    if obj is None or JOINT_INDEX_PROP not in obj:
        return None
    return int(obj[JOINT_INDEX_PROP])
