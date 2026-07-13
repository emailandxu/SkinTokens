from __future__ import annotations

from typing import Sequence


PREVIEW_COLLECTION = "SkinTokens Interactive Preview"
JOINT_INDEX_PROP = "skintokens_joint_index"
JOINT_NAME_PROP = "skintokens_joint_name"


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


def draw_skeleton(
    joints: Sequence[Sequence[float]],
    parents: Sequence[int],
    joint_names: Sequence[str],
    *,
    collection_name: str = PREVIEW_COLLECTION,
) -> None:
    bpy = _bpy()
    collection = ensure_collection(collection_name)
    clear_collection(collection_name)

    for idx, xyz in enumerate(joints):
        name = joint_names[idx] if idx < len(joint_names) else f"bone_{idx}"
        empty = bpy.data.objects.new(
            f"SKT_JOINT_{idx:03d}_{name}",
            None,
        )
        empty[JOINT_INDEX_PROP] = idx
        empty[JOINT_NAME_PROP] = name
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = 0.025
        empty.location = tuple(float(v) for v in xyz)
        collection.objects.link(empty)

    for child, parent in enumerate(parents):
        if int(parent) < 0:
            continue
        curve = bpy.data.curves.new(f"bone_{int(parent)}_{child}", "CURVE")
        curve.dimensions = "3D"
        curve.resolution_u = 1
        curve.bevel_depth = 0.004
        polyline = curve.splines.new("POLY")
        polyline.points.add(1)
        a = joints[int(parent)]
        b = joints[child]
        polyline.points[0].co = (float(a[0]), float(a[1]), float(a[2]), 1.0)
        polyline.points[1].co = (float(b[0]), float(b[1]), float(b[2]), 1.0)
        obj = bpy.data.objects.new(curve.name, curve)
        collection.objects.link(obj)


def _context_without_deleted_joints(context: dict, joints: list[list[float] | None]) -> dict:
    parents = [int(parent) for parent in context.get("parents", [])]
    names = list(context.get("joint_names", []))
    deleted = {idx for idx, joint in enumerate(joints) if joint is None}
    keep = [idx for idx, joint in enumerate(joints) if joint is not None]
    if not keep:
        return {
            **context,
            "joints": [],
            "parents": [],
            "joint_names": [],
            "done": False,
        }

    old_to_new = {old: new for new, old in enumerate(keep)}
    compact_parents = []
    for old in keep:
        parent = parents[old]
        while parent in deleted:
            parent = parents[parent]
        compact_parents.append(-1 if parent == -1 else old_to_new[parent])

    root_count = sum(1 for parent in compact_parents if parent == -1)
    if root_count != 1:
        raise ValueError(
            "deleting this root branch would create multiple roots; delete or reparent its children first"
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


def read_preview_context(context: dict, collection_name: str = PREVIEW_COLLECTION):
    """Return edited context from preview empties.

    Moving empties updates joint coordinates. Deleting an empty removes that
    joint only; surviving children are reparented to the nearest surviving
    ancestor so the skeleton remains a single valid tree.
    """
    bpy = _bpy()
    collection = bpy.data.collections.get(collection_name)
    expected_count = len(context.get("joints", []))
    if collection is None or expected_count == 0:
        return None

    joints = [None] * expected_count
    for obj in collection.objects:
        if JOINT_INDEX_PROP not in obj:
            continue
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
