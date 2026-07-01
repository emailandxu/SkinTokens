from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from src.rig_package.info.asset import Asset


@dataclass(frozen=True)
class RigSpec:
    joints: np.ndarray
    parents: np.ndarray
    joint_names: List[str]
    obj_text_joints: np.ndarray


def build_matrix_local(joints: np.ndarray) -> np.ndarray:
    matrix_local = np.repeat(
        np.eye(4, dtype=np.float32)[None, :, :],
        joints.shape[0],
        axis=0,
    )
    matrix_local[:, :3, 3] = joints
    return matrix_local


def apply_rig(asset: Asset, rig: RigSpec) -> None:
    asset.matrix_local = build_matrix_local(rig.joints)
    asset.parents = rig.parents.copy()
    asset.joint_names = list(rig.joint_names)


def children_from_parents(parents: Sequence[int]) -> List[List[int]]:
    children: List[List[int]] = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent != -1:
            children[int(parent)].append(child)
    return children


def root_from_parents(parents: Sequence[int]) -> int:
    roots = [idx for idx, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {len(roots)}")
    return roots[0]


def subtree_size(node: int, children: Sequence[Sequence[int]]) -> int:
    return 1 + sum(subtree_size(child, children) for child in children[node])


def longest_chain_indices(node: int, children: Sequence[Sequence[int]]) -> List[int]:
    if not children[node]:
        return [node]
    chains = [[node] + longest_chain_indices(child, children) for child in children[node]]
    return max(chains, key=len)


def resample_chain_points(
    joints: np.ndarray,
    chain: Sequence[int],
    samples: int = 16,
    *,
    mirror_x: bool = False,
) -> np.ndarray:
    points = np.asarray([joints[idx] for idx in chain], dtype=np.float64)
    points = points - points[0]
    if mirror_x:
        points[:, 0] *= -1.0
    scale = np.linalg.norm(points[-1])
    if scale < 1e-9:
        scale = np.max(np.linalg.norm(points, axis=1))
    points = points / max(float(scale), 1e-9)
    if points.shape[0] == 1:
        return np.repeat(points, samples, axis=0)

    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = distances[-1]
    if total < 1e-9:
        return np.repeat(points[:1], samples, axis=0)

    out = []
    for target in np.linspace(0.0, total, samples):
        seg = np.searchsorted(distances, target, side="right") - 1
        seg = min(max(seg, 0), len(seg_lengths) - 1)
        alpha = (target - distances[seg]) / max(float(seg_lengths[seg]), 1e-9)
        out.append(points[seg] * (1.0 - alpha) + points[seg + 1] * alpha)
    return np.stack(out)


def chain_lengths(joints: np.ndarray, chain: Sequence[int]) -> np.ndarray:
    lengths = []
    for a, b in zip(chain, chain[1:]):
        lengths.append(float(np.linalg.norm(joints[b] - joints[a])))
    return np.asarray(lengths, dtype=np.float64)


def subtree_similarity(
    joints: np.ndarray,
    children: Sequence[Sequence[int]],
    a: int,
    b: int,
) -> float:
    chain_a = longest_chain_indices(a, children)
    chain_b = longest_chain_indices(b, children)
    lengths_a = chain_lengths(joints, chain_a)
    lengths_b = chain_lengths(joints, chain_b)
    n = min(lengths_a.shape[0], lengths_b.shape[0])
    if n == 0:
        length_sim = 1.0 if lengths_a.shape[0] == lengths_b.shape[0] else 0.0
    else:
        rel = np.abs(lengths_a[:n] - lengths_b[:n]) / np.maximum(
            np.maximum(lengths_a[:n], lengths_b[:n]),
            1e-9,
        )
        len_penalty = abs(lengths_a.shape[0] - lengths_b.shape[0]) / max(
            lengths_a.shape[0],
            lengths_b.shape[0],
            1,
        )
        length_sim = float((1.0 - np.mean(rel)) * (1.0 - len_penalty))

    shape_a = resample_chain_points(joints, chain_a)
    shape_b = resample_chain_points(joints, chain_b)
    shape_b_mirror = resample_chain_points(joints, chain_b, mirror_x=True)
    shape_dist = min(
        float(np.sqrt(np.mean(np.sum((shape_a - shape_b) ** 2, axis=1)))),
        float(np.sqrt(np.mean(np.sum((shape_a - shape_b_mirror) ** 2, axis=1)))),
    )
    shape_sim = float(np.exp(-shape_dist / 0.5))
    size_penalty = 1.0 - abs(subtree_size(a, children) - subtree_size(b, children)) / max(
        subtree_size(a, children),
        subtree_size(b, children),
        1,
    )
    return 0.45 * length_sim + 0.45 * shape_sim + 0.10 * size_penalty


def order_children_by_local_similarity(
    joints: np.ndarray,
    children: Sequence[Sequence[int]],
    child_ids: Sequence[int],
    *,
    threshold: float = 0.85,
) -> List[int]:
    if len(child_ids) <= 2:
        return list(child_ids)

    ordered = list(child_ids)
    original_index = {child: idx for idx, child in enumerate(child_ids)}
    non_leaves = [child for child in child_ids if subtree_size(child, children) > 1]
    pairs = []
    for i, a in enumerate(non_leaves):
        for b in non_leaves[i + 1 :]:
            sim = subtree_similarity(joints, children, a, b)
            if sim >= threshold and abs(original_index[a] - original_index[b]) > 1:
                pairs.append((sim, original_index[a], original_index[b], a, b))

    moved: set[int] = set()
    for _, _, _, a, b in sorted(pairs, reverse=True):
        if a in moved or b in moved:
            continue
        pos_a = ordered.index(a)
        pos_b = ordered.index(b)
        if abs(pos_a - pos_b) <= 1:
            continue
        first, second = (a, b) if pos_a < pos_b else (b, a)
        ordered.remove(second)
        ordered.insert(ordered.index(first) + 1, second)
        moved.add(second)

    return ordered


def similar_subtree_order(
    joints: np.ndarray,
    parents: Sequence[int],
) -> List[int]:
    children = children_from_parents(parents)
    new_children = [list(row) for row in children]
    for node, child_ids in enumerate(children):
        if len(child_ids) > 1:
            new_children[node] = order_children_by_local_similarity(
                joints,
                children,
                child_ids,
            )

    root = root_from_parents(parents)
    order: List[int] = []

    def visit(node: int) -> None:
        order.append(node)
        for child in new_children[node]:
            visit(child)

    visit(root)
    if len(order) != len(parents):
        raise RuntimeError("reordered skeleton did not visit every joint")
    return order


def reorder_rig_spec(rig: RigSpec, order: Sequence[int]) -> RigSpec:
    old_to_new = {old: new for new, old in enumerate(order)}
    parents = np.asarray(
        [-1 if rig.parents[old] == -1 else old_to_new[int(rig.parents[old])] for old in order],
        dtype=np.int32,
    )
    return RigSpec(
        joints=rig.joints[np.asarray(order, dtype=np.int64)].copy(),
        parents=parents,
        joint_names=[rig.joint_names[old] for old in order],
        obj_text_joints=rig.obj_text_joints[np.asarray(order, dtype=np.int64)].copy(),
    )


def rig_with_joint_order(source_rig: RigSpec, target_rig: RigSpec) -> RigSpec:
    source_by_name = {name: idx for idx, name in enumerate(source_rig.joint_names)}
    missing = [name for name in target_rig.joint_names if name not in source_by_name]
    if missing:
        raise ValueError(f"source rig is missing joints: {missing}")
    order = np.asarray(
        [source_by_name[name] for name in target_rig.joint_names],
        dtype=np.int64,
    )
    return RigSpec(
        joints=source_rig.joints[order].copy(),
        parents=target_rig.parents.copy(),
        joint_names=list(target_rig.joint_names),
        obj_text_joints=target_rig.obj_text_joints.copy(),
    )


def remap_skin_to_rig(asset: Asset, source_rig: RigSpec, target_rig: RigSpec) -> None:
    if asset.skin is None:
        raise ValueError("asset is missing skin")
    if asset.skin.shape[1] != len(source_rig.joint_names):
        raise ValueError(
            f"skin columns ({asset.skin.shape[1]}) do not match source rig "
            f"({len(source_rig.joint_names)})"
        )
    source_by_name = {name: idx for idx, name in enumerate(source_rig.joint_names)}
    missing = [name for name in target_rig.joint_names if name not in source_by_name]
    if missing:
        raise ValueError(f"source skin is missing joints: {missing}")
    order = np.asarray(
        [source_by_name[name] for name in target_rig.joint_names],
        dtype=np.int64,
    )
    asset.skin = asset.skin[:, order]


def reorder_asset_joints(asset: Asset, order: Sequence[int]) -> None:
    if asset.skin is None or asset.parents is None or asset.matrix_local is None:
        raise ValueError("asset is missing skin or skeleton fields")
    old_to_new = {old: new for new, old in enumerate(order)}
    parents = np.asarray(
        [-1 if asset.parents[old] == -1 else old_to_new[int(asset.parents[old])] for old in order],
        dtype=np.int32,
    )
    asset.matrix_local = asset.matrix_local[np.asarray(order, dtype=np.int64)].copy()
    asset.parents = parents
    asset.skin = asset.skin[:, np.asarray(order, dtype=np.int64)]
    if asset.joint_names is not None:
        asset.joint_names = [asset.joint_names[old] for old in order]


def transformed_rig_from_asset(asset: Asset, input_rig: Optional[RigSpec]) -> Optional[RigSpec]:
    if input_rig is None:
        return None
    if asset.joints is None or asset.parents is None or asset.joint_names is None:
        raise RuntimeError("rig txt was not preserved through predict transform")
    obj_text_joints_by_name = {
        name: joint for name, joint in zip(input_rig.joint_names, input_rig.obj_text_joints)
    }
    return RigSpec(
        joints=asset.joints.copy(),
        parents=asset.parents.copy(),
        joint_names=list(asset.joint_names),
        obj_text_joints=np.asarray(
            [obj_text_joints_by_name[name] for name in asset.joint_names],
            dtype=np.float32,
        ),
    )
