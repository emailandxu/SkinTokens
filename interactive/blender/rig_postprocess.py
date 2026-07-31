from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .apply_skin import (
    JOINT_ID_PROP,
    MANAGED_VERTEX_GROUPS_PROP,
    armature_for_mesh,
    managed_vertex_group_names,
    set_managed_vertex_group_names,
)


TEMPLATE_BIPED = "BIPED"
TEMPLATE_QUADRUPED = "QUADRUPED"
SUPPORTED_TEMPLATES = {TEMPLATE_BIPED, TEMPLATE_QUADRUPED}

MIRROR_AXIS_X = "X"
MIRROR_AXIS_Y = "Y"
MIRROR_AXIS_Z = "Z"
SUPPORTED_MIRROR_AXES = {MIRROR_AXIS_X, MIRROR_AXIS_Y, MIRROR_AXIS_Z}

MIRROR_DISTANCE_FARTHEST = "FARTHEST"
MIRROR_DISTANCE_NEAREST = "NEAREST"
MIRROR_DISTANCE_AVERAGE = "AVERAGE"
SUPPORTED_MIRROR_DISTANCE_MODES = {
    MIRROR_DISTANCE_FARTHEST,
    MIRROR_DISTANCE_NEAREST,
    MIRROR_DISTANCE_AVERAGE,
}

SEMANTIC_VERSION_PROP = "skintokens_semantic_version"
POSTPROCESS_READY_PROP = "skintokens_postprocess_ready"
SEMANTIC_TEMPLATE_PROP = "skintokens_body_template"
SEMANTIC_CENTER_PROP = "skintokens_symmetry_center"
SEMANTIC_LATERAL_AXIS_PROP = "skintokens_lateral_axis"
SEMANTIC_FORWARD_AXIS_PROP = "skintokens_forward_axis"
SEMANTIC_REGION_PROP = "skintokens_body_region"
SEMANTIC_SIDE_PROP = "skintokens_body_side"
SEMANTIC_GROUP_PROP = "skintokens_body_group"
SEMANTIC_INDEX_PROP = "skintokens_body_index"
SEMANTIC_NAME_PROP = "skintokens_semantic_name"
SEMANTIC_MIRROR_ID_PROP = "skintokens_mirror_joint_id"
SEMANTIC_ORIGINAL_NAME_PROP = "skintokens_original_bone_name"
SEMANTIC_VERSION = 1

_GENERIC_BONE_PATTERN = re.compile(
    r"^bone_\d+(?:_split(?:_\d+)?)?$",
    re.IGNORECASE,
)
_SEMANTIC_BONE_PATTERN = re.compile(
    r"^(?:spine|head|tail)_\d+$"
    r"|^(?:arm|leg)_[lr]_\d+$"
    r"|^leg_(?:front|hind)_[lr]_\d+$",
    re.IGNORECASE,
)
_CENTRAL_REGIONS = {"spine", "head", "tail"}


class RigPostprocessError(RuntimeError):
    pass


@dataclass(frozen=True)
class RigBoneSnapshot:
    joint_id: str
    name: str
    parent: int
    head: tuple[float, float, float]
    tail: tuple[float, float, float]


@dataclass(frozen=True)
class SemanticAssignment:
    joint_id: str
    old_name: str
    target_name: str
    semantic_name: str
    region: str
    side: str
    group: str
    index: int
    mirror_joint_id: str = ""


@dataclass(frozen=True)
class NamingPlan:
    template: str
    center: float
    lateral_axis: tuple[float, float, float]
    forward_axis: tuple[float, float, float]
    assignments: tuple[SemanticAssignment, ...]
    warnings: tuple[str, ...] = ()

    @property
    def rename_count(self) -> int:
        return sum(
            assignment.old_name != assignment.target_name
            for assignment in self.assignments
        )

    @property
    def mirror_pair_count(self) -> int:
        return sum(
            assignment.side == "l" and bool(assignment.mirror_joint_id)
            for assignment in self.assignments
        )


@dataclass(frozen=True)
class RenameResult:
    template: str
    renamed_bones: int
    classified_bones: int
    mirror_pairs: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MirrorResult:
    mirror_pairs: int
    center: float
    paired_bones: tuple[tuple[str, str], ...] = ()
    skipped_bones: tuple[str, ...] = ()


@dataclass(frozen=True)
class MirrorPairCandidate:
    positive_name: str
    negative_name: str
    score: float
    reflected_error: float
    direction_cosine: float


@dataclass(frozen=True)
class MirrorPairInference:
    pairs: tuple[MirrorPairCandidate, ...]
    unmatched_names: tuple[str, ...]
    ambiguous_names: tuple[str, ...]


@dataclass(frozen=True)
class _BranchPair:
    attachment: int
    left_root: int
    right_root: int
    similarity: float
    extent: float

    @property
    def limb_score(self) -> float:
        return self.extent * self.similarity


def _point_tuple(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalize(vector: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        vector = np.asarray(fallback, dtype=np.float64)
        length = float(np.linalg.norm(vector))
    return vector / max(length, 1e-9)


def _deterministic_axis_sign(axis: np.ndarray) -> np.ndarray:
    horizontal = np.asarray(axis[:2], dtype=np.float64)
    dominant = int(np.argmax(np.abs(horizontal)))
    if horizontal[dominant] < 0.0:
        return -axis
    return axis


class _RigAnalysis:
    def __init__(self, bones: Sequence[RigBoneSnapshot], template: str):
        if template not in SUPPORTED_TEMPLATES:
            raise RigPostprocessError(f"不支持的骨架模板：{template}")
        if not bones:
            raise RigPostprocessError("骨架中没有骨骼")
        self.bones = tuple(bones)
        self.template = template
        self.parents = [int(bone.parent) for bone in bones]
        self.children: list[list[int]] = [[] for _ in bones]
        roots = []
        for index, parent in enumerate(self.parents):
            if parent < 0:
                roots.append(index)
            elif parent >= len(bones):
                raise RigPostprocessError(f"骨骼 {bones[index].name} 的父级无效")
            else:
                self.children[parent].append(index)
        if len(roots) != 1:
            raise RigPostprocessError(f"骨架必须只有一个根骨骼，当前为 {len(roots)} 个")
        self.root = roots[0]
        self.heads = np.asarray([bone.head for bone in bones], dtype=np.float64)
        self.tails = np.asarray([bone.tail for bone in bones], dtype=np.float64)
        all_points = np.concatenate([self.heads, self.tails], axis=0)
        self.scale = max(float(np.linalg.norm(np.ptp(all_points, axis=0))), 1e-6)
        self._subtree_cache: dict[int, tuple[int, ...]] = {}
        self._chain_cache: dict[int, tuple[int, ...]] = {}
        self._extent_cache: dict[int, float] = {}
        self.lateral_axis, self.forward_axis = self._body_frame()
        self.lateral = self.heads @ self.lateral_axis
        self.forward = self.heads @ self.forward_axis
        self.center = float(self.lateral[self.root])
        lateral_span = float(np.ptp(self.lateral))
        self.side_epsilon = max(lateral_span * 0.06, self.scale * 1e-4, 1e-6)
        self.center_band = max(lateral_span * 0.18, self.side_epsilon * 2.0)

    def subtree(self, node: int) -> tuple[int, ...]:
        cached = self._subtree_cache.get(node)
        if cached is not None:
            return cached
        result = [node]
        for child in self.children[node]:
            result.extend(self.subtree(child))
        value = tuple(result)
        self._subtree_cache[node] = value
        return value

    def subtree_size(self, node: int) -> int:
        return len(self.subtree(node))

    def longest_chain(self, node: int) -> tuple[int, ...]:
        cached = self._chain_cache.get(node)
        if cached is not None:
            return cached
        if not self.children[node]:
            result = (node,)
        else:
            result = max(
                ((node, *self.longest_chain(child)) for child in self.children[node]),
                key=self.path_length,
            )
        self._chain_cache[node] = result
        return result

    def path_length(self, path: Sequence[int]) -> float:
        if len(path) <= 1:
            node = int(path[0])
            return float(np.linalg.norm(self.tails[node] - self.heads[node]))
        points = self.heads[np.asarray(path, dtype=np.int64)]
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        leaf = int(path[-1])
        return length + float(np.linalg.norm(self.tails[leaf] - self.heads[leaf]))

    def branch_extent(self, node: int) -> float:
        cached = self._extent_cache.get(node)
        if cached is not None:
            return cached
        value = self.path_length(self.longest_chain(node))
        self._extent_cache[node] = value
        return value

    def _body_frame(self) -> tuple[np.ndarray, np.ndarray]:
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        root_children = self.children[self.root]
        main_root = (
            max(root_children, key=self.subtree_size)
            if root_children
            else self.root
        )
        if self.template == TEMPLATE_QUADRUPED:
            descendants = np.asarray(self.subtree(main_root), dtype=np.int64)
            centroid = np.mean(self.heads[descendants], axis=0)
            forward = centroid - self.heads[self.root]
            forward[2] = 0.0
            if float(np.linalg.norm(forward)) < self.scale * 1e-4:
                horizontal = self.heads[descendants] - self.heads[self.root]
                horizontal[:, 2] = 0.0
                forward = horizontal[int(np.argmax(np.linalg.norm(horizontal, axis=1)))]
            forward = _normalize(forward, (0.0, -1.0, 0.0))
            lateral = _normalize(np.cross(up, forward), (1.0, 0.0, 0.0))
            return lateral, forward

        sibling_spans = []
        for child_ids in self.children:
            for position, first in enumerate(child_ids):
                for second in child_ids[position + 1 :]:
                    delta = self.heads[first] - self.heads[second]
                    delta[2] = 0.0
                    sibling_spans.append(delta)
        if sibling_spans:
            lateral = max(sibling_spans, key=lambda item: float(np.linalg.norm(item)))
        else:
            xy = self.heads[:, :2] - np.mean(self.heads[:, :2], axis=0)
            covariance = xy.T @ xy
            values, vectors = np.linalg.eigh(covariance)
            principal = vectors[:, int(np.argmax(values))]
            lateral = np.asarray((principal[0], principal[1], 0.0))
        lateral = _deterministic_axis_sign(
            _normalize(lateral, (1.0, 0.0, 0.0))
        )
        forward = _normalize(np.cross(lateral, up), (0.0, -1.0, 0.0))
        return lateral, forward

    def body_points(self, indices: Sequence[int]) -> np.ndarray:
        points = self.heads[np.asarray(indices, dtype=np.int64)]
        return np.stack(
            [
                points @ self.lateral_axis,
                points @ self.forward_axis,
                points[:, 2],
            ],
            axis=1,
        )

    def chain_body_points(self, node: int) -> np.ndarray:
        chain = self.longest_chain(node)
        points = self.body_points(chain)
        leaf = int(chain[-1])
        tail = self.tails[leaf]
        tail_body = np.asarray(
            (
                float(np.dot(tail, self.lateral_axis)),
                float(np.dot(tail, self.forward_axis)),
                float(tail[2]),
            ),
            dtype=np.float64,
        )
        if float(np.linalg.norm(tail_body - points[-1])) > 1e-9:
            points = np.concatenate([points, tail_body[None, :]], axis=0)
        return points

    @staticmethod
    def _resample(points: np.ndarray, samples: int = 16) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        points = points - points[0]
        if points.shape[0] == 1:
            return np.repeat(points, samples, axis=0)
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distances = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total = float(distances[-1])
        if total < 1e-9:
            return np.repeat(points[:1], samples, axis=0)
        points = points / total
        distances = distances / total
        result = []
        for target in np.linspace(0.0, 1.0, samples):
            segment = int(np.searchsorted(distances, target, side="right") - 1)
            segment = min(max(segment, 0), len(segment_lengths) - 1)
            denominator = max(float(distances[segment + 1] - distances[segment]), 1e-9)
            alpha = (target - distances[segment]) / denominator
            result.append(points[segment] * (1.0 - alpha) + points[segment + 1] * alpha)
        return np.stack(result)

    def pair_similarity(self, left: int, right: int) -> float:
        raw_left = self.chain_body_points(left)
        raw_right = self.chain_body_points(right)
        direction_left = raw_left[-1] - raw_left[0]
        direction_right = raw_right[-1] - raw_right[0]
        direction_right[0] *= -1.0
        direction_denominator = float(
            np.linalg.norm(direction_left) * np.linalg.norm(direction_right)
        )
        if direction_denominator < 1e-9:
            return 0.0
        direction_cosine = float(
            np.dot(direction_left, direction_right) / direction_denominator
        )
        if direction_cosine < 0.15:
            return 0.0
        direction_similarity = min(1.0, max(0.0, direction_cosine))

        points_left = self._resample(raw_left)
        points_right = self._resample(raw_right)
        points_right[:, 0] *= -1.0
        shape_distance = float(
            np.sqrt(np.mean(np.sum((points_left - points_right) ** 2, axis=1)))
        )
        shape_similarity = float(np.exp(-shape_distance / 0.55))

        extent_left = self.branch_extent(left)
        extent_right = self.branch_extent(right)
        length_similarity = 1.0 - abs(extent_left - extent_right) / max(
            extent_left,
            extent_right,
            1e-9,
        )
        size_left = self.subtree_size(left)
        size_right = self.subtree_size(right)
        size_similarity = 1.0 - abs(size_left - size_right) / max(
            size_left,
            size_right,
            1,
        )

        parent_left = self.parents[left]
        parent_right = self.parents[right]
        origin_left = self.heads[parent_left] if parent_left >= 0 else self.heads[left]
        origin_right = self.heads[parent_right] if parent_right >= 0 else self.heads[right]
        root_left = self.heads[left] - origin_left
        root_right = self.heads[right] - origin_right
        root_left_body = np.asarray(
            (
                float(np.dot(root_left, self.lateral_axis)),
                float(np.dot(root_left, self.forward_axis)),
                float(root_left[2]),
            )
        )
        root_right_body = np.asarray(
            (
                -float(np.dot(root_right, self.lateral_axis)),
                float(np.dot(root_right, self.forward_axis)),
                float(root_right[2]),
            )
        )
        root_distance = float(np.linalg.norm(root_left_body - root_right_body)) / self.scale
        root_similarity = float(np.exp(-root_distance / 0.18))
        return max(
            0.0,
            0.35 * shape_similarity
            + 0.15 * length_similarity
            + 0.10 * size_similarity
            + 0.20 * root_similarity
            + 0.20 * direction_similarity,
        )

    def branch_pairs(self) -> list[_BranchPair]:
        result = []
        for attachment, child_ids in enumerate(self.children):
            if abs(float(self.lateral[attachment] - self.center)) > self.center_band:
                continue
            left = [
                child
                for child in child_ids
                if self.lateral[child] > self.center + self.side_epsilon
            ]
            right = [
                child
                for child in child_ids
                if self.lateral[child] < self.center - self.side_epsilon
            ]
            scores = []
            for left_root in left:
                for right_root in right:
                    similarity = self.pair_similarity(left_root, right_root)
                    scores.append((similarity, left_root, right_root))
            used_left: set[int] = set()
            used_right: set[int] = set()
            for similarity, left_root, right_root in sorted(scores, reverse=True):
                if similarity < 0.58:
                    continue
                if left_root in used_left or right_root in used_right:
                    continue
                used_left.add(left_root)
                used_right.add(right_root)
                extent = 0.5 * (
                    self.branch_extent(left_root) + self.branch_extent(right_root)
                )
                result.append(
                    _BranchPair(
                        attachment=attachment,
                        left_root=left_root,
                        right_root=right_root,
                        similarity=similarity,
                        extent=extent,
                    )
                )
        return result

    def _pair_mass(self, node: int, pairs: Sequence[_BranchPair]) -> float:
        descendants = set(self.subtree(node))
        return sum(
            pair.limb_score
            for pair in pairs
            if pair.attachment in descendants
        )

    def main_path(self, pairs: Sequence[_BranchPair]) -> tuple[int, ...]:
        path = [self.root]
        current = self.root
        paired_roots = {
            root
            for pair in pairs
            for root in (pair.left_root, pair.right_root)
        }
        while True:
            candidates = [
                child
                for child in self.children[current]
                if child not in paired_roots
                and abs(float(self.lateral[child] - self.center)) <= self.center_band
            ]
            if not candidates:
                break
            current = max(
                candidates,
                key=lambda child: (
                    self._pair_mass(child, pairs),
                    self.subtree_size(child),
                    self.branch_extent(child),
                    -abs(float(self.lateral[child] - self.center)),
                ),
            )
            path.append(current)
        return tuple(path)

    def tail_nodes(
        self,
        main_path: Sequence[int],
        pairs: Sequence[_BranchPair],
        hind_attachment: int,
    ) -> tuple[int, ...]:
        if len(main_path) < 2:
            return ()
        main_nodes = set(main_path)
        path_position = {node: index for index, node in enumerate(main_path)}
        hind_position = path_position.get(hind_attachment, 0)
        attachment_limit = min(len(main_path) - 1, hind_position + 1)
        paired_roots = {
            root
            for pair in pairs
            for root in (pair.left_root, pair.right_root)
        }

        head_direction = self.tails[int(main_path[-1])] - self.heads[self.root]
        head_direction -= self.lateral_axis * float(
            np.dot(head_direction, self.lateral_axis)
        )
        head_direction = _normalize(head_direction, self.forward_axis)

        candidates = []
        for attachment in main_path[: attachment_limit + 1]:
            attachment_position = path_position[int(attachment)]
            attachment_score = 1.0 / (
                1.0 + abs(float(attachment_position - hind_position))
            )
            for child in self.children[int(attachment)]:
                if child in main_nodes or child in paired_roots:
                    continue
                chain = self.longest_chain(child)
                extent = self.branch_extent(child)
                if extent < self.scale * 0.05:
                    continue

                lateral_offset = abs(
                    float(self.lateral[child] - self.lateral[int(attachment)])
                )
                centrality = float(
                    np.exp(-lateral_offset / max(self.center_band, 1e-6))
                )
                chainness = len(chain) / max(self.subtree_size(child), 1)
                length_score = min(1.0, extent / max(self.scale * 0.25, 1e-6))

                tail_direction = self.tails[int(chain[-1])] - self.heads[child]
                tail_direction -= self.lateral_axis * float(
                    np.dot(tail_direction, self.lateral_axis)
                )
                if float(np.linalg.norm(tail_direction)) < self.scale * 1e-4:
                    direction_score = 0.5
                else:
                    tail_direction = _normalize(tail_direction, -head_direction)
                    direction_score = 0.5 * (
                        1.0 - float(np.dot(tail_direction, head_direction))
                    )
                    direction_score = min(1.0, max(0.0, direction_score))

                # Direction remains a weak hint so hanging or curled tails are valid.
                score = (
                    0.33 * attachment_score
                    + 0.22 * centrality
                    + 0.17 * chainness
                    + 0.17 * length_score
                    + 0.11 * direction_score
                )
                candidates.append((score, extent, chainness, -child, chain))
        if not candidates:
            return ()
        candidates.sort(reverse=True)
        if candidates[0][0] < 0.50:
            return ()
        if (
            len(candidates) >= 2
            and candidates[0][0] - candidates[1][0] < 0.09
        ):
            return ()
        return tuple(candidates[0][-1])

    def major_limb_pairs(
        self,
        main_path: Sequence[int],
        pairs: Sequence[_BranchPair],
    ) -> tuple[_BranchPair, _BranchPair]:
        main_nodes = set(main_path)
        candidates = [
            pair
            for pair in pairs
            if pair.attachment in main_nodes
            and pair.extent >= self.scale * 0.05
            and pair.similarity >= 0.65
        ]
        ranked = sorted(
            candidates,
            key=lambda pair: (
                pair.limb_score,
                pair.extent,
                pair.similarity,
            ),
            reverse=True,
        )
        selected = ranked[:2]
        if len(selected) != 2:
            label = "二足" if self.template == TEMPLATE_BIPED else "四足"
            raise RigPostprocessError(
                f"{label}模板需要识别两组主要左右肢体，当前只识别到 {len(selected)} 组"
            )
        if (
            len(ranked) >= 3
            and ranked[2].limb_score >= ranked[1].limb_score * 0.85
        ):
            raise RigPostprocessError(
                "第二、第三组左右分支尺寸过于接近，无法可靠区分四肢和附件"
            )
        path_position = {node: index for index, node in enumerate(main_path)}

        def anatomical_position(pair: _BranchPair) -> tuple[float, float]:
            roots = 0.5 * (
                self.heads[pair.left_root] + self.heads[pair.right_root]
            )
            return (
                float(path_position[pair.attachment]),
                float(np.dot(roots - self.heads[self.root], self.forward_axis)),
            )

        selected.sort(key=anatomical_position)
        return selected[0], selected[1]

    def match_subtrees(
        self,
        left_root: int,
        right_root: int,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        matches: list[tuple[int, int]] = []
        matched_left: set[int] = set()
        matched_right: set[int] = set()

        def visit(left: int, right: int) -> None:
            matches.append((left, right))
            matched_left.add(left)
            matched_right.add(right)
            scores = []
            for child_left in self.children[left]:
                for child_right in self.children[right]:
                    scores.append(
                        (
                            self.pair_similarity(child_left, child_right),
                            child_left,
                            child_right,
                        )
                    )
            used_left: set[int] = set()
            used_right: set[int] = set()
            chosen = []
            for similarity, child_left, child_right in sorted(scores, reverse=True):
                if similarity < 0.50:
                    continue
                if child_left in used_left or child_right in used_right:
                    continue
                used_left.add(child_left)
                used_right.add(child_right)
                chosen.append((child_left, child_right))
            for child_left, child_right in sorted(
                chosen,
                key=lambda pair: (
                    -self.subtree_size(pair[0]),
                    self.bones[pair[0]].name,
                ),
            ):
                visit(child_left, child_right)

        visit(left_root, right_root)
        left_unmatched = [
            node for node in self.subtree(left_root) if node not in matched_left
        ]
        right_unmatched = [
            node for node in self.subtree(right_root) if node not in matched_right
        ]
        return matches, left_unmatched, right_unmatched


def _renameable(name: str) -> bool:
    return bool(
        _GENERIC_BONE_PATTERN.fullmatch(name)
        or _SEMANTIC_BONE_PATTERN.fullmatch(name)
    )


def _semantic_name(region: str, side: str, group: str, index: int) -> str:
    if region in _CENTRAL_REGIONS:
        return f"{region}_{index}"
    if group:
        return f"{region}_{group}_{side}_{index}"
    return f"{region}_{side}_{index}"


def build_naming_plan(
    bones: Sequence[RigBoneSnapshot],
    template: str,
) -> NamingPlan:
    analysis = _RigAnalysis(bones, template)
    branch_pairs = analysis.branch_pairs()
    main_path = analysis.main_path(branch_pairs)
    if len(main_path) < 2:
        raise RigPostprocessError("无法识别从根骨骼通向头部的主干")
    lower_pair, upper_pair = analysis.major_limb_pairs(main_path, branch_pairs)
    path_position = {node: index for index, node in enumerate(main_path)}
    split_position = path_position[upper_pair.attachment]
    spine_nodes = tuple(main_path[: split_position + 1])
    head_nodes = (
        analysis.subtree(main_path[split_position + 1])
        if split_position + 1 < len(main_path)
        else ()
    )
    tail_nodes = analysis.tail_nodes(
        main_path,
        branch_pairs,
        lower_pair.attachment,
    )

    assignments_by_node: dict[int, SemanticAssignment] = {}

    def assign(
        node: int,
        region: str,
        side: str,
        group: str,
        index: int,
        mirror_node: int | None = None,
    ) -> None:
        if node in assignments_by_node:
            return
        semantic_name = _semantic_name(region, side, group, index)
        bone = bones[node]
        assignments_by_node[node] = SemanticAssignment(
            joint_id=bone.joint_id,
            old_name=bone.name,
            target_name=semantic_name if _renameable(bone.name) else bone.name,
            semantic_name=semantic_name,
            region=region,
            side=side,
            group=group,
            index=index,
            mirror_joint_id="" if mirror_node is None else bones[mirror_node].joint_id,
        )

    for index, node in enumerate(spine_nodes, start=1):
        assign(node, "spine", "center", "", index)
    for index, node in enumerate(head_nodes, start=1):
        assign(node, "head", "center", "", index)
    for index, node in enumerate(tail_nodes, start=1):
        assign(node, "tail", "center", "", index)

    if template == TEMPLATE_BIPED:
        pair_roles = ((lower_pair, "leg", ""), (upper_pair, "arm", ""))
    else:
        pair_roles = (
            (lower_pair, "leg", "hind"),
            (upper_pair, "leg", "front"),
        )
    for pair, region, group in pair_roles:
        matches, left_unmatched, right_unmatched = analysis.match_subtrees(
            pair.left_root,
            pair.right_root,
        )
        segment = 1
        for left, right in matches:
            assign(left, region, "l", group, segment, right)
            assign(right, region, "r", group, segment, left)
            segment += 1
        for node in left_unmatched:
            assign(node, region, "l", group, segment)
            segment += 1
        for node in right_unmatched:
            assign(node, region, "r", group, segment)
            segment += 1

    warnings = []
    generic_unclassified = [
        bone.name
        for index, bone in enumerate(bones)
        if index not in assignments_by_node and _GENERIC_BONE_PATTERN.fullmatch(bone.name)
    ]
    if generic_unclassified:
        warnings.append(f"{len(generic_unclassified)} 根附件骨骼置信度不足，已保留原名")
    if not head_nodes:
        warnings.append("未找到可独立命名的头部子树")
    if not tail_nodes and template == TEMPLATE_QUADRUPED:
        warnings.append("未识别到尾部骨骼")

    assignments = tuple(
        assignments_by_node[index]
        for index in range(len(bones))
        if index in assignments_by_node
    )
    targets = [assignment.target_name for assignment in assignments]
    if len(targets) != len(set(targets)):
        raise RigPostprocessError("语义命名产生了重复名称，未修改骨架")
    return NamingPlan(
        template=template,
        center=analysis.center,
        lateral_axis=_point_tuple(analysis.lateral_axis),
        forward_axis=_point_tuple(analysis.forward_axis),
        assignments=assignments,
        warnings=tuple(warnings),
    )


def resolve_postprocess_armature(context):
    active = getattr(context, "object", None)
    if active is not None and active.type == "ARMATURE":
        return active
    if active is not None and active.type == "MESH":
        armature = armature_for_mesh(active)
        if armature is not None:
            return armature
        source_name = str(active.get("skintokens_source_armature", ""))
        if source_name:
            candidate = context.blend_data.objects.get(source_name)
            if candidate is not None and candidate.type == "ARMATURE":
                return candidate

    candidates = set()
    for obj in getattr(context, "selected_objects", ()):
        if obj.type == "ARMATURE":
            candidates.add(obj)
        elif obj.type == "MESH":
            armature = armature_for_mesh(obj)
            if armature is not None:
                candidates.add(armature)
    return next(iter(candidates)) if len(candidates) == 1 else None


def postprocess_ready(armature_obj) -> bool:
    return bool(
        armature_obj is not None
        and armature_obj.type == "ARMATURE"
        and armature_obj.get(POSTPROCESS_READY_PROP, False)
    )


@dataclass(frozen=True)
class _ContextState:
    active_name: str
    active_mode: str
    selected_names: tuple[str, ...]


def _prepare_armature_context(context, armature_obj) -> _ContextState:
    bpy = __import__("bpy")
    active = context.view_layer.objects.active
    state = _ContextState(
        active_name="" if active is None else str(active.name),
        active_mode="OBJECT" if active is None else str(active.mode),
        selected_names=tuple(str(obj.name) for obj in context.selected_objects),
    )
    if active is not None and active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(context.selected_objects):
        obj.select_set(False)
    armature_obj.select_set(True)
    context.view_layer.objects.active = armature_obj
    if armature_obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    return state


def _restore_context(context, state: _ContextState) -> None:
    bpy = __import__("bpy")
    active = context.view_layer.objects.active
    if active is not None and active.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for obj in list(context.selected_objects):
        obj.select_set(False)
    for name in state.selected_names:
        obj = context.blend_data.objects.get(name)
        if obj is not None:
            obj.select_set(True)
    original = context.blend_data.objects.get(state.active_name)
    if original is None:
        context.view_layer.objects.active = None
        return
    original.select_set(True)
    context.view_layer.objects.active = original
    if state.active_mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode=state.active_mode)
        except RuntimeError:
            pass


def snapshot_armature(armature_obj) -> tuple[RigBoneSnapshot, ...]:
    if armature_obj is None or armature_obj.type != "ARMATURE":
        raise RigPostprocessError("请选择一个骨架或绑定该骨架的网格")
    source_bones = list(armature_obj.data.bones)
    name_to_index = {str(bone.name): index for index, bone in enumerate(source_bones)}
    matrix_world = armature_obj.matrix_world
    seen_ids: set[str] = set()
    snapshots = []
    for bone in source_bones:
        joint_id = str(bone.get(JOINT_ID_PROP, ""))
        if not joint_id or joint_id in seen_ids:
            joint_id = uuid.uuid4().hex
            bone[JOINT_ID_PROP] = joint_id
        seen_ids.add(joint_id)
        parent = -1 if bone.parent is None else name_to_index[str(bone.parent.name)]
        snapshots.append(
            RigBoneSnapshot(
                joint_id=joint_id,
                name=str(bone.name),
                parent=parent,
                head=_point_tuple(matrix_world @ bone.head_local),
                tail=_point_tuple(matrix_world @ bone.tail_local),
            )
        )
    return tuple(snapshots)


def _meshes_for_armature(armature_obj) -> list:
    bpy = __import__("bpy")
    result = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        uses_modifier = any(
            modifier.type == "ARMATURE" and modifier.object is armature_obj
            for modifier in obj.modifiers
        )
        if uses_modifier or (obj.parent is armature_obj and obj.parent_type == "ARMATURE"):
            result.append(obj)
    return result


def _iter_constraints():
    bpy = __import__("bpy")
    for obj in bpy.data.objects:
        for constraint in obj.constraints:
            yield constraint
        pose = getattr(obj, "pose", None)
        if pose is None:
            continue
        for pose_bone in pose.bones:
            for constraint in pose_bone.constraints:
                yield constraint


def _vertex_group_reference_owners(mesh_obj):
    seen: set[int] = set()
    owners = [*mesh_obj.modifiers, *mesh_obj.particle_systems]
    for item in [*mesh_obj.modifiers, *mesh_obj.particle_systems]:
        for attribute in ("settings", "collision_settings"):
            owner = getattr(item, attribute, None)
            if owner is not None:
                owners.append(owner)
    for owner in owners:
        pointer = getattr(owner, "as_pointer", None)
        key = int(pointer()) if pointer is not None else id(owner)
        if key in seen:
            continue
        seen.add(key)
        yield owner


def _vertex_group_references(meshes: Sequence, mapping: dict[str, str]) -> list:
    references = []
    for mesh in meshes:
        for owner in _vertex_group_reference_owners(mesh):
            rna = getattr(owner, "bl_rna", None)
            if rna is None:
                continue
            for prop in rna.properties:
                identifier = str(prop.identifier)
                if (
                    prop.type != "STRING"
                    or "vertex_group" not in identifier
                    or bool(getattr(prop, "is_readonly", False))
                ):
                    continue
                try:
                    value = str(getattr(owner, identifier))
                except (AttributeError, TypeError):
                    continue
                if value in mapping:
                    references.append((owner, identifier, value))
    return references


def _geometry_nodes_vertex_group_references(
    meshes: Sequence,
    mapping: dict[str, str],
) -> list:
    references = []
    for mesh in meshes:
        for modifier in mesh.modifiers:
            if modifier.type != "NODES":
                continue
            inputs = getattr(
                getattr(modifier, "properties", None),
                "inputs",
                None,
            )
            input_rna = getattr(inputs, "bl_rna", None)
            if input_rna is not None:
                for prop in input_rna.properties:
                    identifier = str(prop.identifier)
                    if prop.type != "POINTER":
                        continue
                    socket = getattr(inputs, identifier, None)
                    try:
                        socket_type = str(socket.type)
                        value = socket.attribute_name
                    except (AttributeError, TypeError):
                        continue
                    if socket_type == "ATTRIBUTE" and value in mapping:
                        references.append(
                            (socket, "attribute_name", value, False)
                        )

            # Blender 4.x and earlier store interface inputs as modifier ID
            # properties; Blender 5.2 exposes them through properties.inputs.
            try:
                keys = tuple(modifier.keys())
            except TypeError:
                keys = ()
            for key in keys:
                key = str(key)
                suffix = "_attribute_name"
                if not key.endswith(suffix):
                    continue
                use_attribute_key = f"{key[:-len(suffix)]}_use_attribute"
                try:
                    value = modifier[key]
                except (KeyError, TypeError):
                    continue
                if (
                    isinstance(value, str)
                    and value in mapping
                    and bool(modifier.get(use_attribute_key, False))
                ):
                    references.append((modifier, str(key), value, True))
    return references


def _copy_id_property(value):
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        to_list = getattr(value, "to_list", None)
        return to_list() if to_list is not None else value


def _capture_id_properties(owner, keys: Sequence[str]) -> dict:
    return {
        key: (True, _copy_id_property(owner[key])) if key in owner else (False, None)
        for key in keys
    }


def _restore_id_properties(owner, state: dict) -> None:
    for key, (existed, value) in state.items():
        if existed:
            owner[key] = value
        elif key in owner:
            del owner[key]


def _apply_naming_plan(armature_obj, plan: NamingPlan) -> None:
    bpy = __import__("bpy")
    if not bool(getattr(armature_obj, "is_editable", True)) or not bool(
        getattr(armature_obj.data, "is_editable", True)
    ):
        raise RigPostprocessError("链接或只读骨架不能执行身体分区命名")
    if int(getattr(armature_obj.data, "users", 1)) > 1:
        raise RigPostprocessError(
            "该骨架数据被多个 Armature 对象共享，请先将骨架数据改为单用户"
        )
    data_bones = armature_obj.data.bones
    mapping = {
        assignment.old_name: assignment.target_name
        for assignment in plan.assignments
        if assignment.old_name != assignment.target_name
    }
    if len(mapping) != plan.rename_count:
        raise RigPostprocessError("同一骨骼出现了多个重命名目标")
    existing_names = {str(bone.name) for bone in data_bones}
    source_names = set(mapping)
    for target in mapping.values():
        if target in existing_names and target not in source_names:
            raise RigPostprocessError(f"骨架中已存在名称 {target}，未执行重命名")

    meshes = _meshes_for_armature(armature_obj)
    managed_names_by_mesh = {
        str(mesh.name): managed_vertex_group_names(mesh, armature_obj)
        for mesh in meshes
    }
    for mesh in meshes:
        if not bool(getattr(mesh, "is_editable", True)):
            raise RigPostprocessError(
                f"链接或只读网格 {mesh.name} 不能同步顶点组名称"
            )
        group_names = {str(group.name) for group in mesh.vertex_groups}
        managed_names = managed_names_by_mesh[str(mesh.name)]
        for old_name, target in mapping.items():
            if (
                target in group_names
                and target not in source_names
                and target not in managed_names
            ):
                raise RigPostprocessError(
                    f"网格 {mesh.name} 已存在顶点组 {target}，未执行重命名"
                )

    parent_references = [
        (obj, str(obj.parent_bone))
        for obj in bpy.data.objects
        if obj.parent is armature_obj
        and obj.parent_type == "BONE"
        and str(obj.parent_bone) in mapping
    ]
    constraint_references = []
    for constraint in _iter_constraints():
        if (
            getattr(constraint, "target", None) is armature_obj
            and hasattr(constraint, "subtarget")
            and str(constraint.subtarget) in mapping
        ):
            constraint_references.append((constraint, "subtarget", str(constraint.subtarget)))
        for target in getattr(constraint, "targets", ()):
            if (
                getattr(target, "target", None) is armature_obj
                and hasattr(target, "subtarget")
                and str(target.subtarget) in mapping
            ):
                constraint_references.append((target, "subtarget", str(target.subtarget)))
    vertex_group_references = _vertex_group_references(meshes, mapping)
    geometry_nodes_references = _geometry_nodes_vertex_group_references(
        meshes,
        mapping,
    )

    token = uuid.uuid4().hex
    bone_temporary = {
        old_name: f"__skintokens_{token}_{index}"
        for index, old_name in enumerate(mapping)
    }
    group_temporary = {
        old_name: f"__skintokens_group_{token}_{index}"
        for index, old_name in enumerate(mapping)
    }
    group_references = []
    for mesh in meshes:
        for old_name in mapping:
            group = mesh.vertex_groups.get(old_name)
            if group is not None:
                group_references.append((mesh, group, old_name))

    final_bone_names = (existing_names - source_names) | set(mapping.values())
    source_groups_by_mesh = {
        str(mesh.name): {
            old_name
            for owner_mesh, _group, old_name in group_references
            if owner_mesh is mesh
        }
        for mesh in meshes
    }
    stale_group_references = []
    for mesh in meshes:
        managed_names = managed_names_by_mesh[str(mesh.name)]
        source_group_names = source_groups_by_mesh[str(mesh.name)]
        stale_names = {
            target
            for old_name, target in mapping.items()
            if old_name in source_group_names
            and target not in source_names
            and target in managed_names
            and mesh.vertex_groups.get(target) is not None
        }
        stale_names.update(
            name
            for name in managed_names
            if mapping.get(name, name) not in final_bone_names
        )
        for name in sorted(stale_names):
            group = mesh.vertex_groups.get(name)
            if group is not None:
                stale_group_references.append((mesh, group, name))

    semantic_keys = (
        SEMANTIC_REGION_PROP,
        SEMANTIC_SIDE_PROP,
        SEMANTIC_GROUP_PROP,
        SEMANTIC_INDEX_PROP,
        SEMANTIC_NAME_PROP,
        SEMANTIC_MIRROR_ID_PROP,
        SEMANTIC_ORIGINAL_NAME_PROP,
    )
    armature_semantic_keys = (
        SEMANTIC_VERSION_PROP,
        SEMANTIC_TEMPLATE_PROP,
        SEMANTIC_CENTER_PROP,
        SEMANTIC_LATERAL_AXIS_PROP,
        SEMANTIC_FORWARD_AXIS_PROP,
    )
    original_bone_names = {
        str(bone.get(JOINT_ID_PROP, "")): str(bone.name)
        for bone in data_bones
    }
    bone_property_states = {
        str(bone.get(JOINT_ID_PROP, "")): _capture_id_properties(
            bone,
            semantic_keys,
        )
        for bone in data_bones
    }
    armature_property_state = _capture_id_properties(
        armature_obj.data,
        armature_semantic_keys,
    )
    mesh_managed_property_states = {
        str(mesh.name): _capture_id_properties(
            mesh,
            (MANAGED_VERTEX_GROUPS_PROP,),
        )
        for mesh in meshes
    }

    try:
        for index, (_mesh, group, old_name) in enumerate(stale_group_references):
            temp_name = f"__skintokens_stale_{token}_{index}"
            group.name = temp_name
        for _mesh, group, old_name in group_references:
            group.name = group_temporary[old_name]
        for old_name, temp_name in bone_temporary.items():
            bone = data_bones.get(old_name)
            if bone is None:
                raise RigPostprocessError(f"找不到待重命名骨骼：{old_name}")
            bone.name = temp_name
        for old_name, target in mapping.items():
            data_bones[bone_temporary[old_name]].name = target
        for _mesh, group, old_name in group_references:
            group.name = mapping[old_name]
        for obj, old_name in parent_references:
            obj.parent_bone = mapping[old_name]
        for owner, attribute, old_name in constraint_references:
            setattr(owner, attribute, mapping[old_name])
        for owner, attribute, old_name in vertex_group_references:
            setattr(owner, attribute, mapping[old_name])
        for owner, key, old_name, is_id_property in geometry_nodes_references:
            if is_id_property:
                owner[key] = mapping[old_name]
            else:
                setattr(owner, key, mapping[old_name])

        for bone in data_bones:
            for key in semantic_keys[:-1]:
                if key in bone:
                    del bone[key]
        bones_by_id = {
            str(bone.get(JOINT_ID_PROP, "")): bone
            for bone in data_bones
        }
        for assignment in plan.assignments:
            bone = bones_by_id.get(assignment.joint_id)
            if bone is None:
                raise RigPostprocessError(
                    f"重命名后无法定位骨骼：{assignment.old_name}"
                )
            if SEMANTIC_ORIGINAL_NAME_PROP not in bone:
                bone[SEMANTIC_ORIGINAL_NAME_PROP] = assignment.old_name
            bone[SEMANTIC_REGION_PROP] = assignment.region
            bone[SEMANTIC_SIDE_PROP] = assignment.side
            bone[SEMANTIC_GROUP_PROP] = assignment.group
            bone[SEMANTIC_INDEX_PROP] = assignment.index
            bone[SEMANTIC_NAME_PROP] = assignment.semantic_name
            bone[SEMANTIC_MIRROR_ID_PROP] = assignment.mirror_joint_id
        armature_obj.data[SEMANTIC_VERSION_PROP] = SEMANTIC_VERSION
        armature_obj.data[SEMANTIC_TEMPLATE_PROP] = plan.template
        armature_obj.data[SEMANTIC_CENTER_PROP] = float(plan.center)
        armature_obj.data[SEMANTIC_LATERAL_AXIS_PROP] = list(plan.lateral_axis)
        armature_obj.data[SEMANTIC_FORWARD_AXIS_PROP] = list(plan.forward_axis)

        for mesh in meshes:
            managed_names = managed_names_by_mesh[str(mesh.name)]
            translated = {
                mapping.get(name, name)
                for name in managed_names
            }
            translated.update(
                mapping[name]
                for name in source_groups_by_mesh[str(mesh.name)]
            )
            existing_group_names = {
                str(group.name) for group in mesh.vertex_groups
            }
            set_managed_vertex_group_names(
                mesh,
                translated & final_bone_names & existing_group_names,
            )
        for mesh, group, _old_name in stale_group_references:
            mesh.vertex_groups.remove(group)
    except Exception as original_error:
        try:
            rollback_token = uuid.uuid4().hex
            rollback_groups = []
            for index, (_mesh, group, _old_name) in enumerate(group_references):
                if str(group.name) == _old_name:
                    continue
                group.name = f"__st_rb_group_{rollback_token}_{index}"
                rollback_groups.append((group, _old_name))
            current_by_id = {
                str(bone.get(JOINT_ID_PROP, "")): bone
                for bone in data_bones
            }
            rollback_bones = []
            for index, (joint_id, old_name) in enumerate(original_bone_names.items()):
                bone = current_by_id.get(joint_id)
                if bone is None:
                    continue
                if str(bone.name) == old_name:
                    continue
                temp_name = f"__st_rb_{rollback_token}_{index}"
                bone.name = temp_name
                rollback_bones.append((temp_name, old_name))
            for temp_name, old_name in rollback_bones:
                data_bones[temp_name].name = old_name
            for group, old_name in rollback_groups:
                group.name = old_name
            for _mesh, group, old_name in stale_group_references:
                group.name = old_name
            for obj, old_name in parent_references:
                obj.parent_bone = old_name
            for owner, attribute, old_name in constraint_references:
                setattr(owner, attribute, old_name)
            for owner, attribute, old_name in vertex_group_references:
                setattr(owner, attribute, old_name)
            for owner, key, old_name, is_id_property in geometry_nodes_references:
                if is_id_property:
                    owner[key] = old_name
                else:
                    setattr(owner, key, old_name)
            restored_by_id = {
                str(bone.get(JOINT_ID_PROP, "")): bone
                for bone in data_bones
            }
            for joint_id, property_state in bone_property_states.items():
                bone = restored_by_id.get(joint_id)
                if bone is not None:
                    _restore_id_properties(bone, property_state)
            _restore_id_properties(armature_obj.data, armature_property_state)
            for mesh in meshes:
                _restore_id_properties(
                    mesh,
                    mesh_managed_property_states[str(mesh.name)],
                )
        except Exception as rollback_error:
            raise RigPostprocessError(
                f"身体分区命名失败且回滚失败：{rollback_error}"
            ) from original_error
        raise


def rename_body_regions(context, armature_obj, template: str) -> RenameResult:
    state = _prepare_armature_context(context, armature_obj)
    joint_id_states = [
        (
            bone,
            JOINT_ID_PROP in bone,
            _copy_id_property(bone[JOINT_ID_PROP])
            if JOINT_ID_PROP in bone
            else None,
        )
        for bone in armature_obj.data.bones
    ]
    try:
        plan = build_naming_plan(snapshot_armature(armature_obj), template)
        _apply_naming_plan(armature_obj, plan)
        return RenameResult(
            template=plan.template,
            renamed_bones=plan.rename_count,
            classified_bones=len(plan.assignments),
            mirror_pairs=plan.mirror_pair_count,
            warnings=plan.warnings,
        )
    except Exception:
        for bone, existed, value in joint_id_states:
            if existed:
                bone[JOINT_ID_PROP] = value
            elif JOINT_ID_PROP in bone:
                del bone[JOINT_ID_PROP]
        raise
    finally:
        _restore_context(context, state)


def semantic_mirror_pair_count(armature_obj) -> int:
    if armature_obj is None or armature_obj.type != "ARMATURE":
        return 0
    if int(armature_obj.data.get(SEMANTIC_VERSION_PROP, 0)) != SEMANTIC_VERSION:
        return 0
    pairs = 0
    ids = {
        str(bone.get(JOINT_ID_PROP, "")): bone
        for bone in armature_obj.data.bones
    }
    for bone in armature_obj.data.bones:
        if str(bone.get(SEMANTIC_SIDE_PROP, "")) != "l":
            continue
        mirror_id = str(bone.get(SEMANTIC_MIRROR_ID_PROP, ""))
        mirror = ids.get(mirror_id)
        if mirror is None:
            continue
        if str(mirror.get(SEMANTIC_MIRROR_ID_PROP, "")) != str(
            bone.get(JOINT_ID_PROP, "")
        ):
            continue
        pairs += 1
    return pairs


def _mirror_hierarchy(
    bones: Sequence[RigBoneSnapshot],
) -> tuple[list[list[int]], list[int], list[int]]:
    children: list[list[int]] = [[] for _ in bones]
    parents = [int(bone.parent) for bone in bones]
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= len(bones):
            raise RigPostprocessError(f"骨骼 {bones[index].name} 的父级无效")
        if parent >= 0:
            children[parent].append(index)

    depths = [-1] * len(bones)

    def depth(index: int, visiting: set[int]) -> int:
        cached = depths[index]
        if cached >= 0:
            return cached
        if index in visiting:
            raise RigPostprocessError("骨架层级中存在循环")
        parent = parents[index]
        value = 0 if parent < 0 else depth(parent, {*visiting, index}) + 1
        depths[index] = value
        return value

    subtree_sizes = [0] * len(bones)

    def subtree_size(index: int) -> int:
        cached = subtree_sizes[index]
        if cached:
            return cached
        value = 1 + sum(subtree_size(child) for child in children[index])
        subtree_sizes[index] = value
        return value

    for index in range(len(bones)):
        depth(index, set())
        subtree_size(index)
    return children, depths, subtree_sizes


def _is_ancestor(parents: Sequence[int], ancestor: int, node: int) -> bool:
    parent = int(parents[node])
    while parent >= 0:
        if parent == ancestor:
            return True
        parent = int(parents[parent])
    return False


def _reflected_point(point: np.ndarray, axis_index: int, center: float) -> np.ndarray:
    reflected = np.asarray(point, dtype=np.float64).copy()
    reflected[axis_index] = 2.0 * center - reflected[axis_index]
    return reflected


def _maximum_weight_mirror_matching(
    candidates: Sequence[MirrorPairCandidate],
    positive_names: Sequence[str],
    negative_names: Sequence[str],
) -> tuple[MirrorPairCandidate, ...]:
    if not candidates or not positive_names or not negative_names:
        return ()

    positive_names = tuple(positive_names)
    negative_names = tuple(negative_names)
    positive_index = {name: index for index, name in enumerate(positive_names)}
    negative_index = {name: index for index, name in enumerate(negative_names)}
    by_pair = {
        (item.positive_name, item.negative_name): item
        for item in candidates
    }

    # Each positive-side bone also gets a dummy column, so leaving a bone
    # unmatched costs less than accepting an invalid or non-positive pair.
    row_count = len(positive_names)
    real_column_count = len(negative_names)
    column_count = real_column_count + row_count
    costs = np.ones((row_count, column_count), dtype=np.float64)
    costs[:, :real_column_count] = 1e6
    for item in candidates:
        row = positive_index[item.positive_name]
        column = negative_index[item.negative_name]
        costs[row, column] = 1.0 - float(item.score)

    # Rectangular Hungarian algorithm (rows <= columns).
    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    matched_row = np.zeros(column_count + 1, dtype=np.int64)
    previous_column = np.zeros(column_count + 1, dtype=np.int64)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = int(matched_row[column])
            delta = np.inf
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced_cost = (
                    costs[active_row - 1, candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < minimum[candidate_column]:
                    minimum[candidate_column] = reduced_cost
                    previous_column[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = int(previous_column[column])
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assigned_column = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] > 0:
            assigned_column[int(matched_row[column]) - 1] = column - 1

    result = []
    for row, column in enumerate(assigned_column):
        if column < 0 or column >= real_column_count:
            continue
        item = by_pair.get((positive_names[row], negative_names[column]))
        if item is not None:
            result.append(item)
    return tuple(result)


def infer_selected_mirror_pairs(
    bones: Sequence[RigBoneSnapshot],
    selected_names: Sequence[str],
    *,
    axis: str = MIRROR_AXIS_X,
    center: float = 0.0,
    minimum_score: float = 0.50,
    ambiguity_margin: float = 0.08,
) -> MirrorPairInference:
    axis_index, axis_vector = _mirror_axis(axis)
    center = float(center)
    if not np.isfinite(center):
        raise RigPostprocessError("对称面位置必须是有限数值")
    if len(bones) < 2:
        raise RigPostprocessError("骨架中没有足够的骨骼用于镜像配对")

    name_to_index = {bone.name: index for index, bone in enumerate(bones)}
    selected = tuple(dict.fromkeys(str(name) for name in selected_names))
    missing = sorted(name for name in selected if name not in name_to_index)
    if missing:
        raise RigPostprocessError(f"选中的骨骼不存在：{', '.join(missing)}")
    if len(selected) < 2:
        raise RigPostprocessError("请在编辑模式下至少高亮选择两根骨骼")

    children, depths, subtree_sizes = _mirror_hierarchy(bones)
    parents = [int(bone.parent) for bone in bones]
    heads = np.asarray([bone.head for bone in bones], dtype=np.float64)
    tails = np.asarray([bone.tail for bone in bones], dtype=np.float64)
    midpoints = 0.5 * (heads + tails)
    lengths = np.linalg.norm(tails - heads, axis=1)
    all_points = np.concatenate([heads, tails], axis=0)
    scale = max(float(np.linalg.norm(np.ptp(all_points, axis=0))), 1e-6)
    side_epsilon = max(scale * 1e-5, 1e-6)

    positive = []
    negative = []
    centered = []
    for name in selected:
        index = name_to_index[name]
        coordinate = float(midpoints[index, axis_index]) - center
        if coordinate > side_epsilon:
            positive.append(index)
        elif coordinate < -side_epsilon:
            negative.append(index)
        else:
            centered.append(index)

    if not positive or not negative:
        raise RigPostprocessError(
            "选中的骨骼必须分布在对称面的两侧；位于平面上的骨骼不会参与配对"
        )

    def candidate(positive_index: int, negative_index: int) -> MirrorPairCandidate:
        if _is_ancestor(parents, positive_index, negative_index) or _is_ancestor(
            parents,
            negative_index,
            positive_index,
        ):
            return MirrorPairCandidate(
                bones[positive_index].name,
                bones[negative_index].name,
                -1.0,
                float("inf"),
                -1.0,
            )

        reflected_head = _reflected_point(heads[negative_index], axis_index, center)
        reflected_tail = _reflected_point(tails[negative_index], axis_index, center)
        reflected_error = float(
            np.sqrt(
                0.5
                * (
                    np.sum((heads[positive_index] - reflected_head) ** 2)
                    + np.sum((tails[positive_index] - reflected_tail) ** 2)
                )
            )
        )
        positive_vector = tails[positive_index] - heads[positive_index]
        negative_vector = tails[negative_index] - heads[negative_index]
        reflected_negative_vector = (
            negative_vector
            - 2.0 * axis_vector * float(np.dot(negative_vector, axis_vector))
        )
        direction_denominator = float(
            lengths[positive_index] * lengths[negative_index]
        )
        direction_cosine = (
            -1.0
            if direction_denominator < 1e-9
            else float(
                np.dot(positive_vector, reflected_negative_vector)
                / direction_denominator
            )
        )
        direction_similarity = max(0.0, min(1.0, direction_cosine))
        length_similarity = float(
            min(lengths[positive_index], lengths[negative_index])
            / max(lengths[positive_index], lengths[negative_index], 1e-9)
        )
        pair_scale = max(
            0.5 * (lengths[positive_index] + lengths[negative_index]),
            scale * 0.03,
            1e-6,
        )
        position_similarity = float(np.exp(-reflected_error / (1.5 * pair_scale)))
        positive_distance = abs(float(midpoints[positive_index, axis_index]) - center)
        negative_distance = abs(float(midpoints[negative_index, axis_index]) - center)
        distance_similarity = 1.0 - abs(positive_distance - negative_distance) / max(
            positive_distance,
            negative_distance,
            side_epsilon,
        )

        positive_parent = parents[positive_index]
        negative_parent = parents[negative_index]
        if positive_parent == negative_parent:
            parent_similarity = 1.0
        elif positive_parent < 0 or negative_parent < 0:
            parent_similarity = 0.0
        else:
            reflected_parent = _reflected_point(
                midpoints[negative_parent],
                axis_index,
                center,
            )
            parent_error = float(
                np.linalg.norm(midpoints[positive_parent] - reflected_parent)
            )
            parent_similarity = float(np.exp(-parent_error / (2.0 * pair_scale)))
        depth_similarity = 1.0 / (1.0 + abs(depths[positive_index] - depths[negative_index]))
        child_similarity = 1.0 - abs(
            len(children[positive_index]) - len(children[negative_index])
        ) / max(
            len(children[positive_index]),
            len(children[negative_index]),
            1,
        )
        subtree_similarity = float(
            min(subtree_sizes[positive_index], subtree_sizes[negative_index])
            / max(subtree_sizes[positive_index], subtree_sizes[negative_index])
        )
        topology_similarity = 0.25 * (
            parent_similarity
            + depth_similarity
            + child_similarity
            + subtree_similarity
        )
        score = (
            0.45 * position_similarity
            + 0.20 * direction_similarity
            + 0.15 * length_similarity
            + 0.10 * distance_similarity
            + 0.10 * topology_similarity
        )
        return MirrorPairCandidate(
            positive_name=bones[positive_index].name,
            negative_name=bones[negative_index].name,
            score=float(score),
            reflected_error=reflected_error,
            direction_cosine=direction_cosine,
        )

    if len(selected) == 2:
        if centered:
            raise RigPostprocessError("位于对称面上的骨骼不能作为镜像对")
        explicit = candidate(positive[0], negative[0])
        if explicit.score < 0.0:
            raise RigPostprocessError("同一父子链上的骨骼不能互相镜像对齐")
        return MirrorPairInference((explicit,), (), ())

    candidates = []
    for positive_index in positive:
        for negative_index in negative:
            item = candidate(positive_index, negative_index)
            length_similarity = float(
                min(lengths[positive_index], lengths[negative_index])
                / max(lengths[positive_index], lengths[negative_index], 1e-9)
            )
            if (
                item.score >= minimum_score
                and item.direction_cosine >= 0.15
                and length_similarity >= 0.35
                and abs(depths[positive_index] - depths[negative_index]) <= 1
            ):
                candidates.append(item)

    candidates.sort(key=lambda item: -item.score)
    by_name: dict[str, list[MirrorPairCandidate]] = {}
    for item in candidates:
        by_name.setdefault(item.positive_name, []).append(item)
        by_name.setdefault(item.negative_name, []).append(item)
    for options in by_name.values():
        options.sort(key=lambda item: -item.score)

    assigned = _maximum_weight_mirror_matching(
        candidates,
        tuple(bones[index].name for index in positive),
        tuple(bones[index].name for index in negative),
    )
    assigned_partner = {
        item.positive_name: item.negative_name
        for item in assigned
    }
    assigned_partner.update(
        {item.negative_name: item.positive_name for item in assigned}
    )
    assigned_by_key = {
        (item.positive_name, item.negative_name): item
        for item in assigned
    }
    positive_children_by_parent: dict[int, int] = {}
    negative_children_by_parent: dict[int, int] = {}
    for index in positive:
        positive_children_by_parent[parents[index]] = (
            positive_children_by_parent.get(parents[index], 0) + 1
        )
    for index in negative:
        negative_children_by_parent[parents[index]] = (
            negative_children_by_parent.get(parents[index], 0) + 1
        )

    def locally_unambiguous(item: MirrorPairCandidate) -> bool:
        for name in (item.positive_name, item.negative_name):
            alternatives = [
                option.score
                for option in by_name.get(name, ())
                if option is not item
            ]
            if alternatives and item.score - max(alternatives) < ambiguity_margin:
                return False
        return True

    accepted_keys: set[tuple[str, str]] = set()
    for item in assigned:
        positive_parent = parents[name_to_index[item.positive_name]]
        negative_parent = parents[name_to_index[item.negative_name]]
        shares_unique_parent = (
            positive_parent >= 0
            and positive_parent == negative_parent
            and positive_children_by_parent.get(positive_parent) == 1
            and negative_children_by_parent.get(negative_parent) == 1
        )
        if shares_unique_parent or locally_unambiguous(item):
            accepted_keys.add((item.positive_name, item.negative_name))

    changed = True
    while changed:
        changed = False
        for item in assigned:
            key = (item.positive_name, item.negative_name)
            if key in accepted_keys:
                continue
            positive_parent = parents[name_to_index[item.positive_name]]
            negative_parent = parents[name_to_index[item.negative_name]]
            if positive_parent < 0 or negative_parent < 0:
                continue
            positive_parent_name = bones[positive_parent].name
            negative_parent_name = bones[negative_parent].name
            if assigned_partner.get(positive_parent_name) != negative_parent_name:
                continue
            if (
                positive_children_by_parent.get(positive_parent) != 1
                or negative_children_by_parent.get(negative_parent) != 1
            ):
                continue
            parent_key = (
                (positive_parent_name, negative_parent_name)
                if (positive_parent_name, negative_parent_name) in assigned_by_key
                else (negative_parent_name, positive_parent_name)
            )
            if parent_key in accepted_keys:
                accepted_keys.add(key)
                changed = True

    pairs = [
        item
        for item in assigned
        if (item.positive_name, item.negative_name) in accepted_keys
    ]
    ambiguous = {
        name
        for item in assigned
        if (item.positive_name, item.negative_name) not in accepted_keys
        for name in (item.positive_name, item.negative_name)
    }

    paired_names = {
        name
        for pair in pairs
        for name in (pair.positive_name, pair.negative_name)
    }
    unmatched = tuple(sorted(name for name in selected if name not in paired_names))
    return MirrorPairInference(
        pairs=tuple(pairs),
        unmatched_names=unmatched,
        ambiguous_names=tuple(sorted(ambiguous)),
    )


def _symmetric_points(
    left: np.ndarray,
    right: np.ndarray,
    axis: str,
    center: float,
    distance_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    axis_index, _axis_vector = _mirror_axis(axis)
    distance_mode = _mirror_distance_mode(distance_mode)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_distance = abs(float(left[axis_index]) - center)
    right_distance = abs(float(right[axis_index]) - center)
    if distance_mode == MIRROR_DISTANCE_FARTHEST:
        distance = max(left_distance, right_distance)
    elif distance_mode == MIRROR_DISTANCE_NEAREST:
        distance = min(left_distance, right_distance)
    else:
        distance = 0.5 * (left_distance + right_distance)

    tangent = 0.5 * (left + right)
    aligned_left = tangent.copy()
    aligned_right = tangent.copy()
    aligned_left[axis_index] = center + distance
    aligned_right[axis_index] = center - distance
    return aligned_left, aligned_right


def _mirror_axis(axis: str) -> tuple[int, np.ndarray]:
    if not isinstance(axis, str):
        raise RigPostprocessError("对称面只支持 X、Y 或 Z 轴，不能旋转")
    normalized = axis.upper()
    if normalized not in SUPPORTED_MIRROR_AXES:
        raise RigPostprocessError("对称面只支持 X、Y 或 Z 轴，不能旋转")
    axis_index = {MIRROR_AXIS_X: 0, MIRROR_AXIS_Y: 1, MIRROR_AXIS_Z: 2}[
        normalized
    ]
    axis_vector = np.zeros(3, dtype=np.float64)
    axis_vector[axis_index] = 1.0
    return axis_index, axis_vector


def _mirror_distance_mode(distance_mode: str) -> str:
    if not isinstance(distance_mode, str):
        raise RigPostprocessError("镜像距离策略只支持最远、最近或平均")
    normalized = distance_mode.upper()
    if normalized not in SUPPORTED_MIRROR_DISTANCE_MODES:
        raise RigPostprocessError("镜像距离策略只支持最远、最近或平均")
    return normalized


def _symmetric_bone_targets(
    left_head: np.ndarray,
    left_tail: np.ndarray,
    right_head: np.ndarray,
    right_tail: np.ndarray,
    axis: str,
    center: float,
    distance_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _axis_index, axis_vector = _mirror_axis(axis)
    distance_mode = _mirror_distance_mode(distance_mode)
    left_vector = left_tail - left_head
    right_vector = right_tail - right_head
    left_length = float(np.linalg.norm(left_vector))
    right_length = float(np.linalg.norm(right_vector))
    if min(left_length, right_length) < 1e-9:
        raise RigPostprocessError("镜像配对中存在零长度骨骼")
    reflected_right_vector = (
        right_vector
        - 2.0 * axis_vector * float(np.dot(right_vector, axis_vector))
    )
    direction_cosine = float(
        np.dot(left_vector, reflected_right_vector)
        / (left_length * right_length)
    )
    if direction_cosine < 0.15:
        raise RigPostprocessError("所选骨骼的方向不一致，不符合镜像关系")
    aligned_left_head, aligned_right_head = _symmetric_points(
        left_head,
        right_head,
        axis,
        center,
        distance_mode,
    )
    aligned_left_tail, aligned_right_tail = _symmetric_points(
        left_tail,
        right_tail,
        axis,
        center,
        distance_mode,
    )
    minimum_length = 0.1 * min(left_length, right_length)
    if (
        float(np.linalg.norm(aligned_left_tail - aligned_left_head))
        < minimum_length
        or float(np.linalg.norm(aligned_right_tail - aligned_right_head))
        < minimum_length
    ):
        raise RigPostprocessError("镜像对齐会使骨骼长度退化，操作已取消")
    return (
        aligned_left_head,
        aligned_left_tail,
        aligned_right_head,
        aligned_right_tail,
    )


def selected_edit_bone_names(context, armature_obj) -> tuple[str, ...]:
    if (
        armature_obj is None
        or armature_obj.type != "ARMATURE"
        or getattr(context, "object", None) is not armature_obj
        or str(getattr(context, "mode", "")) != "EDIT_ARMATURE"
        or str(getattr(armature_obj, "mode", "")) != "EDIT"
    ):
        return ()
    return tuple(
        str(bone.name)
        for bone in armature_obj.data.edit_bones
        if bool(bone.select)
    )


def _snapshot_edit_armature(armature_obj) -> tuple[RigBoneSnapshot, ...]:
    edit_bones = list(armature_obj.data.edit_bones)
    name_to_index = {
        str(bone.name): index for index, bone in enumerate(edit_bones)
    }
    return tuple(
        RigBoneSnapshot(
            joint_id=str(bone.get(JOINT_ID_PROP, "")) or str(bone.name),
            name=str(bone.name),
            parent=(
                -1
                if bone.parent is None
                else name_to_index[str(bone.parent.name)]
            ),
            head=_point_tuple(bone.head),
            tail=_point_tuple(bone.tail),
        )
        for bone in edit_bones
    )


def _connected_endpoint_targets(
    edit_bones,
    endpoint_targets: dict[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    bones = list(edit_bones)
    points = np.asarray(
        [
            _point_tuple(point)
            for bone in bones
            for point in (bone.head, bone.tail)
        ],
        dtype=np.float64,
    )
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-6)
    joint_epsilon = max(1e-6, scale * 1e-5)
    children_by_parent: dict[str, list] = {}
    for bone in bones:
        if bone.parent is not None:
            children_by_parent.setdefault(str(bone.parent.name), []).append(bone)

    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for bone in bones:
        if bone.parent is None:
            continue
        siblings = children_by_parent[str(bone.parent.name)]
        touches_parent = float((bone.head - bone.parent.tail).length) <= joint_epsilon
        all_siblings_share_joint = all(
            float((sibling.head - bone.parent.tail).length) <= joint_epsilon
            for sibling in siblings
        )
        virtual_connection = touches_parent and (
            len(siblings) == 1 or all_siblings_share_joint
        )
        if not bone.use_connect and not virtual_connection:
            continue
        parent_tail = (str(bone.parent.name), "tail")
        child_head = (str(bone.name), "head")
        adjacency.setdefault(parent_tail, set()).add(child_head)
        adjacency.setdefault(child_head, set()).add(parent_tail)

    resolved = {
        endpoint: np.asarray(value, dtype=np.float64).copy()
        for endpoint, value in endpoint_targets.items()
    }
    visited: set[tuple[str, str]] = set()
    for start in adjacency:
        if start in visited:
            continue
        component = set()
        pending = [start]
        while pending:
            endpoint = pending.pop()
            if endpoint in component:
                continue
            component.add(endpoint)
            pending.extend(adjacency.get(endpoint, ()))
        visited.update(component)
        requested = [
            endpoint_targets[endpoint]
            for endpoint in component
            if endpoint in endpoint_targets
        ]
        if not requested:
            continue
        shared_target = np.mean(np.asarray(requested, dtype=np.float64), axis=0)
        tolerance = max(1e-6, float(np.linalg.norm(shared_target)) * 1e-6)
        if any(
            float(np.linalg.norm(np.asarray(value) - shared_target)) > tolerance
            for value in requested
        ):
            names = sorted({name for name, _endpoint in component})
            raise RigPostprocessError(
                "连接关节收到不一致的镜像目标，请缩小选择范围："
                + ", ".join(names)
            )
        for endpoint in component:
            resolved[endpoint] = shared_target.copy()
    return resolved


def _mirror_align_edit_bones(
    context,
    armature_obj,
    *,
    selected_names: Sequence[str],
    axis: str = MIRROR_AXIS_X,
    center: float = 0.0,
    distance_mode: str = MIRROR_DISTANCE_AVERAGE,
) -> MirrorResult:
    _mirror_axis(axis)
    axis = axis.upper()
    distance_mode = _mirror_distance_mode(distance_mode)
    center = float(center)
    if not np.isfinite(center):
        raise RigPostprocessError("对称面位置必须是有限数值")
    if (
        getattr(context, "object", None) is not armature_obj
        or str(getattr(context, "mode", "")) != "EDIT_ARMATURE"
        or str(getattr(armature_obj, "mode", "")) != "EDIT"
    ):
        raise RigPostprocessError("镜像对齐只能用于当前活动骨架的编辑模式")
    if not bool(getattr(armature_obj, "is_editable", True)) or not bool(
        getattr(armature_obj.data, "is_editable", True)
    ):
        raise RigPostprocessError("链接或只读骨架不能执行镜像对齐")
    if int(getattr(armature_obj.data, "users", 1)) > 1:
        raise RigPostprocessError(
            "该骨架数据被多个 Armature 对象共享，请先将骨架数据改为单用户"
        )
    edit_bones = armature_obj.data.edit_bones
    inference = infer_selected_mirror_pairs(
        _snapshot_edit_armature(armature_obj),
        selected_names,
        axis=axis,
        center=center,
    )
    if not inference.pairs:
        detail = (
            ", ".join(inference.ambiguous_names or inference.unmatched_names)
            or "当前选择"
        )
        raise RigPostprocessError(
            f"没有识别出可靠的镜像骨骼对，请缩小选择范围：{detail}"
        )

    endpoint_targets: dict[tuple[str, str], np.ndarray] = {}
    paired_names = []
    for pair in inference.pairs:
        positive = edit_bones.get(pair.positive_name)
        negative = edit_bones.get(pair.negative_name)
        if positive is None or negative is None:
            raise RigPostprocessError("选中的镜像骨骼已不存在")
        try:
            (
                positive_head,
                positive_tail,
                negative_head,
                negative_tail,
            ) = _symmetric_bone_targets(
                np.asarray(positive.head, dtype=np.float64),
                np.asarray(positive.tail, dtype=np.float64),
                np.asarray(negative.head, dtype=np.float64),
                np.asarray(negative.tail, dtype=np.float64),
                axis,
                center,
                distance_mode,
            )
        except RigPostprocessError as exc:
            raise RigPostprocessError(
                f"骨骼 {pair.positive_name} / {pair.negative_name}：{exc}"
            ) from exc
        endpoint_targets[(pair.positive_name, "head")] = positive_head
        endpoint_targets[(pair.positive_name, "tail")] = positive_tail
        endpoint_targets[(pair.negative_name, "head")] = negative_head
        endpoint_targets[(pair.negative_name, "tail")] = negative_tail
        paired_names.append((pair.positive_name, pair.negative_name))

    resolved_targets = _connected_endpoint_targets(edit_bones, endpoint_targets)
    original_connections = {
        str(bone.name): bool(bone.use_connect)
        for bone in edit_bones
    }
    original_geometry = {
        str(bone.name): (
            _point_tuple(bone.head),
            _point_tuple(bone.tail),
            float(bone.roll),
        )
        for bone in edit_bones
    }
    affected_names = {name for name, _endpoint in resolved_targets}
    try:
        for bone in edit_bones:
            bone.use_connect = False
        for (name, endpoint), value in resolved_targets.items():
            setattr(edit_bones[name], endpoint, _point_tuple(value))
        for name, was_connected in original_connections.items():
            edit_bones[name].use_connect = was_connected
        if any(
            float((edit_bones[name].tail - edit_bones[name].head).length) < 1e-9
            for name in affected_names
        ):
            raise RigPostprocessError("镜像对齐产生了零长度骨骼，操作已回滚")
    except Exception:
        for bone in edit_bones:
            bone.use_connect = False
        for name, (head, tail, roll) in original_geometry.items():
            bone = edit_bones.get(name)
            if bone is None:
                continue
            bone.head = head
            bone.tail = tail
            bone.roll = roll
        for name, was_connected in original_connections.items():
            bone = edit_bones.get(name)
            if bone is not None:
                bone.use_connect = was_connected
        raise

    armature_obj.data.update_tag()
    return MirrorResult(
        mirror_pairs=len(paired_names),
        center=center,
        paired_bones=tuple(paired_names),
        skipped_bones=inference.unmatched_names,
    )


def mirror_align_selected_edit_bones(
    context,
    armature_obj,
    *,
    axis: str = MIRROR_AXIS_X,
    center: float = 0.0,
    distance_mode: str = MIRROR_DISTANCE_AVERAGE,
) -> MirrorResult:
    selected_names = selected_edit_bone_names(context, armature_obj)
    if len(selected_names) < 2:
        selected_names = tuple(
            str(bone.name) for bone in armature_obj.data.edit_bones
        )
    return _mirror_align_edit_bones(
        context,
        armature_obj,
        selected_names=selected_names,
        axis=axis,
        center=center,
        distance_mode=distance_mode,
    )


def mirror_align_bones(
    context,
    armature_obj,
    *,
    axis: str = MIRROR_AXIS_X,
    center: float = 0.0,
    distance_mode: str = MIRROR_DISTANCE_AVERAGE,
) -> MirrorResult:
    is_session_edit = (
        getattr(context, "object", None) is armature_obj
        and str(getattr(context, "mode", "")) == "EDIT_ARMATURE"
        and str(getattr(armature_obj, "mode", "")) == "EDIT"
    )
    if is_session_edit:
        selected_names = selected_edit_bone_names(context, armature_obj)
        if len(selected_names) < 2:
            selected_names = tuple(
                str(bone.name) for bone in armature_obj.data.edit_bones
            )
        return _mirror_align_edit_bones(
            context,
            armature_obj,
            selected_names=selected_names,
            axis=axis,
            center=center,
            distance_mode=distance_mode,
        )

    if armature_obj is None or getattr(armature_obj, "type", "") != "ARMATURE":
        raise RigPostprocessError("当前会话骨架不存在")
    bpy = __import__("bpy")
    state = _prepare_armature_context(context, armature_obj)
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        selected_names = tuple(
            str(bone.name) for bone in armature_obj.data.edit_bones
        )
        return _mirror_align_edit_bones(
            context,
            armature_obj,
            selected_names=selected_names,
            axis=axis,
            center=center,
            distance_mode=distance_mode,
        )
    finally:
        _restore_context(context, state)


def mirror_align_body_regions(
    context,
    armature_obj,
    *,
    axis: str = MIRROR_AXIS_X,
    center: float = 0.0,
    distance_mode: str = MIRROR_DISTANCE_AVERAGE,
) -> MirrorResult:
    return mirror_align_bones(
        context,
        armature_obj,
        axis=axis,
        center=center,
        distance_mode=distance_mode,
    )


__all__ = [
    "MIRROR_AXIS_X",
    "MIRROR_AXIS_Y",
    "MIRROR_AXIS_Z",
    "MIRROR_DISTANCE_AVERAGE",
    "MIRROR_DISTANCE_FARTHEST",
    "MIRROR_DISTANCE_NEAREST",
    "MirrorPairCandidate",
    "MirrorPairInference",
    "MirrorResult",
    "NamingPlan",
    "POSTPROCESS_READY_PROP",
    "RenameResult",
    "RigBoneSnapshot",
    "RigPostprocessError",
    "TEMPLATE_BIPED",
    "TEMPLATE_QUADRUPED",
    "SUPPORTED_MIRROR_AXES",
    "SUPPORTED_MIRROR_DISTANCE_MODES",
    "build_naming_plan",
    "infer_selected_mirror_pairs",
    "mirror_align_bones",
    "mirror_align_body_regions",
    "mirror_align_selected_edit_bones",
    "postprocess_ready",
    "rename_body_regions",
    "resolve_postprocess_armature",
    "selected_edit_bone_names",
    "semantic_mirror_pair_count",
    "snapshot_armature",
]
