from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from torch import Tensor


@dataclass
class SkeletonContext:
    joints: np.ndarray
    parents: np.ndarray
    joint_names: List[str]
    done: bool = False

    @classmethod
    def empty(cls, done: bool = False) -> "SkeletonContext":
        return cls(
            joints=np.zeros((0, 3), dtype=np.float32),
            parents=np.zeros((0,), dtype=np.int32),
            joint_names=[],
            done=done,
        )

    @classmethod
    def from_payload(cls, payload: dict) -> Optional["SkeletonContext"]:
        if "joints" not in payload or "parents" not in payload:
            return None
        joints = np.asarray(payload["joints"], dtype=np.float32).reshape((-1, 3))
        parents = np.asarray(payload["parents"], dtype=np.int32).reshape((-1,))
        if joints.shape[0] != parents.shape[0]:
            raise ValueError("joints and parents lengths do not match")
        names = payload.get("joint_names")
        if names is None:
            names = [f"bone_{i}" for i in range(joints.shape[0])]
        if len(names) != joints.shape[0]:
            raise ValueError("joint_names and joints lengths do not match")
        return cls(
            joints=joints,
            parents=parents,
            joint_names=[str(name) for name in names],
            done=bool(payload.get("done", False)),
        )

    def to_payload(self) -> dict:
        return {
            "joints": self.joints.astype(float).tolist(),
            "parents": self.parents.astype(int).tolist(),
            "joint_names": list(self.joint_names),
            "done": self.done,
        }

    def validate_hierarchy(self) -> "SkeletonContext":
        count = int(self.parents.shape[0])
        if self.joints.shape[0] != count or len(self.joint_names) != count:
            raise ValueError(
                "skeleton joints, parents, and joint_names must have equal lengths"
            )
        if len(set(self.joint_names)) != count:
            raise ValueError("skeleton joint_names must be unique")
        if count == 0:
            return self
        roots = [idx for idx, parent in enumerate(self.parents) if int(parent) == -1]
        if roots != [0]:
            raise ValueError(f"skeleton must have root at index 0, found roots {roots}")
        children: list[list[int]] = [[] for _ in range(count)]
        for child in range(1, count):
            parent = int(self.parents[child])
            if parent < 0 or parent >= child:
                raise ValueError(
                    f"joint {child} parent must reference an earlier joint, got {parent}"
                )
            children[parent].append(child)
        dfs_order: list[int] = []

        def visit(joint: int) -> None:
            dfs_order.append(joint)
            for child in children[joint]:
                visit(child)

        visit(0)
        if dfs_order != list(range(count)):
            raise ValueError("skeleton joints are not stored in DFS order")
        return self


@dataclass
class SessionRecord:
    session_id: str
    owner_id: str
    asset_id: str
    obj_path: Path
    created_at: float
    updated_at: float
    client_ip: str = "local"
    reserved_joint_names: set[str] = field(default_factory=set)
    next_bone_id: int = 0
    initial_bone_count: int = 0
    latest_bone_count: int = 0
    max_bone_count: int = 0
    vertex_count: int = 0
    skin_generation_count: int = 0
    final_has_skin: bool = False
    start_logged: bool = False
    end_logged: bool = False

    def observe_bones(self, count: int) -> None:
        count = max(0, int(count))
        self.latest_bone_count = count
        self.max_bone_count = max(self.max_bone_count, count)

    def observe_skin(self) -> None:
        self.skin_generation_count += 1
        self.final_has_skin = True


@dataclass
class InteractiveSession:
    session_id: str
    obj_path: Path
    device: str
    cls: str
    vertices: Tensor
    normals: Tensor
    faces: np.ndarray
    learned_mesh_cond: Tensor
    cond_latents: Tensor
    normalized_vertices_cpu: np.ndarray
    blender_vertices: np.ndarray
    obj_text_vertices: np.ndarray
    normalized_to_blender: np.ndarray
    blender_to_normalized: np.ndarray
    normalized_to_obj_text: np.ndarray
    blender_to_obj_text: np.ndarray
    context: SkeletonContext = field(default_factory=SkeletonContext.empty)
    created_at: float = 0.0
    updated_at: float = 0.0

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        return apply_affine(points, self.blender_to_normalized).astype(np.float32)

    def denormalize_points(self, points: np.ndarray) -> np.ndarray:
        return apply_affine(points, self.normalized_to_blender).astype(np.float32)

    def to_obj_text_points(self, points: np.ndarray) -> np.ndarray:
        normalized = self.normalize_points(points)
        return apply_affine(normalized, self.normalized_to_obj_text).astype(np.float32)


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points.astype(np.float64).reshape((0, 3))
    x = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return x.astype(np.float64) @ matrix


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape != dst.shape:
        raise ValueError(f"cannot fit affine for different shapes: {src.shape} vs {dst.shape}")
    x = np.concatenate([src, np.ones((src.shape[0], 1), dtype=src.dtype)], axis=1)
    matrix, *_ = np.linalg.lstsq(x.astype(np.float64), dst.astype(np.float64), rcond=None)
    return matrix
