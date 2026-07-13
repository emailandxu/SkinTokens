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
