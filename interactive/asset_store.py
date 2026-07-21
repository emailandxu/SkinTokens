from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.rig_package.info.asset import Asset

from .rig_text import load_rig_txt_payload
from .session import SkeletonContext


DEFAULT_INFO_LINES = (
    "info Scale 1.000000",
    "info Pivot 0.000000 0.000000 0.000000",
)


@dataclass(frozen=True)
class MeshAsset:
    asset_id: str
    path: Path


class MeshAssetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_obj(self, content: bytes) -> MeshAsset:
        if not content:
            raise ValueError("uploaded OBJ is empty")
        asset_id = hashlib.sha256(content).hexdigest()
        path = self.root / f"{asset_id}.obj"
        if path.is_file():
            return MeshAsset(asset_id=asset_id, path=path)

        temporary = self.root / f".{asset_id}.{uuid.uuid4().hex}.tmp"
        temporary.write_bytes(content)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return MeshAsset(asset_id=asset_id, path=path)


def obj_to_internal_point(x: float, y: float, z: float) -> List[float]:
    return [x, -z, y]


def parse_obj_face_index(token: str, vertex_count: int) -> int:
    raw_index = token.split("/", 1)[0]
    if raw_index == "":
        raise ValueError(f"invalid OBJ face token: {token}")
    index = int(raw_index)
    index = vertex_count + index if index < 0 else index - 1
    if index < 0 or index >= vertex_count:
        raise ValueError(f"OBJ face index out of range: {token}")
    return index


def order_face_like_bpy_parser(
    indices: List[int],
    vertices: List[List[float]],
) -> List[List[int]]:
    edges = [
        tuple(sorted((a, indices[(index + 1) % len(indices)])))
        for index, a in enumerate(indices)
    ]
    adjacency: dict[int, List[int]] = {}
    nodes: list[int] = []
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        nodes.extend((a, b))

    first = min(set(nodes))
    loop: list[int] = []
    current = first
    seen: set[int] = set()
    while True:
        loop.append(current)
        seen.add(current)
        unseen = [node for node in adjacency[current] if node not in seen]
        if not unseen:
            break
        current = unseen[0]

    polygon_vertices = np.asarray([vertices[index] for index in indices], dtype=np.float64)
    polygon_normal = np.cross(
        polygon_vertices[1] - polygon_vertices[0],
        polygon_vertices[2] - polygon_vertices[0],
    )
    faces: List[List[int]] = []
    for second, third in zip(loop[1:], loop[2:]):
        face = [first, second, third]
        v0, v1, v2 = (np.asarray(vertices[index], dtype=np.float64) for index in face)
        if float(np.dot(np.cross(v1 - v0, v2 - v0), polygon_normal)) < 0.0:
            face = [first, third, second]
        faces.append(face)
    return faces


def load_obj_asset(path: Path) -> Asset:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"invalid OBJ vertex line: {line.rstrip()}")
                vertices.append(
                    obj_to_internal_point(
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                )
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    raise ValueError(f"invalid OBJ face line: {line.rstrip()}")
                indices = [parse_obj_face_index(token, len(vertices)) for token in parts]
                faces.extend(order_face_like_bpy_parser(indices, vertices))

    if not vertices:
        raise ValueError(f"input OBJ has no text vertices: {path}")
    if not faces:
        raise ValueError(f"input OBJ has no faces: {path}")

    asset = Asset(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        cls="articulation",
        path=str(path),
    )
    asset.build_normals()
    asset.change_dtype(float_dtype=np.float32, int_dtype=np.int32)
    return asset


def load_obj_text_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r") as obj_file:
        for line in obj_file:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"invalid OBJ vertex line: {line.rstrip()}")
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"input OBJ has no text vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def load_rig_txt(path: Path) -> SkeletonContext:
    context = SkeletonContext.from_payload(load_rig_txt_payload(path))
    if context is None:
        raise ValueError(f"rig txt did not produce a skeleton context: {path}")
    return context


def apply_skeleton(asset: Asset, context: SkeletonContext) -> None:
    matrix_local = np.repeat(
        np.eye(4, dtype=np.float32)[None, :, :],
        context.joints.shape[0],
        axis=0,
    )
    matrix_local[:, :3, 3] = context.joints
    asset.matrix_local = matrix_local
    asset.parents = context.parents.copy()
    asset.joint_names = list(context.joint_names)


def joint_names(asset: Asset) -> List[str]:
    if asset.joint_names is not None:
        return [str(name) for name in asset.joint_names]
    if asset.joints is None:
        raise ValueError("asset has no joints")
    return [f"bone_{index}" for index in range(asset.joints.shape[0])]


def skin_items(
    row: np.ndarray,
    names: Sequence[str],
    topk: int,
    eps: float,
) -> list[tuple[str, float]]:
    clean = np.nan_to_num(
        row.astype(np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    clean[clean < 0.0] = 0.0
    if clean.sum() <= eps:
        clean[int(np.argmax(row))] = 1.0
    count = min(topk, clean.shape[0]) if topk > 0 else clean.shape[0]
    indices = np.argsort(-clean)[:count]
    indices = indices[clean[indices] > eps]
    if indices.shape[0] == 0:
        indices = np.asarray([int(np.argmax(clean))])
    weights = clean[indices]
    weights = weights / max(float(weights.sum()), eps)
    return [(names[int(index)], float(weight)) for index, weight in zip(indices, weights)]


def heter_txt_content(
    asset: Asset,
    obj_text_joints: np.ndarray,
    topk: int,
    eps: float,
) -> str:
    if asset.skin is None or asset.vertices is None or asset.parents is None:
        raise ValueError("asset is missing skin, vertices, or parents")
    if asset.skin.shape[0] != asset.vertices.shape[0]:
        raise ValueError("skin rows do not match asset vertices")
    names = joint_names(asset)
    if len(names) != obj_text_joints.shape[0]:
        raise ValueError("joint name count does not match joints")

    roots = np.where(asset.parents == -1)[0]
    root = int(roots[0]) if roots.shape[0] else 0
    lines: List[str] = []
    for name, xyz in zip(names, obj_text_joints):
        lines.append(f"joints {name} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")
    lines.append(f"root {names[root]}\n")
    for vertex_id, row in enumerate(asset.skin):
        parts = [f"skin {vertex_id}"]
        for name, weight in skin_items(row, names, topk=topk, eps=eps):
            parts.extend([name, f"{weight:.6f}"])
        lines.append(" ".join(parts) + "\n")
    for child, parent in enumerate(asset.parents.tolist()):
        if parent != -1:
            lines.append(f"hier {names[int(parent)]} {names[child]}\n")
    lines.extend(f"{line}\n" for line in DEFAULT_INFO_LINES)
    return "".join(lines)


def write_heter_txt(
    asset: Asset,
    obj_text_joints: np.ndarray,
    output_path: Path,
    topk: int,
    eps: float,
) -> None:
    content = heter_txt_content(
        asset,
        obj_text_joints=obj_text_joints,
        topk=topk,
        eps=eps,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    print(f"[interactive] wrote {output_path}")
