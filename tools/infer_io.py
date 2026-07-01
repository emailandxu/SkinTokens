from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor

from infer_rigpatcher import RigSpec
from src.rig_package.info.asset import Asset


DEFAULT_INFO_LINES = (
    "info Scale 1.000000",
    "info Pivot 0.000000 0.000000 0.000000",
)


def obj_to_internal_point(x: float, y: float, z: float) -> List[float]:
    return [x, -z, y]


def parse_obj_face_index(token: str, vertex_count: int) -> int:
    raw_index = token.split("/", 1)[0]
    if raw_index == "":
        raise ValueError(f"invalid OBJ face token: {token}")
    index = int(raw_index)
    if index < 0:
        index = vertex_count + index
    else:
        index -= 1
    if index < 0 or index >= vertex_count:
        raise ValueError(f"OBJ face index out of range: {token}")
    return index


def order_face_like_bpy_parser(indices: List[int], vertices: List[List[float]]) -> List[List[int]]:
    faces: List[List[int]] = []
    edges = []
    for i, a in enumerate(indices):
        b = indices[(i + 1) % len(indices)]
        edges.append(tuple(sorted((a, b))))

    nodes = []
    adj: dict[int, List[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        nodes.append(a)
        nodes.append(b)

    nodes = list(set(sorted(nodes)))
    first = nodes[0]
    loop = []
    now = first
    seen = {}
    while True:
        loop.append(now)
        seen[now] = True
        if seen.get(adj[now][0]) is None:
            now = adj[now][0]
        elif seen.get(adj[now][1]) is None:
            now = adj[now][1]
        else:
            break

    polygon_vertices = np.asarray([vertices[i] for i in indices], dtype=np.float64)
    polygon_normal = np.cross(
        polygon_vertices[1] - polygon_vertices[0],
        polygon_vertices[2] - polygon_vertices[0],
    )
    for second, third in zip(loop[1:], loop[2:]):
        face = [first, second, third]
        v0, v1, v2 = (np.asarray(vertices[i], dtype=np.float64) for i in face)
        if float(np.dot(np.cross(v1 - v0, v2 - v0), polygon_normal)) < 0.0:
            face = [first, third, second]
        faces.append(face)
    return faces


def load_obj_asset(path: Path) -> Asset:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r") as f:
        for line in f:
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


def load_rig_txt(path: Path) -> RigSpec:
    joints_by_name: dict[str, List[float]] = {}
    obj_text_joints_by_name: dict[str, List[float]] = {}
    parents_by_child: dict[str, str] = {}
    root_name: Optional[str] = None

    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "joints":
                if len(parts) < 5:
                    raise ValueError(f"invalid joints line: {line.rstrip()}")
                obj_text_joints_by_name[parts[1]] = [
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ]
                joints_by_name[parts[1]] = obj_to_internal_point(
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            elif parts[0] == "root":
                if len(parts) < 2:
                    raise ValueError(f"invalid root line: {line.rstrip()}")
                root_name = parts[1]
            elif parts[0] == "hier":
                if len(parts) < 3:
                    raise ValueError(f"invalid hier line: {line.rstrip()}")
                parents_by_child[parts[2]] = parts[1]

    if not joints_by_name:
        raise ValueError(f"rig txt has no joints: {path}")
    if root_name is None:
        root_candidates = set(joints_by_name) - set(parents_by_child)
        if len(root_candidates) != 1:
            raise ValueError(f"rig txt must provide one root: {path}")
        root_name = next(iter(root_candidates))
    if root_name not in joints_by_name:
        raise ValueError(f"root joint {root_name!r} is not declared in {path}")

    children_by_parent: dict[str, List[str]] = {name: [] for name in joints_by_name}
    for child, parent in parents_by_child.items():
        if child not in joints_by_name or parent not in joints_by_name:
            continue
        children_by_parent[parent].append(child)

    names: List[str] = []
    parents: List[int] = []

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

    joints = np.asarray([joints_by_name[name] for name in names], dtype=np.float32)
    obj_text_joints = np.asarray(
        [obj_text_joints_by_name[name] for name in names],
        dtype=np.float32,
    )
    return RigSpec(
        joints=joints,
        parents=np.asarray(parents, dtype=np.int32),
        joint_names=names,
        obj_text_joints=obj_text_joints,
    )


def fit_affine(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit dst ~= [src, 1] @ matrix and return matrix plus max error."""
    if src.shape != dst.shape:
        raise ValueError(f"cannot fit affine for different shapes: {src.shape} vs {dst.shape}")
    x = np.concatenate([src, np.ones((src.shape[0], 1), dtype=src.dtype)], axis=1)
    matrix, *_ = np.linalg.lstsq(x.astype(np.float64), dst.astype(np.float64), rcond=None)
    pred = x.astype(np.float64) @ matrix
    max_err = float(np.abs(pred - dst).max())
    return matrix, max_err


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return x.astype(np.float64) @ matrix


def load_obj_text_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"invalid OBJ vertex line: {line.rstrip()}")
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"input OBJ has no text vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def build_debug_arrays(
    pred,
    batch: dict,
    latent_neighbors: Optional[np.ndarray | list[np.ndarray]],
) -> dict:
    debug_arrays = {
        "sampled_vertices": batch["vertices"][0].detach().float().cpu().numpy(),
    }
    if pred.skin_pred is not None:
        debug_arrays["sampled_skin_pred"] = pred.skin_pred.detach().float().cpu().numpy()
    if pred.output_ids is not None:
        debug_arrays["output_ids"] = pred.output_ids.detach().cpu().numpy()
    if latent_neighbors is not None:
        if isinstance(latent_neighbors, np.ndarray):
            debug_arrays["latent_neighbors"] = latent_neighbors
        else:
            debug_arrays["latent_neighbors"] = np.asarray(
                latent_neighbors,
                dtype=object,
            )
    return debug_arrays


def joint_names(asset) -> List[str]:
    if asset.joint_names is not None:
        return [str(x) for x in asset.joint_names]
    return [f"bone_{i}" for i in range(asset.joints.shape[0])]


def skin_items(row: np.ndarray, names: Sequence[str], topk: int, eps: float):
    clean = np.nan_to_num(row.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    clean[clean < 0.0] = 0.0
    if clean.sum() <= eps:
        clean[int(np.argmax(row))] = 1.0
    k = min(topk, clean.shape[0]) if topk > 0 else clean.shape[0]
    idx = np.argsort(-clean)[:k]
    idx = idx[clean[idx] > eps]
    if idx.shape[0] == 0:
        idx = np.array([int(np.argmax(clean))])
    weights = clean[idx]
    weights = weights / max(float(weights.sum()), eps)
    return [(names[int(i)], float(w)) for i, w in zip(idx, weights)]


def write_heter_txt(asset, obj_text_joints: np.ndarray, output_path: Path, topk: int, eps: float) -> None:
    if asset.skin.shape[0] != asset.vertices.shape[0]:
        raise ValueError("skin rows do not match asset vertices")
    names = joint_names(asset)
    if len(names) != obj_text_joints.shape[0]:
        raise ValueError("joint name count does not match joints")

    root_candidates = np.where(asset.parents == -1)[0]
    root = int(root_candidates[0]) if root_candidates.shape[0] else 0

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
            lines.append(f"hier {names[int(parent)]} {names[int(child)]}\n")
    for line in DEFAULT_INFO_LINES:
        lines.append(line + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines))
    print(f"[infer] wrote {output_path}")


def write_skin_with_txt_template(
    asset,
    template_path: Path,
    output_path: Path,
    topk: int,
    eps: float,
) -> None:
    if asset.skin.shape[0] != asset.vertices.shape[0]:
        raise ValueError("skin rows do not match asset vertices")
    names = joint_names(asset)
    if len(names) != asset.skin.shape[1]:
        raise ValueError("joint name count does not match skin columns")

    skin_lines = []
    for vertex_id, row in enumerate(asset.skin):
        parts = [f"skin {vertex_id}"]
        for name, weight in skin_items(row, names, topk=topk, eps=eps):
            parts.extend([name, f"{weight:.6f}"])
        skin_lines.append(" ".join(parts) + "\n")

    lines: List[str] = []
    skin_inserted = False
    for line in template_path.read_text().splitlines(keepends=True):
        if line.startswith("skin "):
            if not skin_inserted:
                lines.extend(skin_lines)
                skin_inserted = True
            continue
        lines.append(line)
    if not skin_inserted:
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            if line.startswith("hier ") or line.startswith("info "):
                insert_at = idx
                break
        lines[insert_at:insert_at] = skin_lines

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines))
    print(f"[infer] wrote {output_path}")


def write_debug_npz(asset, obj_text_joints: np.ndarray, output_path: Path, debug_arrays: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray(joint_names(asset), dtype=object)
    np.savez(
        output_path,
        vertices=asset.vertices,
        faces=asset.faces,
        joints=asset.joints,
        obj_text_joints=obj_text_joints,
        parents=asset.parents,
        skin=asset.skin,
        joint_names=names,
        **debug_arrays,
    )
    print(f"[infer] wrote {output_path}")
