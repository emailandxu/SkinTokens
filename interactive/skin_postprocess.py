from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from src.data.transform import Transform
from src.rig_package.utils import sample_vertex_groups
from src.rig_package.info.asset import Asset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATENT_CHECKPOINT = str(REPO_ROOT / "experiments" / "articulation-xl.ckpt")
MICHELANGELO_ENCODER_CONFIG = {
    "pretrained_path": None,
    "freeze_encoder": False,
    "device": "cpu",
    "dtype": "float32",
    "num_latents": 512,
    "embed_dim": 64,
    "point_feats": 3,
    "num_freqs": 8,
    "include_pi": False,
    "heads": 8,
    "width": 512,
    "num_encoder_layers": 16,
    "use_ln_post": True,
    "init_scale": 0.25,
    "qkv_bias": False,
    "use_checkpoint": False,
    "flash": False,
    "supervision_type": "sdf",
    "query_method": False,
    "token_num": 1024,
}


@dataclass(frozen=True)
class VaeReconstructionOptions:
    topk: int = 4
    batch_size: int = 4
    seed: int = 1234
    decode_chunk: int = 4096


def normalized_topk_weights(weights: np.ndarray, topk: int) -> np.ndarray:
    clean = np.nan_to_num(
        np.asarray(weights, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    clean[clean < 0.0] = 0.0
    if clean.ndim != 2 or clean.shape[1] == 0:
        raise ValueError(f"skin weights must have shape (vertices, bones), got {clean.shape}")
    if 0 < int(topk) < clean.shape[1]:
        keep = np.argpartition(clean, -int(topk), axis=1)[:, -int(topk) :]
        mask = np.zeros_like(clean, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        clean[~mask] = 0.0
    row_sum = clean.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-12
    if np.any(empty):
        clean[empty, 0] = 1.0
        row_sum = clean.sum(axis=1, keepdims=True)
    return (clean / row_sum).astype(np.float32)


class VaeSkinReconstructor:
    """Project every bone weight field through the pretrained Hard-FSQ VAE."""

    def __init__(self, vae, options: VaeReconstructionOptions) -> None:
        self.vae = vae
        self.options = options

    @torch.no_grad()
    def __call__(
        self,
        asset: Asset,
        bone_indices: Optional[Sequence[int]] = None,
        mesh_cond_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        if asset.vertices is None or asset.faces is None or asset.skin is None:
            raise ValueError("VAE reconstruction requires mesh vertices, faces, and skin")
        if asset.vertex_normals is None or asset.face_normals is None:
            asset.build_normals()

        input_weights = normalized_topk_weights(asset.skin, self.options.topk)
        transform = Transform.parse(
            **self.vae.transform_config["predict_transform"]
        )
        sampler = transform.sampler
        if sampler is None or sampler.num_skin_samples is None:
            raise RuntimeError("VAE predict transform is missing dense skin sampling")

        parameter = next(self.vae.parameters())
        device = parameter.device
        dtype = parameter.dtype
        device_type = device.type
        started = time.perf_counter()
        random_state = np.random.get_state()
        np.random.seed(int(self.options.seed))
        try:
            uniform_vertices, uniform_normals, uniform_weights = sample_vertex_groups(
                vertices=asset.vertices,
                faces=asset.faces,
                num_samples=sampler.num_samples,
                vertex_normals=asset.vertex_normals,
                face_normals=asset.face_normals,
                vertex_groups=input_weights,
                face_mask=None,
                shuffle=True,
                same=True,
            )
            if uniform_normals is None or uniform_weights is None:
                raise RuntimeError("VAE uniform sampling did not return normals and weights")
            uniform_cond_np = np.concatenate(
                [uniform_vertices[:, 0], uniform_normals[:, 0]],
                axis=1,
            ).astype(np.float32)
            query_cond_np = np.concatenate(
                [asset.vertices, asset.vertex_normals],
                axis=1,
            ).astype(np.float32)
            query_cond = torch.from_numpy(query_cond_np).unsqueeze(0).to(device, dtype)

            cond_tokens = mesh_cond_tokens
            if cond_tokens is None:
                uniform_cond = torch.from_numpy(uniform_cond_np).unsqueeze(0).to(
                    device,
                    dtype,
                )
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type == "cuda",
                ):
                    _, cond_tokens = self.vae.model._encode(
                        x=None,
                        cond=uniform_cond,
                        num_tokens=self.vae.sample_tokens,
                        cond_tokens=self.vae.cond_tokens,
                        seed=int(self.options.seed),
                        return_z=False,
                    )
            else:
                cond_tokens = cond_tokens.to(device=device, dtype=dtype)
            if cond_tokens is None:
                raise RuntimeError("VAE did not return mesh condition tokens")

            reconstructed = input_weights.copy()
            bone_count = input_weights.shape[1]
            if bone_indices is None:
                selected_bones = list(range(bone_count))
            else:
                selected_bones = list(dict.fromkeys(int(index) for index in bone_indices))
                if not selected_bones:
                    raise ValueError("VAE reconstruction requires at least one bone")
                invalid = [
                    index
                    for index in selected_bones
                    if index < 0 or index >= bone_count
                ]
                if invalid:
                    raise ValueError(
                        f"VAE reconstruction bone indices out of range: {invalid}"
                    )
            batch_size = max(1, int(self.options.batch_size))
            for start in range(0, len(selected_bones), batch_size):
                batch_bones = selected_bones[start : start + batch_size]
                encoder_rows = []
                for bone in batch_bones:
                    field = input_weights[:, bone]
                    face_mask = sampler.sample_on_skin(
                        skin=field,
                        vertices=asset.vertices,
                        faces=asset.faces,
                    )
                    dense_vertices, dense_normals, dense_weights = sample_vertex_groups(
                        vertices=asset.vertices,
                        faces=asset.faces,
                        num_samples=sampler.num_skin_samples,
                        num_vertex_samples=sampler.num_vertex_samples,
                        vertex_normals=asset.vertex_normals,
                        face_normals=asset.face_normals,
                        vertex_groups=field,
                        face_mask=face_mask,
                        shuffle=True,
                        same=True,
                    )
                    if dense_normals is None or dense_weights is None:
                        raise RuntimeError("VAE dense sampling did not return normals and weights")
                    dense_cond = np.concatenate(
                        [dense_vertices[:, 0], dense_normals[:, 0]],
                        axis=1,
                    ).astype(np.float32)
                    full_cond = np.concatenate([uniform_cond_np, dense_cond], axis=0)
                    full_weights = np.concatenate(
                        [uniform_weights[:, bone], dense_weights[:, 0]],
                        axis=0,
                    ).astype(np.float32)
                    encoder_rows.append(
                        np.concatenate([full_cond, full_weights[:, None]], axis=1)
                    )

                encoder_input = torch.from_numpy(np.stack(encoder_rows)).to(device, dtype)
                current_batch = len(batch_bones)
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type == "cuda",
                ):
                    z, _ = self.vae.model._encode(
                        x=encoder_input,
                        cond=None,
                        num_tokens=self.vae.sample_tokens,
                        seed=int(self.options.seed),
                        return_cond=False,
                    )
                    quantized, _indices, _ = self.vae.model.FSQ(z)
                    decoded = self.vae.model._decode(
                        z=quantized,
                        cond=cond_tokens.expand(current_batch, -1, -1),
                        sampled_points=query_cond.expand(current_batch, -1, -1),
                        num_chunks=max(1, int(self.options.decode_chunk)),
                    )
                reconstructed[:, batch_bones] = decoded[..., 0].permute(
                    1, 0
                ).float().cpu().numpy()
        finally:
            np.random.set_state(random_state)

        output = normalized_topk_weights(reconstructed, self.options.topk)
        residual = np.abs(input_weights - output)
        selected_residual = residual[:, selected_bones]
        report: dict[str, object] = {
            "mode": "vae-reconstruction",
            "bones": len(selected_bones),
            "total_bones": int(input_weights.shape[1]),
            "bone_indices": selected_bones,
            "batch_size": max(1, int(self.options.batch_size)),
            "seed": int(self.options.seed),
            "reused_mesh_condition": mesh_cond_tokens is not None,
            "input_samples": int(sampler.num_samples + sampler.num_skin_samples),
            "mae": float(np.mean(selected_residual)),
            "rmse": float(np.sqrt(np.mean(selected_residual * selected_residual))),
            "full_skin_mae": float(np.mean(residual)),
            "full_skin_rmse": float(np.sqrt(np.mean(residual * residual))),
            "wall_sec": time.perf_counter() - started,
        }
        return output, report


@dataclass(frozen=True)
class LatentSmoothOptions:
    iterations: int
    neighbor_factor: float
    latent_k: int
    threshold_std: Optional[float]
    checkpoint_path: Optional[Path]
    device: str


class MichelangeloVertexFeatureExtractor:
    def __init__(self, *, checkpoint_path: Optional[Path], device: Optional[str]) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None

    @torch.no_grad()
    def __call__(self, *, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        model = self._load_model()
        vertices_norm = self._normalize_vertices(
            np.asarray(vertices, dtype=np.float32)
        ).astype(np.float32)
        normals = self._compute_normals(
            vertices_norm,
            np.asarray(faces, dtype=np.int64),
        ).astype(np.float32)
        pc = torch.from_numpy(vertices_norm).unsqueeze(0).to(self.device)
        feats = torch.from_numpy(normals).unsqueeze(0).to(self.device)

        _, latents, _, _ = model.encode_latents(pc=pc, feats=feats)
        point_data = model.encoder.fourier_embedder(pc)
        point_data = torch.cat([point_data, feats], dim=-1)
        point_tokens = model.encoder.input_proj(point_data)
        vertex_features = model.encoder.cross_attn(point_tokens, latents)
        if model.encoder.ln_post is not None:
            vertex_features = model.encoder.ln_post(vertex_features)
        vertex_features = F.normalize(vertex_features[0].float(), dim=-1)
        return vertex_features.detach().cpu().numpy().astype(np.float32)

    def _load_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        from src.model.michelangelo.get_model import get_encoder_simplified

        config = dict(MICHELANGELO_ENCODER_CONFIG)
        config["device"] = self.device
        model = get_encoder_simplified(**config).to(self.device)
        self._load_optional_weights(model)
        model.eval()
        self._model = model
        return model

    def _load_optional_weights(self, model: torch.nn.Module) -> None:
        if self.checkpoint_path is None:
            return
        checkpoint_path = self.checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Michelangelo checkpoint not found: {checkpoint_path}")

        try:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        status = model.load_state_dict(state, strict=False)
        if not status.missing_keys:
            return

        filtered = {}
        for key, value in state.items():
            for prefix in ("global_encoder.", "model.global_encoder."):
                if key.startswith(prefix):
                    filtered[key[len(prefix) :]] = value
        if not filtered:
            raise RuntimeError(
                f"{checkpoint_path} does not contain direct or global_encoder weights"
            )
        model.load_state_dict(filtered, strict=False)

    @staticmethod
    def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
        centered = vertices - center
        scale = np.max(np.abs(centered))
        if scale <= 1e-8:
            raise RuntimeError("Degenerate vertex bounds")
        return centered / scale

    @staticmethod
    def _compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)


class LatentSkinSmoother:
    def __init__(self, options: LatentSmoothOptions) -> None:
        self.options = options
        self.feature_extractor = MichelangeloVertexFeatureExtractor(
            checkpoint_path=options.checkpoint_path,
            device=options.device,
        )

    def __call__(self, asset: Asset) -> tuple[np.ndarray, np.ndarray | list[np.ndarray]]:
        if asset.vertices is None:
            raise ValueError("asset has no vertices")
        if asset.faces is None:
            raise ValueError("asset has no faces")
        if asset.skin is None:
            raise ValueError("asset has no skin")

        latent_features = self.feature_extractor(
            vertices=asset.vertices,
            faces=asset.faces,
        )
        return self.smooth_weights_by_latent_knn(asset.skin, latent_features)

    def smooth_weights_by_latent_knn(
        self,
        weights: np.ndarray,
        latent_features: np.ndarray,
        *,
        candidate_k: int = 32,
        min_k: int = 3,
    ) -> tuple[np.ndarray, np.ndarray | list[np.ndarray]]:
        if self.options.threshold_std is None:
            neighbors = self.latent_knn_neighbors(
                latent_features,
                k=self.options.latent_k,
            )
        else:
            neighbors = self.adaptive_latent_neighbors(
                latent_features,
                candidate_k=candidate_k,
                max_k=self.options.latent_k,
                min_k=min_k,
                threshold_std=self.options.threshold_std,
            )
        smoothed = self.smooth_weights_with_neighbors(weights, neighbors)
        return smoothed, neighbors

    def smooth_weights_with_neighbors(
        self,
        weights: np.ndarray,
        neighbors: list[np.ndarray] | np.ndarray,
    ) -> np.ndarray:
        smoothed_weights = copy.deepcopy(weights)
        for _ in range(self.options.iterations):
            new_weights = copy.deepcopy(smoothed_weights)
            for i in range(smoothed_weights.shape[0]):
                vertex_neighbors = neighbors[i]
                if len(vertex_neighbors) == 0:
                    continue
                neighbor_weights = np.mean(smoothed_weights[vertex_neighbors], axis=0)
                new_weights[i] = (
                    (1.0 - self.options.neighbor_factor) * smoothed_weights[i]
                    + self.options.neighbor_factor * neighbor_weights
                )
                if np.sum(new_weights[i]) > 0:
                    new_weights[i] /= np.sum(new_weights[i])
            smoothed_weights = new_weights
        return smoothed_weights

    @classmethod
    def latent_knn_neighbors(cls, features: np.ndarray, k: int) -> np.ndarray:
        _, indices = cls._latent_distance_topk(features, k)
        return indices

    @classmethod
    def adaptive_latent_neighbors(
        cls,
        features: np.ndarray,
        *,
        candidate_k: int,
        max_k: int,
        min_k: int,
        threshold_std: float,
    ) -> list[np.ndarray]:
        candidate_k = max(int(candidate_k), int(max_k), int(min_k), 1)
        distances, indices = cls._latent_distance_topk(features, candidate_k)
        max_k = max(1, int(max_k))
        min_k = max(1, min(int(min_k), max_k))
        neighbors: list[np.ndarray] = []
        for row_dist, row_idx in zip(distances, indices):
            mean = float(np.mean(row_dist))
            std = float(np.std(row_dist))
            threshold = mean + float(threshold_std) * std
            selected = row_idx[row_dist <= threshold][:max_k]
            if selected.shape[0] < min_k:
                selected = row_idx[:min_k]
            neighbors.append(selected.astype(np.int64, copy=False))
        return neighbors

    @staticmethod
    def _latent_distance_topk(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        features_t = torch.from_numpy(np.asarray(features, dtype=np.float32))
        features_t = F.normalize(features_t, dim=-1)
        distances = torch.cdist(features_t, features_t)
        distances.fill_diagonal_(float("inf"))
        k = max(1, min(int(k), features_t.shape[0] - 1))
        topk = torch.topk(distances, k=k, largest=False)
        return topk.values.cpu().numpy(), topk.indices.cpu().numpy()


def latent_smooth_options_from_payload(
    payload: dict,
    *,
    device: str,
) -> LatentSmoothOptions:
    threshold = payload.get("latent_threshold_std", -1.5)
    checkpoint = payload.get("latent_checkpoint", DEFAULT_LATENT_CHECKPOINT)
    return LatentSmoothOptions(
        iterations=int(payload.get("latent_smooth_iterations", 10)),
        neighbor_factor=float(payload.get("latent_neighbor_factor", 0.3)),
        latent_k=int(payload.get("latent_k", 12)),
        threshold_std=None if threshold is None else float(threshold),
        checkpoint_path=Path(str(checkpoint)) if checkpoint else None,
        device=device,
    )
