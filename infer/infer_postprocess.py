from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

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


def latent_smooth_options_from_args(args: argparse.Namespace) -> LatentSmoothOptions:
    return LatentSmoothOptions(
        iterations=args.latent_smooth_iterations,
        neighbor_factor=args.latent_neighbor_factor,
        latent_k=args.latent_k,
        threshold_std=args.latent_threshold_std,
        checkpoint_path=Path(args.latent_checkpoint) if args.latent_checkpoint else None,
        device=args.device,
    )
