from __future__ import annotations

import argparse
import os
import time
import traceback
import uuid
from types import SimpleNamespace
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from infer.args import DEFAULT_MODEL_CKPT
from infer.infer import get_model
from infer.infer_io import load_obj_asset, load_obj_text_vertices, write_heter_txt
from infer.infer_postprocess import (
    DEFAULT_LATENT_CHECKPOINT,
    LatentSkinSmoother,
    latent_smooth_options_from_args,
)
from infer.infer_rigpatcher import similar_subtree_order
from src.data.vertex_group import voxel_skin
from src.data.transform import Transform
from src.model.spec import ModelInput
from src.model.tokenrig import encode_mesh_cond
from src.rig_package.info.asset import Asset

from .protocol import AUTHKEY, MODEL_SOCKET_PATH, Request, Response, ensure_runtime_dir, err, ok
from .session import InteractiveSession, SkeletonContext, fit_affine
from .skeleton_stream import GenerationOptions, generate_next, generate_skin


SKIN_REORDER_NONE = "none"
SKIN_REORDER_SIMILAR_SUBTREES = "similar-subtrees"
SKIN_POSTPROCESS_NONE = "none"
SKIN_POSTPROCESS_VOXEL = "voxel"
SKIN_POSTPROCESS_LATENT = "latent"


def _remap_parents(parents: np.ndarray, order: list[int]) -> np.ndarray:
    old_to_new = {old: new for new, old in enumerate(order)}
    return np.asarray(
        [-1 if parents[old] == -1 else old_to_new[int(parents[old])] for old in order],
        dtype=np.int32,
    )


def _reordered_context_for_skin(
    session: InteractiveSession,
    context: SkeletonContext,
    mode: str,
) -> tuple[SkeletonContext, list[int]]:
    identity = list(range(context.joints.shape[0]))
    if mode in ("", SKIN_REORDER_NONE, None):
        return context, identity
    if mode != SKIN_REORDER_SIMILAR_SUBTREES:
        raise ValueError(f"unknown skin reorder mode: {mode}")

    normalized_joints = session.normalize_points(context.joints)
    order = similar_subtree_order(normalized_joints, context.parents)
    if order == identity:
        return context, identity
    order_array = np.asarray(order, dtype=np.int64)
    return (
        SkeletonContext(
            joints=context.joints[order_array].copy(),
            parents=_remap_parents(context.parents, order),
            joint_names=[context.joint_names[old] for old in order],
            done=context.done,
        ),
        order,
    )


def _remap_skin_to_original_order(skin: np.ndarray, order: list[int]) -> np.ndarray:
    if order == list(range(len(order))):
        return skin
    remapped = np.zeros_like(skin)
    for new_idx, old_idx in enumerate(order):
        remapped[:, old_idx] = skin[:, new_idx]
    return remapped


def _apply_skin_postprocess(asset: Asset, payload: Request, device: str) -> str:
    raw_mode = payload.get("skin_postprocess", SKIN_POSTPROCESS_NONE)
    mode = SKIN_POSTPROCESS_NONE if raw_mode in (None, "") else str(raw_mode)
    if mode == SKIN_POSTPROCESS_NONE:
        return SKIN_POSTPROCESS_NONE
    if asset.skin is None:
        raise RuntimeError("asset has no skin to postprocess")
    if asset.joints is None or asset.vertices is None or asset.faces is None:
        raise RuntimeError("asset is missing joints, vertices, or faces for postprocess")

    if mode == SKIN_POSTPROCESS_VOXEL:
        voxel = asset.voxel(resolution=int(payload.get("voxel_resolution", 196)))
        asset.skin *= voxel_skin(
            grid=0,
            grid_coords=voxel.coords,
            joints=asset.joints,
            vertices=asset.vertices,
            faces=asset.faces,
            mode="square",
            voxel_size=voxel.voxel_size,
        )
        asset.normalize_skin()
        return mode

    if mode == SKIN_POSTPROCESS_LATENT:
        args = SimpleNamespace(
            latent_smooth_iterations=int(payload.get("latent_smooth_iterations", 10)),
            latent_neighbor_factor=float(payload.get("latent_neighbor_factor", 0.3)),
            latent_k=int(payload.get("latent_k", 12)),
            latent_threshold_std=float(payload.get("latent_threshold_std", -1.5)),
            latent_checkpoint=payload.get("latent_checkpoint", DEFAULT_LATENT_CHECKPOINT),
            device=device,
        )
        options = latent_smooth_options_from_args(args)
        asset.skin, _ = LatentSkinSmoother(options)(asset)
        asset.normalize_skin()
        return mode

    raise ValueError(f"unknown skin postprocess mode: {mode}")


class InteractiveModelServer:
    def __init__(self, model_ckpt: str, hf_path: Optional[str], device: str) -> None:
        self.model = get_model(model_ckpt, hf_path=hf_path, device=device)
        self.model.eval()
        self.device = device
        self.sessions: Dict[str, InteractiveSession] = {}

    def start(self, payload: Request) -> Response:
        obj_path = Path(str(payload["obj_path"])).expanduser().resolve()
        asset = load_obj_asset(obj_path)
        blender_vertices = asset.vertices.copy() if asset.vertices is not None else None
        obj_text_vertices = load_obj_text_vertices(obj_path)
        transform = Transform.parse(**self.model.transform_config["predict_transform"])
        transform.apply(asset=asset)
        if asset.faces is None:
            raise RuntimeError("transformed asset is missing faces")
        if asset.sampled_vertices is None or asset.sampled_normals is None:
            model_input = ModelInput(asset=asset, tokens=None)
            processed = self.model._process_fn([model_input])[0]
            vertices = torch.from_numpy(processed["vertices"]).float()
            normals = torch.from_numpy(processed["normals"]).float()
            asset = processed["non"]["model_input"].asset
        else:
            vertices = torch.from_numpy(asset.sampled_vertices).float()
            normals = torch.from_numpy(asset.sampled_normals).float()
        vertices = vertices.to(self.device)
        normals = normals.to(self.device)
        cond = torch.cat([vertices, normals], dim=-1).unsqueeze(0)
        device_type = "cuda" if vertices.device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, cond_latents = self.model.vae.model._encode(
                x=None,
                cond=cond,
                num_tokens=self.model.tokens_per_skin,
                cond_tokens=self.model.tokens_skin_cond,
                return_z=False,
            )
            learned_mesh_cond = encode_mesh_cond(
                self.model.mesh_encoder,
                self.model.output_proj,
                self.model.tokens_skin_cond,
                {"vertices": vertices, "normals": normals},
            )
        if cond_latents is None:
            raise RuntimeError("VAE did not return cond_latents")
        if asset.vertices is None:
            raise RuntimeError("transformed asset is missing vertices")
        if blender_vertices is None:
            raise RuntimeError("input asset is missing vertices")
        normalized_vertices = asset.vertices.astype(np.float32)
        normalized_to_blender = fit_affine(normalized_vertices.astype(np.float64), blender_vertices.astype(np.float64))
        blender_to_normalized = fit_affine(blender_vertices.astype(np.float64), normalized_vertices.astype(np.float64))
        normalized_to_obj_text = fit_affine(normalized_vertices.astype(np.float64), obj_text_vertices)
        now = time.time()
        session_id = uuid.uuid4().hex
        session = InteractiveSession(
            session_id=session_id,
            obj_path=obj_path,
            device=self.device,
            cls=str(asset.cls or "articulation"),
            vertices=vertices,
            normals=normals,
            faces=asset.faces.copy(),
            learned_mesh_cond=learned_mesh_cond,
            cond_latents=cond_latents,
            normalized_vertices_cpu=normalized_vertices,
            blender_vertices=blender_vertices.astype(np.float32),
            obj_text_vertices=obj_text_vertices,
            normalized_to_blender=normalized_to_blender,
            blender_to_normalized=blender_to_normalized,
            normalized_to_obj_text=normalized_to_obj_text,
            context=SkeletonContext.empty(),
            created_at=now,
            updated_at=now,
        )
        self.sessions[session_id] = session
        return ok(
            session_id=session_id,
            context=session.context.to_payload(),
            socket=str(MODEL_SOCKET_PATH),
        )

    def _session(self, payload: Request) -> InteractiveSession:
        session_id = str(payload["session_id"])
        if session_id not in self.sessions:
            raise KeyError(f"unknown session_id: {session_id}")
        session = self.sessions[session_id]
        session.updated_at = time.time()
        return session

    def _context(self, session: InteractiveSession, payload: Request) -> SkeletonContext:
        context = SkeletonContext.from_payload(payload)
        if context is None:
            context = session.context
        return context

    def next(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        options = GenerationOptions.from_payload(payload)
        branch_parent = payload.get("branch_parent")
        branch_parent = None if branch_parent in (None, "", -1) else int(branch_parent)
        context, tokens = generate_next(self.model, session, context, options, branch_parent=branch_parent)
        session.context = context
        return ok(
            session_id=session.session_id,
            context=context.to_payload(),
            tokens=tokens.astype(int).tolist(),
        )

    def branch(self, payload: Request) -> Response:
        if "branch_parent" not in payload:
            return err("branch requires branch_parent")
        payload = {**payload, "command": "next"}
        return self.next(payload)

    def sync(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        session.context = context
        return ok(
            session_id=session.session_id,
            context=context.to_payload(),
        )

    def skin(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        if context.joints.shape[0] == 0:
            raise ValueError("cannot generate skin for an empty skeleton")
        options = GenerationOptions.from_payload(payload)
        raw_skin_reorder = payload.get("skin_reorder", SKIN_REORDER_NONE)
        skin_reorder = SKIN_REORDER_NONE if raw_skin_reorder in (None, "") else str(raw_skin_reorder)
        generation_context, skin_order = _reordered_context_for_skin(
            session,
            context,
            skin_reorder,
        )
        result = generate_skin(self.model, session, generation_context, options)
        if result.skin_pred is None:
            raise RuntimeError("model did not generate complete skin tokens")
        generated_skin = result.skin_pred.detach().float().cpu().numpy()
        skin = _remap_skin_to_original_order(generated_skin, skin_order)
        normalized_joints = session.normalize_points(context.joints)
        asset = Asset.from_data(
            vertices=session.normalized_vertices_cpu,
            faces=session.faces,
            sampled_vertices=session.vertices.detach().float().cpu().numpy(),
            sampled_skin=skin,
            joints=normalized_joints,
            parents=context.parents.copy(),
            joint_names=list(context.joint_names),
            cls=session.cls,
            path=str(session.obj_path),
        )
        applied_postprocess = _apply_skin_postprocess(asset, payload, self.device)
        output_path = payload.get("output_path")
        if output_path is not None:
            write_heter_txt(
                asset=asset,
                obj_text_joints=session.to_obj_text_points(context.joints),
                output_path=Path(str(output_path)).expanduser().resolve(),
                topk=int(payload.get("topk_skin", 4)),
                eps=float(payload.get("weight_eps", 1e-8)),
            )
        session.context = SkeletonContext(
            joints=context.joints,
            parents=context.parents,
            joint_names=context.joint_names,
            done=True,
        )
        return ok(
            session_id=session.session_id,
            context=session.context.to_payload(),
            skin_shape=list(generated_skin.shape),
            asset_skin_shape=[] if asset.skin is None else list(asset.skin.shape),
            skin_reorder=skin_reorder,
            skin_order=skin_order,
            skin_postprocess=applied_postprocess,
            output_path=None if output_path is None else str(Path(str(output_path)).expanduser().resolve()),
        )

    def reset(self, payload: Request) -> Response:
        session_id = str(payload["session_id"])
        self.sessions.pop(session_id, None)
        return ok(session_id=session_id)

    def handle(self, payload: Request) -> Response:
        command = str(payload.get("command", ""))
        if command == "start":
            return self.start(payload)
        if command == "next":
            return self.next(payload)
        if command == "branch":
            return self.branch(payload)
        if command == "sync":
            return self.sync(payload)
        if command == "skin":
            return self.skin(payload)
        if command == "reset":
            return self.reset(payload)
        if command == "ping":
            return ok(message="pong")
        return err(f"unknown command: {command}")


def serve(args: argparse.Namespace) -> None:
    ensure_runtime_dir()
    socket_path = Path(args.socket).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = InteractiveModelServer(
        model_ckpt=args.model_ckpt,
        hf_path=args.hf_path,
        device=args.device,
    )
    print(f"[interactive] model server listening on {socket_path}")
    with Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY) as listener:
        while True:
            conn = listener.accept()
            try:
                payload = conn.recv()
                response = server.handle(payload)
            except Exception as exc:
                response = err(f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            try:
                conn.send(response)
            finally:
                conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SkinTokens model server.")
    parser.add_argument("--socket", default=str(MODEL_SOCKET_PATH))
    parser.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser


def main() -> None:
    serve(build_parser().parse_args())


if __name__ == "__main__":
    main()
