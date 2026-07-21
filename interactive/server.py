from __future__ import annotations

import argparse
import os
import re
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Optional

import numpy as np
import torch

os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

from src.data.vertex_group import voxel_skin
from src.data.transform import Transform
from src.model.spec import ModelInput
from src.model.tokenrig import TokenRig, encode_mesh_cond
from src.rig_package.info.asset import Asset

from .asset_store import (
    heter_txt_content,
    load_obj_asset,
    load_obj_text_vertices,
    write_heter_txt,
)
from .skin_postprocess import (
    LatentSkinSmoother,
    VaeReconstructionOptions,
    VaeSkinReconstructor,
    latent_smooth_options_from_payload,
)
from .skin_generation import (
    SKIN_MODE_DFS_ENSEMBLE,
    SkinEnsembleOptions,
    generate_skin_ensemble,
)

from .protocol import (
    AUTHKEY,
    MODEL_SOCKET_PATH,
    RUNTIME_DIR,
    Request,
    Response,
    decode_float32_array,
    encode_float32_array,
    ensure_runtime_dir,
    err,
    ok,
)
from .session import InteractiveSession, SessionRecord, SkeletonContext, fit_affine
from .server_config import DEFAULT_SERVER_CONFIG_PATH, parse_args_with_config
from .skeleton_stream import (
    GenerationOptions,
    generate_next,
    generate_rig,
    generate_skin,
    reorder_skeleton_context,
    similar_subtree_order,
)
from .usage_events import UsageEventLog


MIDPROCESS_NONE = "none"
MIDPROCESS_SIMILAR_SUBTREES = "similar-subtrees"
MIDPROCESS_DFS_ENSEMBLE = SKIN_MODE_DFS_ENSEMBLE
DEFAULT_MIDPROCESS = MIDPROCESS_DFS_ENSEMBLE
SKIN_POSTPROCESS_NONE = "none"
SKIN_POSTPROCESS_VOXEL = "voxel"
SKIN_POSTPROCESS_LATENT = "latent"
SKIN_POSTPROCESS_VAE_RECONSTRUCTION = "vae-reconstruction"
SKIN_POSTPROCESS_MODES = (
    SKIN_POSTPROCESS_NONE,
    SKIN_POSTPROCESS_VOXEL,
    SKIN_POSTPROCESS_LATENT,
    SKIN_POSTPROCESS_VAE_RECONSTRUCTION,
)
DEFAULT_MAX_CONTEXT_BONES = 96
DEFAULT_MAX_SESSIONS = 512
DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS = 3600.0
DEFAULT_SESSION_CLEANUP_INTERVAL_SECONDS = 60.0
DEFAULT_MODEL_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_USAGE_DIR = RUNTIME_DIR / "usage"
PROTOCOL_VERSION = 1
SERVER_VERSION = "1.8.2"
GENERIC_BONE_NAME = re.compile(r"^bone_(\d+)(?:\.\d+)?$")


def get_model(
    checkpoint_path: str,
    hf_path: Optional[str] = None,
    device: str = "cuda",
) -> TokenRig:
    model = TokenRig.load_from_system_checkpoint(checkpoint_path=checkpoint_path)
    if hf_path is not None:
        from transformers import AutoModel

        hf_model = AutoModel.from_pretrained(
            hf_path,
            local_files_only=True,
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model.transformer.model.load_state_dict(hf_model.state_dict())
    return model.to(device)


def _reordered_context_for_skin(
    session: InteractiveSession,
    context: SkeletonContext,
    mode: str,
) -> tuple[SkeletonContext, list[int]]:
    identity = list(range(context.joints.shape[0]))
    if mode in ("", MIDPROCESS_NONE, None):
        return context, identity
    if mode != MIDPROCESS_SIMILAR_SUBTREES:
        raise ValueError(f"unknown midprocess mode: {mode}")

    normalized_joints = session.normalize_points(context.joints)
    order = similar_subtree_order(normalized_joints, context.parents)
    if order == identity:
        return context, identity
    return reorder_skeleton_context(context, order), order


def _remap_skin_to_original_order(skin: np.ndarray, order: list[int]) -> np.ndarray:
    if order == list(range(len(order))):
        return skin
    remapped = np.zeros_like(skin)
    for new_idx, old_idx in enumerate(order):
        remapped[:, old_idx] = skin[:, new_idx]
    return remapped


def _apply_skin_postprocess(
    asset: Asset,
    payload: Request,
    model,
    device: str,
    default_mode: str = SKIN_POSTPROCESS_NONE,
) -> str:
    raw_mode = payload.get("skin_postprocess", default_mode)
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
        options = latent_smooth_options_from_payload(
            payload,
            device=device,
        )
        asset.skin, _ = LatentSkinSmoother(options)(asset)
        asset.normalize_skin()
        return mode

    if mode == SKIN_POSTPROCESS_VAE_RECONSTRUCTION:
        asset.skin, report = VaeSkinReconstructor(
            model.vae,
            VaeReconstructionOptions(
                topk=int(payload.get("topk_skin", 4)),
                batch_size=max(
                    1,
                    int(payload.get("vae_reconstruction_batch_size", 4)),
                ),
                seed=int(payload.get("vae_reconstruction_seed", 1234)),
                decode_chunk=max(
                    1,
                    int(payload.get("vae_reconstruction_decode_chunk", 4096)),
                ),
            ),
        )(asset)
        print(
            "[interactive] VAE reconstruction "
            f"bones={report['bones']} mae={report['mae']:.6f} "
            f"wall={report['wall_sec']:.3f}s"
        )
        return mode

    raise ValueError(f"unknown skin postprocess mode: {mode}")


class InteractiveServiceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def exception_response(exc: Exception) -> Response:
    if isinstance(exc, InteractiveServiceError):
        return err(str(exc), code=exc.code)
    return err(f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())


class InteractiveModelServer:
    def __init__(
        self,
        model_ckpt: str,
        hf_path: Optional[str],
        device: str,
        max_runtime_sessions: int = 8,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_context_bones: int = DEFAULT_MAX_CONTEXT_BONES,
        default_skin_postprocess: str = SKIN_POSTPROCESS_NONE,
        usage_dir: str | Path | None = DEFAULT_USAGE_DIR,
        session_idle_timeout_seconds: float = DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
        session_cleanup_interval_seconds: float = DEFAULT_SESSION_CLEANUP_INTERVAL_SECONDS,
    ) -> None:
        self.model = get_model(model_ckpt, hf_path=hf_path, device=device)
        self.model.eval()
        self.device = device
        self.max_runtime_sessions = max(1, int(max_runtime_sessions))
        self.max_sessions = max(1, int(max_sessions))
        self.max_context_bones = max(1, int(max_context_bones))
        if default_skin_postprocess not in SKIN_POSTPROCESS_MODES:
            raise ValueError(
                f"unknown default skin postprocess mode: {default_skin_postprocess}"
            )
        self.default_skin_postprocess = default_skin_postprocess
        self.session_idle_timeout_seconds = max(
            0.0,
            float(session_idle_timeout_seconds),
        )
        self.session_cleanup_interval_seconds = max(
            0.0,
            float(session_cleanup_interval_seconds),
        )
        self.model_version = Path(model_ckpt).name
        self.usage_events = None if usage_dir is None else UsageEventLog(usage_dir)
        self._state_lock = threading.RLock()
        self._maintenance_stop = threading.Event()
        self.sessions: OrderedDict[str, SessionRecord] = OrderedDict()
        self.runtime_cache: OrderedDict[str, InteractiveSession] = OrderedDict()
        self._maintenance_thread: threading.Thread | None = None
        if (
            self.session_idle_timeout_seconds > 0
            and self.session_cleanup_interval_seconds > 0
        ):
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                name="skintokens-session-cleanup",
                daemon=True,
            )
            self._maintenance_thread.start()

    def _lock(self):
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._state_lock = lock
        return lock

    def _maintenance_loop(self) -> None:
        stop = self._maintenance_stop
        while not stop.wait(self.session_cleanup_interval_seconds):
            self.expire_idle_sessions()

    def expire_idle_sessions(self, *, now: float | None = None) -> int:
        timeout = float(getattr(self, "session_idle_timeout_seconds", 0.0))
        if timeout <= 0:
            return 0
        current_time = time.time() if now is None else float(now)
        expired: list[SessionRecord] = []
        with self._lock():
            for session_id, record in list(self.sessions.items()):
                if current_time - float(record.updated_at) < timeout:
                    break
                removed = self.sessions.pop(session_id, None)
                self.runtime_cache.pop(session_id, None)
                if removed is not None:
                    expired.append(removed)
        for record in expired:
            self._log_session_end(record, "idle_timeout")
        return len(expired)

    def _log_session_start(self, record: SessionRecord) -> None:
        if record.start_logged:
            return
        record.start_logged = True
        usage_events = getattr(self, "usage_events", None)
        if usage_events is None:
            return
        usage_events.append(
            "session_start",
            server_version=SERVER_VERSION,
            model_version=str(getattr(self, "model_version", "unknown")),
            device=str(getattr(self, "device", "unknown")),
            session_id=record.session_id,
            client_id=record.owner_id,
            client_ip=record.client_ip,
            blender_extension_version=record.blender_extension_version,
            initial_bone_count=record.initial_bone_count,
            initial_has_skin=False,
            vertex_count=record.vertex_count,
        )

    def _log_session_end(self, record: SessionRecord, reason: str) -> None:
        if record.end_logged:
            return
        record.end_logged = True
        usage_events = getattr(self, "usage_events", None)
        if usage_events is None:
            return
        ended_at = time.time()
        usage_events.append(
            "session_end",
            server_version=SERVER_VERSION,
            model_version=str(getattr(self, "model_version", "unknown")),
            device=str(getattr(self, "device", "unknown")),
            session_id=record.session_id,
            client_id=record.owner_id,
            client_ip=record.client_ip,
            blender_extension_version=record.blender_extension_version,
            end_reason=str(reason or "reset")[:64],
            duration_seconds=max(0.0, ended_at - float(record.created_at)),
            initial_bone_count=record.initial_bone_count,
            final_bone_count=record.latest_bone_count,
            max_bone_count=record.max_bone_count,
            net_bone_change=(
                record.latest_bone_count - record.initial_bone_count
            ),
            final_added_bones=max(
                0,
                record.latest_bone_count - record.initial_bone_count,
            ),
            final_removed_bones=max(
                0,
                record.initial_bone_count - record.latest_bone_count,
            ),
            peak_added_bones=max(
                0,
                record.max_bone_count - record.initial_bone_count,
            ),
            skin_generation_count=record.skin_generation_count,
            final_has_skin=record.final_has_skin,
            vertex_count=record.vertex_count,
        )

    @staticmethod
    def _observe_context(record: SessionRecord | None, context: SkeletonContext) -> None:
        if record is not None:
            record.observe_bones(int(context.joints.shape[0]))

    def close(self, end_reason: str = "server_shutdown") -> None:
        stop = getattr(self, "_maintenance_stop", None)
        if stop is not None:
            stop.set()
        thread = getattr(self, "_maintenance_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock():
            records = list(getattr(self, "sessions", {}).values())
            getattr(self, "sessions", {}).clear()
            getattr(self, "runtime_cache", {}).clear()
        for record in records:
            self._log_session_end(record, end_reason)

    def _build_runtime(self, record: SessionRecord) -> InteractiveSession:
        obj_path = record.obj_path
        if not obj_path.is_file():
            raise InteractiveServiceError(
                "ASSET_EXPIRED",
                f"mesh asset is unavailable: {record.asset_id}",
            )
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
        blender_to_obj_text = fit_affine(
            blender_vertices.astype(np.float64),
            obj_text_vertices,
        )
        session = InteractiveSession(
            session_id=record.session_id,
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
            blender_to_obj_text=blender_to_obj_text,
            context=SkeletonContext.empty(),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
        self._put_runtime(session)
        return session

    def _put_runtime(self, session: InteractiveSession) -> None:
        session_id = session.session_id
        with self._lock():
            self.runtime_cache[session_id] = session
            self.runtime_cache.move_to_end(session_id)
            while len(self.runtime_cache) > self.max_runtime_sessions:
                self.runtime_cache.popitem(last=False)

    def _touch_session(self, session_id: str) -> None:
        with self._lock():
            move_to_end = getattr(self.sessions, "move_to_end", None)
            if move_to_end is not None and session_id in self.sessions:
                move_to_end(session_id)

    def _enforce_session_limit(self) -> None:
        limit = max(1, int(getattr(self, "max_sessions", DEFAULT_MAX_SESSIONS)))
        removed: list[SessionRecord] = []
        with self._lock():
            while len(self.sessions) > limit:
                if hasattr(self.sessions, "popitem") and hasattr(
                    self.sessions,
                    "move_to_end",
                ):
                    session_id, record = self.sessions.popitem(last=False)
                else:
                    session_id = next(iter(self.sessions))
                    record = self.sessions.pop(session_id)
                removed.append(record)
                self.runtime_cache.pop(session_id, None)
        for record in removed:
            self._log_session_end(record, "lru_evicted")

    def start(self, payload: Request) -> Response:
        obj_path = Path(str(payload["obj_path"])).expanduser().resolve()
        if not obj_path.is_file():
            raise FileNotFoundError(f"mesh not found: {obj_path}")
        now = time.time()
        self.expire_idle_sessions(now=now)
        session_id = uuid.uuid4().hex
        record = SessionRecord(
            session_id=session_id,
            owner_id=str(payload.get("owner_id", "local")),
            asset_id=str(payload.get("asset_id", obj_path.stem)),
            obj_path=obj_path,
            created_at=now,
            updated_at=now,
            client_ip=(str(payload.get("client_ip", "local")).strip() or "unknown")[:64],
            blender_extension_version=(
                str(payload.get("blender_extension_version", "unknown")).strip()
                or "unknown"
            )[:32],
            initial_bone_count=max(0, int(payload.get("initial_bone_count", 0))),
        )
        record.latest_bone_count = record.initial_bone_count
        record.max_bone_count = record.initial_bone_count
        with self._lock():
            self.sessions[session_id] = record
        try:
            session = self._build_runtime(record)
        except Exception:
            with self._lock():
                self.sessions.pop(session_id, None)
            raise
        record.vertex_count = int(session.blender_vertices.shape[0])
        self._touch_session(session_id)
        self._enforce_session_limit()
        self._log_session_start(record)
        return ok(
            session_id=session_id,
            asset_id=record.asset_id,
            context=session.context.to_payload(),
            blender_to_obj_text=session.blender_to_obj_text.tolist(),
        )

    def _record(self, payload: Request) -> SessionRecord:
        session_id = str(payload["session_id"])
        self.expire_idle_sessions()
        with self._lock():
            if session_id not in self.sessions:
                raise InteractiveServiceError(
                    "SESSION_EXPIRED",
                    f"unknown or expired session_id: {session_id}",
                )
            record = self.sessions[session_id]
            owner_id = str(payload.get("owner_id", "local"))
            if record.owner_id != owner_id:
                raise InteractiveServiceError("FORBIDDEN", "session belongs to another client")
            record.updated_at = time.time()
            move_to_end = getattr(self.sessions, "move_to_end", None)
            if move_to_end is not None:
                move_to_end(session_id)
            return record

    def _session(self, payload: Request) -> InteractiveSession:
        record = self._record(payload)
        session = self.runtime_cache.get(record.session_id)
        if session is None:
            session = self._build_runtime(record)
        else:
            self.runtime_cache.move_to_end(record.session_id)
        session.updated_at = record.updated_at
        return session

    def _context(self, session: InteractiveSession, payload: Request) -> SkeletonContext:
        context = SkeletonContext.from_payload(payload)
        if context is None:
            context = session.context
        context = self._validate_context(context)
        record = getattr(self, "sessions", {}).get(session.session_id)
        if record is not None:
            self._reserve_context_names(record, context)
        return context

    @staticmethod
    def _reserve_context_names(
        record: SessionRecord,
        context: SkeletonContext,
    ) -> None:
        for name in context.joint_names:
            name = str(name)
            record.reserved_joint_names.add(name)
            match = GENERIC_BONE_NAME.fullmatch(name)
            if match is not None:
                record.next_bone_id = max(
                    record.next_bone_id,
                    int(match.group(1)) + 1,
                )

    @classmethod
    def _resolve_generated_joint_names(
        cls,
        record: SessionRecord,
        context: SkeletonContext,
        previous_count: int,
    ) -> SkeletonContext:
        count = int(context.joints.shape[0])
        if len(context.parents) != count or len(context.joint_names) != count:
            raise ValueError(
                "generated skeleton joints, parents, and names have unequal lengths"
            )
        if previous_count < 0 or previous_count > count:
            raise ValueError(
                f"invalid generated skeleton prefix length {previous_count} for {count} bones"
            )

        names = [str(name) for name in context.joint_names]
        resolved = list(names[:previous_count])
        used = set(record.reserved_joint_names)
        used.update(resolved)
        for raw_name in names[previous_count:]:
            candidate = raw_name.strip()
            if not candidate or candidate in used:
                while f"bone_{record.next_bone_id}" in used:
                    record.next_bone_id += 1
                candidate = f"bone_{record.next_bone_id}"
                record.next_bone_id += 1
            resolved.append(candidate)
            used.add(candidate)
            match = GENERIC_BONE_NAME.fullmatch(candidate)
            if match is not None:
                record.next_bone_id = max(
                    record.next_bone_id,
                    int(match.group(1)) + 1,
                )
        record.reserved_joint_names.update(used)
        if resolved == names:
            return context
        return SkeletonContext(
            joints=context.joints,
            parents=context.parents,
            joint_names=resolved,
            done=context.done,
        )

    def _validate_context(self, context: SkeletonContext) -> SkeletonContext:
        context.validate_hierarchy()
        bone_count = int(context.joints.shape[0])
        max_bones = int(getattr(self, "max_context_bones", DEFAULT_MAX_CONTEXT_BONES))
        if bone_count > max_bones:
            raise InteractiveServiceError(
                "ARMATURE_TOO_LARGE",
                f"armature has {bone_count} bones; service limit is {max_bones}. "
                "The original armature was not modified.",
            )
        return context

    def next(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        previous_count = int(context.joints.shape[0])
        options = GenerationOptions.from_payload(payload)
        branch_parent = payload.get("branch_parent")
        branch_parent = None if branch_parent in (None, "", -1) else int(branch_parent)
        context, tokens = generate_next(self.model, session, context, options, branch_parent=branch_parent)
        record = getattr(self, "sessions", {}).get(session.session_id)
        if record is not None:
            context = self._resolve_generated_joint_names(
                record,
                context,
                previous_count,
            )
        self._validate_context(context)
        session.context = context
        self._observe_context(record, context)
        return ok(
            session_id=session.session_id,
            context=context.to_payload(),
            tokens=tokens.astype(int).tolist(),
        )

    def rig(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        previous_count = int(context.joints.shape[0])
        token_budget = max(
            int(payload.get("max_new_tokens", 0)),
            int(getattr(self, "max_context_bones", DEFAULT_MAX_CONTEXT_BONES)) * 8
            + 32,
        )
        options = GenerationOptions.from_payload(
            {**payload, "max_new_tokens": token_budget}
        )
        context, tokens = generate_rig(self.model, session, context, options)
        record = getattr(self, "sessions", {}).get(session.session_id)
        if record is not None:
            context = self._resolve_generated_joint_names(
                record,
                context,
                previous_count,
            )
        self._validate_context(context)
        session.context = context
        self._observe_context(record, context)
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
        record = self._record(payload)
        context = SkeletonContext.from_payload(payload)
        if context is None:
            session = self.runtime_cache.get(record.session_id)
            context = SkeletonContext.empty() if session is None else session.context
        self._validate_context(context)
        self._reserve_context_names(record, context)
        self._observe_context(record, context)
        session = self.runtime_cache.get(record.session_id)
        if session is not None:
            session.context = context
            self.runtime_cache.move_to_end(record.session_id)
        return ok(
            session_id=record.session_id,
            context=context.to_payload(),
        )

    def skin(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        if context.joints.shape[0] == 0:
            raise ValueError("cannot generate skin for an empty skeleton")
        raw_midprocess = payload.get("midprocess", DEFAULT_MIDPROCESS)
        midprocess = DEFAULT_MIDPROCESS if raw_midprocess in (None, "") else str(raw_midprocess)
        normalized_joints = session.normalize_points(context.joints)
        ensemble_report = None
        if midprocess == MIDPROCESS_DFS_ENSEMBLE:
            rig = SkeletonContext(
                joints=normalized_joints,
                parents=context.parents.copy(),
                joint_names=list(context.joint_names),
                done=context.done,
            )
            ensemble = generate_skin_ensemble(
                self.model,
                vertices=session.vertices,
                normals=session.normals,
                rig=rig,
                cls=session.cls,
                options=SkinEnsembleOptions.from_payload(
                    payload,
                    default_num_beams=10,
                ),
                learned_mesh_cond=session.learned_mesh_cond,
                cond_latents=session.cond_latents,
            )
            generated_skin = ensemble.selected.sampled_skin
            skin = generated_skin
            skin_order = list(ensemble.selected.order)
            ensemble_report = ensemble.report(include_candidates=False)
        elif midprocess in (MIDPROCESS_NONE, MIDPROCESS_SIMILAR_SUBTREES):
            options = GenerationOptions.from_payload(payload)
            generation_context, skin_order = _reordered_context_for_skin(
                session,
                context,
                midprocess,
            )
            result = generate_skin(self.model, session, generation_context, options)
            if result.skin_pred is None:
                raise RuntimeError("model did not generate complete skin tokens")
            generated_skin = result.skin_pred.detach().float().cpu().numpy()
            skin = _remap_skin_to_original_order(generated_skin, skin_order)
        else:
            raise ValueError(f"unknown midprocess mode: {midprocess}")

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
        applied_postprocess = _apply_skin_postprocess(
            asset,
            payload,
            self.model,
            self.device,
            getattr(
                self,
                "default_skin_postprocess",
                SKIN_POSTPROCESS_NONE,
            ),
        )
        if asset.skin is None:
            raise RuntimeError("skin interpolation did not produce full-resolution weights")
        output_path = payload.get("output_path")
        obj_text_joints = session.to_obj_text_points(context.joints)
        if output_path is not None:
            write_heter_txt(
                asset=asset,
                obj_text_joints=obj_text_joints,
                output_path=Path(str(output_path)).expanduser().resolve(),
                topk=int(payload.get("topk_skin", 4)),
                eps=float(payload.get("weight_eps", 1e-8)),
            )
        txt_content = None
        if bool(payload.get("include_txt", False)):
            txt_content = heter_txt_content(
                asset,
                obj_text_joints=obj_text_joints,
                topk=int(payload.get("topk_skin", 4)),
                eps=float(payload.get("weight_eps", 1e-8)),
            )
        session.context = SkeletonContext(
            joints=context.joints,
            parents=context.parents,
            joint_names=context.joint_names,
            done=True,
        )
        record = getattr(self, "sessions", {}).get(session.session_id)
        self._observe_context(record, session.context)
        if record is not None:
            record.observe_skin()
        return ok(
            session_id=session.session_id,
            context=session.context.to_payload(),
            skin_shape=list(generated_skin.shape),
            asset_skin_shape=list(asset.skin.shape),
            skin=encode_float32_array(asset.skin),
            midprocess=midprocess,
            midprocess_order=skin_order,
            skin_postprocess=applied_postprocess,
            skin_ensemble=ensemble_report,
            output_path=None if output_path is None else str(Path(str(output_path)).expanduser().resolve()),
            **({} if txt_content is None else {"txt_content": txt_content}),
        )

    def reconstruct(self, payload: Request) -> Response:
        session = self._session(payload)
        context = self._context(session, payload)
        bone_count = int(context.joints.shape[0])
        vertex_count = int(session.normalized_vertices_cpu.shape[0])
        if bone_count == 0:
            raise ValueError("cannot reconstruct skin for an empty skeleton")

        raw_names = payload.get("bone_names")
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError("VAE reconstruction requires selected bone_names")
        selected_names = list(dict.fromkeys(str(name) for name in raw_names))
        name_to_index = {name: index for index, name in enumerate(context.joint_names)}
        unknown = [name for name in selected_names if name not in name_to_index]
        if unknown:
            raise ValueError(f"selected bones are not in the current skeleton: {unknown}")
        selected_indices = [name_to_index[name] for name in selected_names]
        trajectory_levels = int(payload.get("trajectory_levels", 0))
        if trajectory_levels < 0 or trajectory_levels > 10:
            raise ValueError("trajectory_levels must be between 0 and 10")
        if trajectory_levels and len(selected_indices) != 1:
            raise ValueError("VAE reconstruction trajectories require exactly one bone")

        current_skin = decode_float32_array(
            payload.get("skin"),
            (vertex_count, bone_count),
        )
        asset = Asset.from_data(
            vertices=session.normalized_vertices_cpu,
            faces=session.faces,
            skin=current_skin,
            joints=session.normalize_points(context.joints),
            parents=context.parents.copy(),
            joint_names=list(context.joint_names),
            cls=session.cls,
            path=str(session.obj_path),
        )
        reconstructor = VaeSkinReconstructor(
            self.model.vae,
            VaeReconstructionOptions(
                topk=max(1, int(payload.get("topk_skin", 4))),
                batch_size=max(
                    1,
                    int(payload.get("vae_reconstruction_batch_size", 4)),
                ),
                seed=int(payload.get("vae_reconstruction_seed", 1234)),
                decode_chunk=max(
                    1,
                    int(payload.get("vae_reconstruction_decode_chunk", 4096)),
                ),
            ),
        )
        fields = [current_skin[:, selected_indices].copy()]
        reports = []
        iterations = trajectory_levels if trajectory_levels else 1
        for _level in range(iterations):
            asset.skin, level_report = reconstructor(
                asset,
                bone_indices=selected_indices,
                mesh_cond_tokens=getattr(session, "cond_latents", None),
            )
            reports.append(level_report)
            if trajectory_levels:
                fields.append(asset.skin[:, selected_indices].copy())

        report = dict(reports[-1])
        if trajectory_levels:
            report.update(
                {
                    "levels": trajectory_levels,
                    "wall_sec": float(
                        sum(float(item.get("wall_sec", 0.0)) for item in reports)
                    ),
                    "level_mae": [float(item.get("mae", 0.0)) for item in reports],
                    "level_wall_sec": [
                        float(item.get("wall_sec", 0.0)) for item in reports
                    ],
                }
            )
        session.context = context
        print(
            "[interactive] reconstructed selected skin fields "
            f"bones={selected_names} levels={iterations} mae={report['mae']:.6f} "
            f"wall={report['wall_sec']:.3f}s"
        )
        response = ok(
            session_id=session.session_id,
            context=context.to_payload(),
            bone_names=selected_names,
            skin=encode_float32_array(asset.skin),
            vae_reconstruction=report,
        )
        if trajectory_levels:
            trajectory = np.stack(fields, axis=0).astype(np.float32, copy=False)
            response["skin_fields"] = encode_float32_array(trajectory)
            response["skin_fields_shape"] = list(trajectory.shape)
        return response

    def reset(self, payload: Request) -> Response:
        record = self._record(payload)
        with self._lock():
            self.sessions.pop(record.session_id, None)
            self.runtime_cache.pop(record.session_id, None)
        self._log_session_end(record, str(payload.get("end_reason", "reset")))
        return ok(session_id=record.session_id)

    def status(self) -> Response:
        self.expire_idle_sessions()
        with self._lock():
            session_count = len(self.sessions)
            runtime_count = len(self.runtime_cache)
        return ok(
            message="pong",
            protocol_version=PROTOCOL_VERSION,
            server_version=SERVER_VERSION,
            features=[
                MIDPROCESS_DFS_ENSEMBLE,
                SKIN_POSTPROCESS_VAE_RECONSTRUCTION,
            ],
            sessions=session_count,
            max_sessions=self.max_sessions,
            runtime_sessions=runtime_count,
            max_runtime_sessions=self.max_runtime_sessions,
            session_idle_timeout_seconds=self.session_idle_timeout_seconds,
            session_cleanup_interval_seconds=self.session_cleanup_interval_seconds,
            max_context_bones=self.max_context_bones,
            default_skin_postprocess=self.default_skin_postprocess,
            usage_dir=(
                None
                if getattr(self, "usage_events", None) is None
                else str(self.usage_events.root)
            ),
        )

    def handle(self, payload: Request) -> Response:
        command = str(payload.get("command", ""))
        if command == "start":
            return self.start(payload)
        if command == "next":
            return self.next(payload)
        if command == "rig":
            return self.rig(payload)
        if command == "branch":
            return self.branch(payload)
        if command == "sync":
            return self.sync(payload)
        if command == "skin":
            return self.skin(payload)
        if command == "reconstruct":
            return self.reconstruct(payload)
        if command == "reset":
            return self.reset(payload)
        if command == "ping":
            return self.status()
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
        max_runtime_sessions=args.max_runtime_sessions,
        max_sessions=args.max_sessions,
        max_context_bones=args.max_context_bones,
        default_skin_postprocess=args.skin_postprocess,
        usage_dir=args.usage_dir,
        session_idle_timeout_seconds=args.session_idle_timeout_seconds,
        session_cleanup_interval_seconds=args.session_cleanup_interval_seconds,
    )
    print(f"[interactive] model server listening on {socket_path}")
    try:
        with Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY) as listener:
            while True:
                conn = listener.accept()
                try:
                    payload = conn.recv()
                    response = server.handle(payload)
                except Exception as exc:
                    response = exception_response(exc)
                try:
                    conn.send(response)
                finally:
                    conn.close()
    finally:
        server.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SkinTokens model server.")
    parser.add_argument("--config", default=str(DEFAULT_SERVER_CONFIG_PATH))
    parser.add_argument("--socket", default=str(MODEL_SOCKET_PATH))
    parser.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-runtime-sessions", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument(
        "--session-idle-timeout-seconds",
        type=float,
        default=DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--session-cleanup-interval-seconds",
        type=float,
        default=DEFAULT_SESSION_CLEANUP_INTERVAL_SECONDS,
    )
    parser.add_argument("--max-context-bones", type=int, default=DEFAULT_MAX_CONTEXT_BONES)
    parser.add_argument("--usage-dir", default=str(DEFAULT_USAGE_DIR))
    parser.add_argument(
        "--skin-postprocess",
        choices=SKIN_POSTPROCESS_MODES,
        default=SKIN_POSTPROCESS_NONE,
    )
    return parser


def main() -> None:
    serve(parse_args_with_config(build_parser()))


if __name__ == "__main__":
    main()
