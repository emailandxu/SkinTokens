from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .transport import PROTOCOL_VERSION, decode_float32_array, encode_float32_array

from .apply_skin import (
    active_armature_bone_name,
    apply_armature_context,
    apply_mesh_skin_weights,
    armature_by_name,
    armature_for_mesh,
    capture_armature_bone_states,
    ensure_mesh_armature,
    import_skin_output,
    parse_armature_object,
    parse_mesh_armature,
    parse_rig_txt,
    read_mesh_skin_weights,
    selected_skin_bone_names,
    write_mesh_skin_txt,
)
from .client import model_request
from .mesh import import_obj


DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL = 3


def same_skeleton_context(left: dict, right: dict) -> bool:
    left_parents = [int(parent) for parent in left.get("parents", [])]
    right_parents = [int(parent) for parent in right.get("parents", [])]
    if left_parents != right_parents:
        return False

    left_names = [str(name) for name in left.get("joint_names", [])]
    right_names = [str(name) for name in right.get("joint_names", [])]
    if left_names != right_names:
        return False

    left_joints = left.get("joints", [])
    right_joints = right.get("joints", [])
    if len(left_joints) != len(right_joints):
        return False
    return all(
        len(left_joint) == len(right_joint)
        and all(
            math.isclose(
                float(left_value),
                float(right_value),
                rel_tol=1e-7,
                abs_tol=1e-6,
            )
            for left_value, right_value in zip(left_joint, right_joint)
        )
        for left_joint, right_joint in zip(left_joints, right_joints)
    )


def same_armature_bone_states(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for joint_id, left_state in left.items():
        right_state = right[joint_id]
        if left_state[:2] != right_state[:2]:
            return False
        for left_point, right_point in zip(left_state[2:], right_state[2:]):
            if any(
                not math.isclose(
                    float(left_value),
                    float(right_value),
                    rel_tol=1e-7,
                    abs_tol=1e-6,
                )
                for left_value, right_value in zip(left_point, right_point)
            ):
                return False
    return True


def scene_armature_context(
    mesh_object_name: str | None,
    armature_object_name: str | None = None,
):
    if not mesh_object_name:
        return None

    import bpy  # type: ignore

    mesh_obj = bpy.data.objects.get(mesh_object_name)
    if mesh_obj is None or mesh_obj.type != "MESH":
        raise ValueError(f"mesh object not found: {mesh_object_name}")
    armature_obj = bpy.data.objects.get(armature_object_name) if armature_object_name else None
    if armature_obj is not None:
        if armature_obj.type != "ARMATURE":
            raise ValueError(f"object is not an armature: {armature_object_name}")
        joints, parents, names = parse_armature_object(armature_obj)
    else:
        if armature_for_mesh(mesh_obj) is None:
            return None
        armature_obj = armature_for_mesh(mesh_obj)
        joints, parents, names = parse_mesh_armature(mesh_obj)
    return mesh_obj, armature_obj, {
        "joints": joints,
        "parents": parents,
        "joint_names": names,
        "done": False,
    }


def normalized_skin_weights(skin: np.ndarray) -> np.ndarray:
    weights = np.nan_to_num(
        np.asarray(skin, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weights[weights < 0.0] = 0.0
    if weights.ndim != 2 or weights.shape[1] == 0:
        raise ValueError(f"skin must have shape (vertices, bones), got {weights.shape}")
    totals = weights.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 1e-12
    if np.any(empty):
        weights[empty, 0] = 1.0
        totals = weights.sum(axis=1, keepdims=True)
    return (weights / totals).astype(np.float32)


@dataclass
class BlenderInteractiveSession:
    blender_session_id: str
    model_session_id: str
    obj_path: Path
    context: dict
    mesh_object_name: str | None = None
    armature_object_name: str | None = None
    armature_show_in_front_before: bool = False
    blender_to_obj_text: Optional[np.ndarray] = field(default=None, repr=False)
    skin_generated: bool = False
    vae_base_skin: Optional[np.ndarray] = field(default=None, repr=False)
    vae_weight_fields: Dict[str, list[np.ndarray]] = field(
        default_factory=dict,
        repr=False,
    )
    vae_levels: Dict[str, int] = field(default_factory=dict, repr=False)
    vae_last_applied_skin: Optional[np.ndarray] = field(default=None, repr=False)
    original_bone_ids: set[str] = field(default_factory=set, repr=False)
    tail_follow_bone_ids: set[str] = field(default_factory=set, repr=False)
    bone_edit_snapshot: dict = field(default_factory=dict, repr=False)

    def clear_vae_cache(self) -> None:
        self.vae_base_skin = None
        self.vae_weight_fields.clear()
        self.vae_levels.clear()
        self.vae_last_applied_skin = None


class BlenderInteractiveCore:
    def __init__(
        self,
        model_socket: str | Path,
        owner_id: str | None = None,
        request_timeout: float = 300.0,
    ):
        self.model_socket = str(model_socket)
        self.owner_id = owner_id or uuid.uuid4().hex
        self.request_timeout = max(1.0, float(request_timeout))
        self.sessions: Dict[str, BlenderInteractiveSession] = {}

    @staticmethod
    def _clear_vae_state(session: BlenderInteractiveSession) -> None:
        session.clear_vae_cache()

    @staticmethod
    def _update_coordinate_mapping(
        session: BlenderInteractiveSession,
        response: dict,
    ) -> None:
        raw = response.get("blender_to_obj_text")
        if raw is None:
            return
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (4, 3):
            raise ValueError(
                f"model returned invalid Blender-to-OBJ transform {matrix.shape}"
            )
        session.blender_to_obj_text = matrix

    def model_request(self, payload: dict) -> dict:
        return model_request(
            payload,
            socket_path=self.model_socket,
            owner_id=self.owner_id,
            timeout=self.request_timeout,
        )

    def server_status(self) -> dict:
        response = self.model_request({"command": "ping"})
        if not response.get("ok"):
            return response
        remote_version = response.get("protocol_version")
        if remote_version != PROTOCOL_VERSION:
            return {
                "ok": False,
                "code": "PROTOCOL_MISMATCH",
                "error": (
                    f"server protocol {remote_version!r} is incompatible with "
                    f"extension protocol {PROTOCOL_VERSION}"
                ),
            }
        return response

    @staticmethod
    def prepare_session_request(
        session: BlenderInteractiveSession,
        payload: dict,
    ) -> dict:
        return {
            "blender_session_id": session.blender_session_id,
            "model_session_id": session.model_session_id,
            "obj_path": str(session.obj_path),
            "payload": copy.deepcopy(payload),
        }

    def execute_session_request(self, prepared: dict) -> dict:
        model_session_id = str(prepared["model_session_id"])
        payload = {
            **prepared["payload"],
            "session_id": model_session_id,
        }
        response = self.model_request(payload)
        restarted = None
        if response.get("code") in {"SESSION_EXPIRED", "ASSET_EXPIRED"}:
            source_context = prepared.get("payload", {})
            restarted = self.model_request({
                "command": "start",
                "obj_path": str(prepared["obj_path"]),
                "initial_bone_count": len(source_context.get("joints", [])),
            })
            if not restarted.get("ok"):
                response = restarted
            else:
                model_session_id = str(restarted["session_id"])
                response = self.model_request(
                    {**prepared["payload"], "session_id": model_session_id}
                )
        return {
            "response": response,
            "model_session_id": model_session_id,
            "restart_response": restarted,
        }

    def apply_session_request(self, prepared: dict, result: dict) -> dict:
        session = self.sessions.get(str(prepared["blender_session_id"]))
        if session is None:
            return {
                "ok": False,
                "code": "STALE_RESULT",
                "error": "interactive session ended while the request was running",
            }
        if session.model_session_id != str(prepared["model_session_id"]):
            return {
                "ok": False,
                "code": "STALE_RESULT",
                "error": "interactive session changed while the request was running",
            }
        session.model_session_id = str(result["model_session_id"])
        restarted = result.get("restart_response")
        if isinstance(restarted, dict) and restarted.get("ok"):
            self._update_coordinate_mapping(session, restarted)
        return result["response"]

    def session_request(self, session: BlenderInteractiveSession, payload: dict) -> dict:
        prepared = self.prepare_session_request(session, payload)
        result = self.execute_session_request(prepared)
        if self.sessions.get(session.blender_session_id) is not session:
            session.model_session_id = str(result["model_session_id"])
            restarted = result.get("restart_response")
            if isinstance(restarted, dict) and restarted.get("ok"):
                self._update_coordinate_mapping(session, restarted)
            return result["response"]
        return self.apply_session_request(prepared, result)

    @staticmethod
    def sync_context_from_armature(session: BlenderInteractiveSession) -> None:
        armature_obj = armature_by_name(session.armature_object_name)
        if armature_obj is None:
            raise RuntimeError("interactive session armature is unavailable")
        current_bone_states = capture_armature_bone_states(armature_obj)
        session.bone_edit_snapshot = current_bone_states
        joints, parents, names = parse_armature_object(armature_obj)
        edited = {
            **session.context,
            "joints": joints,
            "parents": parents,
            "joint_names": names,
            "done": False,
        }
        if same_skeleton_context(session.context, edited):
            edited["done"] = bool(session.context.get("done", False))
        session.context = edited

    def prepared_context_is_current(self, prepared: dict) -> bool:
        session = self.sessions.get(str(prepared["blender_session_id"]))
        expected = prepared.get("source_context")
        if session is None or expected is None:
            return False
        armature_obj = armature_by_name(session.armature_object_name)
        if armature_obj is None:
            return False
        if session.bone_edit_snapshot and not same_armature_bone_states(
            session.bone_edit_snapshot,
            capture_armature_bone_states(armature_obj),
        ):
            return False
        joints, parents, names = parse_armature_object(armature_obj)
        current = {
            "joints": joints,
            "parents": parents,
            "joint_names": names,
        }
        return same_skeleton_context(expected, current)

    def prepared_vae_skin_is_current(self, prepared: dict) -> bool:
        if not self.prepared_context_is_current(prepared):
            return False
        session = self.sessions.get(str(prepared["blender_session_id"]))
        if session is None or not session.mesh_object_name:
            return False
        current = read_mesh_skin_weights(
            session.mesh_object_name,
            [str(name) for name in prepared["joint_names"]],
        )
        expected = np.asarray(prepared["visible_skin"])
        return current.shape == expected.shape and np.allclose(
            current,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

    @staticmethod
    def _result_mode(session: BlenderInteractiveSession) -> str:
        armature_obj = armature_by_name(session.armature_object_name)
        if armature_obj is not None and armature_obj.mode in {"EDIT", "POSE"}:
            return str(armature_obj.mode)
        return "EDIT"

    @staticmethod
    def _apply_context_to_armature(
        session: BlenderInteractiveSession,
        context: dict,
        *,
        select_name: str | None,
        remove_missing: bool = False,
        mode_after: str | None = None,
        allow_remove_names: set[str] | None = None,
        allow_reparent_names: set[str] | None = None,
    ) -> None:
        if not session.armature_object_name:
            raise RuntimeError("interactive session has no armature")
        apply_armature_context(
            session.armature_object_name,
            context.get("joints", []),
            context.get("parents", []),
            context.get("joint_names", []),
            select_name=select_name,
            remove_missing=remove_missing,
            mode_after=mode_after,
            session_id=session.blender_session_id,
            original_bone_ids=session.original_bone_ids,
            tail_follow_bone_ids=session.tail_follow_bone_ids,
            allow_remove_names=allow_remove_names,
            allow_reparent_names=allow_reparent_names,
        )
        armature_obj = armature_by_name(session.armature_object_name)
        if armature_obj is None:
            raise RuntimeError("interactive session armature is unavailable")
        joints, parents, names = parse_armature_object(armature_obj)
        context["joints"] = joints
        context["parents"] = parents
        context["joint_names"] = names
        session.bone_edit_snapshot = capture_armature_bone_states(armature_obj)

    def sync_context_to_model(self, session: BlenderInteractiveSession) -> dict:
        payload = {
            "command": "sync",
            "session_id": session.model_session_id,
            **session.context,
            "done": False,
        }
        return self.session_request(session, payload)

    def prepare_history_sync(self, blender_session_id: str) -> dict:
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "error": f"unknown blender_session_id: {blender_session_id}"}
        previous = copy.deepcopy(session.context)
        self.sync_context_from_armature(session)
        if same_skeleton_context(previous, session.context):
            session.context = previous
            return {
                "ok": True,
                "needs_remote": False,
                "context": session.context,
                "synced": False,
            }
        payload = {
            "command": "sync",
            **session.context,
            "done": False,
        }
        return {
            "ok": True,
            "needs_remote": True,
            "blender_session_id": blender_session_id,
            "previous_context": previous,
            "source_context": copy.deepcopy(session.context),
            "select_name": active_armature_bone_name(
                session.armature_object_name
            ),
            "mode_after": self._result_mode(session),
            "remote": self.prepare_session_request(session, payload),
        }

    def apply_history_sync(self, prepared: dict, response: dict) -> dict:
        session = self.sessions.get(str(prepared["blender_session_id"]))
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        if not response.get("ok"):
            session.context = prepared["previous_context"]
            return response
        session.context = response["context"]
        self._apply_context_to_armature(
            session,
            session.context,
            select_name=prepared.get("select_name"),
            remove_missing=True,
            mode_after=str(prepared["mode_after"]),
        )
        self._clear_vae_state(session)
        return {"ok": True, "context": session.context, "synced": True}

    def sync_history_snapshot(self, blender_session_id: str) -> dict:
        prepared = self.prepare_history_sync(blender_session_id)
        if not prepared.get("ok") or not prepared.get("needs_remote"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        return self.apply_history_sync(prepared, response)

    @staticmethod
    def dfs_stack(parents: list[int]) -> list[int]:
        if not parents:
            return []
        stack = []
        current = len(parents) - 1
        while current != -1:
            stack.append(int(current))
            current = int(parents[current])
        stack.reverse()
        return stack

    @staticmethod
    def focus_context_on_joint(context: dict, selected_index: int) -> tuple[dict, int]:
        joints = list(context.get("joints", []))
        parents = [int(parent) for parent in context.get("parents", [])]
        names = list(context.get("joint_names", []))
        count = len(joints)
        if selected_index < 0 or selected_index >= count:
            raise ValueError(f"selected parent {selected_index} is outside current skeleton")
        if count == 0:
            return context, selected_index

        roots = [idx for idx, parent in enumerate(parents) if parent == -1]
        if len(roots) != 1:
            raise ValueError(f"expected one root, found {len(roots)}")
        root = roots[0]

        children = [[] for _ in range(count)]
        for child, parent in enumerate(parents):
            if parent != -1:
                children[parent].append(child)

        path = []
        current = selected_index
        while current != -1:
            path.append(current)
            current = parents[current]
        path.reverse()

        for parent, child_on_path in zip(path, path[1:]):
            row = children[parent]
            if child_on_path in row:
                row.remove(child_on_path)
                row.append(child_on_path)

        order = []

        def visit(node: int) -> None:
            order.append(node)
            for child in children[node]:
                visit(child)

        visit(root)
        if len(order) != count:
            raise ValueError("focused DFS order did not visit every joint")

        old_to_new = {old: new for new, old in enumerate(order)}
        new_context = {
            **context,
            "joints": [joints[old] for old in order],
            "parents": [
                -1 if parents[old] == -1 else old_to_new[parents[old]]
                for old in order
            ],
            "joint_names": [
                names[old] if old < len(names) else f"bone_{old}"
                for old in order
            ],
            "done": False,
        }
        return new_context, old_to_new[selected_index]

    def prepare_start(
        self,
        obj_path: str | Path,
        *,
        import_mesh: bool = True,
        mesh_object_name: str | None = None,
        armature_object_name: str | None = None,
        **options,
    ) -> dict:
        mesh_object = import_obj(obj_path) if import_mesh else None
        if mesh_object is not None:
            mesh_object_name = mesh_object.name
        if not mesh_object_name:
            return {"ok": False, "error": "start requires a mesh object"}
        armature_context = scene_armature_context(
            mesh_object_name,
            armature_object_name,
        )
        if armature_context is None:
            initial_context = {
                "joints": [],
                "parents": [],
                "joint_names": [],
                "done": False,
            }
            context_source = "empty-armature"
            resolved_armature_name = None
            result_mode = "OBJECT"
        else:
            _mesh_obj, armature_obj, initial_context = armature_context
            context_source = "scene-armature"
            resolved_armature_name = str(armature_obj.name)
            result_mode = (
                str(armature_obj.mode)
                if armature_obj.mode in {"EDIT", "POSE"}
                else "OBJECT"
            )
        initial_bone_states = (
            {}
            if armature_context is None
            else capture_armature_bone_states(armature_obj)
        )
        return {
            "ok": True,
            "obj_path": str(Path(obj_path).expanduser().resolve()),
            "mesh_object_name": str(mesh_object_name),
            "armature_object_name": resolved_armature_name,
            "initial_context": copy.deepcopy(initial_context),
            "context_source": context_source,
            "result_mode": result_mode,
            "initial_bone_states": initial_bone_states,
            "armature_show_in_front_before": (
                False
                if armature_context is None
                else bool(getattr(armature_obj, "show_in_front", False))
            ),
            "payload": {
                "command": "start",
                "obj_path": str(Path(obj_path).expanduser().resolve()),
                "initial_bone_count": len(initial_context.get("joints", [])),
                **copy.deepcopy(options),
            },
        }

    def execute_start_request(self, prepared: dict) -> dict:
        status = self.server_status()
        if not status.get("ok"):
            return {"response": status, "start_response": status}
        started = self.model_request(prepared["payload"])
        if not started.get("ok"):
            return {"response": started, "start_response": started}
        response = started
        initial_context = prepared["initial_context"]
        if initial_context.get("joints"):
            response = self.model_request(
                {
                    "command": "sync",
                    "session_id": str(started["session_id"]),
                    **initial_context,
                    "done": False,
                }
            )
            if not response.get("ok"):
                self.model_request(
                    {
                        "command": "reset",
                        "session_id": str(started["session_id"]),
                        "end_reason": "start_failed",
                    }
                )
        return {"response": response, "start_response": started}

    @staticmethod
    def prepared_start_is_current(prepared: dict) -> bool:
        armature_name = prepared.get("armature_object_name")
        if armature_name is None:
            return True
        try:
            armature_context = scene_armature_context(
                str(prepared["mesh_object_name"]),
                str(armature_name),
            )
        except Exception:
            return False
        if armature_context is None:
            return False
        if not same_skeleton_context(
            prepared["initial_context"],
            armature_context[2],
        ):
            return False
        expected_bone_states = prepared.get("initial_bone_states", {})
        return not expected_bone_states or same_armature_bone_states(
            expected_bone_states,
            capture_armature_bone_states(armature_context[1]),
        )

    def apply_start_request(self, prepared: dict, result: dict) -> dict:
        response = result["response"]
        if not response.get("ok"):
            return response
        started = result["start_response"]
        mesh_object_name = str(prepared["mesh_object_name"])
        armature_object_name = prepared.get("armature_object_name")
        if armature_object_name is None:
            armature_obj = ensure_mesh_armature(mesh_object_name)
        else:
            armature_obj = ensure_mesh_armature(
                mesh_object_name,
                str(armature_object_name),
            )
        armature_show_in_front_before = bool(
            prepared.get("armature_show_in_front_before", False)
        )
        armature_obj.show_in_front = True
        blender_session_id = uuid.uuid4().hex
        initial_bone_snapshot = capture_armature_bone_states(armature_obj)
        session = BlenderInteractiveSession(
            blender_session_id=blender_session_id,
            model_session_id=str(started["session_id"]),
            obj_path=Path(str(prepared["obj_path"])),
            context=started.get("context", {}),
            mesh_object_name=mesh_object_name,
            armature_object_name=str(armature_obj.name),
            armature_show_in_front_before=armature_show_in_front_before,
            original_bone_ids=set(initial_bone_snapshot),
            bone_edit_snapshot=initial_bone_snapshot,
        )
        self._update_coordinate_mapping(session, started)
        self.sessions[blender_session_id] = session
        initial_context = prepared["initial_context"]
        session.context = (
            response.get("context", initial_context)
            if initial_context.get("joints")
            else {**session.context, **initial_context}
        )
        try:
            self._apply_context_to_armature(
                session,
                session.context,
                select_name=(
                    active_armature_bone_name(session.armature_object_name)
                    if initial_context.get("joints")
                    else None
                ),
                mode_after=str(prepared["result_mode"]),
            )
        except Exception:
            self.sessions.pop(blender_session_id, None)
            armature_obj.show_in_front = armature_show_in_front_before
            raise
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "armature_object_name": session.armature_object_name,
            "context": session.context,
            "context_source": prepared["context_source"],
        }

    def start(
        self,
        obj_path: str | Path,
        *,
        import_mesh: bool = True,
        mesh_object_name: str | None = None,
        armature_object_name: str | None = None,
        **options,
    ) -> dict:
        prepared = self.prepare_start(
            obj_path,
            import_mesh=import_mesh,
            mesh_object_name=mesh_object_name,
            armature_object_name=armature_object_name,
            **options,
        )
        if not prepared.get("ok"):
            return prepared
        return self.apply_start_request(
            prepared,
            self.execute_start_request(prepared),
        )

    def prepare_next(
        self,
        blender_session_id: str,
        *,
        force_parent: bool = False,
        **options,
    ) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        previous_context = session.context
        previous_count = len(previous_context.get("joints", []))
        selected_name = active_armature_bone_name(session.armature_object_name)
        selected_parent = None
        session.context = {**previous_context, "done": False}
        options.pop("branch_parent", None)
        if previous_count and force_parent:
            if selected_name is None:
                return {
                    "ok": False,
                    "error": "select one active parent bone in Edit or Pose mode",
                }
            names = [str(name) for name in previous_context.get("joint_names", [])]
            if selected_name not in names:
                return {"ok": False, "error": f"active bone is not in context: {selected_name}"}
            selected_parent = names.index(selected_name)
            parents = [int(parent) for parent in previous_context.get("parents", [])]
            if selected_parent not in self.dfs_stack(parents):
                session.context, selected_parent = self.focus_context_on_joint(
                    previous_context,
                    selected_parent,
                )
            else:
                session.context = {**previous_context, "done": False}
            options["branch_parent"] = selected_parent
        payload = {
            "command": "next",
            "session_id": session.model_session_id,
            **session.context,
            **options,
        }
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": copy.deepcopy(previous_context),
            "previous_count": previous_count,
            "selected_name": selected_name,
            "selected_parent": selected_parent,
            "mode_after": self._result_mode(session),
            "remote": self.prepare_session_request(session, payload),
        }

    def apply_next_request(self, prepared: dict, response: dict) -> dict:
        if not response.get("ok"):
            return response
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        previous_count = int(prepared["previous_count"])
        selected_name = prepared.get("selected_name")
        selected_parent = prepared.get("selected_parent")
        result_context = response["context"]
        result_count = len(result_context.get("joints", []))
        if result_count not in {previous_count, previous_count + 1}:
            raise RuntimeError(
                f"Next returned {result_count - previous_count} new bones"
            )
        result_names = [str(name) for name in result_context.get("joint_names", [])]
        if result_count > previous_count:
            if selected_parent is not None:
                generated_parent = int(result_context["parents"][-1])
                if generated_parent != selected_parent:
                    raise RuntimeError(
                        f"generated parent {generated_parent} does not match active bone {selected_parent}"
                    )
            selected_name = result_names[-1]
        self._apply_context_to_armature(
            session,
            result_context,
            select_name=selected_name,
            remove_missing=True,
            mode_after=str(prepared["mode_after"]),
        )
        session.context = result_context
        if result_count != previous_count:
            self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def next(
        self,
        blender_session_id: str,
        *,
        force_parent: bool = False,
        **options,
    ) -> dict:
        prepared = self.prepare_next(
            blender_session_id,
            force_parent=force_parent,
            **options,
        )
        if not prepared.get("ok"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        return self.apply_next_request(prepared, response)

    def prepare_rig(self, blender_session_id: str, **options) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        source_context = copy.deepcopy(session.context)
        previous_count = len(session.context.get("joints", []))
        active_name = active_armature_bone_name(session.armature_object_name)
        payload = {
            "command": "rig",
            **session.context,
            **options,
        }
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": source_context,
            "previous_count": previous_count,
            "active_name": active_name,
            "mode_after": self._result_mode(session),
            "remote": self.prepare_session_request(session, payload),
        }

    def apply_rig_request(self, prepared: dict, response: dict) -> dict:
        if not response.get("ok"):
            return response
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        previous_count = int(prepared["previous_count"])
        active_name = prepared.get("active_name")
        result_context = response["context"]
        result_names = [str(name) for name in result_context.get("joint_names", [])]
        select_name = result_names[-1] if len(result_names) > previous_count else active_name
        self._apply_context_to_armature(
            session,
            result_context,
            select_name=select_name,
            remove_missing=True,
            mode_after=str(prepared["mode_after"]),
        )
        session.context = result_context
        if len(result_names) != previous_count:
            self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def rig(self, blender_session_id: str, **options) -> dict:
        prepared = self.prepare_rig(blender_session_id, **options)
        if not prepared.get("ok"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        return self.apply_rig_request(prepared, response)

    def branch(self, blender_session_id: str, **options) -> dict:
        return self.next(blender_session_id, force_parent=True, **options)

    def prepare_refresh(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        payload = {
            "command": "sync",
            **session.context,
            "done": False,
        }
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": copy.deepcopy(session.context),
            "select_name": active_armature_bone_name(session.armature_object_name),
            "mode_after": self._result_mode(session),
            "remote": self.prepare_session_request(session, payload),
        }

    def apply_refresh_request(self, prepared: dict, response: dict) -> dict:
        if not response.get("ok"):
            return response
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        session.context = response["context"]
        self._apply_context_to_armature(
            session,
            session.context,
            select_name=prepared.get("select_name"),
            remove_missing=True,
            mode_after=str(prepared["mode_after"]),
        )
        self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def refresh(self, blender_session_id: str) -> dict:
        prepared = self.prepare_refresh(blender_session_id)
        if not prepared.get("ok"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        return self.apply_refresh_request(prepared, response)

    def prepare_split(self, blender_session_id: str, ratio: float = 0.5) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        source_context = copy.deepcopy(session.context)
        selected_name = active_armature_bone_name(session.armature_object_name)
        if selected_name is None:
            return {"ok": False, "error": "select a bone to split in Edit or Pose mode"}

        joints = [list(joint) for joint in session.context.get("joints", [])]
        parents = [int(parent) for parent in session.context.get("parents", [])]
        names = list(session.context.get("joint_names", []))
        if selected_name not in names:
            return {"ok": False, "error": f"active bone is not in context: {selected_name}"}
        selected = names.index(selected_name)
        count = len(joints)
        if selected < 0 or selected >= count:
            return {"ok": False, "error": f"selected joint {selected} is outside current skeleton"}
        children = [
            index for index, parent in enumerate(parents) if parent == selected
        ]
        if not children:
            return {
                "ok": False,
                "error": "the selected terminal bone has no downstream joint to split",
            }
        child = children[0]

        ratio = max(0.01, min(0.99, float(ratio)))
        parent_joint = joints[selected]
        child_joint = joints[child]
        mid_joint = [
            float(parent_joint[axis]) * (1.0 - ratio) + float(child_joint[axis]) * ratio
            for axis in range(3)
        ]
        base_mid_name = f"{selected_name}_split"
        mid_name = base_mid_name
        suffix = 1
        while mid_name in names:
            mid_name = f"{base_mid_name}_{suffix}"
            suffix += 1

        insert_at = child
        new_joints = joints[:insert_at] + [mid_joint] + joints[insert_at:]
        new_names = names[:insert_at] + [mid_name] + names[insert_at:]

        def remap_index(index: int) -> int:
            return index + 1 if index >= insert_at else index

        new_mid_index = insert_at
        new_parents = []
        for old_parent in parents:
            if old_parent == selected:
                new_parents.append(new_mid_index)
            elif old_parent == -1:
                new_parents.append(-1)
            else:
                new_parents.append(remap_index(old_parent))
        new_parents = (
            new_parents[:insert_at]
            + [remap_index(selected)]
            + new_parents[insert_at:]
        )

        session.context = {
            **session.context,
            "joints": new_joints,
            "parents": new_parents,
            "joint_names": new_names,
            "done": False,
        }
        payload = {
            "command": "sync",
            "session_id": session.model_session_id,
            **session.context,
        }
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": source_context,
            "mid_name": mid_name,
            "selected_name": selected_name,
            "mode_after": self._result_mode(session),
            "split_joint_index": new_mid_index,
            "split_child_name": names[child],
            "split_reparent_names": [names[index] for index in children],
            "remote": self.prepare_session_request(session, payload),
        }

    def apply_split_request(self, prepared: dict, response: dict) -> dict:
        if not response.get("ok"):
            return response
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        self._apply_context_to_armature(
            session,
            response["context"],
            select_name=str(prepared["selected_name"]),
            remove_missing=True,
            mode_after=str(prepared["mode_after"]),
            allow_reparent_names=set(prepared["split_reparent_names"]),
        )
        session.context = response["context"]
        self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "split_joint_index": int(prepared["split_joint_index"]),
            "split_child_name": str(prepared["split_child_name"]),
        }

    def split_selected_bone(self, blender_session_id: str, ratio: float = 0.5) -> dict:
        prepared = self.prepare_split(blender_session_id, ratio=ratio)
        if not prepared.get("ok"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        return self.apply_split_request(prepared, response)

    def prepare_delete(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        source_context = copy.deepcopy(session.context)
        selected_name = active_armature_bone_name(session.armature_object_name)
        if selected_name is None:
            return {
                "ok": False,
                "error": "select a bone to delete in Edit or Pose mode",
            }

        joints = [list(joint) for joint in session.context.get("joints", [])]
        parents = [int(parent) for parent in session.context.get("parents", [])]
        names = [str(name) for name in session.context.get("joint_names", [])]
        if selected_name not in names:
            return {
                "ok": False,
                "error": f"active bone is not in context: {selected_name}",
            }
        selected = names.index(selected_name)
        children = [
            index for index, parent in enumerate(parents) if parent == selected
        ]
        if children:
            removed = children[0]
            replacement_parent = selected
            select_name = selected_name
            operation = "dissolve"
        else:
            removed = selected
            replacement_parent = parents[selected]
            select_name = (
                None
                if replacement_parent == -1
                else names[replacement_parent]
            )
            operation = "delete"
        removed_name = names[removed]
        reparented_names = [
            names[index]
            for index, parent in enumerate(parents)
            if parent == removed
        ]
        keep = [index for index in range(len(joints)) if index != removed]
        old_to_new = {old: new for new, old in enumerate(keep)}
        new_parents = []
        for old in keep:
            parent = parents[old]
            if parent == removed:
                parent = replacement_parent
            new_parents.append(-1 if parent == -1 else old_to_new[parent])
        new_context = {
            **session.context,
            "joints": [joints[index] for index in keep],
            "parents": new_parents,
            "joint_names": [names[index] for index in keep],
            "done": False,
        }
        payload = {
                "command": "sync",
                "session_id": session.model_session_id,
                **new_context,
        }
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": source_context,
            "select_name": select_name,
            "mode_after": self._result_mode(session),
            "deleted_bone_name": removed_name,
            "delete_operation": operation,
            "reparented_names": reparented_names,
            "remote": self.prepare_session_request(session, payload),
        }

    def apply_delete_request(self, prepared: dict, response: dict) -> dict:
        if not response.get("ok"):
            return response
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        self._apply_context_to_armature(
            session,
            response["context"],
            select_name=prepared.get("select_name"),
            remove_missing=True,
            mode_after=str(prepared["mode_after"]),
            allow_remove_names={str(prepared["deleted_bone_name"])},
            allow_reparent_names=set(prepared["reparented_names"]),
        )
        session.context = response["context"]
        self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "deleted_bone_name": str(prepared["deleted_bone_name"]),
            "delete_operation": str(prepared["delete_operation"]),
        }

    def delete_selected_bone(self, blender_session_id: str) -> dict:
        prepared = self.prepare_delete(blender_session_id)
        if not prepared.get("ok"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        return self.apply_delete_request(prepared, response)

    def dissolve_selected_bone(self, blender_session_id: str) -> dict:
        return self.delete_selected_bone(blender_session_id)

    def import_txt(self, blender_session_id: str, txt_path: str | Path) -> dict:
        session = self.sessions[blender_session_id]
        resolved = Path(txt_path).expanduser().resolve()
        if not resolved.is_file():
            return {"ok": False, "error": f"txt not found: {resolved}"}
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}

        joints, parents, names = parse_rig_txt(resolved)
        session.context = {
            **session.context,
            "joints": joints,
            "parents": parents,
            "joint_names": names,
            "done": False,
        }
        response = self.sync_context_to_model(session)
        if not response.get("ok"):
            return response
        session.context = response["context"]
        self._apply_context_to_armature(
            session,
            session.context,
            select_name=None,
            remove_missing=True,
            mode_after="OBJECT",
        )
        self._clear_vae_state(session)
        import_skin_output(
            resolved,
            mesh_object_name=session.mesh_object_name,
            joints=session.context.get("joints", []),
            parents=session.context.get("parents", []),
            joint_names=session.context.get("joint_names", []),
            armature_object_name=session.armature_object_name,
        )

        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "txt_path": str(resolved),
        }

    def prepare_skin(
        self,
        blender_session_id: str,
        output_path: str | Path | None = None,
        **options,
    ) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        source_context = copy.deepcopy(session.context)
        payload = {
            "command": "skin",
            "session_id": session.model_session_id,
            **session.context,
            "done": False,
            **options,
        }
        resolved_output = None
        if output_path is not None:
            resolved_output = Path(output_path).expanduser().resolve()
            payload["output_path"] = str(resolved_output)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": source_context,
            "resolved_output": (
                None if resolved_output is None else str(resolved_output)
            ),
            "remote": self.prepare_session_request(session, payload),
        }

    def execute_skin_request(self, prepared: dict) -> dict:
        remote_result = self.execute_session_request(prepared["remote"])
        response = remote_result["response"]
        skin = None
        if response.get("ok"):
            shape = tuple(
                int(value) for value in response.get("asset_skin_shape", [])
            )
            if len(shape) != 2:
                response = {
                    "ok": False,
                    "error": "model response has no full-resolution skin",
                }
                remote_result["response"] = response
            else:
                skin = decode_float32_array(response.get("skin"), shape)
        return {"remote_result": remote_result, "skin": skin}

    def apply_skin_request(self, prepared: dict, result: dict) -> dict:
        response = self.apply_session_request(
            prepared["remote"],
            result["remote_result"],
        )
        if not response.get("ok"):
            return response
        return self._apply_skin_response(prepared, response, result["skin"])

    def _apply_skin_response(
        self,
        prepared: dict,
        response: dict,
        skin: np.ndarray | None,
    ) -> dict:
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        session.context = response["context"]
        if skin is None:
            return {"ok": False, "error": "model response has no decoded skin"}
        names = [str(name) for name in session.context.get("joint_names", [])]
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}
        apply_mesh_skin_weights(session.mesh_object_name, names, skin)
        session.skin_generated = True
        self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "output_path": prepared["resolved_output"],
            "skin_shape": response.get("skin_shape"),
            "midprocess": response.get("midprocess"),
            "skin_ensemble": response.get("skin_ensemble"),
        }

    def skin(
        self,
        blender_session_id: str,
        output_path: str | Path | None = None,
        **options,
    ) -> dict:
        prepared = self.prepare_skin(
            blender_session_id,
            output_path=output_path,
            **options,
        )
        if not prepared.get("ok"):
            return prepared
        session = self.sessions[blender_session_id]
        response = self.session_request(session, prepared["remote"]["payload"])
        if not response.get("ok"):
            return response
        shape = tuple(int(value) for value in response.get("asset_skin_shape", []))
        skin = (
            decode_float32_array(response.get("skin"), shape)
            if len(shape) == 2
            else None
        )
        return self._apply_skin_response(prepared, response, skin)

    def export_skin(self, blender_session_id: str, output_path: str | Path) -> dict:
        session = self.sessions[blender_session_id]
        if not session.skin_generated:
            return {"ok": False, "error": "no generated Skin is available"}
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}
        if session.blender_to_obj_text is None:
            return {
                "ok": False,
                "error": "session has no Blender-to-OBJ coordinate transform",
            }
        self.sync_context_from_armature(session)
        resolved = write_mesh_skin_txt(
            output_path,
            mesh_object_name=session.mesh_object_name,
            joints=session.context.get("joints", []),
            parents=session.context.get("parents", []),
            joint_names=session.context.get("joint_names", []),
            blender_to_obj_text=session.blender_to_obj_text,
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "output_path": str(resolved),
        }

    def reconstruct_selected_skin(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}

        joint_names = [str(name) for name in session.context.get("joint_names", [])]
        if not joint_names:
            return {"ok": False, "error": "current skeleton is empty"}
        selected_names = selected_skin_bone_names(session.mesh_object_name)
        if not selected_names:
            return {
                "ok": False,
                "error": "activate a skin vertex group or select one or more pose bones",
            }
        unknown = [name for name in selected_names if name not in joint_names]
        if unknown:
            return {
                "ok": False,
                "error": f"selected bones are not part of the SkinTokens skeleton: {unknown}",
            }

        current_skin = read_mesh_skin_weights(
            session.mesh_object_name,
            joint_names,
        )
        if not bool((current_skin > 0.0).any()):
            return {
                "ok": False,
                "error": "mesh has no skin weights; run Skin or import a skin first",
            }
        response = self.session_request(
            session,
            {
                "command": "reconstruct",
                **session.context,
                "bone_names": selected_names,
                "skin": encode_float32_array(current_skin),
            },
        )
        if not response.get("ok"):
            return response

        reconstructed = decode_float32_array(
            response.get("skin"),
            current_skin.shape,
        )
        self._clear_vae_state(session)
        apply_mesh_skin_weights(
            session.mesh_object_name,
            joint_names,
            reconstructed,
        )
        session.context = response.get("context", session.context)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "bone_names": response.get("bone_names", selected_names),
            "vae_reconstruction": response.get("vae_reconstruction"),
        }

    def vae_reconstruction_level(
        self,
        blender_session_id: str,
        bone_name: str,
    ) -> int:
        session = self.sessions.get(blender_session_id)
        if session is None:
            return 0
        return int(session.vae_levels.get(str(bone_name), 0))

    def set_cached_vae_reconstruction_level(
        self,
        blender_session_id: str,
        level: int,
    ) -> dict:
        session = self.sessions[blender_session_id]
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}
        joint_names = [str(name) for name in session.context.get("joint_names", [])]
        selected_names = selected_skin_bone_names(session.mesh_object_name)
        if len(selected_names) != 1:
            return {
                "ok": False,
                "error": "activate exactly one skin vertex group or select exactly one pose bone",
            }
        bone_name = selected_names[0]
        fields = session.vae_weight_fields.get(bone_name)
        if session.vae_base_skin is None or fields is None:
            return {
                "ok": False,
                "code": "VAE_CACHE_MISSING",
                "error": "generate the reconstruction cache before changing levels",
            }
        level = int(level)
        max_level = len(fields) - 1
        if level < 0 or level > max_level:
            return {
                "ok": False,
                "error": f"VAE reconstruction level must be between 0 and {max_level}",
            }
        if bone_name not in joint_names:
            return {
                "ok": False,
                "error": f"selected bone is not part of the SkinTokens skeleton: {bone_name}",
            }
        return self._apply_vae_level({
            "blender_session_id": blender_session_id,
            "bone_name": bone_name,
            "bone_index": joint_names.index(bone_name),
            "joint_names": joint_names,
            "level": level,
            "max_level": max_level,
            "cache_invalidated": False,
        })

    def prepare_vae_reconstruction_level(
        self,
        blender_session_id: str,
        level: int,
        *,
        max_level: int = DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL,
    ) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}

        level = int(level)
        max_level = int(max_level)
        if level < 0 or level > max_level or max_level < 1 or max_level > 10:
            return {
                "ok": False,
                "error": f"VAE reconstruction level must be between 0 and {max_level}",
            }

        joint_names = [str(name) for name in session.context.get("joint_names", [])]
        if not joint_names:
            return {"ok": False, "error": "current skeleton is empty"}
        selected_names = selected_skin_bone_names(session.mesh_object_name)
        if len(selected_names) != 1:
            return {
                "ok": False,
                "error": "activate exactly one skin vertex group or select exactly one pose bone",
            }
        bone_name = selected_names[0]
        if bone_name not in joint_names:
            return {
                "ok": False,
                "error": f"selected bone is not part of the SkinTokens skeleton: {bone_name}",
            }
        bone_index = joint_names.index(bone_name)

        current_skin = read_mesh_skin_weights(session.mesh_object_name, joint_names)
        if not bool((current_skin > 0.0).any()):
            return {
                "ok": False,
                "error": "mesh has no skin weights; run Skin or import a skin first",
            }

        cache_invalidated = False
        shape_changed = (
            session.vae_base_skin is not None
            and session.vae_base_skin.shape != current_skin.shape
        )
        externally_edited = (
            session.vae_last_applied_skin is not None
            and session.vae_last_applied_skin.shape == current_skin.shape
            and not np.allclose(
                current_skin,
                session.vae_last_applied_skin,
                rtol=1e-5,
                atol=1e-6,
            )
        )
        if session.vae_base_skin is None or shape_changed or externally_edited:
            cache_invalidated = shape_changed or externally_edited
            self._clear_vae_state(session)
            session.vae_base_skin = current_skin.copy()
            session.vae_last_applied_skin = current_skin.copy()

        assert session.vae_base_skin is not None
        needs_remote = level > 0 and bone_name not in session.vae_weight_fields
        payload = {
            "command": "reconstruct",
            **session.context,
            "bone_names": [bone_name],
            "trajectory_levels": max_level,
        }
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "source_context": copy.deepcopy(session.context),
            "bone_name": bone_name,
            "bone_index": bone_index,
            "joint_names": joint_names,
            "level": level,
            "max_level": max_level,
            "cache_invalidated": cache_invalidated,
            "base_skin": session.vae_base_skin.copy(),
            "visible_skin": current_skin.copy(),
            "needs_remote": needs_remote,
            "remote": self.prepare_session_request(session, payload),
        }

    @staticmethod
    def _decode_vae_trajectory(prepared: dict, response: dict) -> np.ndarray:
        expected_shape = (
            int(prepared["max_level"]) + 1,
            int(prepared["base_skin"].shape[0]),
            1,
        )
        raw_shape = response.get("skin_fields_shape")
        if (
            raw_shape is not None
            and tuple(int(value) for value in raw_shape) != expected_shape
        ):
            raise ValueError(
                f"VAE trajectory shape {raw_shape} does not match "
                f"{list(expected_shape)}"
            )
        return decode_float32_array(response.get("skin_fields"), expected_shape)

    def execute_vae_request(self, prepared: dict) -> dict:
        remote = copy.deepcopy(prepared["remote"])
        remote["payload"]["skin"] = encode_float32_array(prepared["base_skin"])
        remote_result = self.execute_session_request(remote)
        response = remote_result["response"]
        trajectory = None
        if response.get("ok"):
            try:
                trajectory = self._decode_vae_trajectory(prepared, response)
            except Exception as exc:
                remote_result["response"] = {
                    "ok": False,
                    "error": str(exc),
                }
        return {
            "remote": remote,
            "remote_result": remote_result,
            "trajectory": trajectory,
        }

    def _apply_vae_level(
        self,
        prepared: dict,
        *,
        response: dict | None = None,
        trajectory: np.ndarray | None = None,
    ) -> dict:
        blender_session_id = str(prepared["blender_session_id"])
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "code": "STALE_RESULT", "error": "session ended"}
        bone_name = str(prepared["bone_name"])
        bone_index = int(prepared["bone_index"])
        joint_names = [str(name) for name in prepared["joint_names"]]
        level = int(prepared["level"])
        max_level = int(prepared["max_level"])
        generated = trajectory is not None
        report = None if response is None else response.get("vae_reconstruction")
        if trajectory is not None:
            session.vae_weight_fields[bone_name] = [
                trajectory[index, :, 0].copy()
                for index in range(max_level + 1)
            ]
            if response is not None:
                session.context = response.get("context", session.context)
        elif bone_name not in session.vae_weight_fields:
            assert session.vae_base_skin is not None
            session.vae_weight_fields[bone_name] = [
                session.vae_base_skin[:, bone_index].copy()
            ]

        assert session.vae_base_skin is not None
        session.vae_levels[bone_name] = level
        composed = session.vae_base_skin.copy()
        name_to_index = {name: index for index, name in enumerate(joint_names)}
        for cached_name, cached_level in session.vae_levels.items():
            cached_fields = session.vae_weight_fields.get(cached_name)
            cached_index = name_to_index.get(cached_name)
            if (
                cached_fields is None
                or cached_index is None
                or cached_level < 0
                or cached_level >= len(cached_fields)
            ):
                continue
            composed[:, cached_index] = cached_fields[cached_level]
        composed = normalized_skin_weights(composed)
        apply_mesh_skin_weights(
            session.mesh_object_name,
            joint_names,
            composed,
        )
        session.vae_last_applied_skin = composed.copy()
        cache_bytes = sum(
            field.nbytes
            for fields in session.vae_weight_fields.values()
            for field in fields
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "bone_name": bone_name,
            "level": level,
            "max_level": max_level,
            "generated": generated,
            "cache_invalidated": bool(prepared["cache_invalidated"]),
            "cached_bones": len(session.vae_weight_fields),
            "cache_bytes": cache_bytes,
            "vae_reconstruction": report,
        }

    def apply_vae_request(self, prepared: dict, result: dict) -> dict:
        response = self.apply_session_request(
            result["remote"],
            result["remote_result"],
        )
        if not response.get("ok"):
            return response
        return self._apply_vae_level(
            prepared,
            response=response,
            trajectory=result["trajectory"],
        )

    def set_selected_skin_reconstruction_level(
        self,
        blender_session_id: str,
        level: int,
        *,
        max_level: int = DEFAULT_VAE_RECONSTRUCTION_MAX_LEVEL,
    ) -> dict:
        prepared = self.prepare_vae_reconstruction_level(
            blender_session_id,
            level,
            max_level=max_level,
        )
        if not prepared.get("ok"):
            return prepared
        if not prepared["needs_remote"]:
            return self._apply_vae_level(prepared)
        session = self.sessions[blender_session_id]
        payload = {
            **prepared["remote"]["payload"],
            "skin": encode_float32_array(prepared["base_skin"]),
        }
        response = self.session_request(session, payload)
        if not response.get("ok"):
            return response
        try:
            trajectory = self._decode_vae_trajectory(prepared, response)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return self._apply_vae_level(
            prepared,
            response=response,
            trajectory=trajectory,
        )

    def apply_vae_reconstruction_levels(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}
        applied_levels = {
            name: int(level)
            for name, level in session.vae_levels.items()
            if name in session.vae_weight_fields
        }
        if not applied_levels or session.vae_last_applied_skin is None:
            return {"ok": False, "error": "choose a VAE level above 0 first"}

        self._clear_vae_state(session)
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "applied_levels": applied_levels,
            "level": 0,
        }

    def build_armature(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_armature(session)
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}
        armature = ensure_mesh_armature(
            session.mesh_object_name,
            session.armature_object_name,
        )
        session.armature_object_name = armature.name
        self._apply_context_to_armature(
            session,
            session.context,
            select_name=None,
            mode_after="OBJECT",
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "armature_object_name": armature.name,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def reset(self, blender_session_id: str, end_reason: str = "reset") -> dict:
        payload = self.detach_session(
            blender_session_id,
            end_reason=end_reason,
        )
        return self.model_request(payload)

    def detach_session(
        self,
        blender_session_id: str,
        *,
        end_reason: str = "reset",
    ) -> dict:
        session = self.sessions.pop(blender_session_id)
        self._clear_vae_state(session)
        armature = armature_by_name(session.armature_object_name)
        if armature is not None:
            armature.show_in_front = session.armature_show_in_front_before
        return {
            "command": "reset",
            "session_id": session.model_session_id,
            "end_reason": str(end_reason),
        }
