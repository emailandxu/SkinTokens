from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .apply_skin import create_armature, import_skin_output, parse_mesh_armature, parse_rig_txt
from .client import model_request
from .mesh import import_obj
from .preview import draw_skeleton, read_preview_context, selected_preview_joint_index


@dataclass
class BlenderInteractiveSession:
    blender_session_id: str
    model_session_id: str
    obj_path: Path
    context: dict
    mesh_object_name: str | None = None
    output_path: Optional[Path] = None


class BlenderInteractiveCore:
    def __init__(self, model_socket: str | Path):
        self.model_socket = Path(model_socket)
        self.sessions: Dict[str, BlenderInteractiveSession] = {}
        self.undo_stacks: Dict[str, list[dict]] = {}

    def sync_context_from_preview(self, session: BlenderInteractiveSession) -> None:
        edited = read_preview_context(session.context)
        if edited is not None:
            session.context = edited

    def push_undo(self, session: BlenderInteractiveSession) -> None:
        self.undo_stacks.setdefault(session.blender_session_id, []).append(
            copy.deepcopy(session.context)
        )

    def discard_undo(self, session: BlenderInteractiveSession) -> None:
        stack = self.undo_stacks.get(session.blender_session_id)
        if stack:
            stack.pop()

    def sync_context_to_model(self, session: BlenderInteractiveSession) -> dict:
        payload = {
            "command": "sync",
            "session_id": session.model_session_id,
            **session.context,
            "done": False,
        }
        return model_request(payload, socket_path=self.model_socket)

    def undo_last(self, blender_session_id: str) -> dict:
        session = self.sessions.get(blender_session_id)
        if session is None:
            return {"ok": False, "error": f"unknown blender_session_id: {blender_session_id}"}
        stack = self.undo_stacks.get(blender_session_id)
        if not stack:
            return {"ok": True, "context": session.context, "undone": False}
        session.context = stack.pop()
        response = self.sync_context_to_model(session)
        if not response.get("ok"):
            return response
        session.context = response["context"]
        return {"ok": True, "context": session.context, "undone": True}

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

    def start(
        self,
        obj_path: str | Path,
        *,
        import_mesh: bool = True,
        mesh_object_name: str | None = None,
        **options,
    ) -> dict:
        mesh_object = import_obj(obj_path) if import_mesh else None
        if mesh_object is not None:
            mesh_object_name = mesh_object.name
        response = model_request(
            {"command": "start", "obj_path": str(Path(obj_path).expanduser().resolve()), **options},
            socket_path=self.model_socket,
        )
        if not response.get("ok"):
            return response
        blender_session_id = uuid.uuid4().hex
        session = BlenderInteractiveSession(
            blender_session_id=blender_session_id,
            model_session_id=str(response["session_id"]),
            obj_path=Path(obj_path).expanduser().resolve(),
            context=response.get("context", {}),
            mesh_object_name=mesh_object_name,
        )
        self.sessions[blender_session_id] = session
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def next(self, blender_session_id: str, **options) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_preview(session)
        self.push_undo(session)
        selected_parent = options.pop("branch_parent", None)
        if selected_parent not in (None, "", -1):
            selected_parent = int(selected_parent)
            parents = [int(parent) for parent in session.context.get("parents", [])]
            if parents and selected_parent not in self.dfs_stack(parents):
                session.context, selected_parent = self.focus_context_on_joint(
                    session.context,
                    selected_parent,
                )
                draw_skeleton(
                    session.context.get("joints", []),
                    session.context.get("parents", []),
                    session.context.get("joint_names", []),
                )
            options["branch_parent"] = selected_parent
        payload = {
            "command": "next",
            "session_id": session.model_session_id,
            **session.context,
            **options,
        }
        response = model_request(payload, socket_path=self.model_socket)
        if not response.get("ok"):
            self.discard_undo(session)
            return response
        session.context = response["context"]
        draw_skeleton(
            session.context.get("joints", []),
            session.context.get("parents", []),
            session.context.get("joint_names", []),
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def branch(self, blender_session_id: str, **options) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_preview(session)
        return model_request(
            {
                "command": "branch",
                "session_id": session.model_session_id,
                **session.context,
                **options,
            },
            socket_path=self.model_socket,
        )

    def refresh(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        self.push_undo(session)
        self.sync_context_from_preview(session)
        response = self.sync_context_to_model(session)
        if not response.get("ok"):
            self.discard_undo(session)
            return response
        session.context = response["context"]
        draw_skeleton(
            session.context.get("joints", []),
            session.context.get("parents", []),
            session.context.get("joint_names", []),
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def split_selected_bone(self, blender_session_id: str, ratio: float = 0.5) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_preview(session)
        self.push_undo(session)
        selected = selected_preview_joint_index()
        if selected is None:
            self.discard_undo(session)
            return {"ok": False, "error": "select the child SKT_JOINT_* of the bone to split"}

        joints = [list(joint) for joint in session.context.get("joints", [])]
        parents = [int(parent) for parent in session.context.get("parents", [])]
        names = list(session.context.get("joint_names", []))
        count = len(joints)
        if selected < 0 or selected >= count:
            self.discard_undo(session)
            return {"ok": False, "error": f"selected joint {selected} is outside current skeleton"}
        parent = parents[selected]
        if parent == -1:
            self.discard_undo(session)
            return {"ok": False, "error": "cannot split the root joint; select a child joint instead"}

        ratio = max(0.01, min(0.99, float(ratio)))
        parent_joint = joints[parent]
        child_joint = joints[selected]
        mid_joint = [
            float(parent_joint[axis]) * (1.0 - ratio) + float(child_joint[axis]) * ratio
            for axis in range(3)
        ]
        child_name = names[selected] if selected < len(names) else f"bone_{selected}"
        mid_name = f"{child_name}_split"

        insert_at = selected
        new_joints = joints[:insert_at] + [mid_joint] + joints[insert_at:]
        new_names = names[:insert_at] + [mid_name] + names[insert_at:]

        def remap_index(index: int) -> int:
            return index + 1 if index >= insert_at else index

        new_mid_index = insert_at
        new_parents = []
        for old_index, old_parent in enumerate(parents):
            if old_index == selected:
                new_parents.append(new_mid_index)
            elif old_parent == -1:
                new_parents.append(-1)
            else:
                new_parents.append(remap_index(old_parent))
        new_parents = (
            new_parents[:insert_at]
            + [remap_index(parent)]
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
        response = model_request(payload, socket_path=self.model_socket)
        if not response.get("ok"):
            self.discard_undo(session)
            return response
        session.context = response["context"]
        draw_skeleton(
            session.context.get("joints", []),
            session.context.get("parents", []),
            session.context.get("joint_names", []),
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "split_joint_index": new_mid_index,
        }

    def import_txt(self, blender_session_id: str, txt_path: str | Path) -> dict:
        session = self.sessions[blender_session_id]
        resolved = Path(txt_path).expanduser().resolve()
        if not resolved.is_file():
            return {"ok": False, "error": f"txt not found: {resolved}"}
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}

        self.push_undo(session)
        try:
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
                self.discard_undo(session)
                return response
            session.context = response["context"]
            draw_skeleton(
                session.context.get("joints", []),
                session.context.get("parents", []),
                session.context.get("joint_names", []),
            )
            import_skin_output(
                resolved,
                mesh_object_name=session.mesh_object_name,
                joints=session.context.get("joints", []),
                parents=session.context.get("parents", []),
                joint_names=session.context.get("joint_names", []),
            )
        except Exception:
            self.discard_undo(session)
            raise

        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "txt_path": str(resolved),
        }

    def import_scene_armature(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}

        import bpy  # type: ignore

        mesh_obj = bpy.data.objects.get(session.mesh_object_name)
        if mesh_obj is None or mesh_obj.type != "MESH":
            return {"ok": False, "error": f"mesh object not found: {session.mesh_object_name}"}

        self.push_undo(session)
        try:
            joints, parents, names = parse_mesh_armature(mesh_obj)
            session.context = {
                **session.context,
                "joints": joints,
                "parents": parents,
                "joint_names": names,
                "done": False,
            }
            response = self.sync_context_to_model(session)
            if not response.get("ok"):
                self.discard_undo(session)
                return response
            session.context = response["context"]
            draw_skeleton(
                session.context.get("joints", []),
                session.context.get("parents", []),
                session.context.get("joint_names", []),
            )
            create_armature(
                mesh_obj,
                session.context.get("joints", []),
                session.context.get("parents", []),
                session.context.get("joint_names", []),
            )
        except Exception:
            self.discard_undo(session)
            raise

        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def skin(self, blender_session_id: str, output_path: str | Path | None = None, **options) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_preview(session)
        self.push_undo(session)
        if output_path is None:
            output_path = session.obj_path.with_name(f"{session.obj_path.stem}_interactive_skin.txt")
        session.output_path = Path(output_path).expanduser().resolve()
        payload = {
            "command": "skin",
            "session_id": session.model_session_id,
            "output_path": str(session.output_path),
            **session.context,
            "done": False,
            **options,
        }
        response = model_request(payload, socket_path=self.model_socket)
        if not response.get("ok"):
            self.discard_undo(session)
            return response
        session.context = response["context"]
        import_skin_output(
            session.output_path,
            mesh_object_name=session.mesh_object_name,
            joints=session.context.get("joints", []),
            parents=session.context.get("parents", []),
            joint_names=session.context.get("joint_names", []),
        )
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "model_session_id": session.model_session_id,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
            "output_path": str(session.output_path),
            "skin_shape": response.get("skin_shape"),
        }

    def build_armature(self, blender_session_id: str) -> dict:
        session = self.sessions[blender_session_id]
        self.sync_context_from_preview(session)
        if not session.mesh_object_name:
            return {"ok": False, "error": "session has no mesh object"}
        import bpy  # type: ignore

        mesh_obj = bpy.data.objects.get(session.mesh_object_name)
        if mesh_obj is None:
            return {"ok": False, "error": f"mesh object not found: {session.mesh_object_name}"}
        armature = create_armature(
            mesh_obj,
            session.context.get("joints", []),
            session.context.get("parents", []),
            session.context.get("joint_names", []),
        )
        if armature is None:
            return {"ok": False, "error": "current skeleton is empty"}
        return {
            "ok": True,
            "blender_session_id": blender_session_id,
            "armature_object_name": armature.name,
            "mesh_object_name": session.mesh_object_name,
            "context": session.context,
        }

    def reset(self, blender_session_id: str) -> dict:
        session = self.sessions.pop(blender_session_id)
        return model_request(
            {"command": "reset", "session_id": session.model_session_id},
            socket_path=self.model_socket,
        )
