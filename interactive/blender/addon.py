from __future__ import annotations

bl_info = {
    "name": "SkinTokens Interactive MVP",
    "author": "SkinTokens",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > SkinTokens",
    "description": "Interactive Start / Next / Skin MVP for SkinTokens.",
    "category": "Rigging",
}

import tempfile
import traceback
from pathlib import Path

import bpy  # type: ignore

from interactive.protocol import MODEL_SOCKET_PATH

from .core import BlenderInteractiveCore


_CORE: BlenderInteractiveCore | None = None
_LAST_UNDO_MARKER = 0


def get_core(context) -> BlenderInteractiveCore:
    global _CORE
    socket_path = Path(context.scene.skintokens_model_socket).expanduser()
    if _CORE is None or _CORE.model_socket != socket_path:
        _CORE = BlenderInteractiveCore(model_socket=socket_path)
    return _CORE


def set_status(context, message: str) -> None:
    context.scene.skintokens_status = message
    print(f"[SkinTokens Interactive] {message}")


def selected_mesh(context):
    obj = context.object
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Select a mesh object first")
    return obj


def export_selected_obj(context, obj) -> Path:
    path = Path(tempfile.gettempdir()) / f"skintokens_interactive_{obj.name}.obj"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    context.view_layer.objects.active = obj
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
    else:
        bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)
    return path


def source_obj_path(context, obj) -> Path:
    raw = obj.get("skintokens_source_obj")
    if raw:
        path = Path(str(raw)).expanduser().resolve()
        if path.exists():
            return path
    return export_selected_obj(context, obj)


def dfs_stack_from_parents(parents):
    if not parents:
        return []
    stack = []
    current = len(parents) - 1
    while current != -1:
        stack.append(int(current))
        current = int(parents[current])
    stack.reverse()
    return stack


def active_session(context):
    core = get_core(context)
    session_id = context.scene.skintokens_blender_session_id
    if not session_id:
        return None
    return core.sessions.get(session_id)


def branch_parent_items(self, context):
    blank = [("-1", " ", "Free generation")]
    session = active_session(context)
    if session is None:
        return blank
    parents = session.context.get("parents", [])
    names = session.context.get("joint_names", [])
    count = len(parents)
    if count == 0:
        return blank
    stack = dfs_stack_from_parents(parents)
    stack_set = set(stack)
    items = list(blank)
    last = count - 1
    for idx in range(count):
        name = names[idx] if idx < len(names) else f"bone_{idx}"
        tags = []
        if idx == last:
            tags.append("last")
        if idx in stack_set:
            tags.append("stack")
        suffix = f" ({', '.join(tags)})" if tags else ""
        items.append((str(idx), f"{idx}: {name}{suffix}", "If needed, Next will reorder DFS so this bone is branchable"))
    return items


def selected_branch_parent(context):
    raw = context.scene.skintokens_branch_parent
    if raw in {"", "-1"}:
        return None
    parent = int(raw)
    session = active_session(context)
    if session is None:
        return None
    if parent < 0 or parent >= len(session.context.get("parents", [])):
        return None
    return parent


def clear_branch_parent(context) -> None:
    context.scene.skintokens_branch_parent = "-1"


def mark_undo_step(context) -> None:
    global _LAST_UNDO_MARKER
    context.scene.skintokens_undo_marker += 1
    _LAST_UNDO_MARKER = context.scene.skintokens_undo_marker


def sync_model_after_undo(_dummy=None) -> None:
    global _LAST_UNDO_MARKER
    if _CORE is None:
        return
    scene = bpy.context.scene
    if not hasattr(scene, "skintokens_undo_marker"):
        return
    marker = scene.skintokens_undo_marker
    if marker >= _LAST_UNDO_MARKER:
        _LAST_UNDO_MARKER = marker
        return
    session_id = scene.skintokens_blender_session_id
    if not session_id:
        _LAST_UNDO_MARKER = marker
        return
    steps = _LAST_UNDO_MARKER - marker
    for _ in range(steps):
        response = _CORE.undo_last(session_id)
        if not response.get("ok"):
            print(f"[SkinTokens Interactive] Undo sync failed: {response.get('error')}")
            break
    _LAST_UNDO_MARKER = marker
    print(f"[SkinTokens Interactive] Synced model after undo: {session_id[:8]}")


class SKINTOKENS_OT_start(bpy.types.Operator):
    bl_idname = "skintokens_interactive.start"
    bl_label = "Start"
    bl_description = "Create a SkinTokens interactive session for the selected mesh"

    def execute(self, context):
        try:
            obj = selected_mesh(context)
            obj_path = source_obj_path(context, obj)
            core = get_core(context)
            response = core.start(
                obj_path,
                import_mesh=False,
                mesh_object_name=obj.name,
                top_k=context.scene.skintokens_top_k,
                top_p=context.scene.skintokens_top_p,
                temperature=context.scene.skintokens_temperature,
                repetition_penalty=context.scene.skintokens_repetition_penalty,
                num_beams=context.scene.skintokens_num_beams,
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "start failed"))
            context.scene.skintokens_blender_session_id = response["blender_session_id"]
            obj["skintokens_blender_session_id"] = response["blender_session_id"]
            obj["skintokens_model_session_id"] = response["model_session_id"]
            obj["skintokens_source_obj"] = str(obj_path)
            clear_branch_parent(context)
            set_status(context, f"Started session {response['blender_session_id'][:8]}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Start failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_next(bpy.types.Operator):
    bl_idname = "skintokens_interactive.next"
    bl_label = "Next"
    bl_description = "Generate the next joint or skeleton EOS"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            core = get_core(context)
            response = core.next(
                session_id,
                branch_parent=selected_branch_parent(context),
                max_new_tokens=context.scene.skintokens_max_new_tokens,
                top_k=context.scene.skintokens_top_k,
                top_p=context.scene.skintokens_top_p,
                temperature=context.scene.skintokens_temperature,
                repetition_penalty=context.scene.skintokens_repetition_penalty,
                num_beams=context.scene.skintokens_num_beams,
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "next failed"))
            count = len(response.get("context", {}).get("joints", []))
            done = response.get("context", {}).get("done", False)
            clear_branch_parent(context)
            mark_undo_step(context)
            set_status(context, f"Generated joints: {count}{' (done)' if done else ''}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Next failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_import_txt(bpy.types.Operator):
    bl_idname = "skintokens_interactive.import_txt"
    bl_label = "Import TXT"
    bl_description = "Import heter-skinning txt skeleton and skin into the current session"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            if not context.scene.skintokens_import_txt_path.strip():
                raise RuntimeError("Choose a heter-skinning txt first")
            txt_path = Path(context.scene.skintokens_import_txt_path).expanduser()
            response = get_core(context).import_txt(session_id, txt_path)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "import txt failed"))
            count = len(response.get("context", {}).get("joints", []))
            clear_branch_parent(context)
            mark_undo_step(context)
            set_status(context, f"Imported TXT joints: {count}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Import TXT failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_import_fbx_rig(bpy.types.Operator):
    bl_idname = "skintokens_interactive.import_fbx_rig"
    bl_label = "Import FBX Rig"
    bl_description = "Import the selected mesh armature and vertex groups into the current session"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            response = get_core(context).import_scene_armature(session_id)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "import fbx rig failed"))
            count = len(response.get("context", {}).get("joints", []))
            clear_branch_parent(context)
            mark_undo_step(context)
            set_status(context, f"Imported FBX rig joints: {count}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Import FBX Rig failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_skin(bpy.types.Operator):
    bl_idname = "skintokens_interactive.skin"
    bl_label = "Skin"
    bl_description = "Generate skin for the current interactive skeleton"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            output_path = Path(context.scene.skintokens_output_path).expanduser()
            core = get_core(context)
            response = core.skin(
                session_id,
                output_path=output_path,
                max_new_tokens=context.scene.skintokens_skin_max_new_tokens,
                top_k=context.scene.skintokens_top_k,
                top_p=context.scene.skintokens_top_p,
                temperature=context.scene.skintokens_temperature,
                repetition_penalty=context.scene.skintokens_repetition_penalty,
                num_beams=context.scene.skintokens_num_beams,
                skin_reorder=context.scene.skintokens_skin_reorder,
                skin_postprocess=context.scene.skintokens_skin_postprocess,
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "skin failed"))
            mark_undo_step(context)
            set_status(
                context,
                "Skin written: "
                f"{response.get('output_path')} "
                f"[reorder={response.get('skin_reorder')}, "
                f"post={response.get('skin_postprocess')}]",
            )
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Skin failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_refresh(bpy.types.Operator):
    bl_idname = "skintokens_interactive.refresh"
    bl_label = "Refresh"
    bl_description = "Sync edited/deleted preview joints without generating"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            response = get_core(context).refresh(session_id)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "refresh failed"))
            count = len(response.get("context", {}).get("joints", []))
            mark_undo_step(context)
            set_status(context, f"Refreshed joints: {count}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Refresh failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_OT_split(bpy.types.Operator):
    bl_idname = "skintokens_interactive.split"
    bl_label = "Split"
    bl_description = "Insert a midpoint joint into the selected bone"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            session_id = context.scene.skintokens_blender_session_id
            if not session_id:
                raise RuntimeError("Start a session first")
            response = get_core(context).split_selected_bone(session_id)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "split failed"))
            count = len(response.get("context", {}).get("joints", []))
            clear_branch_parent(context)
            mark_undo_step(context)
            set_status(context, f"Split bone. Joints: {count}")
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            set_status(context, f"Split failed: {exc}")
            return {"CANCELLED"}


class SKINTOKENS_PT_interactive(bpy.types.Panel):
    bl_label = "SkinTokens Interactive"
    bl_idname = "SKINTOKENS_PT_interactive"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SkinTokens"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "skintokens_model_socket")
        layout.prop(scene, "skintokens_output_path")
        layout.prop(scene, "skintokens_import_txt_path")
        layout.prop(scene, "skintokens_branch_parent")
        layout.prop(scene, "skintokens_skin_reorder")
        layout.prop(scene, "skintokens_skin_postprocess")
        layout.operator("skintokens_interactive.start", icon="PLAY")
        layout.operator("skintokens_interactive.import_txt", icon="IMPORT")
        layout.operator("skintokens_interactive.import_fbx_rig", icon="ARMATURE_DATA")
        layout.operator("skintokens_interactive.next", icon="TRACKING_FORWARDS")
        layout.operator("skintokens_interactive.refresh", icon="FILE_REFRESH")
        layout.operator("skintokens_interactive.split", icon="ADD")
        layout.operator("skintokens_interactive.skin", icon="MOD_ARMATURE")
        layout.separator()
        layout.prop(scene, "skintokens_max_new_tokens")
        layout.prop(scene, "skintokens_skin_max_new_tokens")
        layout.prop(scene, "skintokens_top_k")
        layout.prop(scene, "skintokens_top_p")
        layout.prop(scene, "skintokens_temperature")
        layout.prop(scene, "skintokens_repetition_penalty")
        layout.prop(scene, "skintokens_num_beams")
        layout.separator()
        layout.label(text=f"Session: {scene.skintokens_blender_session_id[:8] or '-'}")
        layout.label(text=scene.skintokens_status)


classes = (
    SKINTOKENS_OT_start,
    SKINTOKENS_OT_next,
    SKINTOKENS_OT_import_txt,
    SKINTOKENS_OT_import_fbx_rig,
    SKINTOKENS_OT_skin,
    SKINTOKENS_OT_refresh,
    SKINTOKENS_OT_split,
    SKINTOKENS_PT_interactive,
)


def register():
    global _LAST_UNDO_MARKER
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.skintokens_model_socket = bpy.props.StringProperty(
        name="Model Socket",
        default=str(MODEL_SOCKET_PATH),
    )
    bpy.types.Scene.skintokens_output_path = bpy.props.StringProperty(
        name="Output",
        default=str(Path("results/xiaobaozi_interactive_skin.txt").resolve()),
        subtype="FILE_PATH",
    )
    bpy.types.Scene.skintokens_import_txt_path = bpy.props.StringProperty(
        name="Import TXT",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.skintokens_blender_session_id = bpy.props.StringProperty(default="")
    bpy.types.Scene.skintokens_branch_parent = bpy.props.EnumProperty(
        name="Next Parent",
        description="Optional one-shot parent constraint for the next generation step",
        items=branch_parent_items,
    )
    bpy.types.Scene.skintokens_skin_reorder = bpy.props.EnumProperty(
        name="Skin Reorder",
        description="Optional infer-style skeleton ordering used internally during Skin",
        items=[
            ("none", "None", "Use the current interactive skeleton order"),
            ("similar-subtrees", "Similar Subtrees", "Group similar sibling subtrees before skin generation, then remap back"),
        ],
        default="none",
    )
    bpy.types.Scene.skintokens_skin_postprocess = bpy.props.EnumProperty(
        name="Skin Postprocess",
        description="Optional infer-style postprocess applied after skin generation",
        items=[
            ("none", "None", "Write raw generated skin"),
            ("voxel", "Voxel", "Apply voxel visibility weighting and renormalize"),
            ("latent", "Latent", "Apply latent-neighbor smoothing and renormalize"),
        ],
        default="none",
    )
    bpy.types.Scene.skintokens_status = bpy.props.StringProperty(default="Idle")
    bpy.types.Scene.skintokens_max_new_tokens = bpy.props.IntProperty(name="Next Tokens", default=16, min=1, max=64)
    bpy.types.Scene.skintokens_skin_max_new_tokens = bpy.props.IntProperty(name="Skin Tokens", default=2048, min=8, max=8192)
    bpy.types.Scene.skintokens_top_k = bpy.props.IntProperty(name="Top K", default=5, min=0, max=100)
    bpy.types.Scene.skintokens_top_p = bpy.props.FloatProperty(name="Top P", default=0.95, min=0.0, max=1.0)
    bpy.types.Scene.skintokens_temperature = bpy.props.FloatProperty(name="Temperature", default=1.5, min=0.01, max=5.0)
    bpy.types.Scene.skintokens_repetition_penalty = bpy.props.FloatProperty(name="Repetition Penalty", default=1.2, min=0.1, max=5.0)
    bpy.types.Scene.skintokens_num_beams = bpy.props.IntProperty(name="Beams", default=1, min=1, max=16)
    bpy.types.Scene.skintokens_undo_marker = bpy.props.IntProperty(default=0, options={"HIDDEN"})
    _LAST_UNDO_MARKER = 0
    if sync_model_after_undo not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(sync_model_after_undo)


def unregister():
    if sync_model_after_undo in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(sync_model_after_undo)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    for name in (
        "skintokens_model_socket",
        "skintokens_output_path",
        "skintokens_import_txt_path",
        "skintokens_blender_session_id",
        "skintokens_branch_parent",
        "skintokens_skin_reorder",
        "skintokens_skin_postprocess",
        "skintokens_status",
        "skintokens_max_new_tokens",
        "skintokens_skin_max_new_tokens",
        "skintokens_top_k",
        "skintokens_top_p",
        "skintokens_temperature",
        "skintokens_repetition_penalty",
        "skintokens_num_beams",
        "skintokens_undo_marker",
    ):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)


if __name__ == "__main__":
    register()
