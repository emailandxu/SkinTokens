from __future__ import annotations

from pathlib import Path


def _bpy():
    import bpy  # type: ignore

    return bpy


def import_obj(path: str | Path):
    bpy = _bpy()
    resolved = Path(path).expanduser().resolve()
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(resolved))
    else:
        bpy.ops.import_scene.obj(filepath=str(resolved))
    created = [obj for obj in bpy.data.objects if obj not in before]
    mesh_objects = [obj for obj in created if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"OBJ import created no mesh objects: {resolved}")
    mesh = mesh_objects[0]
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    return mesh

