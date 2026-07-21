from __future__ import annotations

import sys
import argparse
from pathlib import Path

import bpy  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH_PATH = REPO_ROOT / "examples" / "xiaobaozi.obj"


def parse_launcher_args() -> argparse.Namespace:
    raw_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Launch SkinTokens interactive Blender scene with an OBJ or FBX.",
    )
    parser.add_argument("mesh", nargs="?", default=None, help="OBJ/FBX path to import.")
    parser.add_argument("--obj", dest="mesh_option", default=None, help="OBJ/FBX path to import.")
    parser.add_argument("--fbx", dest="mesh_option", default=None, help="FBX path to import.")
    parser.add_argument("--txt", default=None, help="Optional heter-skinning txt to prefill.")
    parser.add_argument(
        "--output",
        default=None,
        help="Save the latest generated skin TXT here when Finish is pressed.",
    )
    parser.add_argument("--server-url", default=None, help="Interactive HTTP model server URL.")
    parser.add_argument("--next-tokens", type=int, default=None)
    parser.add_argument("--skin-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    args = parser.parse_args(raw_args)
    mesh = args.mesh_option or args.mesh or str(DEFAULT_MESH_PATH)
    mesh_path = Path(mesh).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = (REPO_ROOT / mesh_path).resolve()
    else:
        mesh_path = mesh_path.resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    if mesh_path.suffix.lower() not in {".obj", ".fbx"}:
        raise ValueError(f"expected .obj or .fbx, got: {mesh_path}")
    args.mesh_path = mesh_path
    args.output_path = None
    if args.output is not None:
        output_path = Path(args.output).expanduser()
        args.output_path = (
            (REPO_ROOT / output_path).resolve()
            if not output_path.is_absolute()
            else output_path.resolve()
        )
    if args.txt is None:
        args.txt_path = None
    else:
        txt_path = Path(args.txt).expanduser()
        args.txt_path = (
            (REPO_ROOT / txt_path).resolve()
            if not txt_path.is_absolute()
            else txt_path.resolve()
        )
        if not args.txt_path.is_file():
            raise FileNotFoundError(f"TXT not found: {args.txt_path}")
    return args


def import_mesh(path: Path):
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".obj" and hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == ".obj":
        bpy.ops.import_scene.obj(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"unsupported mesh format: {path}")
    created = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in created if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"failed to import mesh: {path}")
    return max(meshes, key=lambda obj: len(obj.data.vertices))


def main():
    args = parse_launcher_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    for obj in list(bpy.context.scene.objects):
        obj.select_set(True)
    bpy.ops.object.delete()

    from interactive.blender import addon

    try:
        addon.unregister()
    except Exception:
        pass
    addon.register()

    mesh = import_mesh(args.mesh_path)
    if args.mesh_path.suffix.lower() == ".obj":
        mesh["skintokens_source_obj"] = str(args.mesh_path)
    else:
        mesh["skintokens_source_fbx"] = str(args.mesh_path)
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.context.scene.skintokens_output_path = (
        "" if args.output_path is None else str(args.output_path)
    )
    if args.server_url is not None:
        bpy.context.scene.skintokens_model_socket = args.server_url.rstrip("/")
    parameter_overrides = {
        "skintokens_max_new_tokens": args.next_tokens,
        "skintokens_skin_max_new_tokens": args.skin_tokens,
        "skintokens_top_k": args.top_k,
        "skintokens_top_p": args.top_p,
        "skintokens_temperature": args.temperature,
        "skintokens_repetition_penalty": args.repetition_penalty,
        "skintokens_num_beams": args.num_beams,
    }
    for property_name, value in parameter_overrides.items():
        if value is not None:
            setattr(bpy.context.scene, property_name, value)
    if args.txt_path is not None:
        bpy.context.scene.skintokens_import_txt_path = str(args.txt_path)
    bpy.context.scene.skintokens_status = (
        f"Loaded {args.mesh_path.name}. Start the model server, then press Start."
    )

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            override = {"area": area, "region": next((r for r in area.regions if r.type == "WINDOW"), None)}
            try:
                bpy.ops.view3d.view_axis(override, type="FRONT", align_active=False)
                bpy.ops.view3d.view_selected(override, use_all_regions=False)
            except Exception:
                pass
            break


main()
