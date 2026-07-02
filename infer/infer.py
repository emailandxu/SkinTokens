#!/usr/bin/env python3
"""Run SkinTokens inference and write a heter_skinning-style skin txt.

The output skin rows are indexed by the input OBJ vertex order.  Do not use a
round-tripped FBX/GLB mesh for this contract: export/import can split vertices.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["XFORMERS_IGNORE_FLASH_VERSION_CHECK"] = "1"

if __package__:
    from .infer_io import (  # noqa: E402
        apply_affine,
        build_debug_arrays,
        fit_affine,
        load_obj_asset,
        load_obj_text_vertices,
        load_rig_txt,
        write_debug_npz,
        write_heter_txt,
        write_skin_with_txt_template,
    )
    from .infer_postprocess import (  # type: ignore[import-not-found]  # noqa: E402
        DEFAULT_LATENT_CHECKPOINT,
        LatentSkinSmoother,
        latent_smooth_options_from_args,
    )
    from .infer_rigpatcher import (  # type: ignore[import-not-found]  # noqa: E402
        RigSpec,
        apply_rig,
        remap_skin_to_rig,
        reorder_rig_spec,
        rig_with_joint_order,
        similar_subtree_order,
        transformed_rig_from_asset,
    )
    from .infer_scoring import (  # type: ignore[import-not-found]  # noqa: E402
        generate_output_ids,
        generation_kwargs_from_args,
        get_skeleton_tokens_from_batch,
        print_skeleton_score,
        print_skin_score,
        score_generated_skin_tokens,
        score_skeleton_tokens,
    )
else:
    from infer_io import (  # noqa: E402
        apply_affine,
        build_debug_arrays,
        fit_affine,
        load_obj_asset,
        load_obj_text_vertices,
        load_rig_txt,
        write_debug_npz,
        write_heter_txt,
        write_skin_with_txt_template,
    )
    from infer_postprocess import (  # noqa: E402
        DEFAULT_LATENT_CHECKPOINT,
        LatentSkinSmoother,
        latent_smooth_options_from_args,
    )
    from infer_rigpatcher import (  # noqa: E402
        RigSpec,
        apply_rig,
        remap_skin_to_rig,
        reorder_rig_spec,
        rig_with_joint_order,
        similar_subtree_order,
        transformed_rig_from_asset,
    )
    from infer_scoring import (  # noqa: E402
        generate_output_ids,
        generation_kwargs_from_args,
        get_skeleton_tokens_from_batch,
        print_skeleton_score,
        print_skin_score,
        score_generated_skin_tokens,
        score_skeleton_tokens,
    )
from src.data.transform import Transform  # noqa: E402
from src.data.vertex_group import voxel_skin  # noqa: E402
from src.model.spec import ModelInput  # noqa: E402
from src.model.tokenrig import TokenRig  # noqa: E402
from src.tokenizer.spec import TokenizeInput  # noqa: E402


DEFAULT_MODEL_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"


def get_model(
    ckpt_path: str,
    hf_path: Optional[str] = None,
    device: str = "cuda",
) -> TokenRig:
    model = TokenRig.load_from_system_checkpoint(checkpoint_path=ckpt_path)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer rig/skin with SkinTokens and export heter_skinning txt.",
    )
    parser.add_argument("--input", required=True, help="Input OBJ path.")
    parser.add_argument("--output", default=None, help="Output *_skin.txt path.")
    parser.add_argument(
        "--debug-npz",
        default=None,
        help="Optional debug npz with predicted asset arrays.",
    )
    parser.add_argument(
        "--txt",
        default=None,
        help="Optional heter-skinning rig txt used as a skeleton condition.",
    )
    parser.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--topk-skin", type=int, default=4)
    parser.add_argument("--weight-eps", type=float, default=1e-8)
    parser.add_argument("--postprocess", default="none", choices=["none", "latent", "voxel"])
    parser.add_argument(
        "--score-skeleton",
        action="store_true",
        help="Score --txt skeleton token likelihood and exit without generating skin.",
    )
    parser.add_argument(
        "--reorder-generated-skeleton",
        default="similar-subtrees",
        choices=["none", "similar-subtrees"],
        help="For auto mode, reorder generated skeleton siblings before generating skin.",
    )
    parser.add_argument("--latent-smooth-iterations", type=int, default=10)
    parser.add_argument("--latent-neighbor-factor", type=float, default=0.3)
    parser.add_argument("--latent-k", type=int, default=12)
    parser.add_argument("--latent-threshold-std", type=float, default=-1.5)
    parser.add_argument(
        "--latent-checkpoint",
        default=DEFAULT_LATENT_CHECKPOINT,
    )
    return parser.parse_args()


def collate_single(processed: list[Dict]) -> dict:
    if len(processed) != 1:
        raise ValueError("infer.py only supports one input asset")
    item = processed[0]
    batch: dict = {}
    non = item.get("non", {})
    for key, value in item.items():
        if key == "non":
            continue
        if isinstance(value, np.ndarray):
            batch[key] = torch.from_numpy(value).unsqueeze(0)
        elif isinstance(value, Tensor):
            batch[key] = value.unsqueeze(0)
        else:
            raise ValueError(f"cannot collate key {key} with type {type(value)}")
    for key, value in non.items():
        batch[key] = [value]
    return batch


def build_batch(
    input_path: Path,
    model,
    rig_txt_path: Optional[Path] = None,
    reorder_skeleton: bool = False,
) -> tuple[dict, Optional[RigSpec], Optional[RigSpec]]:
    asset = load_obj_asset(input_path)
    input_rig = load_rig_txt(rig_txt_path) if rig_txt_path is not None else None
    token_rig = input_rig
    if input_rig is not None and reorder_skeleton:
        order = similar_subtree_order(input_rig.joints, input_rig.parents)
        if order != list(range(len(order))):
            token_rig = reorder_rig_spec(input_rig, order)
            print("[infer] reordered input skeleton internally by similar subtrees")
        else:
            print("[infer] input skeleton sibling order unchanged")
    if token_rig is not None:
        apply_rig(asset, token_rig)

    transform = Transform.parse(**model.transform_config["predict_transform"])
    transform.apply(asset=asset)
    transformed_token_rig = transformed_rig_from_asset(asset, token_rig)
    transformed_output_rig = (
        rig_with_joint_order(transformed_token_rig, input_rig)
        if input_rig is not None and transformed_token_rig is not None
        else None
    )

    tokens = None
    if asset.parents is not None:
        tokens = model.tokenizer.tokenize(
            input=TokenizeInput(
                joints=asset.joints,
                parents=asset.parents.tolist(),
                cls=asset.cls,
                joint_names=asset.joint_names,
            )
        )
    model_input = ModelInput(asset=asset, tokens=None)
    model_input.tokens = tokens
    processed = model._process_fn([model_input])
    return collate_single(processed), transformed_token_rig, transformed_output_rig


def move_tensor_batch_to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def infer_asset(args: argparse.Namespace):
    input_path = Path(args.input).resolve()
    if input_path.suffix.lower() != ".obj":
        raise ValueError("infer.py currently expects an input .obj")
    rig_txt_path = Path(args.txt).resolve() if args.txt is not None else None
    if args.output is None and not args.score_skeleton:
        raise ValueError("--output is required unless --score-skeleton is used")
    if args.score_skeleton and rig_txt_path is None:
        raise ValueError("--score-skeleton requires --txt")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    obj_text_vertices = load_obj_text_vertices(input_path)
    obj_text_vertex_count = obj_text_vertices.shape[0]

    model = get_model(args.model_ckpt, hf_path=args.hf_path, device=args.device)
    model.eval()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    reorder_input_skeleton = (
        rig_txt_path is not None
        and args.reorder_generated_skeleton == "similar-subtrees"
    )
    batch, token_rig, output_rig = build_batch(
        input_path,
        model=model,
        rig_txt_path=rig_txt_path,
        reorder_skeleton=reorder_input_skeleton,
    )
    batch = move_tensor_batch_to_device(batch, args.device)

    if args.score_skeleton:
        batch["generate_kwargs"] = generation_kwargs_from_args(args)
        skeleton_tokens = get_skeleton_tokens_from_batch(batch)
        score = score_skeleton_tokens(model, batch)
        print_skeleton_score(score)
        output_ids = generate_output_ids(model, batch, skeleton_tokens)
        skin_score = score_generated_skin_tokens(model, batch, output_ids)
        print_skin_score(skin_score)
        return None

    if rig_txt_path is None:
        batch.pop("skeleton_tokens", None)
        batch.pop("skeleton_mask", None)

    batch["generate_kwargs"] = generation_kwargs_from_args(args)

    if "skeleton_tokens" in batch and "skeleton_mask" in batch:
        mask = batch["skeleton_mask"][0] == 1
        skeleton_tokens = batch["skeleton_tokens"][0][mask].cpu().numpy()
    else:
        skeleton_tokens = None

    pred = model.predict_step(
        batch,
        skeleton_tokens=[skeleton_tokens] if skeleton_tokens is not None else None,
        make_asset=True,
    )["results"][0]

    if (
        args.reorder_generated_skeleton == "similar-subtrees"
        and rig_txt_path is None
        and pred.detokenize_output is not None
    ):
        generated_joints = pred.detokenize_output.joints.astype(np.float32)
        generated_parents = np.asarray(pred.detokenize_output.parents, dtype=np.int32)
        generated_names = (
            list(pred.detokenize_output.joint_names)
            if pred.detokenize_output.joint_names is not None
            else [f"bone_{i}" for i in range(generated_joints.shape[0])]
        )
        order = similar_subtree_order(generated_joints, generated_parents)
        if order != list(range(len(order))):
            reordered_rig = RigSpec(
                joints=generated_joints,
                parents=generated_parents,
                joint_names=generated_names,
                obj_text_joints=generated_joints.copy(),
            )
            reordered_rig = reorder_rig_spec(reordered_rig, order)
            skeleton_tokens = model.tokenizer.tokenize(
                input=TokenizeInput(
                    joints=reordered_rig.joints,
                    parents=reordered_rig.parents.tolist(),
                    cls=pred.detokenize_output.cls,
                    joint_names=reordered_rig.joint_names,
                )
            )
            print("[infer] reordered generated skeleton by similar subtrees")
            pred = model.predict_step(
                batch,
                skeleton_tokens=[skeleton_tokens],
                make_asset=True,
            )["results"][0]
            if pred.asset is not None:
                apply_rig(pred.asset, reordered_rig)
        else:
            print("[infer] generated skeleton sibling order unchanged")

    asset = pred.asset
    if asset is None:
        raise RuntimeError("SkinTokens did not return an asset")
    if asset.vertices is None or asset.skin is None or asset.joints is None or asset.parents is None:
        raise RuntimeError("predicted asset is missing vertices, skin, joints, or parents")
    if asset.vertices.shape[0] != obj_text_vertex_count:
        raise RuntimeError(
            f"vertex contract failed: input OBJ has {obj_text_vertex_count} vertices, "
            f"predicted asset has {asset.vertices.shape[0]}"
        )
    if output_rig is not None and token_rig is not None:
        if asset.skin.shape[1] != token_rig.parents.shape[0]:
            raise RuntimeError(
                f"skin-only rig contract failed: predicted skin has {asset.skin.shape[1]} "
                f"joints, but --txt rig has {token_rig.parents.shape[0]}"
            )
        remap_skin_to_rig(asset, source_rig=token_rig, target_rig=output_rig)
        apply_rig(asset, output_rig)

    latent_neighbors = None
    if args.postprocess == "voxel":
        voxel = asset.voxel(resolution=196)
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
    elif args.postprocess == "latent":
        options = latent_smooth_options_from_args(args)
        print(
            "[infer] latent postprocess "
            f"iterations={options.iterations} "
            f"k={options.latent_k} "
            f"neighbor_factor={options.neighbor_factor} "
            f"threshold_std={options.threshold_std} "
            f"checkpoint={options.checkpoint_path}"
        )
        asset.skin, latent_neighbors = LatentSkinSmoother(options)(asset)
        asset.normalize_skin()
    else:
        print("[infer] no postprocess applied")

    normalized_to_obj_text, max_affine_err = fit_affine(asset.vertices, obj_text_vertices)
    if output_rig is not None:
        obj_text_joints = output_rig.obj_text_joints.copy()
    else:
        obj_text_joints = apply_affine(asset.joints, normalized_to_obj_text)
    debug_arrays = build_debug_arrays(pred, batch, latent_neighbors)

    print(f"[infer] input vertices: {obj_text_vertex_count}")
    print(f"[infer] output skin rows: {asset.skin.shape[0]}")
    print(f"[infer] joints: {asset.joints.shape[0]}")
    print(f"[infer] normalized->obj-text affine max error: {max_affine_err:.6e}")
    return asset, obj_text_joints, debug_arrays


def main() -> None:
    args = parse_args()
    result = infer_asset(args)
    if result is None:
        return
    asset, obj_text_joints, debug_arrays = result
    if args.debug_npz is not None:
        write_debug_npz(
            asset=asset,
            obj_text_joints=obj_text_joints,
            output_path=Path(args.debug_npz).resolve(),
            debug_arrays=debug_arrays,
        )
    if args.txt is not None:
        write_skin_with_txt_template(
            asset=asset,
            template_path=Path(args.txt).resolve(),
            output_path=Path(args.output).resolve(),
            topk=args.topk_skin,
            eps=args.weight_eps,
        )
    else:
        write_heter_txt(
            asset=asset,
            obj_text_joints=obj_text_joints,
            output_path=Path(args.output).resolve(),
            topk=args.topk_skin,
            eps=args.weight_eps,
        )


if __name__ == "__main__":
    main()
