from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_LATENT_CHECKPOINT = str(REPO_ROOT / "experiments" / "articulation-xl.ckpt")
DISABLE_SERVER_ENV = "SKINTOKENS_NO_SERVER"


def build_parser(*, require_input: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer rig/skin with SkinTokens and export heter_skinning txt.",
    )
    parser.add_argument("--input", required=require_input, help="Input OBJ path.")
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
    parser.add_argument("--latent-checkpoint", default=DEFAULT_LATENT_CHECKPOINT)
    return parser


def normalize_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve(strict=False))
