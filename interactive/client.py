from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import DEFAULT_MODEL_URL, request


def print_response(response: dict) -> None:
    print(json.dumps(response, indent=2, ensure_ascii=False))


def model_smoke(args: argparse.Namespace) -> None:
    start = request(args.endpoint, {"command": "start", "obj_path": args.obj})
    print_response(start)
    if not start.get("ok"):
        return
    session_id = start["session_id"]
    context = start.get("context", {})
    for _ in range(args.steps):
        response = request(
            args.endpoint,
            {
                "command": "next",
                "session_id": session_id,
                **context,
                "max_new_tokens": args.max_new_tokens,
            },
        )
        print_response(response)
        if not response.get("ok"):
            return
        context = response.get("context", context)
        if context.get("done"):
            break
    if args.skin:
        response = request(
            args.endpoint,
            {
                "command": "skin",
                "session_id": session_id,
                **context,
                "output_path": args.output,
                "midprocess": args.midprocess,
                "skin_num_beams": args.skin_num_beams,
                "skin_postprocess": args.skin_postprocess,
            },
        )
        print_response(response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SkinTokens model client.")
    sub = parser.add_subparsers(dest="target", required=True)

    model = sub.add_parser("model", help="Talk directly to the interactive model server.")
    model.add_argument("--endpoint", default=DEFAULT_MODEL_URL)
    model.add_argument("--obj", default="examples/xiaobaozi.obj")
    model.add_argument("--steps", type=int, default=2)
    model.add_argument("--max-new-tokens", type=int, default=16)
    model.add_argument("--skin", action="store_true")
    model.add_argument(
        "--midprocess",
        choices=["none", "similar-subtrees", "dfs-ensemble"],
        default="dfs-ensemble",
    )
    model.add_argument("--skin-num-beams", type=int, default=10)
    model.add_argument(
        "--skin-postprocess",
        choices=["none", "voxel", "latent", "vae-reconstruction"],
        default="none",
    )
    model.add_argument("--output", default="results/xiaobaozi_interactive_skin.txt")
    model.set_defaults(func=model_smoke)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
