from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import BLENDER_SOCKET_PATH, MODEL_SOCKET_PATH, request


def print_response(response: dict) -> None:
    print(json.dumps(response, indent=2, ensure_ascii=False))


def model_smoke(args: argparse.Namespace) -> None:
    start = request(args.socket, {"command": "start", "obj_path": args.obj})
    print_response(start)
    if not start.get("ok"):
        return
    session_id = start["session_id"]
    context = start.get("context", {})
    for _ in range(args.steps):
        response = request(
            args.socket,
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
            args.socket,
            {
                "command": "skin",
                "session_id": session_id,
                **context,
                "output_path": args.output,
            },
        )
        print_response(response)


def blender_smoke(args: argparse.Namespace) -> None:
    start = request(args.socket, {"command": "start", "obj_path": args.obj})
    print_response(start)
    if not start.get("ok"):
        return
    blender_session_id = start["blender_session_id"]
    for _ in range(args.steps):
        response = request(
            args.socket,
            {
                "command": "next",
                "blender_session_id": blender_session_id,
                "options": {"max_new_tokens": args.max_new_tokens},
            },
        )
        print_response(response)
        if not response.get("ok"):
            return
        if response.get("context", {}).get("done"):
            break
    if args.skin:
        response = request(
            args.socket,
            {
                "command": "skin",
                "blender_session_id": blender_session_id,
                "output_path": args.output,
            },
        )
        print_response(response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SkinTokens local client.")
    sub = parser.add_subparsers(dest="target", required=True)

    model = sub.add_parser("model", help="Talk directly to the interactive model server.")
    model.add_argument("--socket", default=str(MODEL_SOCKET_PATH))
    model.add_argument("--obj", default="examples/xiaobaozi.obj")
    model.add_argument("--steps", type=int, default=2)
    model.add_argument("--max-new-tokens", type=int, default=16)
    model.add_argument("--skin", action="store_true")
    model.add_argument("--output", default="results/xiaobaozi_interactive_skin.txt")
    model.set_defaults(func=model_smoke)

    blender = sub.add_parser("blender", help="Talk to the Blender validation server.")
    blender.add_argument("--socket", default=str(BLENDER_SOCKET_PATH))
    blender.add_argument("--obj", default="examples/xiaobaozi.obj")
    blender.add_argument("--steps", type=int, default=2)
    blender.add_argument("--max-new-tokens", type=int, default=16)
    blender.add_argument("--skin", action="store_true")
    blender.add_argument("--output", default="results/xiaobaozi_interactive_skin.txt")
    blender.set_defaults(func=blender_smoke)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

