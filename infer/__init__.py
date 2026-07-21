from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Sequence

from interactive.protocol import request
from interactive.rig_text import load_rig_txt_payload
from interactive.server_config import DEFAULT_SERVER_CONFIG_PATH, load_server_config


MIDPROCESS_NONE = "none"
MIDPROCESS_SIMILAR_SUBTREES = "similar-subtrees"
MIDPROCESS_DFS_ENSEMBLE = "dfs-ensemble"
SKIN_POSTPROCESS_NONE = "none"
SKIN_POSTPROCESS_MODES = (
    SKIN_POSTPROCESS_NONE,
    "voxel",
    "latent",
    "vae-reconstruction",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one rig-and-skin job through the interactive HTTP service.",
    )
    parser.add_argument("--input", required=True, help="Input OBJ path.")
    parser.add_argument("--output", required=True, help="Output skin TXT path.")
    parser.add_argument(
        "--txt",
        default=None,
        help="Optional heter-skinning TXT used as the skeleton context.",
    )
    parser.add_argument("--config", default=str(DEFAULT_SERVER_CONFIG_PATH))
    parser.add_argument(
        "--server-url",
        default=None,
        help="HTTP model service URL; defaults to host/port from --config.",
    )
    parser.add_argument("--model-ckpt", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--hf-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-context-bones",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--skin-num-beams", type=int, default=10)
    parser.add_argument("--topk-skin", type=int, default=4)
    parser.add_argument("--weight-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--midprocess",
        "--skin-mode",
        dest="midprocess",
        choices=[
            "single",
            MIDPROCESS_NONE,
            MIDPROCESS_SIMILAR_SUBTREES,
            MIDPROCESS_DFS_ENSEMBLE,
        ],
        default=MIDPROCESS_DFS_ENSEMBLE,
    )
    parser.add_argument(
        "--skin-postprocess",
        "--postprocess",
        dest="skin_postprocess",
        choices=SKIN_POSTPROCESS_MODES,
        default=SKIN_POSTPROCESS_NONE,
    )
    return parser


def _checked(endpoint: str, payload: dict) -> dict:
    try:
        response = request(endpoint, payload)
    except Exception as exc:
        raise RuntimeError(f"interactive request to {endpoint} failed: {exc}") from exc
    if response.get("ok"):
        return response
    code = response.get("code", "UNKNOWN")
    raise RuntimeError(f"interactive request failed [{code}]: {response.get('error')}")


def _server_url(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "server_url", "") or "").strip().rstrip("/")
    if explicit:
        if not explicit.startswith(("http://", "https://")):
            raise ValueError("server URL must start with http:// or https://")
        return explicit

    config = load_server_config(getattr(args, "config", DEFAULT_SERVER_CONFIG_PATH))
    host = str(config["host"]).strip()
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{int(config['port'])}"


def _warn_ignored_server_options(args: argparse.Namespace) -> None:
    ignored = [
        option
        for option in ("device", "model_ckpt", "hf_path", "max_context_bones")
        if getattr(args, option, None) is not None
    ]
    if ignored:
        print(
            "[infer] warning: server-side options are ignored by the HTTP client: "
            + ", ".join(f"--{option.replace('_', '-')}" for option in ignored),
            file=sys.stderr,
        )


def _generation_options(args: argparse.Namespace) -> dict:
    midprocess = MIDPROCESS_NONE if args.midprocess == "single" else args.midprocess
    return {
        "max_new_tokens": max(1, int(args.max_new_tokens)),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "repetition_penalty": float(args.repetition_penalty),
        "num_beams": max(1, int(args.num_beams)),
        "skin_num_beams": max(1, int(args.skin_num_beams)),
        "topk_skin": max(1, int(args.topk_skin)),
        "weight_eps": float(args.weight_eps),
        "midprocess": midprocess,
        "skin_postprocess": str(args.skin_postprocess),
        **({} if args.seed is None else {"seed": int(args.seed)}),
    }


def infer_asset(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    txt_path = None if args.txt is None else Path(args.txt).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input OBJ not found: {input_path}")
    if input_path.suffix.lower() != ".obj":
        raise ValueError(f"input must be an OBJ file: {input_path}")
    if txt_path is not None and not txt_path.is_file():
        raise FileNotFoundError(f"skeleton TXT not found: {txt_path}")
    initial_context = None if txt_path is None else load_rig_txt_payload(txt_path)
    endpoint = _server_url(args)
    _warn_ignored_server_options(args)
    owner_id = f"infer-cli-{uuid.uuid4().hex}"
    session_id = None
    completed = False
    options = _generation_options(args)

    try:
        started = _checked(
            endpoint,
            {
                "command": "start",
                "obj_path": str(input_path),
                "asset_id": input_path.stem,
                "owner_id": owner_id,
                "initial_bone_count": (
                    0
                    if initial_context is None
                    else len(initial_context["joints"])
                ),
            },
        )
        session_id = str(started["session_id"])
        context = dict(started["context"])

        if initial_context is not None:
            context = initial_context
            synced = _checked(
                endpoint,
                {
                    "command": "sync",
                    "session_id": session_id,
                    "owner_id": owner_id,
                    **context,
                },
            )
            context = dict(synced["context"])
        else:
            rigged = _checked(
                endpoint,
                {
                    **options,
                    **context,
                    "command": "rig",
                    "session_id": session_id,
                    "owner_id": owner_id,
                },
            )
            context = dict(rigged["context"])

        _checked(
            endpoint,
            {
                **options,
                **context,
                "command": "skin",
                "session_id": session_id,
                "owner_id": owner_id,
                "output_path": str(output_path),
            },
        )
        if not output_path.is_file():
            raise RuntimeError(f"interactive skin did not write output: {output_path}")
        completed = True
        return output_path
    finally:
        if session_id is not None:
            try:
                _checked(
                    endpoint,
                    {
                        "command": "reset",
                        "session_id": session_id,
                        "owner_id": owner_id,
                        "end_reason": "finish" if completed else "infer_failed",
                    },
                )
            except Exception as exc:
                print(f"[infer] session cleanup failed: {exc}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> None:
    output_path = infer_asset(build_parser().parse_args(argv))
    print(f"[infer] wrote {output_path}")


__all__ = ["build_parser", "infer_asset", "main"]
