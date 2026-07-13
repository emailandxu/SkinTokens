from __future__ import annotations

import argparse
import os
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

from interactive.protocol import (
    AUTHKEY,
    BLENDER_SOCKET_PATH,
    MODEL_SOCKET_PATH,
    ensure_runtime_dir,
    err,
    ok,
)

from .core import BlenderInteractiveCore


def serve(args: argparse.Namespace) -> None:
    ensure_runtime_dir()
    socket_path = Path(args.socket).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    core = BlenderInteractiveCore(model_socket=args.model_socket)
    print(f"[interactive] blender validation server listening on {socket_path}")
    with Listener(str(socket_path), family="AF_UNIX", authkey=AUTHKEY) as listener:
        while True:
            conn = listener.accept()
            try:
                payload = conn.recv()
                command = str(payload.get("command", ""))
                if command == "start":
                    response = core.start(payload["obj_path"], **payload.get("options", {}))
                elif command == "next":
                    response = core.next(payload["blender_session_id"], **payload.get("options", {}))
                elif command == "branch":
                    response = core.branch(payload["blender_session_id"], **payload.get("options", {}))
                elif command == "skin":
                    response = core.skin(
                        payload["blender_session_id"],
                        output_path=payload.get("output_path"),
                        **payload.get("options", {}),
                    )
                elif command == "reset":
                    response = core.reset(payload["blender_session_id"])
                elif command == "ping":
                    response = ok(message="pong")
                else:
                    response = err(f"unknown command: {command}")
            except Exception as exc:
                response = err(f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            try:
                conn.send(response)
            finally:
                conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SkinTokens Blender validation server.")
    parser.add_argument("--socket", default=str(BLENDER_SOCKET_PATH))
    parser.add_argument("--model-socket", default=str(MODEL_SOCKET_PATH))
    return parser


def main() -> None:
    serve(build_parser().parse_args())


if __name__ == "__main__":
    main()
