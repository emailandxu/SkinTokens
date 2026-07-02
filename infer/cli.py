from __future__ import annotations

import json
import os
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Optional

from .args import (
    DISABLE_SERVER_ENV,
    build_parser,
    normalize_optional_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = REPO_ROOT / ".runtime"
DEFAULT_SOCKET_PATH = DEFAULT_RUNTIME_DIR / "infer.sock"
DEFAULT_INFO_PATH = DEFAULT_RUNTIME_DIR / "infer_server.json"
AUTHKEY = b"skintokens-infer"


def load_info(info_path: Path = DEFAULT_INFO_PATH) -> Optional[dict[str, Any]]:
    try:
        return json.loads(info_path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def send_request(socket_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    with Client(str(socket_path), family="AF_UNIX", authkey=AUTHKEY) as conn:
        conn.send(request)
        return conn.recv()


def try_server_infer(args) -> bool:
    if os.environ.get(DISABLE_SERVER_ENV, "").lower() in {"1", "true", "yes", "on"}:
        return False

    info = load_info()
    if not info or not info.get("ready"):
        return False

    try:
        pid = int(info.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if not pid_alive(pid):
        return False

    if normalize_optional_path(args.model_ckpt) != normalize_optional_path(info.get("model_ckpt")):
        return False
    if normalize_optional_path(args.hf_path) != normalize_optional_path(info.get("hf_path")):
        return False

    try:
        response = send_request(
            Path(info["socket"]),
            {"command": "infer", "args": vars(args)},
        )
    except Exception:
        return False

    if response.get("ok"):
        device = response.get("device")
        if device:
            print(f"[infer] completed by infer.server pid={pid} device={device}")
        else:
            print(f"[infer] completed by infer.server pid={pid}")
        return True
    raise RuntimeError(response.get("error", "infer.server request failed"))


def main() -> None:
    args = build_parser(require_input=True).parse_args()
    if try_server_infer(args):
        return

    from .infer import infer_asset, write_infer_outputs

    result = infer_asset(args)
    write_infer_outputs(args, result)


if __name__ == "__main__":
    main()
