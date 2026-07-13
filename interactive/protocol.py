from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / ".runtime"
MODEL_SOCKET_PATH = RUNTIME_DIR / "interactive_model.sock"
BLENDER_SOCKET_PATH = RUNTIME_DIR / "interactive_blender.sock"
AUTHKEY = b"skintokens-interactive"


Request = Dict[str, Any]
Response = Dict[str, Any]


@dataclass(frozen=True)
class SocketConfig:
    path: Path = MODEL_SOCKET_PATH
    authkey: bytes = AUTHKEY


def ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def ok(**payload: Any) -> Response:
    return {"ok": True, **payload}


def err(message: str, **payload: Any) -> Response:
    return {"ok": False, "error": message, **payload}


def request(socket_path: str | Path, payload: Request, authkey: bytes = AUTHKEY) -> Response:
    with Client(str(socket_path), family="AF_UNIX", authkey=authkey) as conn:
        conn.send(payload)
        response = conn.recv()
    if not isinstance(response, dict):
        raise RuntimeError(f"interactive server returned non-dict response: {type(response)}")
    return response

