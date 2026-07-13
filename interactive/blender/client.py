from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from interactive.protocol import AUTHKEY, MODEL_SOCKET_PATH, request


def model_request(payload: Dict[str, Any], socket_path: str | Path = MODEL_SOCKET_PATH) -> Dict[str, Any]:
    return request(socket_path, payload, authkey=AUTHKEY)

