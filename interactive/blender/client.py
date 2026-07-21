from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .transport import DEFAULT_MODEL_URL, request


def model_request(
    payload: Dict[str, Any],
    socket_path: str | Path = DEFAULT_MODEL_URL,
    owner_id: str = "anonymous",
    timeout: float = 300.0,
) -> Dict[str, Any]:
    try:
        import bpy  # type: ignore

        if not bool(getattr(bpy.app, "online_access", True)):
            forced_offline = bool(
                getattr(bpy.app, "online_access_override", False)
            )
            return {
                "ok": False,
                "code": "ONLINE_ACCESS_DISABLED",
                "error": (
                    "Blender was launched with --offline-mode; restart it without "
                    "--offline-mode to use SkinTokens"
                    if forced_offline
                    else "Enable Online Access to use SkinTokens"
                ),
            }
    except ImportError:
        pass
    return request(
        socket_path,
        {**payload, "owner_id": owner_id},
        timeout=timeout,
    )
