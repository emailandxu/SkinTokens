from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import numpy as np


DEFAULT_MODEL_URL = "http://172.17.60.251:8765"
PROTOCOL_VERSION = 1
EXTENSION_PACKAGE_ID = "h3d_skintokens"
EXTENSION_VERSION = "1.0.0"

Request = Dict[str, Any]
Response = Dict[str, Any]


def encode_float32_array(values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype="<f4")
    compressed = zlib.compress(array.tobytes(order="C"), level=6)
    return {
        "shape": list(array.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
        "encoding": "f32-zlib-base64",
    }


def decode_float32_array(payload: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("encoded float array must be an object")
    shape = tuple(int(value) for value in payload.get("shape", []))
    if shape != expected_shape:
        raise ValueError(f"float array shape {shape} does not match {expected_shape}")
    if payload.get("encoding") != "f32-zlib-base64":
        raise ValueError(f"unsupported float array encoding: {payload.get('encoding')}")
    encoded = payload.get("data")
    if not isinstance(encoded, str):
        raise ValueError("encoded float array data must be a base64 string")

    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * 4
    try:
        compressed = base64.b64decode(encoded, validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_bytes + 1)
    except (ValueError, zlib.error) as exc:
        raise ValueError("invalid compressed float array") from exc
    if (
        len(raw) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("compressed float array has an invalid decoded size")
    return np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()


def request(
    endpoint: str | Path,
    payload: Request,
    *,
    timeout: float = 300.0,
) -> Response:
    base_url = str(endpoint).strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("server URL must start with http:// or https://")
    command = str(payload.get("command", ""))
    owner_id = str(payload.get("owner_id", "anonymous"))
    headers = {
        "X-SkinTokens-Owner": owner_id,
        "X-SkinTokens-Blender-Extension-Version": EXTENSION_VERSION,
    }

    if command == "start":
        obj_path = Path(str(payload["obj_path"])).expanduser().resolve()
        body = obj_path.read_bytes()
        url = f"{base_url}/v1/sessions"
        headers.update(
            {
                "Content-Type": "application/octet-stream",
                "X-SkinTokens-Filename": obj_path.name,
                "X-SkinTokens-Initial-Bones": str(
                    max(0, int(payload.get("initial_bone_count", 0)))
                ),
            }
        )
    elif command == "extension_event":
        url = f"{base_url}/v1/extensions/events"
        body = json.dumps({
            key: value
            for key, value in payload.items()
            if key not in {"command", "owner_id"}
        }).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        session_id = quote(str(payload.get("session_id", "")), safe="")
        if command == "ping":
            url = f"{base_url}/health"
            body = None
        else:
            url = f"{base_url}/v1/sessions/{session_id}/{command}"
            wire_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"command", "session_id", "output_path", "owner_id"}
            }
            if command == "skin" and payload.get("output_path") is not None:
                wire_payload["include_txt"] = True
            body = json.dumps(wire_payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

    method = "GET" if command == "ping" else "POST"
    if command == "reset":
        method = "DELETE"
    request_obj = UrlRequest(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request_obj, timeout=max(1.0, float(timeout))) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError:
            response_payload = {"ok": False, "error": raw or str(exc)}
    except URLError as exc:
        raise RuntimeError(f"cannot connect to SkinTokens server at {base_url}: {exc}") from exc

    if not isinstance(response_payload, dict):
        raise RuntimeError(
            "SkinTokens server returned a non-object response: "
            f"{type(response_payload).__name__}"
        )

    output_path = payload.get("output_path")
    txt_content = response_payload.pop("txt_content", None)
    if response_payload.get("ok") and output_path is not None and txt_content is not None:
        resolved_output = Path(str(output_path)).expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(str(txt_content), encoding="utf-8")
        response_payload["output_path"] = str(resolved_output)
    return response_payload
