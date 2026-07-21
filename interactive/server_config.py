from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .protocol import RUNTIME_DIR


DEFAULT_SERVER_CONFIG_PATH = RUNTIME_DIR / "server.json"
DEFAULT_SERVER_CONFIG: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8765,
    "socket": "interactive_model.sock",
    "model_ckpt": (
        "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
    ),
    "hf_path": None,
    "device": "cuda",
    "asset_dir": "interactive_assets",
    "result_dir": "interactive_results",
    "usage_dir": "usage",
    "blender_extensions_enabled": True,
    "blender_extensions_dir": "blender_extensions",
    "max_runtime_sessions": 8,
    "max_sessions": 512,
    "session_idle_timeout_seconds": 3600,
    "session_cleanup_interval_seconds": 60,
    "max_context_bones": 96,
    "skin_postprocess": "none",
    "max_pending_gpu_requests": 8,
    "max_upload_mb": 100,
}
_RUNTIME_PATH_KEYS = {
    "asset_dir",
    "blender_extensions_dir",
    "result_dir",
    "socket",
    "usage_dir",
}


def _write_default_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(DEFAULT_SERVER_CONFIG, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_server_config(
    path: str | Path = DEFAULT_SERVER_CONFIG_PATH,
    *,
    create: bool = True,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if create and not config_path.is_file():
        _write_default_config(config_path)
    if not config_path.is_file():
        return dict(DEFAULT_SERVER_CONFIG)

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"server config must contain a JSON object: {config_path}")
    unknown = sorted(set(loaded) - set(DEFAULT_SERVER_CONFIG))
    if unknown:
        raise ValueError(f"unknown server config keys: {', '.join(unknown)}")

    config = {**DEFAULT_SERVER_CONFIG, **loaded}
    for key in _RUNTIME_PATH_KEYS:
        value = Path(str(config[key])).expanduser()
        if not value.is_absolute():
            value = config_path.parent / value
        config[key] = str(value.resolve())
    return config


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_SERVER_CONFIG_PATH))
    preliminary, _ = config_parser.parse_known_args(argv)
    config = load_server_config(preliminary.config)
    parser_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{
        key: value
        for key, value in config.items()
        if key in parser_destinations
    })
    return parser.parse_args(argv)
