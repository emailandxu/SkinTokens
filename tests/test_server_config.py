from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from interactive.http_server import build_parser
from interactive.server_config import load_server_config, parse_args_with_config


class ServerConfigTest(unittest.TestCase):
    def test_missing_config_is_created_with_one_hour_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.json"

            config = load_server_config(path)

            self.assertTrue(path.is_file())
            self.assertEqual(config["session_idle_timeout_seconds"], 3600)
            self.assertEqual(config["session_cleanup_interval_seconds"], 60)
            self.assertEqual(config["max_runtime_sessions"], 8)
            self.assertEqual(config["max_sessions"], 512)
            self.assertEqual(config["usage_dir"], str(Path(temporary) / "usage"))
            self.assertTrue(config["blender_extensions_enabled"])
            self.assertEqual(
                config["blender_extensions_dir"],
                str(Path(temporary) / "blender_extensions"),
            )

    def test_config_supplies_defaults_and_cli_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.json"
            path.write_text(
                json.dumps({
                    "host": "0.0.0.0",
                    "port": 9000,
                    "max_sessions": 24,
                    "session_idle_timeout_seconds": 120,
                }),
                encoding="utf-8",
            )

            args = parse_args_with_config(
                build_parser(),
                ["--config", str(path), "--port", "9100"],
            )

            self.assertEqual(args.host, "0.0.0.0")
            self.assertEqual(args.port, 9100)
            self.assertEqual(args.max_sessions, 24)
            self.assertEqual(args.session_idle_timeout_seconds, 120)
            self.assertEqual(
                args.asset_dir,
                str(Path(temporary) / "interactive_assets"),
            )

    def test_unknown_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.json"
            path.write_text('{"session_timout": 3600}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "session_timout"):
                load_server_config(path)


if __name__ == "__main__":
    unittest.main()
