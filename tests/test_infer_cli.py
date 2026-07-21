from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infer import build_parser, infer_asset


class FakeInteractiveEndpoint:
    def __init__(self, *, fail_skin: bool = False) -> None:
        self.fail_skin = fail_skin
        self.calls: list[dict] = []
        self.endpoints: list[str] = []

    def request(self, endpoint: str, payload: dict) -> dict:
        self.endpoints.append(endpoint)
        self.calls.append(payload)
        command = payload["command"]
        if command == "start":
            return {
                "ok": True,
                "session_id": "session",
                "context": {
                    "joints": [],
                    "parents": [],
                    "joint_names": [],
                    "done": False,
                },
            }
        if command == "sync":
            return {
                "ok": True,
                "context": {
                    "joints": payload["joints"],
                    "parents": payload["parents"],
                    "joint_names": payload["joint_names"],
                    "done": payload["done"],
                },
            }
        if command == "rig":
            return {
                "ok": True,
                "context": {
                    "joints": [[0.0, 0.0, 0.0]],
                    "parents": [-1],
                    "joint_names": ["root"],
                    "done": True,
                },
            }
        if command == "skin":
            if self.fail_skin:
                raise RuntimeError("skin failed")
            output_path = Path(payload["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("root root\n", encoding="utf-8")
            return {"ok": True}
        if command == "reset":
            return {"ok": True}
        raise AssertionError(f"unexpected command: {command}")


class InferCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.obj_path = self.root / "mesh.obj"
        self.obj_path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, *extra: str, server_url: str | None = "http://model.example:8765"):
        values = [
            "--input",
            str(self.obj_path),
            "--output",
            str(self.root / "mesh_skin.txt"),
        ]
        if server_url is not None:
            values.extend(["--server-url", server_url])
        values.extend(extra)
        return build_parser().parse_args(values)

    def run_fake(self, endpoint: FakeInteractiveEndpoint, args) -> Path:
        with patch("infer.request", side_effect=endpoint.request):
            return infer_asset(args)

    def test_without_txt_generates_rig_then_skin_through_http(self) -> None:
        endpoint = FakeInteractiveEndpoint()

        output = self.run_fake(endpoint, self.args())

        self.assertEqual(output.read_text(encoding="utf-8"), "root root\n")
        self.assertEqual(
            [payload["command"] for payload in endpoint.calls],
            ["start", "rig", "skin", "reset"],
        )
        self.assertEqual(set(endpoint.endpoints), {"http://model.example:8765"})
        self.assertEqual(endpoint.calls[2]["midprocess"], "dfs-ensemble")
        self.assertEqual(endpoint.calls[2]["skin_num_beams"], 10)
        self.assertEqual(endpoint.calls[-1]["end_reason"], "finish")

    def test_txt_is_parsed_and_synced_before_skin(self) -> None:
        txt_path = self.root / "mesh.txt"
        txt_path.write_text(
            "joints root 1 2 3\n"
            "joints child 4 5 6\n"
            "root root\n"
            "hier root child\n",
            encoding="utf-8",
        )
        endpoint = FakeInteractiveEndpoint()

        self.run_fake(endpoint, self.args("--txt", str(txt_path)))

        self.assertEqual(
            [payload["command"] for payload in endpoint.calls],
            ["start", "sync", "skin", "reset"],
        )
        self.assertEqual(
            endpoint.calls[1]["joints"],
            [[1.0, -3.0, 2.0], [4.0, -6.0, 5.0]],
        )
        self.assertEqual(endpoint.calls[1]["parents"], [-1, 0])
        self.assertEqual(endpoint.calls[0]["initial_bone_count"], 2)

    def test_reset_runs_when_skin_fails(self) -> None:
        endpoint = FakeInteractiveEndpoint(fail_skin=True)
        with patch("infer.request", side_effect=endpoint.request):
            with self.assertRaisesRegex(RuntimeError, "skin failed"):
                infer_asset(self.args())

        self.assertEqual(endpoint.calls[-1]["command"], "reset")
        self.assertEqual(endpoint.calls[-1]["end_reason"], "infer_failed")

    def test_server_url_defaults_to_runtime_config(self) -> None:
        config_path = self.root / "server.json"
        config_path.write_text(
            json.dumps({"host": "0.0.0.0", "port": 9123}),
            encoding="utf-8",
        )
        endpoint = FakeInteractiveEndpoint()

        self.run_fake(
            endpoint,
            self.args("--config", str(config_path), server_url=None),
        )

        self.assertEqual(set(endpoint.endpoints), {"http://127.0.0.1:9123"})

    def test_legacy_device_option_is_accepted_but_ignored(self) -> None:
        endpoint = FakeInteractiveEndpoint()
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            self.run_fake(endpoint, self.args("--device", "cpu"))

        self.assertIn("--device", stderr.getvalue())
        self.assertEqual(set(endpoint.endpoints), {"http://model.example:8765"})


if __name__ == "__main__":
    unittest.main()
