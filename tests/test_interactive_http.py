from __future__ import annotations

import gc
import json
import tempfile
import threading
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import urlopen
from wsgiref.simple_server import WSGIRequestHandler, make_server

import numpy as np
import torch
from bottle import BaseRequest

from infer import build_parser as build_infer_parser
from infer import infer_asset
from interactive.asset_store import MeshAssetStore
from interactive.blender.apply_skin import selected_skin_bone_names
from interactive.blender.transport import request as blender_request
from interactive.blender.core import (
    BlenderInteractiveCore,
    BlenderInteractiveSession,
    normalized_skin_weights,
    same_skeleton_context,
)
from interactive.blender.preview import _context_without_deleted_joints, preview_style
from interactive.server import InteractiveHttpApi, ThreadingWSGIServer
from interactive.protocol import decode_float32_array, encode_float32_array, request
from interactive.server import InteractiveModelServer, InteractiveServiceError
from interactive.session import InteractiveSession, SessionRecord, SkeletonContext
from interactive.usage_events import UsageEventLog, load_usage_events
from interactive.skeleton_stream import (
    GenerationOptions,
    generate_rig,
    preserve_context_prefix,
)


class FakeModelService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def status(self) -> dict:
        return {
            "ok": True,
            "message": "pong",
            "protocol_version": 1,
            "server_version": "1.8.2",
        }

    def handle(self, payload: dict) -> dict:
        self.calls.append(payload)
        command = payload["command"]
        if command == "start":
            return {
                "ok": True,
                "session_id": "model-session",
                "asset_id": payload["asset_id"],
                "context": {"joints": [], "parents": [], "joint_names": [], "done": False},
                "blender_to_obj_text": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                ],
            }
        if command == "next":
            return {
                "ok": True,
                "session_id": payload["session_id"],
                "context": {
                    "joints": [[1.0, 2.0, 3.0]],
                    "parents": [-1],
                    "joint_names": ["root"],
                    "done": False,
                },
            }
        if command == "rig":
            return {
                "ok": True,
                "session_id": payload["session_id"],
                "context": {
                    "joints": [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]],
                    "parents": [-1, 0],
                    "joint_names": ["root", "child"],
                    "done": True,
                },
            }
        if command == "skin":
            bone_count = len(payload.get("joint_names", []))
            skin = np.full((3, bone_count), 1.0 / bone_count, dtype=np.float32)
            response = {
                "ok": True,
                "session_id": payload["session_id"],
                "context": {**payload, "done": True},
                "skin_shape": list(skin.shape),
                "asset_skin_shape": list(skin.shape),
                "skin": encode_float32_array(skin),
            }
            if payload.get("include_txt"):
                response["txt_content"] = "bones 1\nroot -1\n"
            return response
        if command == "reconstruct":
            return {
                "ok": True,
                "session_id": payload["session_id"],
                "context": {
                    "joints": payload.get("joints", []),
                    "parents": payload.get("parents", []),
                    "joint_names": payload.get("joint_names", []),
                    "done": bool(payload.get("done", False)),
                },
                "bone_names": payload["bone_names"],
                "skin": payload["skin"],
                "vae_reconstruction": {"bones": len(payload["bone_names"])},
            }
        if command == "reset":
            return {"ok": True, "session_id": payload["session_id"]}
        raise AssertionError(f"unexpected command: {command}")


class InteractiveHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.service = FakeModelService()
        self.service.usage_events = UsageEventLog(root / "usage")
        self.extension_archive = b"blender-extension-archive"
        self.extension_archive_name = "h3d_skintokens-1.0.0.zip"
        self.extension_repo = root / "blender_extensions"
        self.extension_repo.mkdir()
        (self.extension_repo / self.extension_archive_name).write_bytes(
            self.extension_archive
        )
        (self.extension_repo / "index.json").write_text(
            json.dumps({
                "version": "v1",
                "blocklist": [],
                "data": [{
                    "schema_version": "1.0.0",
                    "id": "h3d_skintokens",
                    "name": "H3D Skintokens",
                    "version": "1.0.0",
                    "type": "add-on",
                    "archive_url": f"./{self.extension_archive_name}",
                    "archive_size": len(self.extension_archive),
                    "archive_hash": "sha256:test",
                }],
            }),
            encoding="utf-8",
        )
        self.api = InteractiveHttpApi(
            service=self.service,  # type: ignore[arg-type]
            asset_store=MeshAssetStore(root / "assets"),
            result_dir=root / "results",
            max_upload_bytes=1024 * 1024,
            max_pending_gpu_requests=2,
            blender_extensions_dir=self.extension_repo,
        )
        self.server = make_server(
            "127.0.0.1",
            0,
            self.api.app,
            server_class=ThreadingWSGIServer,
            handler_class=WSGIRequestHandler,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.obj_path = root / "mesh.obj"
        self.obj_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_remote_start_next_skin_and_reset(self) -> None:
        started = request(
            self.endpoint,
            {
                "command": "start",
                "obj_path": str(self.obj_path),
                "owner_id": "client-a",
                "initial_bone_count": 7,
            },
        )
        self.assertTrue(started["ok"])
        self.assertEqual(len(list((Path(self.temporary.name) / "assets").glob("*.obj"))), 1)
        self.assertEqual(self.service.calls[0]["initial_bone_count"], 7)
        self.assertEqual(self.service.calls[0]["client_ip"], "127.0.0.1")

        context = {"joints": [], "parents": [], "joint_names": [], "done": False}
        generated = request(
            self.endpoint,
            {
                "command": "next",
                "session_id": started["session_id"],
                "owner_id": "client-a",
                **context,
            },
        )
        self.assertEqual(generated["context"]["joint_names"], ["root"])

        rigged = request(
            self.endpoint,
            {
                "command": "rig",
                "session_id": started["session_id"],
                "owner_id": "client-a",
                **generated["context"],
            },
        )
        self.assertTrue(rigged["context"]["done"])
        self.assertEqual(rigged["context"]["joint_names"], ["root", "child"])

        output_path = Path(self.temporary.name) / "client" / "skin.txt"
        skinned = request(
            self.endpoint,
            {
                "command": "skin",
                "session_id": started["session_id"],
                "owner_id": "client-a",
                "output_path": str(output_path),
                **rigged["context"],
            },
        )
        self.assertTrue(skinned["ok"])
        self.assertEqual(output_path.read_text(encoding="utf-8"), "bones 1\nroot -1\n")
        self.assertEqual(
            decode_float32_array(skinned["skin"], (3, 2)).shape,
            (3, 2),
        )
        self.assertEqual(list((Path(self.temporary.name) / "results").glob("*.txt")), [])

        skin_payload = encode_float32_array(np.ones((3, 1), dtype=np.float32))
        reconstructed = request(
            self.endpoint,
            {
                "command": "reconstruct",
                "session_id": started["session_id"],
                "owner_id": "client-a",
                "bone_names": ["root"],
                "skin": skin_payload,
                "padding": "x" * 150_000,
                **generated["context"],
            },
        )
        self.assertTrue(reconstructed["ok"])
        self.assertEqual(reconstructed["bone_names"], ["root"])
        self.assertEqual(reconstructed["skin"], skin_payload)

        reset = request(
            self.endpoint,
            {
                "command": "reset",
                "session_id": started["session_id"],
                "owner_id": "client-a",
                "end_reason": "finish",
            },
        )
        self.assertTrue(reset["ok"])
        self.assertEqual(self.service.calls[-1]["end_reason"], "finish")
        self.assertTrue(Path(self.service.calls[0]["obj_path"]).is_file())

    def test_json_parser_limit_matches_configured_upload_limit(self) -> None:
        self.assertGreaterEqual(BaseRequest.MEMFILE_MAX, 1024 * 1024)

    def test_blender_extension_repository_serves_index_and_archive(self) -> None:
        with urlopen(f"{self.endpoint}/blender/extensions/index.json") as result:
            index = json.loads(result.read().decode("utf-8"))
            self.assertEqual(result.headers["Cache-Control"], "no-cache")
        self.assertEqual(index["data"][0]["version"], "1.0.0")

        with urlopen(
            f"{self.endpoint}/blender/extensions/{self.extension_archive_name}"
        ) as result:
            archive = result.read()
            cache_control = result.headers["Cache-Control"]
        self.assertEqual(archive, self.extension_archive)
        self.assertIn("immutable", cache_control)

        health = request(self.endpoint, {"command": "ping"})
        repository = health["blender_extensions"]
        self.assertTrue(repository["ready"])
        self.assertEqual(repository["package_id"], "h3d_skintokens")
        self.assertEqual(repository["latest_version"], "1.0.0")
        self.assertEqual(repository["repository_path"], "/blender/extensions/")

    def test_blender_extension_repository_rejects_unpublished_files(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{self.endpoint}/blender/extensions/not-an-extension.txt")
        self.assertEqual(raised.exception.code, 404)

    def test_blender_transport_reports_extension_version(self) -> None:
        started = blender_request(
            self.endpoint,
            {
                "command": "start",
                "obj_path": str(self.obj_path),
                "owner_id": "blender-client",
            },
        )

        self.assertTrue(started["ok"])
        self.assertEqual(
            self.service.calls[0]["blender_extension_version"],
            "1.0.0",
        )

    def test_extension_install_and_update_events_are_logged(self) -> None:
        for action, previous_version, version in (
            ("install", "", "1.0.0"),
            ("update", "1.0.0", "1.0.1"),
        ):
            result = blender_request(
                self.endpoint,
                {
                    "command": "extension_event",
                    "owner_id": "installation-a",
                    "action": action,
                    "package_id": "h3d_skintokens",
                    "extension_version": version,
                    "previous_version": previous_version,
                    "blender_version": "5.2.0",
                    "installation_id": "installation-a",
                    "installation_source": "repository",
                },
            )
            self.assertTrue(result["ok"])

        events = load_usage_events(self.service.usage_events.root)
        self.assertEqual(
            [event["event"] for event in events],
            ["extension_install", "extension_update"],
        )
        self.assertEqual(events[0]["client_ip"], "127.0.0.1")
        self.assertEqual(events[1]["previous_version"], "1.0.0")

    def test_infer_cli_uses_running_http_service(self) -> None:
        output_path = Path(self.temporary.name) / "infer" / "skin.txt"
        args = build_infer_parser().parse_args([
            "--input",
            str(self.obj_path),
            "--output",
            str(output_path),
            "--server-url",
            self.endpoint,
        ])

        result = infer_asset(args)

        self.assertEqual(result, output_path)
        self.assertEqual(output_path.read_text(encoding="utf-8"), "bones 1\nroot -1\n")
        self.assertEqual(
            [payload["command"] for payload in self.service.calls],
            ["start", "rig", "skin", "reset"],
        )
        self.assertEqual(self.service.calls[-1]["end_reason"], "finish")

    def test_asset_store_deduplicates_mesh_content(self) -> None:
        store = MeshAssetStore(Path(self.temporary.name) / "deduplicated")
        first = store.save_obj(b"v 0 0 0\n")
        second = store.save_obj(b"v 0 0 0\n")
        self.assertEqual(first, second)
        self.assertEqual(len(list(store.root.glob("*.obj"))), 1)

    def test_idle_session_locks_are_released_automatically(self) -> None:
        first = self.api._session_lock("same-session")
        second = self.api._session_lock("same-session")
        self.assertIs(first, second)
        self.assertEqual(len(self.api.session_locks), 1)

        del first, second
        gc.collect()

        self.assertEqual(len(self.api.session_locks), 0)


class RuntimeLruTest(unittest.TestCase):
    def test_runtime_cache_evicts_least_recently_used(self) -> None:
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.max_runtime_sessions = 2
        service.runtime_cache = OrderedDict()

        service._put_runtime(SimpleNamespace(session_id="a"))
        service._put_runtime(SimpleNamespace(session_id="b"))
        service.runtime_cache.move_to_end("a")
        service._put_runtime(SimpleNamespace(session_id="c"))

        self.assertEqual(list(service.runtime_cache), ["a", "c"])

    def test_session_cache_evicts_least_recently_used(self) -> None:
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.max_sessions = 2
        service.sessions = OrderedDict()
        service.runtime_cache = OrderedDict()

        for session_id in ("a", "b"):
            service.sessions[session_id] = SessionRecord(
                session_id=session_id,
                owner_id="client-a",
                asset_id="asset",
                obj_path=Path("mesh.obj"),
                created_at=0.0,
                updated_at=0.0,
            )
            service.runtime_cache[session_id] = SimpleNamespace(
                session_id=session_id,
            )
        service._record({"session_id": "a", "owner_id": "client-a"})
        service.sessions["c"] = SessionRecord(
            session_id="c",
            owner_id="client-a",
            asset_id="asset",
            obj_path=Path("mesh.obj"),
            created_at=0.0,
            updated_at=0.0,
        )
        service.runtime_cache["c"] = SimpleNamespace(session_id="c")

        service._enforce_session_limit()

        self.assertEqual(list(service.sessions), ["a", "c"])
        self.assertEqual(list(service.runtime_cache), ["a", "c"])

    def test_missing_session_has_explicit_expired_code(self) -> None:
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.sessions = {}
        with self.assertRaises(InteractiveServiceError) as raised:
            service._record({"session_id": "missing", "owner_id": "client-a"})
        self.assertEqual(raised.exception.code, "SESSION_EXPIRED")

    def test_session_name_registry_does_not_reuse_a_deleted_name(self) -> None:
        record = SessionRecord(
            session_id="session",
            owner_id="client-a",
            asset_id="asset",
            obj_path=Path("mesh.obj"),
            created_at=0.0,
            updated_at=0.0,
        )
        historical = SkeletonContext(
            joints=np.zeros((3, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 1], dtype=np.int32),
            joint_names=["root", "bone_1", "bone_2"],
        )
        current = SkeletonContext(
            joints=np.zeros((2, 3), dtype=np.float32),
            parents=np.asarray([-1, 0], dtype=np.int32),
            joint_names=["root", "bone_1"],
        )
        generated = SkeletonContext(
            joints=np.zeros((3, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 1], dtype=np.int32),
            joint_names=["root", "bone_1", "bone_2"],
        )
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.sessions = {record.session_id: record}
        service.runtime_cache = OrderedDict()
        service.model = object()
        service.max_context_bones = 96
        session = SimpleNamespace(session_id="session", context=current)
        service._session = Mock(return_value=session)
        service._context = Mock(return_value=current)
        service._reserve_context_names(record, historical)

        with patch(
            "interactive.server.generate_next",
            return_value=(generated, np.asarray([1, 2, 3], dtype=np.int64)),
        ):
            response = service.next({"session_id": "session"})

        self.assertTrue(response["ok"])
        self.assertEqual(
            response["context"]["joint_names"],
            ["root", "bone_1", "bone_3"],
        )
        self.assertIn("bone_2", record.reserved_joint_names)
        self.assertIn("bone_3", record.reserved_joint_names)
        self.assertEqual(record.next_bone_id, 4)

        service.max_runtime_sessions = 1
        service._put_runtime(SimpleNamespace(session_id="other"))
        self.assertIn("bone_3", record.reserved_joint_names)

    def test_duplicate_context_names_are_rejected(self) -> None:
        context = SkeletonContext(
            joints=np.zeros((2, 3), dtype=np.float32),
            parents=np.asarray([-1, 0], dtype=np.int32),
            joint_names=["root", "root"],
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            context.validate_hierarchy()

    def test_oversized_armature_is_rejected_without_reduction(self) -> None:
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.max_context_bones = 2
        context = SkeletonContext(
            joints=np.zeros((3, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 1], dtype=np.int32),
            joint_names=["root", "middle", "leaf"],
        )

        with self.assertRaises(InteractiveServiceError) as raised:
            service._validate_context(context)

        self.assertEqual(raised.exception.code, "ARMATURE_TOO_LARGE")
        self.assertIn("3 bones", str(raised.exception))
        self.assertIn("limit is 2", str(raised.exception))

    def test_non_dfs_context_is_rejected_before_tokenization(self) -> None:
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.max_context_bones = 96
        context = SkeletonContext(
            joints=np.zeros((4, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 0, 1], dtype=np.int32),
            joint_names=["root", "a", "b", "a_child"],
        )

        with self.assertRaisesRegex(ValueError, "not stored in DFS order"):
            service._validate_context(context)


class PreviewStyleTest(unittest.TestCase):
    def test_generate_rig_stops_at_skeleton_eos(self) -> None:
        tokenizer = SimpleNamespace(bos=1, eos=9, pad=0)
        generated = torch.tensor([[7, 8, 9, 99]], dtype=torch.long)
        transformer = SimpleNamespace(
            get_input_embeddings=lambda: (
                lambda tokens: torch.zeros(
                    (tokens.shape[0], tokens.shape[1], 4),
                    dtype=torch.float32,
                )
            ),
            generate=Mock(return_value=generated),
        )
        model = SimpleNamespace(
            tokenizer=tokenizer,
            transformer=transformer,
            eos=100,
            tokens_per_skin=32,
        )
        context = SkeletonContext.empty()
        completed = SkeletonContext(
            joints=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            parents=np.asarray([-1], dtype=np.int32),
            joint_names=["root"],
            done=True,
        )
        session = SimpleNamespace(
            vertices=torch.zeros((3, 3)),
            learned_mesh_cond=torch.zeros((1, 2, 4)),
        )

        with (
            patch(
                "interactive.skeleton_stream.start_tokens_without_eos",
                return_value=np.asarray([1, 2], dtype=np.int64),
            ),
            patch(
                "interactive.skeleton_stream.decode_skeleton_tokens",
                return_value=completed,
            ),
            patch(
                "interactive.skeleton_stream.preserve_context_prefix",
                return_value=completed,
            ),
        ):
            result_context, tokens = generate_rig(
                model,
                session,
                context,
                GenerationOptions(max_new_tokens=128),
            )

        self.assertIs(result_context, completed)
        self.assertEqual(tokens.tolist(), [1, 2, 7, 8, 9])
        self.assertEqual(transformer.generate.call_args.kwargs["eos_token_id"], 9)

    def test_preview_scales_with_skeleton_dimensions(self) -> None:
        small = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        large = [[coordinate * 100.0 for coordinate in joint] for joint in small]

        small_joint, small_bone = preview_style(small)
        large_joint, large_bone = preview_style(large)

        self.assertAlmostEqual(large_joint / small_joint, 100.0)
        self.assertAlmostEqual(large_bone / small_bone, 100.0)

    def test_preview_scale_stays_fixed_for_a_mesh_reference(self) -> None:
        partial = [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]
        complete = [[0.0, 0.0, 0.0], [10.0, 5.0, 2.0]]

        partial_style = preview_style(partial, reference_diagonal=3.5)
        complete_style = preview_style(complete, reference_diagonal=3.5)

        self.assertEqual(partial_style, complete_style)

    def test_deleting_internal_joint_removes_its_subtree(self) -> None:
        context = {
            "joints": [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]],
            "parents": [-1, 0, 1, 0],
            "joint_names": ["root", "branch", "branch_leaf", "sibling"],
            "done": False,
        }
        visible_joints = [context["joints"][0], None, context["joints"][2], context["joints"][3]]

        compact = _context_without_deleted_joints(context, visible_joints)

        self.assertEqual(compact["joint_names"], ["root", "sibling"])
        self.assertEqual(compact["parents"], [-1, 0])

    def test_next_preserves_existing_hierarchy_after_token_decode(self) -> None:
        previous = SkeletonContext(
            joints=np.asarray(
                [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]],
                dtype=np.float32,
            ),
            parents=np.asarray([-1, 0, 1, 1], dtype=np.int32),
            joint_names=["root", "a", "same_position", "branch"],
        )
        decoded = SkeletonContext(
            joints=np.asarray(
                [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                dtype=np.float32,
            ),
            parents=np.asarray([-1, 0, 1, 2, 3], dtype=np.int32),
            joint_names=["bone_0", "bone_1", "bone_2", "bone_3", "bone_4"],
        )

        merged = preserve_context_prefix(previous, decoded)

        self.assertEqual(merged.parents.tolist(), [-1, 0, 1, 1, 3])
        self.assertEqual(
            merged.joint_names,
            ["root", "a", "same_position", "branch", "bone_4"],
        )


class BlenderSkinSelectionTest(unittest.TestCase):
    def test_active_mesh_vertex_group_selects_one_skin_field(self) -> None:
        active_group = SimpleNamespace(name="foot")
        mesh = SimpleNamespace(
            type="MESH",
            vertex_groups=SimpleNamespace(active=active_group),
        )
        bpy = SimpleNamespace(
            context=SimpleNamespace(object=mesh),
            data=SimpleNamespace(
                objects=SimpleNamespace(get=Mock(return_value=mesh)),
            ),
        )

        with patch("interactive.blender.apply_skin._bpy", return_value=bpy):
            selected = selected_skin_bone_names("Mesh")

        self.assertEqual(selected, ["foot"])

    def test_active_pose_armature_selects_all_pose_bones(self) -> None:
        bones = [
            SimpleNamespace(name="root", select=False),
            SimpleNamespace(name="left_foot", select=True),
            SimpleNamespace(name="right_foot", select=True),
        ]
        armature = SimpleNamespace(
            mode="POSE",
            data=SimpleNamespace(bones=bones),
        )
        mesh = SimpleNamespace(type="MESH")
        bpy = SimpleNamespace(
            context=SimpleNamespace(object=armature),
            data=SimpleNamespace(
                objects=SimpleNamespace(get=Mock(return_value=mesh)),
            ),
        )

        with (
            patch("interactive.blender.apply_skin._bpy", return_value=bpy),
            patch(
                "interactive.blender.apply_skin.armature_for_mesh",
                return_value=armature,
            ),
        ):
            selected = selected_skin_bone_names("Mesh")

        self.assertEqual(selected, ["left_foot", "right_foot"])


class InteractiveEnsembleTest(unittest.TestCase):
    def test_rig_generation_uses_full_skeleton_budget(self) -> None:
        partial = SkeletonContext(
            joints=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            parents=np.asarray([-1], dtype=np.int32),
            joint_names=["root"],
        )
        complete = SkeletonContext(
            joints=np.asarray(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            parents=np.asarray([-1, 0], dtype=np.int32),
            joint_names=["root", "child"],
            done=True,
        )
        session = SimpleNamespace(session_id="session", context=partial)
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.model = object()
        service.max_context_bones = 96
        service._session = Mock(return_value=session)
        service._context = Mock(return_value=partial)

        with patch(
            "interactive.server.generate_rig",
            return_value=(complete, np.asarray([1, 2, 3], dtype=np.int64)),
        ) as generated:
            response = service.rig({"session_id": "session"})

        options = generated.call_args.args[3]
        self.assertEqual(options.max_new_tokens, 800)
        self.assertTrue(response["context"]["done"])
        self.assertIs(session.context, complete)

    def test_default_midprocess_reuses_session_conditions_with_ten_beams(self) -> None:
        affine = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        context = SkeletonContext(
            joints=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            parents=np.asarray([-1], dtype=np.int32),
            joint_names=["root"],
        )
        learned_mesh_cond = torch.randn(1, 2, 3)
        cond_latents = torch.randn(1, 2, 3)
        session = InteractiveSession(
            session_id="session",
            obj_path=Path("mesh.obj"),
            device="cpu",
            cls="articulation",
            vertices=torch.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            normals=torch.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            learned_mesh_cond=learned_mesh_cond,
            cond_latents=cond_latents,
            normalized_vertices_cpu=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            blender_vertices=np.zeros((3, 3), dtype=np.float32),
            obj_text_vertices=np.zeros((3, 3), dtype=np.float32),
            normalized_to_blender=affine,
            blender_to_normalized=affine,
            normalized_to_obj_text=affine,
            blender_to_obj_text=affine,
            context=context,
        )
        selected = SimpleNamespace(
            sampled_skin=np.ones((3, 1), dtype=np.float32),
            order=(0,),
        )
        ensemble = SimpleNamespace(
            selected=selected,
            report=Mock(
                return_value={
                    "mode": "dfs-ensemble",
                    "candidate_count": 1,
                    "selected_candidate": "baseline",
                }
            ),
        )
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.model = object()
        service.device = "cpu"
        service._session = Mock(return_value=session)
        service._context = Mock(return_value=context)

        with patch(
            "interactive.server.generate_skin_ensemble",
            return_value=ensemble,
        ) as generated:
            response = service.skin({"session_id": "session"})

        kwargs = generated.call_args.kwargs
        self.assertIs(kwargs["learned_mesh_cond"], learned_mesh_cond)
        self.assertIs(kwargs["cond_latents"], cond_latents)
        self.assertEqual(kwargs["options"].num_beams, 10)
        self.assertEqual(response["midprocess"], "dfs-ensemble")
        self.assertEqual(response["skin_ensemble"]["selected_candidate"], "baseline")
        self.assertTrue(response["context"]["done"])

    def test_skin_response_limits_weights_to_top_four_by_default(self) -> None:
        bone_count = 6
        affine = np.eye(4, dtype=np.float64)[:, :3]
        context = SkeletonContext(
            joints=np.zeros((bone_count, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 1, 2, 3, 4], dtype=np.int32),
            joint_names=[f"bone_{index}" for index in range(bone_count)],
        )
        session = InteractiveSession(
            session_id="session",
            obj_path=Path("mesh.obj"),
            device="cpu",
            cls="articulation",
            vertices=torch.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            normals=torch.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            learned_mesh_cond=None,
            cond_latents=None,
            normalized_vertices_cpu=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            blender_vertices=np.zeros((3, 3), dtype=np.float32),
            obj_text_vertices=np.zeros((3, 3), dtype=np.float32),
            normalized_to_blender=affine,
            blender_to_normalized=affine,
            normalized_to_obj_text=affine,
            blender_to_obj_text=affine,
            context=context,
        )
        dense_skin = np.asarray(
            [
                [0.30, 0.25, 0.20, 0.10, 0.08, 0.07],
                [0.05, 0.06, 0.07, 0.08, 0.34, 0.40],
                [0.10, 0.15, 0.25, 0.20, 0.18, 0.12],
            ],
            dtype=np.float32,
        )
        ensemble = SimpleNamespace(
            selected=SimpleNamespace(
                sampled_skin=dense_skin,
                order=tuple(range(bone_count)),
            ),
            report=Mock(return_value={}),
        )
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.model = object()
        service.device = "cpu"
        service._session = Mock(return_value=session)
        service._context = Mock(return_value=context)

        with patch(
            "interactive.server.generate_skin_ensemble",
            return_value=ensemble,
        ):
            response = service.skin({"session_id": "session"})

        returned = decode_float32_array(
            response["skin"],
            tuple(response["asset_skin_shape"]),
        )
        self.assertLessEqual(int((returned > 1e-8).sum(axis=1).max()), 4)
        np.testing.assert_allclose(returned.sum(axis=1), 1.0, atol=1e-6)


class BlenderRecoveryTest(unittest.TestCase):
    def test_server_status_rejects_incompatible_protocol(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        core.model_request = Mock(
            return_value={
                "ok": True,
                "protocol_version": 2,
                "server_version": "2.0.0",
            }
        )

        response = core.server_status()

        self.assertFalse(response["ok"])
        self.assertEqual(response["code"], "PROTOCOL_MISMATCH")

    def test_history_comparison_ignores_storage_precision_only(self) -> None:
        context = {
            "joints": [[0.123456789, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
        }
        float32_snapshot = {
            **context,
            "joints": [[0.12345679, 0.0, 0.0], [1.0, 0.0, 0.0]],
        }
        moved_snapshot = {
            **context,
            "joints": [[0.12345679, 0.0, 0.0], [1.01, 0.0, 0.0]],
        }

        self.assertTrue(same_skeleton_context(context, float32_snapshot))
        self.assertFalse(same_skeleton_context(context, moved_snapshot))

    def test_history_sync_skips_an_unchanged_armature(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        context = {
            "joints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
            "done": True,
        }
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock()

        with (
            patch("interactive.blender.core.armature_by_name", return_value=object()),
            patch(
                "interactive.blender.core.parse_armature_object",
                return_value=(
                    context["joints"],
                    context["parents"],
                    context["joint_names"],
                ),
            ),
            patch(
                "interactive.blender.core.capture_armature_bone_states",
                return_value={},
            ),
        ):
            response = core.sync_history_snapshot(session.blender_session_id)

        self.assertTrue(response["ok"])
        self.assertFalse(response["synced"])
        core.session_request.assert_not_called()

    def test_rig_applies_new_bones_and_selects_the_last_one(self) -> None:
        partial = {
            "joints": [[0.0, 0.0, 0.0]],
            "parents": [-1],
            "joint_names": ["root"],
            "done": False,
        }
        complete = {
            "joints": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
            "done": True,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=partial,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(return_value={"ok": True, "context": complete})

        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="root",
            ),
        ):
            response = core.rig(session.blender_session_id, top_k=5)

        self.assertTrue(response["ok"])
        self.assertEqual(core.session_request.call_args.args[1]["command"], "rig")
        applied.assert_called_once_with(
            session,
            complete,
            select_name="child",
            remove_missing=True,
            mode_after="EDIT",
        )

    def test_skin_applies_full_resolution_weights_without_output_file(self) -> None:
        context = {
            "joints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
            "done": False,
        }
        completed = {**context, "done": True}
        skin = np.asarray(
            [[0.8, 0.2], [0.25, 0.75], [1.0, 0.0]],
            dtype=np.float32,
        )
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(
            return_value={
                "ok": True,
                "context": completed,
                "skin_shape": list(skin.shape),
                "asset_skin_shape": list(skin.shape),
                "skin": encode_float32_array(skin),
                "midprocess": "dfs-ensemble",
            }
        )

        with (
            patch.object(core, "sync_context_from_armature"),
            patch("interactive.blender.core.apply_mesh_skin_weights") as applied,
        ):
            response = core.skin(session.blender_session_id, top_k=5)

        self.assertTrue(response["ok"])
        self.assertTrue(session.skin_generated)
        payload = core.session_request.call_args.args[1]
        self.assertNotIn("output_path", payload)
        applied.assert_called_once()
        self.assertEqual(applied.call_args.args[:2], ("Mesh", ["root", "child"]))
        np.testing.assert_array_equal(applied.call_args.args[2], skin)

    def test_force_next_uses_active_armature_bone_as_parent(self) -> None:
        context = {
            "joints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
            "done": False,
        }
        generated = {
            "joints": context["joints"] + [[0.0, 1.0, 0.0]],
            "parents": [-1, 0, 0],
            "joint_names": ["root", "child", "branch"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(return_value={"ok": True, "context": generated})

        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="POSE"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="root",
            ),
        ):
            response = core.next(
                session.blender_session_id,
                force_parent=True,
                top_k=5,
            )

        self.assertTrue(response["ok"])
        payload = core.session_request.call_args.args[1]
        self.assertEqual(payload["branch_parent"], 0)
        applied.assert_called_once_with(
            session,
            generated,
            select_name="branch",
            remove_missing=True,
            mode_after="POSE",
        )

    def test_next_allows_model_to_choose_a_different_branch(self) -> None:
        context = {
            "joints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
            "done": False,
        }
        generated = {
            "joints": context["joints"] + [[0.0, 1.0, 0.0]],
            "parents": [-1, 0, 0],
            "joint_names": ["root", "child", "sibling"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(return_value={"ok": True, "context": generated})

        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="child",
            ),
        ):
            response = core.next(session.blender_session_id)

        self.assertTrue(response["ok"])
        payload = core.session_request.call_args.args[1]
        self.assertNotIn("branch_parent", payload)
        applied.assert_called_once_with(
            session,
            generated,
            select_name="sibling",
            remove_missing=True,
            mode_after="EDIT",
        )

    def test_force_next_focuses_dfs_without_rebuilding_existing_bones(self) -> None:
        context = {
            "joints": [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0, 0],
            "joint_names": ["root", "left", "right"],
            "done": True,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        focused, focused_parent = core.focus_context_on_joint(context, 1)
        generated = {
            **focused,
            "joints": focused["joints"] + [[-2.0, 0.0, 0.0]],
            "parents": focused["parents"] + [focused_parent],
            "joint_names": focused["joint_names"] + ["left_child"],
            "done": False,
        }
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(return_value={"ok": True, "context": generated})
        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="left",
            ),
        ):
            response = core.next(session.blender_session_id, force_parent=True)

        self.assertTrue(response["ok"])
        payload = core.session_request.call_args.args[1]
        self.assertEqual(payload["joint_names"], ["root", "right", "left"])
        self.assertEqual(payload["branch_parent"], 2)
        applied.assert_called_once_with(
            session,
            generated,
            select_name="left_child",
            remove_missing=True,
            mode_after="EDIT",
        )

    def test_force_next_requires_active_parent_for_nonempty_armature(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context={
                "joints": [[0.0, 0.0, 0.0]],
                "parents": [-1],
                "joint_names": ["root"],
                "done": False,
            },
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock()
        with (
            patch.object(core, "sync_context_from_armature"),
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value=None,
            ),
        ):
            response = core.next(session.blender_session_id, force_parent=True)

        self.assertFalse(response["ok"])
        self.assertIn("active parent", response["error"])
        core.session_request.assert_not_called()

    def test_first_next_on_empty_armature_creates_and_selects_root(self) -> None:
        generated = {
            "joints": [[0.0, 0.0, 0.0]],
            "parents": [-1],
            "joint_names": ["root"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context={
                "joints": [],
                "parents": [],
                "joint_names": [],
                "done": False,
            },
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(return_value={"ok": True, "context": generated})
        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value=None,
            ),
        ):
            response = core.next(session.blender_session_id)

        self.assertTrue(response["ok"])
        self.assertNotIn("branch_parent", core.session_request.call_args.args[1])
        applied.assert_called_once_with(
            session,
            generated,
            select_name="root",
            remove_missing=True,
            mode_after="EDIT",
        )

    def test_split_uses_and_keeps_active_bone_selected(self) -> None:
        context = {
            "joints": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "child"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session

        def echo(_session, payload):
            return {"ok": True, "context": payload}

        core.session_request = Mock(side_effect=echo)
        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="root",
            ),
        ):
            response = core.split_selected_bone(session.blender_session_id)

        self.assertTrue(response["ok"])
        self.assertEqual(response["context"]["joint_names"], ["root", "root_split", "child"])
        self.assertEqual(response["context"]["parents"], [-1, 0, 1])
        self.assertEqual(response["context"]["joints"][1], [1.0, 0.0, 0.0])
        self.assertEqual(response["split_child_name"], "child")
        applied.assert_called_once_with(
            session,
            response["context"],
            select_name="root",
            remove_missing=True,
            mode_after="EDIT",
            allow_reparent_names={"child"},
        )

    def test_split_reparents_all_direct_children_to_midpoint(self) -> None:
        context = {
            "joints": [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            "parents": [-1, 0, 1, 0],
            "joint_names": ["root", "branch", "leaf", "sibling"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(
            side_effect=lambda _session, payload: {"ok": True, "context": payload}
        )
        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature"),
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="root",
            ),
        ):
            response = core.split_selected_bone(session.blender_session_id)

        self.assertTrue(response["ok"])
        self.assertEqual(
            response["context"]["joint_names"],
            ["root", "root_split", "branch", "leaf", "sibling"],
        )
        self.assertEqual(response["context"]["parents"], [-1, 0, 1, 2, 1])

    def test_delete_dissolves_active_bone_downstream_segment(self) -> None:
        context = {
            "joints": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "parents": [-1, 0, 1, 0],
            "joint_names": ["root", "branch", "leaf", "sibling"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session

        def echo(_session, payload):
            return {"ok": True, "context": payload}

        core.session_request = Mock(side_effect=echo)
        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="root",
            ),
        ):
            response = core.delete_selected_bone(session.blender_session_id)

        self.assertTrue(response["ok"])
        self.assertEqual(response["deleted_bone_name"], "branch")
        self.assertEqual(response["delete_operation"], "dissolve")
        self.assertEqual(
            response["context"]["joint_names"],
            ["root", "leaf", "sibling"],
        )
        self.assertEqual(response["context"]["parents"], [-1, 0, 0])
        applied.assert_called_once_with(
            session,
            response["context"],
            select_name="root",
            remove_missing=True,
            mode_after="EDIT",
            allow_remove_names={"branch"},
            allow_reparent_names={"leaf"},
        )

    def test_delete_removes_the_active_leaf_itself(self) -> None:
        context = {
            "joints": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "leaf"],
            "done": False,
        }
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
            armature_object_name="Armature",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock(
            side_effect=lambda _session, payload: {"ok": True, "context": payload}
        )
        with (
            patch.object(core, "sync_context_from_armature"),
            patch.object(core, "_result_mode", return_value="EDIT"),
            patch.object(core, "_apply_context_to_armature") as applied,
            patch(
                "interactive.blender.core.active_armature_bone_name",
                return_value="leaf",
            ),
        ):
            response = core.delete_selected_bone(session.blender_session_id)

        self.assertTrue(response["ok"])
        self.assertEqual(response["deleted_bone_name"], "leaf")
        self.assertEqual(response["delete_operation"], "delete")
        self.assertEqual(response["context"]["joint_names"], ["root"])
        self.assertEqual(response["context"]["parents"], [-1])
        applied.assert_called_once_with(
            session,
            response["context"],
            select_name="root",
            remove_missing=True,
            mode_after="EDIT",
            allow_remove_names={"leaf"},
            allow_reparent_names=set(),
        )

    def test_start_automatically_uses_scene_armature_as_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            obj_path = Path(temporary) / "mesh.obj"
            obj_path.write_text("v 0 0 0\n", encoding="utf-8")
            core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
            initial_context = {
                "joints": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                "parents": [-1, 0],
                "joint_names": ["root", "child"],
                "done": False,
            }
            core.model_request = Mock(side_effect=[
                {
                    "ok": True,
                    "session_id": "model-session",
                    "context": {
                        "joints": [],
                        "parents": [],
                        "joint_names": [],
                        "done": False,
                    },
                },
                {"ok": True, "context": initial_context},
            ])
            mesh_obj = object()
            armature_obj = SimpleNamespace(name="ArtistArmature", mode="OBJECT")

            with (
                patch.object(
                    core,
                    "server_status",
                    return_value={"ok": True, "protocol_version": 1},
                ),
                patch(
                    "interactive.blender.core.scene_armature_context",
                    return_value=(mesh_obj, armature_obj, initial_context),
                ),
                patch(
                    "interactive.blender.core.ensure_mesh_armature",
                    return_value=armature_obj,
                ) as ensured,
                patch.object(core, "_apply_context_to_armature") as applied,
                patch(
                    "interactive.blender.core.capture_armature_bone_states",
                    return_value={},
                ),
                patch(
                    "interactive.blender.core.active_armature_bone_name",
                    return_value=None,
                ),
            ):
                result = core.start(
                    obj_path,
                    import_mesh=False,
                    mesh_object_name="Mesh",
                )

            sync_payload = core.model_request.call_args_list[1].args[0]
            self.assertEqual(sync_payload["command"], "sync")
            self.assertEqual(sync_payload["joints"], initial_context["joints"])
            self.assertEqual(sync_payload["parents"], initial_context["parents"])
            self.assertEqual(result["context_source"], "scene-armature")
            self.assertEqual(result["context"], initial_context)
            self.assertEqual(result["armature_object_name"], "ArtistArmature")
            ensured.assert_called_once_with("Mesh", "ArtistArmature")
            applied.assert_called_once_with(
                core.sessions[result["blender_session_id"]],
                initial_context,
                select_name=None,
                mode_after="OBJECT",
            )

    def test_start_creates_empty_working_armature_for_unrigged_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            obj_path = Path(temporary) / "mesh.obj"
            obj_path.write_text("v 0 0 0\n", encoding="utf-8")
            core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
            core.model_request = Mock(
                return_value={
                    "ok": True,
                    "session_id": "model-session",
                    "context": {
                        "joints": [],
                        "parents": [],
                        "joint_names": [],
                        "done": False,
                    },
                }
            )
            armature_obj = SimpleNamespace(name="EmptyArmature", mode="OBJECT")

            with (
                patch.object(
                    core,
                    "server_status",
                    return_value={"ok": True, "protocol_version": 1},
                ),
                patch(
                    "interactive.blender.core.scene_armature_context",
                    return_value=None,
                ),
                patch(
                    "interactive.blender.core.ensure_mesh_armature",
                    return_value=armature_obj,
                ) as ensured,
                patch.object(core, "_apply_context_to_armature") as applied,
                patch(
                    "interactive.blender.core.capture_armature_bone_states",
                    return_value={},
                ),
            ):
                result = core.start(
                    obj_path,
                    import_mesh=False,
                    mesh_object_name="Mesh",
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["context_source"], "empty-armature")
            self.assertEqual(result["armature_object_name"], "EmptyArmature")
            ensured.assert_called_once_with("Mesh")
            self.assertEqual(core.model_request.call_count, 1)
            applied.assert_called_once_with(
                core.sessions[result["blender_session_id"]],
                result["context"],
                select_name=None,
                mode_after="OBJECT",
            )

    def test_selected_bone_reconstruction_updates_current_mesh_skin(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        context = {
            "joints": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "foot"],
            "done": True,
        }
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
        )
        core.sessions[session.blender_session_id] = session
        current_skin = np.asarray(
            [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]],
            dtype=np.float32,
        )
        reconstructed = np.asarray(
            [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]],
            dtype=np.float32,
        )
        core.session_request = Mock(
            return_value={
                "ok": True,
                "context": context,
                "bone_names": ["foot"],
                "skin": encode_float32_array(reconstructed),
                "vae_reconstruction": {"wall_sec": 0.5},
            }
        )

        with (
            patch.object(core, "sync_context_from_armature"),
            patch(
                "interactive.blender.core.selected_skin_bone_names",
                return_value=["foot"],
            ),
            patch(
                "interactive.blender.core.read_mesh_skin_weights",
                return_value=current_skin,
            ),
            patch("interactive.blender.core.apply_mesh_skin_weights") as applied,
        ):
            result = core.reconstruct_selected_skin(session.blender_session_id)

        self.assertTrue(result["ok"])
        self.assertEqual(result["bone_names"], ["foot"])
        request_payload = core.session_request.call_args.args[1]
        self.assertEqual(request_payload["command"], "reconstruct")
        self.assertEqual(request_payload["bone_names"], ["foot"])
        applied.assert_called_once()
        np.testing.assert_array_equal(applied.call_args.args[2], reconstructed)

    def test_selected_bone_reconstruction_rejects_missing_skin(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context={
                "joints": [[0.0, 0.0, 0.0]],
                "parents": [-1],
                "joint_names": ["root"],
                "done": False,
            },
            mesh_object_name="Mesh",
        )
        core.sessions[session.blender_session_id] = session
        core.session_request = Mock()

        with (
            patch.object(core, "sync_context_from_armature"),
            patch(
                "interactive.blender.core.selected_skin_bone_names",
                return_value=["root"],
            ),
            patch(
                "interactive.blender.core.read_mesh_skin_weights",
                return_value=np.zeros((3, 1), dtype=np.float32),
            ),
        ):
            result = core.reconstruct_selected_skin(session.blender_session_id)

        self.assertFalse(result["ok"])
        self.assertIn("no skin weights", result["error"])
        core.session_request.assert_not_called()

    def test_vae_slider_caches_all_levels_for_one_bone(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        context = {
            "joints": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            "parents": [-1, 0],
            "joint_names": ["root", "foot"],
            "done": True,
        }
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context=context,
            mesh_object_name="Mesh",
        )
        core.sessions[session.blender_session_id] = session
        base_skin = np.asarray(
            [[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]],
            dtype=np.float32,
        )
        trajectory = np.empty((4, 3, 1), dtype=np.float32)
        for level in range(4):
            trajectory[level, :, 0] = np.clip(
                base_skin[:, 1] - 0.04 * level,
                0.0,
                1.0,
            )
        core.session_request = Mock(
            return_value={
                "ok": True,
                "context": context,
                "bone_names": ["foot"],
                "skin_fields": encode_float32_array(trajectory),
                "skin_fields_shape": list(trajectory.shape),
                "vae_reconstruction": {"wall_sec": 0.6, "levels": 3},
            }
        )

        with (
            patch.object(core, "sync_context_from_armature"),
            patch(
                "interactive.blender.core.selected_skin_bone_names",
                return_value=["foot"],
            ),
            patch(
                "interactive.blender.core.read_mesh_skin_weights",
                return_value=base_skin,
            ) as read_skin,
            patch("interactive.blender.core.apply_mesh_skin_weights") as applied,
        ):
            first = core.set_selected_skin_reconstruction_level(
                session.blender_session_id,
                3,
            )
            first_applied = applied.call_args.args[2].copy()
            read_skin.return_value = first_applied
            second = core.set_selected_skin_reconstruction_level(
                session.blender_session_id,
                2,
            )
            second_applied = applied.call_args.args[2].copy()
            manually_edited = second_applied.copy()
            manually_edited[0] = [0.65, 0.35]
            read_skin.return_value = manually_edited
            committed = core.apply_vae_reconstruction_levels(
                session.blender_session_id,
            )
            level_after_apply = core.vae_reconstruction_level(
                session.blender_session_id,
                "foot",
            )
            read_skin.return_value = manually_edited
            after_edit = core.set_selected_skin_reconstruction_level(
                session.blender_session_id,
                3,
            )

        self.assertTrue(first["ok"])
        self.assertTrue(first["generated"])
        self.assertFalse(second["generated"])
        expected_second = base_skin.copy()
        expected_second[:, 1] = trajectory[2, :, 0]
        np.testing.assert_allclose(
            second_applied,
            normalized_skin_weights(expected_second),
        )
        self.assertTrue(committed["ok"])
        self.assertEqual(committed["level"], 0)
        self.assertEqual(committed["applied_levels"], {"foot": 2})
        self.assertEqual(level_after_apply, 0)
        self.assertTrue(after_edit["generated"])
        self.assertEqual(core.session_request.call_count, 2)
        self.assertEqual(applied.call_count, 3)
        request_payload = core.session_request.call_args_list[0].args[1]
        self.assertEqual(request_payload["trajectory_levels"], 3)
        self.assertEqual(request_payload["bone_names"], ["foot"])
        self.assertEqual(len(session.vae_weight_fields["foot"]), 4)
        self.assertEqual(
            core.vae_reconstruction_level(session.blender_session_id, "foot"),
            3,
        )
        rebuilt_payload = core.session_request.call_args_list[1].args[1]
        np.testing.assert_allclose(
            decode_float32_array(rebuilt_payload["skin"], manually_edited.shape),
            manually_edited,
        )
        expected = manually_edited.copy()
        expected[:, 1] = trajectory[3, :, 0]
        expected = normalized_skin_weights(expected)
        np.testing.assert_allclose(applied.call_args.args[2], expected)
        np.testing.assert_allclose(applied.call_args.args[2].sum(axis=1), 1.0)

    def test_expired_remote_session_is_recreated_and_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            obj_path = Path(temporary) / "mesh.obj"
            obj_path.write_text("v 0 0 0\n", encoding="utf-8")
            core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
            core.model_request = Mock(side_effect=[
                {"ok": False, "code": "SESSION_EXPIRED"},
                {"ok": True, "session_id": "new-session", "context": {}},
                {"ok": True, "context": {"joints": [], "parents": []}},
            ])
            session = BlenderInteractiveSession(
                blender_session_id="blender-session",
                model_session_id="old-session",
                obj_path=obj_path,
                context={"joints": [], "parents": [], "joint_names": [], "done": False},
            )

            result = core.session_request(
                session,
                {"command": "next", **session.context},
            )

            self.assertTrue(result["ok"])
            self.assertEqual(session.model_session_id, "new-session")
            retry_payload = core.model_request.call_args_list[2].args[0]
            self.assertEqual(retry_payload["session_id"], "new-session")
            self.assertEqual(retry_payload["joints"], [])

    def test_reset_releases_local_session_and_remote_model_session(self) -> None:
        core = BlenderInteractiveCore("http://model.example", owner_id="client-a")
        session = BlenderInteractiveSession(
            blender_session_id="blender-session",
            model_session_id="model-session",
            obj_path=Path("mesh.obj"),
            context={"joints": [], "parents": [], "joint_names": [], "done": False},
        )
        session.vae_levels["foot"] = 3
        core.sessions[session.blender_session_id] = session
        core.model_request = Mock(return_value={"ok": True})

        result = core.reset(session.blender_session_id)

        self.assertTrue(result["ok"])
        self.assertNotIn(session.blender_session_id, core.sessions)
        self.assertEqual(session.vae_levels, {})
        core.model_request.assert_called_once_with(
            {
                "command": "reset",
                "session_id": "model-session",
                "end_reason": "reset",
            }
        )

if __name__ == "__main__":
    unittest.main()
