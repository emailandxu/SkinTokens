from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from interactive.server import InteractiveModelServer
from interactive.session import SessionRecord
from interactive.usage_events import UsageEventLog, load_usage_events
from interactive.usage_report import build_usage_report, main as usage_report_main


class UsageEventLogTest(unittest.TestCase):
    def test_writer_and_report_cover_completed_and_incomplete_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "usage"
            events = UsageEventLog(root)
            events.append(
                "session_start",
                session_id="complete",
                client_ip="10.0.0.1",
                initial_bone_count=2,
            )
            events.append(
                "session_end",
                session_id="complete",
                client_ip="10.0.0.1",
                end_reason="finish",
                duration_seconds=4.0,
                final_has_skin=True,
                skin_generation_count=2,
                net_bone_change=3,
                final_added_bones=3,
                final_removed_bones=0,
                peak_added_bones=4,
            )
            events.append(
                "session_start",
                session_id="incomplete",
                client_ip="10.0.0.2",
                initial_bone_count=0,
            )
            (root / "1900-01-01.jsonl").write_text(
                "{malformed\n",
                encoding="utf-8",
            )

            loaded = load_usage_events(root)
            report = build_usage_report(loaded)

            # The deliberately malformed file is ignored without affecting valid days.
            self.assertEqual(len(loaded), 3)
            self.assertEqual(report["sessions_started"], 2)
            self.assertEqual(report["sessions_ended"], 1)
            self.assertEqual(report["sessions_incomplete"], 1)
            self.assertEqual(report["unique_client_ips"], 2)
            self.assertEqual(report["sessions_finished"], 1)
            self.assertEqual(report["skinned_assets"], 1)
            self.assertEqual(report["skin_generation_count"], 2)
            self.assertEqual(report["net_bone_change"], 3)
            self.assertEqual(report["final_added_bones"], 3)
            self.assertEqual(report["final_removed_bones"], 0)
            self.assertEqual(report["peak_added_bones"], 4)
            self.assertEqual(report["duration_seconds"]["mean"], 4.0)

    def test_concurrent_writes_remain_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = UsageEventLog(temporary)
            with ThreadPoolExecutor(max_workers=8) as executor:
                paths = list(executor.map(
                    lambda index: events.append(
                        "session_start",
                        session_id=f"session-{index}",
                    ),
                    range(200),
                ))

            loaded = load_usage_events(temporary)
            self.assertTrue(all(path is not None for path in paths))
            self.assertEqual(len(loaded), 200)
            self.assertEqual(len({event["event_id"] for event in loaded}), 200)

    def test_report_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "usage"
            output = Path(temporary) / "report.json"
            UsageEventLog(root).append("session_start", session_id="active")

            usage_report_main([
                "--usage-dir",
                str(root),
                "--output",
                str(output),
            ])

            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["sessions_started"], 1)
            self.assertEqual(report["sessions_incomplete"], 1)


class UsageLifecycleTest(unittest.TestCase):
    def make_service(self, usage_root: Path) -> InteractiveModelServer:
        service = InteractiveModelServer.__new__(InteractiveModelServer)
        service.sessions = OrderedDict()
        service.runtime_cache = OrderedDict()
        service.max_sessions = 512
        service.max_runtime_sessions = 8
        service.session_idle_timeout_seconds = 3600.0
        service.usage_events = UsageEventLog(usage_root)
        service.model_version = "test.ckpt"
        service.device = "cpu"
        return service

    @staticmethod
    def make_record(session_id: str, *, bones: int = 0) -> SessionRecord:
        return SessionRecord(
            session_id=session_id,
            owner_id="anonymous-client",
            client_ip="192.0.2.10",
            asset_id="asset",
            obj_path=Path("not-logged.obj"),
            created_at=time.time() - 1.0,
            updated_at=time.time(),
            initial_bone_count=bones,
            latest_bone_count=bones,
            max_bone_count=bones,
            vertex_count=123,
        )

    def test_finish_records_counts_once_without_asset_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = self.make_service(root)
            record = self.make_record("session", bones=2)
            service.sessions[record.session_id] = record
            service._log_session_start(record)
            record.observe_bones(5)
            record.observe_skin()
            record.observe_skin()

            response = service.reset({
                "session_id": record.session_id,
                "owner_id": record.owner_id,
                "end_reason": "finish",
            })
            service.close()

            self.assertTrue(response["ok"])
            events = load_usage_events(root)
            self.assertEqual([event["event"] for event in events], [
                "session_start",
                "session_end",
            ])
            ended = events[1]
            self.assertEqual(events[0]["client_ip"], "192.0.2.10")
            self.assertEqual(ended["client_ip"], "192.0.2.10")
            self.assertEqual(ended["end_reason"], "finish")
            self.assertEqual(ended["initial_bone_count"], 2)
            self.assertEqual(ended["final_bone_count"], 5)
            self.assertEqual(ended["net_bone_change"], 3)
            self.assertEqual(ended["final_added_bones"], 3)
            self.assertEqual(ended["final_removed_bones"], 0)
            self.assertEqual(ended["skin_generation_count"], 2)
            self.assertTrue(ended["final_has_skin"])
            serialized = json.dumps(events)
            self.assertNotIn("not-logged.obj", serialized)
            self.assertNotIn("obj_path", serialized)
            self.assertNotIn("joint_names", serialized)

    def test_runtime_lru_does_not_end_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            service.max_runtime_sessions = 1
            record = self.make_record("session")
            service.sessions[record.session_id] = record
            service._log_session_start(record)

            service._put_runtime(SimpleNamespace(session_id="session"))
            service._put_runtime(SimpleNamespace(session_id="other"))

            events = load_usage_events(temporary)
            self.assertEqual([event["event"] for event in events], ["session_start"])
            self.assertFalse(record.end_logged)

    def test_session_lru_and_shutdown_have_distinct_end_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            service.max_sessions = 1
            old = self.make_record("old")
            current = self.make_record("current")
            for record in (old, current):
                service.sessions[record.session_id] = record
                service._log_session_start(record)

            service._enforce_session_limit()
            service.close()
            service.close()

            ends = [
                event
                for event in load_usage_events(temporary)
                if event["event"] == "session_end"
            ]
            self.assertEqual(len(ends), 2)
            self.assertEqual(
                {event["session_id"]: event["end_reason"] for event in ends},
                {"old": "lru_evicted", "current": "server_shutdown"},
            )

    def test_idle_timeout_removes_session_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            now = time.time()
            expired = self.make_record("expired")
            expired.updated_at = now - 3600.1
            current = self.make_record("current")
            current.updated_at = now - 3599.0
            for record in (expired, current):
                service.sessions[record.session_id] = record
                service.runtime_cache[record.session_id] = SimpleNamespace(
                    session_id=record.session_id,
                )
                service._log_session_start(record)

            count = service.expire_idle_sessions(now=now)

            self.assertEqual(count, 1)
            self.assertEqual(list(service.sessions), ["current"])
            self.assertEqual(list(service.runtime_cache), ["current"])
            ends = [
                event
                for event in load_usage_events(temporary)
                if event["event"] == "session_end"
            ]
            self.assertEqual(len(ends), 1)
            self.assertEqual(ends[0]["session_id"], "expired")
            self.assertEqual(ends[0]["end_reason"], "idle_timeout")
            report = build_usage_report(load_usage_events(temporary))
            self.assertEqual(report["sessions_idle_timed_out"], 1)

    def test_session_access_refreshes_idle_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            record = self.make_record("session")
            record.updated_at = time.time() - 3599.0
            service.sessions[record.session_id] = record

            returned = service._record({
                "session_id": record.session_id,
                "owner_id": record.owner_id,
            })
            count = service.expire_idle_sessions(now=returned.updated_at + 3599.0)

            self.assertIs(returned, record)
            self.assertEqual(count, 0)
            self.assertIn(record.session_id, service.sessions)

    def test_background_cleanup_expires_idle_session(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = self.make_service(Path(temporary))
            service.session_idle_timeout_seconds = 0.01
            service.session_cleanup_interval_seconds = 0.01
            service._maintenance_stop = threading.Event()
            record = self.make_record("session")
            record.updated_at = time.time() - 1.0
            service.sessions[record.session_id] = record
            service._log_session_start(record)
            service._maintenance_thread = threading.Thread(
                target=service._maintenance_loop,
                daemon=True,
            )
            service._maintenance_thread.start()

            deadline = time.monotonic() + 1.0
            while record.session_id in service.sessions and time.monotonic() < deadline:
                time.sleep(0.01)
            service.close()

            self.assertNotIn(record.session_id, service.sessions)
            ends = [
                event
                for event in load_usage_events(temporary)
                if event["event"] == "session_end"
            ]
            self.assertEqual(len(ends), 1)
            self.assertEqual(ends[0]["end_reason"], "idle_timeout")


if __name__ == "__main__":
    unittest.main()
