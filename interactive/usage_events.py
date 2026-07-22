from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USAGE_SCHEMA_VERSION = 1
USAGE_LOG_FILENAME = "events.jsonl"


class UsageEventLog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._lock = threading.Lock()

    def append(self, event: str, **fields: Any) -> Path | None:
        now = datetime.now(timezone.utc)
        record = {
            "schema_version": USAGE_SCHEMA_VERSION,
            "event_id": uuid.uuid4().hex,
            "event": str(event),
            "time_utc": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            **fields,
        }
        data = (
            json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        path = self.root / USAGE_LOG_FILENAME
        try:
            with self._lock:
                self.root.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    path,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                    0o644,
                )
                try:
                    remaining = memoryview(data)
                    while remaining:
                        written = os.write(descriptor, remaining)
                        if written <= 0:
                            raise OSError("usage event write made no progress")
                        remaining = remaining[written:]
                finally:
                    os.close(descriptor)
            return path
        except OSError:
            return None


def load_usage_events(
    root: str | Path,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(Path(root).expanduser().resolve().glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                day = str(event.get("time_utc", ""))[:10]
                if date_from is not None and day < date_from:
                    continue
                if date_to is not None and day > date_to:
                    continue
                events.append(event)
    return events
