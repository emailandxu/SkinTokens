from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

from .protocol import RUNTIME_DIR
from .usage_events import load_usage_events


DEFAULT_USAGE_DIR = RUNTIME_DIR / "usage"


def build_usage_report(events: list[dict]) -> dict:
    starts = [event for event in events if event.get("event") == "session_start"]
    ends = [event for event in events if event.get("event") == "session_end"]
    installs = [
        event for event in events if event.get("event") == "extension_install"
    ]
    updates = [
        event for event in events if event.get("event") == "extension_update"
    ]
    started_ids = {str(event.get("session_id", "")) for event in starts}
    ended_ids = {str(event.get("session_id", "")) for event in ends}
    client_ips = {
        str(event.get("client_ip", ""))
        for event in starts
        if str(event.get("client_ip", ""))
    }
    durations = [float(event.get("duration_seconds", 0.0)) for event in ends]
    reasons = Counter(str(event.get("end_reason", "unknown")) for event in ends)
    return {
        "sessions_started": len(starts),
        "sessions_ended": len(ends),
        "sessions_incomplete": len(started_ids - ended_ids),
        "sessions_finished": int(reasons.get("finish", 0)),
        "extension_installs": len(installs),
        "extension_updates": len(updates),
        "extension_installations": len({
            str(event.get("installation_id", ""))
            for event in [*installs, *updates]
            if str(event.get("installation_id", ""))
        }),
        "sessions_lru_evicted": int(reasons.get("lru_evicted", 0)),
        "sessions_idle_timed_out": int(reasons.get("idle_timeout", 0)),
        "unique_client_ips": len(client_ips),
        "end_reasons": dict(sorted(reasons.items())),
        "skinned_assets": sum(bool(event.get("final_has_skin", False)) for event in ends),
        "skin_generation_count": sum(
            int(event.get("skin_generation_count", 0)) for event in ends
        ),
        "net_bone_change": sum(int(event.get("net_bone_change", 0)) for event in ends),
        "final_added_bones": sum(
            int(event.get("final_added_bones", 0)) for event in ends
        ),
        "final_removed_bones": sum(
            int(event.get("final_removed_bones", 0)) for event in ends
        ),
        "peak_added_bones": sum(int(event.get("peak_added_bones", 0)) for event in ends),
        "duration_seconds": {
            "total": float(sum(durations)),
            "mean": float(statistics.fmean(durations)) if durations else 0.0,
            "median": float(statistics.median(durations)) if durations else 0.0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize SkinTokens usage events.")
    parser.add_argument("--usage-dir", default=str(DEFAULT_USAGE_DIR))
    parser.add_argument("--from", dest="date_from", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = build_usage_report(
        load_usage_events(
            args.usage_dir,
            date_from=args.date_from,
            date_to=args.date_to,
        )
    )
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(content)
        return
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{content}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
