#!/usr/bin/env python3
"""Unix-socket inference server for keeping SkinTokens warm on CPU."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any, Optional

from .args import (
    DEFAULT_MODEL_CKPT,
    build_parser,
    normalize_optional_path,
)
from .cli import AUTHKEY, DEFAULT_INFO_PATH, DEFAULT_RUNTIME_DIR, DEFAULT_SOCKET_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_gpu_ids(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or raw.strip() == "":
        return None
    ids = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    return ids


def gpu_free_bytes(device_id: int) -> int:
    import torch

    with torch.cuda.device(device_id):
        free, _total = torch.cuda.mem_get_info()
    return int(free)


def select_device(gpu_ids: Optional[list[int]], min_free_gb: float) -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"

    count = torch.cuda.device_count()
    candidates = gpu_ids if gpu_ids is not None else list(range(count))
    valid = [idx for idx in candidates if 0 <= idx < count]
    if not valid:
        raise RuntimeError(f"no valid CUDA devices from {candidates}; visible count={count}")

    free_by_id = [(gpu_free_bytes(idx), idx) for idx in valid]
    free, best_id = max(free_by_id)
    min_free = int(float(min_free_gb) * 1024**3)
    if free < min_free:
        details = ", ".join(
            f"cuda:{idx}={free_bytes / 1024**3:.2f}GiB"
            for free_bytes, idx in sorted(free_by_id, reverse=True)
        )
        raise RuntimeError(
            f"no CUDA device has at least {min_free_gb:.2f}GiB free ({details})"
        )
    return f"cuda:{best_id}"


def load_info(info_path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(info_path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_runtime(socket_path: Path, info_path: Path) -> None:
    info = load_info(info_path)
    if info is not None and pid_alive(int(info.get("pid", -1))):
        return
    with contextlib.suppress(FileNotFoundError):
        socket_path.unlink()
    with contextlib.suppress(FileNotFoundError):
        info_path.unlink()


def wait_until_ready(info_path: Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = load_info(info_path)
        if info and info.get("ready"):
            return True
        time.sleep(0.2)
    return False


def make_infer_args(payload: dict[str, Any]) -> argparse.Namespace:
    parser = build_parser(require_input=False)
    args = parser.parse_args([])
    for key, value in payload.items():
        setattr(args, key.replace("-", "_"), value)
    return args


class InferServer:
    def __init__(
        self,
        *,
        socket_path: Path,
        info_path: Path,
        model_ckpt: str,
        hf_path: Optional[str],
        gpu_ids: Optional[list[int]],
        min_free_gb: float,
    ) -> None:
        self.socket_path = socket_path
        self.info_path = info_path
        self.model_ckpt = normalize_optional_path(model_ckpt) or model_ckpt
        self.hf_path = normalize_optional_path(hf_path)
        self.gpu_ids = gpu_ids
        self.min_free_gb = min_free_gb
        self.model = None
        self.listener: Optional[Listener] = None
        self.should_stop = False

    def run(self) -> int:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        remove_stale_runtime(self.socket_path, self.info_path)
        if self.socket_path.exists():
            raise RuntimeError(f"socket already exists: {self.socket_path}")

        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)

        from .infer import get_model

        print(f"[infer.server] loading model on CPU: {self.model_ckpt}", flush=True)
        self.model = get_model(self.model_ckpt, hf_path=self.hf_path, device="cpu")
        self.model.eval()

        listener = Listener(str(self.socket_path), family="AF_UNIX", authkey=AUTHKEY)
        self.listener = listener
        self._write_info(ready=True)
        print(f"[infer.server] ready socket={self.socket_path}", flush=True)
        try:
            while not self.should_stop:
                try:
                    conn = self.listener.accept()
                except (OSError, socket.error):
                    if self.should_stop:
                        break
                    raise
                with conn:
                    self._handle_connection(conn)
        finally:
            listener.close()
            self.listener = None
            self._cleanup_runtime()
        return 0

    def _handle_stop(self, _signum, _frame) -> None:
        self.should_stop = True
        if self.listener is not None:
            with contextlib.suppress(Exception):
                self.listener.close()

    def _write_info(self, *, ready: bool) -> None:
        info = {
            "pid": os.getpid(),
            "socket": str(self.socket_path),
            "ready": ready,
            "model_ckpt": self.model_ckpt,
            "hf_path": self.hf_path,
            "gpu_ids": self.gpu_ids,
            "min_free_gb": self.min_free_gb,
        }
        self.info_path.write_text(json.dumps(info, indent=2) + "\n")

    def _cleanup_runtime(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.socket_path.unlink()
        with contextlib.suppress(FileNotFoundError):
            self.info_path.unlink()

    def _handle_connection(self, conn) -> None:
        try:
            request = conn.recv()
        except EOFError:
            return
        command = request.get("command")
        if command == "ping":
            conn.send({"ok": True, "pid": os.getpid(), "ready": True})
            return
        if command == "stop":
            self.should_stop = True
            conn.send({"ok": True, "stopping": True})
            return
        if command != "infer":
            conn.send({"ok": False, "error": f"unknown command: {command!r}"})
            return

        try:
            result_payload = self._run_infer(request.get("args", {}))
        except Exception as exc:  # Keep the server alive after a failed job.
            conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            conn.send({"ok": True, **result_payload})

    def _run_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("server model is not loaded")
        args = make_infer_args(payload)
        device = select_device(self.gpu_ids, self.min_free_gb)
        args.device = device
        print(f"[infer.server] job input={args.input} device={device}", flush=True)
        from .infer import infer_asset_with_model, write_infer_outputs

        self.model.to(device)
        try:
            result = infer_asset_with_model(args, self.model)
            write_infer_outputs(args, result)
        finally:
            self.model.to("cpu")
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"output": args.output, "device": device}


def send_request(socket_path: Path, request: dict[str, Any]) -> dict[str, Any]:
    with Client(str(socket_path), family="AF_UNIX", authkey=AUTHKEY) as conn:
        conn.send(request)
        return conn.recv()


def run_command(args: argparse.Namespace) -> int:
    server = InferServer(
        socket_path=Path(args.socket).expanduser().resolve(),
        info_path=Path(args.info).expanduser().resolve(),
        model_ckpt=args.model_ckpt,
        hf_path=args.hf_path,
        gpu_ids=parse_gpu_ids(args.gpus),
        min_free_gb=args.min_free_gb,
    )
    return server.run()


def start_command(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket).expanduser().resolve()
    info_path = Path(args.info).expanduser().resolve()
    info_path.parent.mkdir(parents=True, exist_ok=True)
    remove_stale_runtime(socket_path, info_path)
    existing = load_info(info_path)
    if existing and pid_alive(int(existing.get("pid", -1))):
        print(f"[infer.server] already running pid={existing['pid']}")
        return 0

    command = [
        sys.executable,
        "-m",
        "infer.server",
        "--socket",
        str(socket_path),
        "--info",
        str(info_path),
        "run",
        "--model-ckpt",
        args.model_ckpt,
        "--min-free-gb",
        str(args.min_free_gb),
    ]
    if args.hf_path is not None:
        command.extend(["--hf-path", args.hf_path])
    if args.gpus is not None:
        command.extend(["--gpus", args.gpus])

    log_path = Path(args.log).expanduser().resolve() if args.log else info_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    if not wait_until_ready(info_path, args.timeout):
        print(f"[infer.server] start timed out; log={log_path}", file=sys.stderr)
        return 1
    info = load_info(info_path) or {}
    print(f"[infer.server] started pid={info.get('pid')} socket={socket_path}")
    print(f"[infer.server] log={log_path}")
    return 0


def status_command(args: argparse.Namespace) -> int:
    info_path = Path(args.info).expanduser().resolve()
    info = load_info(info_path)
    if not info:
        print("[infer.server] not running")
        return 1
    alive = pid_alive(int(info.get("pid", -1)))
    print(json.dumps({**info, "alive": alive}, indent=2))
    return 0 if alive else 1


def stop_command(args: argparse.Namespace) -> int:
    socket_path = Path(args.socket).expanduser().resolve()
    info_path = Path(args.info).expanduser().resolve()
    info = load_info(info_path)
    if not info:
        print("[infer.server] not running")
        return 0
    try:
        response = send_request(socket_path, {"command": "stop"})
        print(json.dumps(response, indent=2))
    except Exception as exc:
        print(f"[infer.server] stop request failed: {exc}", file=sys.stderr)
        return 1
    return 0


def infer_command(args: argparse.Namespace) -> int:
    payload = vars(args).copy()
    socket_path = Path(payload.pop("socket")).expanduser().resolve()
    payload.pop("command_func", None)
    response = send_request(socket_path, {"command": "infer", "args": payload})
    print(json.dumps(response, indent=2))
    return 0 if response.get("ok") else 1


def build_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or control the SkinTokens infer server.")
    parser.set_defaults(command_func=None)
    parser.add_argument("--socket", default=str(DEFAULT_SOCKET_PATH))
    parser.add_argument("--info", default=str(DEFAULT_INFO_PATH))

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the server in the foreground.")
    add_server_options(run)
    run.set_defaults(command_func=run_command)

    start = subparsers.add_parser("start", help="Start the server in the background.")
    add_server_options(start)
    start.add_argument("--log", default=None)
    start.add_argument("--timeout", type=float, default=120.0)
    start.set_defaults(command_func=start_command)

    status = subparsers.add_parser("status", help="Print server status.")
    status.set_defaults(command_func=status_command)

    stop = subparsers.add_parser("stop", help="Stop the server.")
    stop.set_defaults(command_func=stop_command)

    infer = subparsers.add_parser("infer", help="Send one inference request to the server.")
    infer_parent = build_parser(require_input=True)
    for action in infer_parent._actions:
        if not action.option_strings or action.dest == "help":
            continue
        infer._add_action(action)
    infer.set_defaults(command_func=infer_command)

    return parser


def add_server_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    parser.add_argument("--hf-path", default=None)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated visible CUDA ids to consider, e.g. 0,1,3.",
    )
    parser.add_argument("--min-free-gb", type=float, default=6.0)


def main() -> None:
    parser = build_server_parser()
    args = parser.parse_args()
    raise SystemExit(args.command_func(args))


if __name__ == "__main__":
    main()
