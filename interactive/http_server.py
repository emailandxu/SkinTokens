from __future__ import annotations

import argparse
import threading
import weakref
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from bottle import BaseRequest, Bottle, request, response

from .asset_store import MeshAssetStore
from .protocol import RUNTIME_DIR, Request, Response, err
from .server import (
    DEFAULT_MAX_CONTEXT_BONES,
    DEFAULT_MAX_SESSIONS,
    DEFAULT_MODEL_CKPT,
    DEFAULT_SESSION_CLEANUP_INTERVAL_SECONDS,
    DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    DEFAULT_USAGE_DIR,
    SKIN_POSTPROCESS_MODES,
    SKIN_POSTPROCESS_NONE,
    InteractiveModelServer,
    InteractiveServiceError,
    exception_response,
)
from .server_config import DEFAULT_SERVER_CONFIG_PATH, parse_args_with_config


DEFAULT_ASSET_DIR = RUNTIME_DIR / "interactive_assets"
DEFAULT_RESULT_DIR = RUNTIME_DIR / "interactive_results"


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class InteractiveHttpApi:
    def __init__(
        self,
        service: InteractiveModelServer,
        asset_store: MeshAssetStore,
        result_dir: str | Path,
        max_upload_bytes: int,
        max_pending_gpu_requests: int,
    ) -> None:
        self.service = service
        self.asset_store = asset_store
        self.result_dir = Path(result_dir).expanduser().resolve()
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = max(1, int(max_upload_bytes))
        BaseRequest.MEMFILE_MAX = max(
            int(BaseRequest.MEMFILE_MAX),
            self.max_upload_bytes,
        )
        self.gpu_slots = threading.BoundedSemaphore(max(1, int(max_pending_gpu_requests)))
        self.gpu_lock = threading.Lock()
        self.session_locks: weakref.WeakValueDictionary[str, threading.Lock] = (
            weakref.WeakValueDictionary()
        )
        self.session_locks_guard = threading.Lock()
        self.app = Bottle()
        self._register_routes()

    def _owner_id(self) -> str:
        return request.headers.get("X-SkinTokens-Owner", "anonymous")

    @staticmethod
    def _client_ip() -> str:
        return str(request.environ.get("REMOTE_ADDR") or "unknown")[:64]

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self.session_locks_guard:
            return self.session_locks.setdefault(session_id, threading.Lock())

    @staticmethod
    def _set_status(payload: Response) -> Response:
        if payload.get("ok"):
            response.status = 200
            return payload
        code = payload.get("code")
        response.status = {
            "BUSY": 429,
            "FORBIDDEN": 403,
            "SESSION_EXPIRED": 410,
            "ASSET_EXPIRED": 410,
            "ARMATURE_TOO_LARGE": 422,
            "UPLOAD_TOO_LARGE": 413,
        }.get(code, 400)
        return payload

    def _call(self, payload: Request, *, gpu: bool = False) -> Response:
        try:
            if not gpu:
                return self._set_status(self.service.handle(payload))
            if not self.gpu_slots.acquire(blocking=False):
                return self._set_status(err("GPU request queue is full", code="BUSY"))
            try:
                with self.gpu_lock:
                    return self._set_status(self.service.handle(payload))
            finally:
                self.gpu_slots.release()
        except Exception as exc:
            return self._set_status(exception_response(exc))

    def _json_payload(self, session_id: str, command: str) -> Request:
        length = request.content_length
        if length is not None and length > self.max_upload_bytes:
            raise InteractiveServiceError(
                "UPLOAD_TOO_LARGE",
                "session JSON request exceeds "
                f"{self.max_upload_bytes / (1024 * 1024):.1f} MB",
            )
        body = request.json
        if body is None:
            body = {}
        if not isinstance(body, dict):
            raise ValueError("request JSON must be an object")
        return {
            **body,
            "command": command,
            "session_id": session_id,
            "owner_id": self._owner_id(),
        }

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        def health() -> Response:
            return self._set_status(self.service.status())

        @app.post("/v1/sessions")
        def start() -> Response:
            length = request.content_length
            if length is not None and length > self.max_upload_bytes:
                response.status = 413
                return err("uploaded OBJ exceeds size limit", code="UPLOAD_TOO_LARGE")
            content = request.body.read(self.max_upload_bytes + 1)
            if len(content) > self.max_upload_bytes:
                response.status = 413
                return err("uploaded OBJ exceeds size limit", code="UPLOAD_TOO_LARGE")
            try:
                asset = self.asset_store.save_obj(content)
            except Exception as exc:
                return self._set_status(exception_response(exc))
            try:
                initial_bone_count = max(
                    0,
                    int(request.headers.get("X-SkinTokens-Initial-Bones", "0")),
                )
            except (TypeError, ValueError):
                initial_bone_count = 0
            return self._call(
                {
                    "command": "start",
                    "obj_path": str(asset.path),
                    "asset_id": asset.asset_id,
                    "owner_id": self._owner_id(),
                    "client_ip": self._client_ip(),
                    "initial_bone_count": initial_bone_count,
                },
                gpu=True,
            )

        def session_command(session_id: str, command: str) -> Response:
            try:
                payload = self._json_payload(session_id, command)
            except Exception as exc:
                return self._set_status(exception_response(exc))
            with self._session_lock(session_id):
                return self._call(
                    payload,
                    gpu=command in {"next", "branch", "rig", "reconstruct"},
                )

        app.post(
            "/v1/sessions/<session_id>/next",
            callback=lambda session_id: session_command(session_id, "next"),
        )
        app.post(
            "/v1/sessions/<session_id>/branch",
            callback=lambda session_id: session_command(session_id, "branch"),
        )
        app.post(
            "/v1/sessions/<session_id>/rig",
            callback=lambda session_id: session_command(session_id, "rig"),
        )
        app.post(
            "/v1/sessions/<session_id>/sync",
            callback=lambda session_id: session_command(session_id, "sync"),
        )
        app.post(
            "/v1/sessions/<session_id>/reconstruct",
            callback=lambda session_id: session_command(session_id, "reconstruct"),
        )

        @app.post("/v1/sessions/<session_id>/skin")
        def skin(session_id: str) -> Response:
            try:
                payload = self._json_payload(session_id, "skin")
            except Exception as exc:
                return self._set_status(exception_response(exc))
            with self._session_lock(session_id):
                return self._call(payload, gpu=True)

        @app.delete("/v1/sessions/<session_id>/reset")
        def reset(session_id: str) -> Response:
            try:
                payload = self._json_payload(session_id, "reset")
            except Exception as exc:
                return self._set_status(exception_response(exc))
            with self._session_lock(session_id):
                return self._call(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SkinTokens HTTP model server.")
    parser.add_argument("--config", default=str(DEFAULT_SERVER_CONFIG_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--asset-dir", default=str(DEFAULT_ASSET_DIR))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--max-runtime-sessions", type=int, default=8)
    parser.add_argument("--max-sessions", type=int, default=DEFAULT_MAX_SESSIONS)
    parser.add_argument(
        "--session-idle-timeout-seconds",
        type=float,
        default=DEFAULT_SESSION_IDLE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--session-cleanup-interval-seconds",
        type=float,
        default=DEFAULT_SESSION_CLEANUP_INTERVAL_SECONDS,
    )
    parser.add_argument("--max-context-bones", type=int, default=DEFAULT_MAX_CONTEXT_BONES)
    parser.add_argument("--usage-dir", default=str(DEFAULT_USAGE_DIR))
    parser.add_argument(
        "--skin-postprocess",
        choices=SKIN_POSTPROCESS_MODES,
        default=SKIN_POSTPROCESS_NONE,
    )
    parser.add_argument("--max-pending-gpu-requests", type=int, default=8)
    parser.add_argument("--max-upload-mb", type=int, default=100)
    return parser


def serve(args: argparse.Namespace) -> None:
    service = InteractiveModelServer(
        model_ckpt=args.model_ckpt,
        hf_path=args.hf_path,
        device=args.device,
        max_runtime_sessions=args.max_runtime_sessions,
        max_sessions=args.max_sessions,
        max_context_bones=args.max_context_bones,
        default_skin_postprocess=args.skin_postprocess,
        usage_dir=args.usage_dir,
        session_idle_timeout_seconds=args.session_idle_timeout_seconds,
        session_cleanup_interval_seconds=args.session_cleanup_interval_seconds,
    )
    api = InteractiveHttpApi(
        service=service,
        asset_store=MeshAssetStore(args.asset_dir),
        result_dir=args.result_dir,
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
        max_pending_gpu_requests=args.max_pending_gpu_requests,
    )
    print(f"[interactive] HTTP model server listening on http://{args.host}:{args.port}")
    try:
        with make_server(
            args.host,
            args.port,
            api.app,
            server_class=ThreadingWSGIServer,
            handler_class=WSGIRequestHandler,
        ) as server:
            server.serve_forever()
    finally:
        service.close()


def main() -> None:
    serve(parse_args_with_config(build_parser()))


if __name__ == "__main__":
    main()
