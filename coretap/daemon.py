from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from coretap.runtime import CoretapError, ensure_state, response_error, response_ok, run_id


REQUEST_SCHEMA = "coretap.daemon.request.v1"
RESPONSE_SCHEMA = "coretap.response.v1"


def default_socket_path() -> Path:
    return ensure_state()["state"] / "coretapd.sock"


def default_pid_path() -> Path:
    return ensure_state()["state"] / "coretapd.pid"


def default_log_path() -> Path:
    return ensure_state()["state"] / "coretapd.log"


def request_daemon(
    argv: list[str],
    *,
    cwd: str | None = None,
    socket_path: str | Path | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    payload = {"schema": REQUEST_SCHEMA, "action": "argv", "argv": argv, "cwd": cwd or os.getcwd()}
    return _send_request(payload, socket_path=socket_path, timeout=timeout)


def ping_daemon(*, socket_path: str | Path | None = None, timeout: float = 2.0) -> dict[str, Any]:
    return _send_request({"schema": REQUEST_SCHEMA, "action": "ping"}, socket_path=socket_path, timeout=timeout)


def stop_daemon(*, socket_path: str | Path | None = None, timeout: float = 2.0) -> dict[str, Any]:
    return _send_request({"schema": REQUEST_SCHEMA, "action": "shutdown"}, socket_path=socket_path, timeout=timeout)


def _send_request(payload: dict[str, Any], *, socket_path: str | Path | None, timeout: float) -> dict[str, Any]:
    path = Path(socket_path) if socket_path else default_socket_path()
    started = time.monotonic()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(path))
            with client.makefile("rwb") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
                stream.flush()
                line = stream.readline()
    except OSError as exc:
        raise CoretapError(
            "DAEMON_UNAVAILABLE",
            f"coretapd is not reachable at {path}",
            stage="daemon",
            category="infrastructure",
            retryable=True,
            details={"socket": str(path), "error": str(exc)},
        ) from exc
    if not line:
        raise CoretapError(
            "DAEMON_REQUEST_FAILED",
            "coretapd closed the connection without a response",
            stage="daemon",
            category="infrastructure",
            retryable=True,
            details={"socket": str(path), "durationMs": round((time.monotonic() - started) * 1000)},
        )
    try:
        data = json.loads(line.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CoretapError(
            "DAEMON_REQUEST_FAILED",
            "coretapd returned invalid JSON",
            stage="daemon",
            category="infrastructure",
            details={"socket": str(path), "raw": line.decode("utf-8", errors="replace")},
        ) from exc
    return data


def start_daemon(*, socket_path: str | Path | None = None, timeout: float = 5.0) -> dict[str, Any]:
    path = Path(socket_path) if socket_path else default_socket_path()
    with suppress(CoretapError):
        status = ping_daemon(socket_path=path, timeout=0.5)
        return {"started": False, "alreadyRunning": True, "socket": str(path), "status": status.get("result")}

    log_path = default_log_path()
    pid_path = default_pid_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "coretap.daemon", "serve", "--socket", str(path)],
            cwd=os.getcwd(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            last_error = f"coretapd exited with code {proc.returncode}"
            break
        try:
            status = ping_daemon(socket_path=path, timeout=0.5)
            return {
                "started": True,
                "alreadyRunning": False,
                "pid": proc.pid,
                "socket": str(path),
                "pidFile": str(pid_path),
                "log": str(log_path),
                "status": status.get("result"),
            }
        except CoretapError as exc:
            last_error = str(exc)
            time.sleep(0.1)
    raise CoretapError(
        "DAEMON_START_FAILED",
        "coretapd did not become ready before the timeout",
        stage="daemon",
        category="infrastructure",
        retryable=True,
        details={"socket": str(path), "pid": proc.pid, "log": str(log_path), "lastError": last_error},
    )


def handle_argv(argv: list[str], *, cwd: str | None = None) -> dict[str, Any]:
    from coretap.cli import EXIT_CODES, build_parser, dispatch, normalize_global_args

    started = time.monotonic()
    command = "unknown"
    old_cwd = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        parser = build_parser()
        args = parser.parse_args(normalize_global_args(argv))
        command = args.command
        result = dispatch(args)
        data = response_ok(command, result)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        data["daemon"] = {"pid": os.getpid(), "socket": str(default_socket_path())}
        data["exitCode"] = 0
        return data
    except CoretapError as exc:
        data = response_error(command, exc)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        data["daemon"] = {"pid": os.getpid(), "socket": str(default_socket_path())}
        data["exitCode"] = EXIT_CODES.get(exc.code, 70)
        return data
    except SystemExit as exc:
        error = CoretapError(
            "INVALID_ARGUMENT",
            f"Daemon request parse failed with exit code {exc.code}",
            category="usage",
            stage="daemon-parse",
            details={"argv": argv, "exitCode": exc.code},
        )
        data = response_error(command, error)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        data["daemon"] = {"pid": os.getpid(), "socket": str(default_socket_path())}
        data["exitCode"] = 2
        return data
    except BaseException as exc:
        error = CoretapError(
            "DAEMON_REQUEST_FAILED",
            f"Daemon request failed: {exc}",
            stage="daemon",
            category="infrastructure",
            details={"argv": argv, "errorType": type(exc).__name__},
        )
        data = response_error(command, error)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        data["daemon"] = {"pid": os.getpid(), "socket": str(default_socket_path())}
        data["exitCode"] = 70
        return data
    finally:
        if cwd:
            os.chdir(old_cwd)


def serve(*, socket_path: str | Path | None = None) -> int:
    path = Path(socket_path) if socket_path else default_socket_path()
    pid_path = default_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            ping_daemon(socket_path=path, timeout=0.2)
            raise CoretapError(
                "DAEMON_ALREADY_RUNNING",
                f"coretapd is already running at {path}",
                stage="daemon",
                category="infrastructure",
                details={"socket": str(path)},
            )
        except CoretapError as exc:
            if exc.code != "DAEMON_UNAVAILABLE":
                raise
            path.unlink(missing_ok=True)

    shutting_down = False

    def _request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal shutting_down
        shutting_down = True

    old_term = signal.signal(signal.SIGTERM, _request_shutdown)
    old_int = signal.signal(signal.SIGINT, _request_shutdown)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            server.listen(16)
            server.settimeout(0.5)
            pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
            while not shutting_down:
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    if _handle_connection(conn):
                        shutting_down = True
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        path.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
    return 0


def _handle_connection(conn: socket.socket) -> bool:
    shutting_down = False
    with conn.makefile("rwb") as stream:
        line = stream.readline()
        try:
            request = json.loads(line.decode("utf-8"))
            action = request.get("action")
            if action == "ping":
                response = response_ok("daemon", {"running": True, "pid": os.getpid(), "socket": str(default_socket_path())})
            elif action == "shutdown":
                response = response_ok("daemon", {"stopping": True, "pid": os.getpid(), "socket": str(default_socket_path())})
                shutting_down = True
            elif action == "argv":
                response = handle_argv(list(request.get("argv") or []), cwd=request.get("cwd"))
            else:
                raise CoretapError(
                    "INVALID_ARGUMENT",
                    f"Unsupported daemon action: {action}",
                    category="usage",
                    stage="daemon",
                    details={"action": action},
                )
        except CoretapError as exc:
            response = response_error("daemon", exc)
            response["exitCode"] = 70
        except BaseException as exc:
            error = CoretapError(
                "DAEMON_REQUEST_FAILED",
                f"Daemon protocol failed: {exc}",
                stage="daemon",
                category="infrastructure",
                details={"errorType": type(exc).__name__},
            )
            response = response_error("daemon", error)
            response["exitCode"] = 70
        response["schema"] = RESPONSE_SCHEMA
        response["requestId"] = response.get("requestId") or run_id("req")
        stream.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        stream.flush()
    return shutting_down


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coretapd")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_cmd = sub.add_parser("serve")
    serve_cmd.add_argument("--socket", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve(socket_path=args.socket)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(0)
