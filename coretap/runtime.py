from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import string
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CoretapError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: str = "infrastructure",
        stage: str = "runtime",
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.category = category
        self.stage = stage
        self.details = details or {}
        self.retryable = retryable


@dataclass(frozen=True)
class Completed:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int


def run_command(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 10.0,
    max_output: int = 1_000_000,
) -> Completed:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CoretapError(
            "COMMAND_NOT_FOUND",
            f"Command not found: {argv[0]}",
            stage="subprocess",
            details={"argv": argv},
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CoretapError(
            "COMMAND_TIMEOUT",
            f"Command timed out after {timeout}s: {' '.join(argv)}",
            stage="subprocess",
            details={"argv": argv, "timeout": timeout},
            retryable=True,
        ) from exc

    stdout = proc.stdout[-max_output:]
    stderr = proc.stderr[-max_output:]
    return Completed(
        argv=argv,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=round((time.monotonic() - started) * 1000),
    )


def require_success(done: Completed, *, code: str, stage: str) -> Completed:
    if done.returncode != 0:
        raise CoretapError(
            code,
            f"Command failed with exit code {done.returncode}: {' '.join(done.argv)}",
            stage=stage,
            details={
                "argv": done.argv,
                "returncode": done.returncode,
                "stdout": done.stdout,
                "stderr": done.stderr,
                "durationMs": done.duration_ms,
            },
        )
    return done


def default_developer_dir() -> str | None:
    xcode = Path("/Applications/Xcode.app/Contents/Developer")
    if xcode.exists():
        return str(xcode)
    return None


def command_env(developer_dir: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    dev_dir = developer_dir or env.get("DEVELOPER_DIR") or default_developer_dir()
    if dev_dir:
        env["DEVELOPER_DIR"] = dev_dir
    return env


def state_root() -> Path:
    return Path.home() / "Library" / "Application Support" / "Coretap"


def cache_root() -> Path:
    return Path.home() / "Library" / "Caches" / "Coretap"


def ensure_state() -> dict[str, Path]:
    roots = {
        "state": state_root(),
        "models": state_root() / "models",
        "receipts": state_root() / "install-receipts",
        "cache": cache_root(),
        "downloads": cache_root() / "downloads",
        "artifacts": cache_root() / "artifacts",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return roots


def run_id(prefix: str = "run") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "".join(random.choice(string.hexdigits.lower()) for _ in range(8))
    return f"{prefix}_{stamp}_{suffix}"


def artifact_dir(root: Path | None = None) -> Path:
    base = root or (cache_root() / "artifacts")
    path = base / run_id()
    path.mkdir(parents=True, exist_ok=False)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise CoretapError("INVALID_PNG", f"Not a PNG file: {path}", stage="image")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def which(name: str) -> str | None:
    return shutil.which(name)


def response_ok(command: str, result: Any, *, artifacts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": "coretap.response.v1",
        "ok": True,
        "command": command,
        "requestId": run_id("req"),
        "durationMs": 0,
        "result": result,
        "artifacts": artifacts or [],
        "warnings": [],
    }


def response_error(command: str, error: CoretapError) -> dict[str, Any]:
    return {
        "schema": "coretap.response.v1",
        "ok": False,
        "command": command,
        "requestId": run_id("req"),
        "durationMs": 0,
        "error": {
            "code": error.code,
            "category": error.category,
            "stage": error.stage,
            "message": str(error),
            "retryable": error.retryable,
            "details": error.details,
        },
        "artifacts": [],
    }
