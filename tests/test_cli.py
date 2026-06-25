from __future__ import annotations

import json
import subprocess
import sys

from coretap.daemon import handle_argv


def run_coretap(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "coretap", "--format", "json", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.stdout, proc.stderr
    return json.loads(proc.stdout)


def test_status_json_envelope() -> None:
    data = run_coretap("status")

    assert data["schema"] == "coretap.response.v1"
    assert data["ok"] is True
    assert data["requestId"].startswith("req_")
    assert data["result"]["version"] == "0.1.0"


def test_model_status_json_envelope() -> None:
    data = run_coretap("model", "status")

    assert data["ok"] is True
    assert data["result"]["profile"] == "builtin:mai-ui-2b-mlx-6bit@1"


def test_internal_fixture_profile_is_not_default() -> None:
    data = run_coretap("--profile", "internal:test-fixture-grounder", "model", "status")

    assert data["ok"] is True
    assert data["result"]["implementation"] == "internal-ocr-fixture-grounder"


def test_daemon_handle_argv_reuses_cli_dispatch(tmp_path) -> None:
    data = handle_argv(["--format", "json", "status"], cwd=str(tmp_path))

    assert data["schema"] == "coretap.response.v1"
    assert data["ok"] is True
    assert data["exitCode"] == 0
    assert data["daemon"]["pid"] > 0
    assert data["result"]["version"] == "0.1.0"
