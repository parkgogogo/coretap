from __future__ import annotations

import json
import subprocess
import sys

import pytest

from coretap.daemon import handle_argv
from coretap.grounding import DEFAULT_GROUNDING_IMAGE_LONG_SIDE
from coretap.ocr import DEFAULT_OCR_LANG
from coretap.runtime import CoretapError


def run_coretap(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "coretap", "--daemon", "off", "--format", "json", *args],
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


def test_press_button_dry_run_json() -> None:
    data = run_coretap("--backend", "device", "--device", "device-udid", "press", "power", "--dry-run")

    assert data["ok"] is True
    assert data["result"]["button"] == "lock"
    assert data["result"]["requestedButton"] == "power"
    assert data["result"]["state"] == "press"
    assert data["result"]["attempted"] is False
    assert data["result"]["dryRun"] is True


def test_type_text_dry_run_json() -> None:
    data = run_coretap("--backend", "device", "--device", "device-udid", "type", "hello@example.com", "--dry-run")

    assert data["ok"] is True
    assert data["result"]["attempted"] is False
    assert data["result"]["dryRun"] is True
    assert data["result"]["text"]["length"] == len("hello@example.com")
    assert data["result"]["inputMethod"] == "coredevice-pasteboard-edit-menu"


def test_type_text_supports_non_ascii_dry_run_json() -> None:
    data = run_coretap("--backend", "device", "--device", "device-udid", "type", "搜索", "--dry-run")

    assert data["ok"] is True
    assert data["result"]["attempted"] is False
    assert data["result"]["text"]["asciiOnly"] is False
    assert data["result"]["inputMethod"] == "coredevice-pasteboard-edit-menu"


def test_type_text_dry_run_accepts_paste_anchor_json() -> None:
    data = run_coretap("--backend", "device", "--device", "device-udid", "type", "hello", "--paste-at", "0.2,0.925", "--dry-run")

    assert data["ok"] is True
    assert data["result"]["pasteAt"] == {"x": 0.2, "y": 0.925}


def test_model_install_dry_run_json() -> None:
    data = run_coretap("model", "install", "--dry-run")

    assert data["ok"] is True
    assert data["result"]["dryRun"] is True
    assert data["result"]["changed"] is False


def test_daemon_handle_argv_reuses_cli_dispatch(tmp_path) -> None:
    data = handle_argv(["--format", "json", "status"], cwd=str(tmp_path))

    assert data["schema"] == "coretap.response.v1"
    assert data["ok"] is True
    assert data["exitCode"] == 0
    assert data["daemon"]["pid"] > 0
    assert data["daemon"]["workers"]["model"]["kind"] == "mlx-vlm-process-resident"
    assert data["daemon"]["workers"]["coredevice"]["kind"] == "coredevice-userspace-persistent"
    assert data["result"]["version"] == "0.1.0"


def test_default_cli_auto_starts_daemon_and_forwards(monkeypatch, capsys) -> None:
    import coretap.daemon
    from coretap.cli import main

    calls = []
    starts = []

    def fake_request_daemon(argv, *, cwd=None, socket_path=None, timeout=300.0):
        calls.append({"argv": argv, "cwd": cwd, "socket_path": socket_path, "timeout": timeout})
        if len(calls) == 1:
            raise CoretapError("DAEMON_UNAVAILABLE", "not running", stage="daemon")
        return {
            "schema": "coretap.response.v1",
            "ok": True,
            "command": "status",
            "requestId": "req_test",
            "durationMs": 1,
            "result": {"version": "0.1.0"},
            "artifacts": [],
            "warnings": [],
            "daemon": {
                "pid": 123,
                "workers": {
                    "model": {"kind": "mlx-vlm-process-resident", "loaded": False},
                    "coredevice": {"kind": "coredevice-userspace-persistent", "running": True},
                },
            },
            "exitCode": 0,
        }

    def fake_start_daemon(*, socket_path=None, timeout=5.0):
        starts.append({"socket_path": socket_path, "timeout": timeout})
        return {"started": True}

    monkeypatch.setattr(coretap.daemon, "request_daemon", fake_request_daemon)
    monkeypatch.setattr(coretap.daemon, "start_daemon", fake_start_daemon)

    with pytest.raises(SystemExit) as exc:
        main(["--format", "json", "status"])

    assert exc.value.code == 0
    assert starts == [{"socket_path": None, "timeout": 5.0}]
    assert [call["argv"] for call in calls] == [["--format", "json", "status"], ["--format", "json", "status"]]
    data = json.loads(capsys.readouterr().out)
    assert data["daemon"]["workers"]["model"]["kind"] == "mlx-vlm-process-resident"
    assert data["daemon"]["workers"]["coredevice"]["running"] is True


def test_daemon_on_requires_existing_daemon(monkeypatch, capsys) -> None:
    import coretap.daemon
    from coretap.cli import main

    starts = []

    def fake_request_daemon(argv, *, cwd=None, socket_path=None, timeout=300.0):
        raise CoretapError("DAEMON_UNAVAILABLE", "not running", stage="daemon")

    def fake_start_daemon(*, socket_path=None, timeout=5.0):
        starts.append({"socket_path": socket_path, "timeout": timeout})
        return {"started": True}

    monkeypatch.setattr(coretap.daemon, "request_daemon", fake_request_daemon)
    monkeypatch.setattr(coretap.daemon, "start_daemon", fake_start_daemon)

    with pytest.raises(SystemExit) as exc:
        main(["--daemon", "on", "--format", "json", "status"])

    assert exc.value.code == 14
    assert starts == []
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False
    assert data["error"]["code"] == "DAEMON_UNAVAILABLE"


def test_text_ocr_commands_default_to_chinese_and_english() -> None:
    from coretap.cli import build_parser, normalize_global_args

    parser = build_parser()

    tap = parser.parse_args(normalize_global_args(["tap", "text", "搜索", "--dry-run"]))
    assert tap.lang == DEFAULT_OCR_LANG

    assert_text = parser.parse_args(normalize_global_args(["assert", "text", "--text", "搜索"]))
    assert assert_text.lang == DEFAULT_OCR_LANG

    wait_text = parser.parse_args(normalize_global_args(["wait", "text", "--text", "搜索"]))
    assert wait_text.lang == DEFAULT_OCR_LANG


def test_screenshot_defaults_to_preview_long_side() -> None:
    from coretap.cli import build_parser, normalize_global_args

    parser = build_parser()

    screenshot = parser.parse_args(normalize_global_args(["screenshot"]))
    assert screenshot.max_long_side == DEFAULT_GROUNDING_IMAGE_LONG_SIDE
    assert screenshot.full_size is False
