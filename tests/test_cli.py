from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from coretap.daemon import handle_argv
from coretap.ocr import OcrToken
from coretap.runtime import CoretapError


def run_coretap(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "coretap", "--daemon", "off", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.stdout, proc.stderr
    return json.loads(proc.stdout)


def test_status_json_envelope() -> None:
    data = run_coretap("status")

    assert data["ok"] is True
    assert data["status"] == "success"
    assert data["requestId"].startswith("req_")
    assert data["result"]["version"] == "0.1.0"


def test_cli_has_single_json_output_form() -> None:
    from coretap.cli import build_parser

    args = build_parser().parse_args(["status"])

    assert not hasattr(args, "format")


def test_trace_logging_records_step_event(tmp_path: Path) -> None:
    from coretap.cli import build_parser, record_trace

    argv = [
        "--artifact-root",
        str(tmp_path),
        "--trace-id",
        "通用 搜索链路",
        "--trace-title",
        "通用搜索链路",
        "step",
        "--action",
        '{"type":"tap","target":"the App Store search field"}',
    ]
    args = build_parser().parse_args(argv)
    data = {
        "schema": "coretap.response.v1",
        "ok": True,
        "command": "step",
        "requestId": "req_test",
        "durationMs": 123,
        "result": {
            "schema": "coretap.step.result.v1",
            "status": "executed",
            "artifactDir": "artifacts/coretap/run_test",
            "action": {"schema": "coretap.action.v2", "type": "tap", "target": "the App Store search field"},
            "before": {"frame": {"path": "before.png", "widthPx": 632, "heightPx": 1368, "sha256": "before"}},
            "execution": {
                "schema": "coretap.step.execution.v1",
                "status": "executed",
                "actionType": "tap",
                "strategy": "vlm_grounding",
                "modelInput": {"path": "source.model-input.png", "widthPx": 632, "heightPx": 1368, "resized": True, "maxLongSidePx": 1368, "scale": 0.5},
                "grounding": {
                    "status": "found",
                    "point": {"normalized": {"x": 0.5, "y": 0.25}, "framePx": {"x": 316, "y": 342}},
                },
                "point": {"normalized": {"x": 0.5, "y": 0.25}, "screenshotPx": {"x": 316, "y": 342}, "hidU16": {"x": 32768, "y": 16384}},
                "tap": {"attempted": True, "dryRun": False, "deliveryStatus": "delivered"},
            },
        },
        "artifacts": [],
        "warnings": [],
    }

    trace = record_trace(args, data, argv=argv, cwd=str(tmp_path))

    assert trace is not None
    assert trace["id"] == "通用-搜索链路"
    trace_path = Path(trace["tracePath"])
    events_path = Path(trace["eventsPath"])
    response_path = Path(trace["responsePath"])
    assert trace_path.exists()
    assert response_path.exists()
    trace_doc = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_doc["eventCount"] == 1
    assert trace_doc["title"] == "通用搜索链路"
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["summary"]["action"]["target"] == "the App Store search field"
    assert event["summary"]["execution"]["strategy"] == "vlm_grounding"
    assert event["summary"]["before"]["frame"]["path"] == "before.png"
    assert event["responsePath"] == str(response_path)


def test_trace_global_args_can_follow_subcommand() -> None:
    from coretap.cli import build_parser, normalize_global_args

    normalized = normalize_global_args(["step", "--action", "{}", "--trace-id", "generic-search", "--trace-title", "通用搜索链路", "--keep-artifacts"])
    args = build_parser().parse_args(normalized)

    assert args.trace_id == "generic-search"
    assert args.trace_title == "通用搜索链路"
    assert args.keep_artifacts is True


def test_model_status_json_envelope() -> None:
    data = run_coretap("model", "status")

    assert data["ok"] is True
    assert data["result"]["profile"] == "builtin:mai-ui-2b-mlx-6bit@1"


def test_internal_fixture_profile_is_not_default() -> None:
    data = run_coretap("--profile", "internal:test-fixture-grounder", "model", "status")

    assert data["ok"] is True
    assert data["result"]["implementation"] == "internal-vision-fixture-grounder"


def test_model_install_dry_run_json() -> None:
    data = run_coretap("model", "install", "--dry-run")

    assert data["ok"] is True
    assert data["result"]["dryRun"] is True
    assert data["result"]["changed"] is False


def test_daemon_handle_argv_reuses_cli_dispatch(tmp_path) -> None:
    data = handle_argv(["status"], cwd=str(tmp_path))

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

    def fake_ping_daemon(*, socket_path=None, timeout=2.0):
        raise CoretapError("DAEMON_UNAVAILABLE", "not running", stage="daemon")

    def fake_request_daemon(argv, *, cwd=None, socket_path=None, timeout=300.0):
        calls.append({"argv": argv, "cwd": cwd, "socket_path": socket_path, "timeout": timeout})
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
                "code": {"fingerprint": "client"},
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
    monkeypatch.setattr(coretap.daemon, "ping_daemon", fake_ping_daemon)
    monkeypatch.setattr(coretap.daemon, "start_daemon", fake_start_daemon)

    with pytest.raises(SystemExit) as exc:
        main(["status"])

    assert exc.value.code == 0
    assert starts == [{"socket_path": None, "timeout": 5.0}]
    assert [call["argv"] for call in calls] == [["status"]]
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True
    assert data["status"] == "success"
    assert data["result"]["version"] == "0.1.0"
    assert "daemon" not in data


def test_default_cli_restarts_stale_daemon_before_forwarding(monkeypatch, capsys) -> None:
    import coretap.daemon
    from coretap.cli import main

    calls = []
    stops = []
    starts = []

    def fake_source_fingerprint() -> dict:
        return {"schema": "coretap.source-fingerprint.v1", "fingerprint": "client"}

    def fake_ping_daemon(*, socket_path=None, timeout=2.0):
        return {
            "schema": "coretap.response.v1",
            "ok": True,
            "result": {
                "running": True,
                "pid": 111,
                "code": {"schema": "coretap.source-fingerprint.v1", "fingerprint": "daemon-old"},
            },
        }

    def fake_stop_daemon(*, socket_path=None, timeout=2.0):
        stops.append({"socket_path": socket_path, "timeout": timeout})
        return {"ok": True}

    def fake_start_daemon(*, socket_path=None, timeout=5.0):
        starts.append({"socket_path": socket_path, "timeout": timeout})
        return {"started": True}

    def fake_request_daemon(argv, *, cwd=None, socket_path=None, timeout=300.0):
        calls.append({"argv": argv, "cwd": cwd, "socket_path": socket_path, "timeout": timeout})
        return {
            "schema": "coretap.response.v1",
            "ok": True,
            "command": "status",
            "requestId": "req_test",
            "durationMs": 1,
            "result": {"version": "0.1.0"},
            "artifacts": [],
            "warnings": [],
            "daemon": {"pid": 222, "code": {"fingerprint": "client"}, "workers": {}},
            "exitCode": 0,
        }

    monkeypatch.setattr(coretap.daemon, "source_fingerprint", fake_source_fingerprint)
    monkeypatch.setattr(coretap.daemon, "ping_daemon", fake_ping_daemon)
    monkeypatch.setattr(coretap.daemon, "stop_daemon", fake_stop_daemon)
    monkeypatch.setattr(coretap.daemon, "start_daemon", fake_start_daemon)
    monkeypatch.setattr(coretap.daemon, "request_daemon", fake_request_daemon)

    with pytest.raises(SystemExit) as exc:
        main(["status"])

    assert exc.value.code == 0
    assert stops == [{"socket_path": None, "timeout": 2.0}]
    assert starts == [{"socket_path": None, "timeout": 5.0}]
    assert [call["argv"] for call in calls] == [["status"]]
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True
    assert data["status"] == "success"
    assert data["result"]["version"] == "0.1.0"
    assert "daemon" not in data


def test_daemon_on_requires_existing_daemon(monkeypatch, capsys) -> None:
    import coretap.daemon
    from coretap.cli import main

    starts = []

    def fake_ping_daemon(*, socket_path=None, timeout=2.0):
        raise CoretapError("DAEMON_UNAVAILABLE", "not running", stage="daemon")

    def fake_start_daemon(*, socket_path=None, timeout=5.0):
        starts.append({"socket_path": socket_path, "timeout": timeout})
        return {"started": True}

    monkeypatch.setattr(coretap.daemon, "ping_daemon", fake_ping_daemon)
    monkeypatch.setattr(coretap.daemon, "start_daemon", fake_start_daemon)

    with pytest.raises(SystemExit) as exc:
        main(["--daemon", "on", "status"])

    assert exc.value.code == 14
    assert starts == []
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False
    assert data["error"]["code"] == "DAEMON_UNAVAILABLE"


def test_daemon_stop_is_idempotent_when_already_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    import coretap.daemon
    from coretap.cli import command_daemon

    def fake_stop_daemon(*, socket_path=None, timeout=2.0):
        raise CoretapError(
            "DAEMON_UNAVAILABLE",
            "not running",
            stage="daemon",
            details={"socket": str(socket_path), "error": "missing"},
        )

    monkeypatch.setattr(coretap.daemon, "stop_daemon", fake_stop_daemon)

    result = command_daemon(argparse.Namespace(daemon_command="stop", socket="/tmp/coretap-test.sock", timeout_ms=10))

    assert result["alreadyStopped"] is True
    assert result["running"] is False
    assert result["socket"] == "/tmp/coretap-test.sock"


def test_stale_daemon_pid_parser_only_matches_same_socket() -> None:
    from coretap.daemon import _stale_daemon_pids_from_ps

    output = """
      101 /usr/bin/python -m coretap.daemon serve --socket /tmp/coretapd.sock
      102 /usr/bin/python -m coretap.daemon serve --socket /tmp/other.sock
      103 /usr/bin/python -m other.daemon serve --socket /tmp/coretapd.sock
      104 /usr/bin/python -m coretap.daemon serve --socket /tmp/coretapd.sock
    """

    pids = _stale_daemon_pids_from_ps(output, socket_path=Path("/tmp/coretapd.sock"), current_pid=104)

    assert pids == [101]


def test_text_ocr_commands_use_builtin_vision_without_engine_options() -> None:
    from coretap.cli import build_parser, normalize_global_args

    parser = build_parser()

    assert_text = parser.parse_args(normalize_global_args(["assert", "text", "--text", "搜索"]))
    assert not hasattr(assert_text, "ocr_engine")
    assert not hasattr(assert_text, "lang")
    assert not hasattr(assert_text, "psm")

    wait_text = parser.parse_args(normalize_global_args(["wait", "text", "--text", "搜索"]))
    assert not hasattr(wait_text, "ocr_engine")
    assert not hasattr(wait_text, "lang")
    assert not hasattr(wait_text, "psm")


def test_step_parser_accepts_single_action_runtime_options() -> None:
    from coretap.cli import build_parser, normalize_global_args

    action = '{"type":"tap","target":"Search"}'
    args = build_parser().parse_args(
        normalize_global_args(
            [
                "step",
                "--action",
                action,
                "--page-wait-ms",
                "6000",
                "--no-vlm",
            ]
        )
    )

    assert args.command == "step"
    assert args.action == action
    assert args.page_wait_ms == 6000
    assert args.no_vlm is True
    assert args.max_long_side == 1368


def test_step_action_accepts_mobile_use_actions_without_schema() -> None:
    from coretap.cli import _load_step_action, _normalize_step_action

    assert _normalize_step_action({"type": "tap", "target": "Search"}) == {
        "schema": "coretap.action.v2",
        "type": "tap",
        "target": "Search",
    }
    assert _normalize_step_action(
        {
            "type": "tap",
            "target": "download Xiaohongshu from the non-ad second app result by Xingin",
        }
    ) == {
        "schema": "coretap.action.v2",
        "type": "tap",
        "target": "download Xiaohongshu from the non-ad second app result by Xingin",
    }
    with pytest.raises(CoretapError, match="constraints were removed"):
        _normalize_step_action({"type": "tap", "target": "Search", "constraints": {"region": "bottomTabBar"}})
    with pytest.raises(CoretapError, match="postconditions were removed"):
        _normalize_step_action({"type": "tap", "target": "Search", "postconditions": [{"type": "screenChanged"}]})
    assert _normalize_step_action({"type": "tapPoint", "x": 0.5, "y": 0.5}) == {
        "schema": "coretap.action.v2",
        "type": "tapPoint",
        "point": {"x": 0.5, "y": 0.5, "space": "normalized", "reference": "source"},
    }
    assert _normalize_step_action({"type": "longPress", "point": {"x": 0.25, "y": 0.75}}) == {
        "schema": "coretap.action.v2",
        "type": "longPress",
        "point": {"x": 0.25, "y": 0.75, "space": "normalized", "reference": "source"},
        "durationMs": 1200,
        "steps": 12,
    }
    assert _normalize_step_action({"type": "terminateApp", "bundleId": "com.apple.AppStore"}) == {
        "schema": "coretap.action.v2",
        "type": "terminateApp",
        "bundleId": "com.apple.AppStore",
        "signal": 9,
    }
    assert _normalize_step_action({"type": "uninstallApp", "name": "小红书"}) == {
        "schema": "coretap.action.v2",
        "type": "uninstallApp",
        "name": "小红书",
        "bundleId": "com.xingin.discover",
        "ignoreMissing": True,
    }
    assert _normalize_step_action({"type": "openUrl", "url": "https://example.com", "timeoutSec": 5}) == {
        "schema": "coretap.action.v2",
        "type": "openUrl",
        "url": "https://example.com",
        "timeoutSec": 5.0,
    }
    with pytest.raises(CoretapError) as alias_exc:
        _normalize_step_action({"type": "type", "text": "hello"})

    assert alias_exc.value.code == "ACTION_UNSUPPORTED"
    assert _load_step_action(argparse.Namespace(action='{"type":"tap","target":"Search"}', action_file=None)) == {
        "type": "tap",
        "target": "Search",
    }
    with pytest.raises(CoretapError) as schema_exc:
        _load_step_action(argparse.Namespace(action='{"schema":"coretap.action.v1","type":"tap","target":"Search"}', action_file=None))

    assert schema_exc.value.code == "ACTION_SCHEMA_INVALID"


@pytest.mark.parametrize(
    "argv",
    [
        ["tap", "target", "--target", "Search"],
        ["tap", "text", "Search"],
        ["locate", "--target", "Search"],
        ["act", "--goal", "install sample app"],
        ["press", "home"],
        ["type", "hello"],
        ["key", "enter"],
        ["clear"],
        ["drag", "--from", "0.5,0.7", "--to", "0.5,0.2"],
        ["scroll", "down"],
        ["run", "flow.json"],
        ["replay", "artifacts/coretap/run"],
        ["test", "--", "echo", "ok"],
        ["ocr", "status"],
    ],
)
def test_removed_public_commands_are_not_registered(argv: list[str]) -> None:
    from coretap.cli import build_parser, normalize_global_args

    with pytest.raises(SystemExit):
        build_parser().parse_args(normalize_global_args(argv))


def test_screenshot_command_writes_full_size_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_screenshot

    captured: dict[str, object] = {}

    def fake_capture_to(args: argparse.Namespace, *, label: str, run_dir: Path, out: Path, write_frame: bool = True) -> argparse.Namespace:
        captured["label"] = label
        captured["runDir"] = run_dir
        captured["out"] = out
        captured["writeFrame"] = write_frame
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"full-size-png")
        return argparse.Namespace(frame_id=f"frame_{out.stem}", path=out, width=1260, height=2736, backend=args.backend, device=args.device)

    monkeypatch.setattr(coretap.cli, "_capture_to", fake_capture_to)
    out = tmp_path / "shot.png"
    args = argparse.Namespace(
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        label="shot",
        out=str(out),
        no_artifacts=False,
        keep_artifacts=False,
        artifact_root=None,
        trace_id=None,
    )

    result = command_screenshot(args)

    assert out.exists()
    assert captured["out"] == out
    assert result["schema"] == "coretap.screenshot.result.v1"
    assert result["frame"]["path"] == str(out)
    assert result["frame"]["widthPx"] == 1260
    assert result["frame"]["heightPx"] == 2736
    assert result["frame"]["resized"] is False
    assert result["frame"]["maxLongSidePx"] is None
    assert result["frame"]["scale"] == 1.0
    assert "artifactDir" not in result


def test_screenshot_command_defaults_to_cache_artifact_when_out_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_screenshot

    def fake_capture_to(args: argparse.Namespace, *, label: str, run_dir: Path, out: Path, write_frame: bool = True) -> argparse.Namespace:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"full-size-png")
        return argparse.Namespace(frame_id=f"frame_{out.stem}", path=out, width=1125, height=2436, backend=args.backend, device=args.device)

    monkeypatch.setattr(coretap.cli, "_capture_to", fake_capture_to)
    args = argparse.Namespace(
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        label="screenshot",
        out=None,
        no_artifacts=False,
        keep_artifacts=False,
        artifact_root=str(tmp_path),
        trace_id=None,
    )

    result = command_screenshot(args)

    frame_path = Path(result["frame"]["path"])
    assert frame_path.exists()
    assert frame_path.name == "screenshot.png"
    assert str(frame_path).startswith(str(tmp_path))
    assert result["artifactDir"].startswith(str(tmp_path))
    assert result["frame"]["resized"] is False


def test_observe_returns_structured_ocr_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_observe

    frame_path = tmp_path / "observe.png"
    screenshot = {
        "artifactDir": str(tmp_path),
        "frame": {
            "frameId": "frame_observe",
            "path": str(frame_path),
            "widthPx": 100,
            "heightPx": 200,
            "backend": "device",
            "device": "device-udid",
            "resized": True,
            "maxLongSidePx": 1368,
            "scale": 0.5,
        },
        "sourceFrame": {"path": str(tmp_path / "observe.source.png"), "widthPx": 200, "heightPx": 400},
    }

    monkeypatch.setattr(coretap.cli, "artifact_dir", lambda _root=None: tmp_path)
    monkeypatch.setattr(coretap.cli, "_screenshot_into", lambda _args, *, run_dir, label, out=None: screenshot)
    monkeypatch.setattr(
        coretap.cli,
        "run_ocr",
        lambda image: ([OcrToken("◎ 搜索", 30.0, 50, 120, 20, 10, "vision")], {"engines": ["vision"], "visionJson": "[]", "errors": []}),
    )
    monkeypatch.setattr(
        coretap.cli,
        "run_visual_observe_model",
        lambda image, *, profile: {
            "schema": "coretap.visual.observe.v1",
            "enabled": True,
            "status": "ready",
            "profile": profile,
            "promptVersion": "visual-observe-v1",
            "summary": "Home screen with icons",
            "elements": [
                {
                    "type": "visual",
                    "source": "vlm",
                    "label": "ChatGPT app icon",
                    "role": "appIcon",
                    "confidence": 0.8,
                    "center": {"x": 0.8, "y": 0.5},
                    "bbox": {"x": 0.72, "y": 0.44, "width": 0.16, "height": 0.12},
                },
                {
                    "type": "visual",
                    "source": "vlm",
                    "label": "Search text label",
                    "role": "button",
                    "confidence": 0.7,
                    "center": {"x": 0.6, "y": 0.625},
                    "bbox": {"x": 0.5, "y": 0.6, "width": 0.2, "height": 0.05},
                }
            ],
        },
    )

    result = command_observe(
        argparse.Namespace(
            label="observe",
            out=None,
            max_long_side=1368,
            full_size=False,
            min_confidence=0.0,
            no_ocr=False,
            no_vlm=False,
            profile="builtin:mai-ui-2b-mlx-6bit@1",
            artifact_root=str(tmp_path),
            keep_artifacts=False,
            no_artifacts=False,
        )
    )

    assert result["schema"] == "coretap.observe.result.v1"
    assert result["ocr"]["schema"] == "coretap.ocr.page.v1"
    assert result["ocr"]["engineMode"] == "vision"
    assert result["ocr"]["selectedEngine"] == "vision"
    assert result["ocr"]["engines"] == ["vision"]
    assert result["ocr"]["plainText"] == "◎ 搜索"
    assert result["ocr"]["tokens"][0]["text"] == "◎ 搜索"
    assert result["ocr"]["tokens"][0]["normalizedText"] == "◎ 搜索"
    assert result["ocr"]["tokens"][0]["centerPx"] == {"x": 60.0, "y": 125.0}
    assert result["ocr"]["tokens"][0]["normalized"] == {"x": 0.6, "y": 0.625}
    assert result["visual"]["enabled"] is True
    assert result["visual"]["summary"] == "Home screen with icons"
    assert [item["label"] for item in result["visual"]["elements"]] == ["ChatGPT app icon"]
    assert result["visual"]["ocrFilteredElementCount"] == 1
    assert (tmp_path / "observe.result.json").exists()


def test_visual_ocr_filter_keeps_icon_adjacent_to_text() -> None:
    from coretap.cli import _filter_visual_elements_against_ocr

    visual = {
        "enabled": True,
        "elements": [
            {
                "type": "visual",
                "source": "vlm",
                "label": "Search icon",
                "role": "button",
                "center": {"x": 0.46, "y": 0.56},
                "bbox": {"x": 0.42, "y": 0.52, "width": 0.08, "height": 0.08},
            }
        ],
    }
    ocr_tokens = [{"text": "搜索", "bboxNormalized": {"x": 0.5, "y": 0.6, "width": 0.2, "height": 0.05}}]

    result = _filter_visual_elements_against_ocr(visual, ocr_tokens)

    assert result["elements"][0]["label"] == "Search icon"
    assert "ocrFilteredElementCount" not in result


def test_observe_no_vlm_skips_visual_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_observe

    screenshot = {
        "artifactDir": str(tmp_path),
        "frame": {"path": str(tmp_path / "observe.png"), "widthPx": 100, "heightPx": 200},
    }
    monkeypatch.setattr(coretap.cli, "artifact_dir", lambda _root=None: tmp_path)
    monkeypatch.setattr(coretap.cli, "_screenshot_into", lambda _args, *, run_dir, label, out=None: screenshot)
    monkeypatch.setattr(coretap.cli, "run_ocr", lambda image: ([], {"engines": ["vision"], "visionJson": "[]", "errors": []}))
    monkeypatch.setattr(
        coretap.cli,
        "run_visual_observe_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("VLM should be disabled")),
    )

    result = command_observe(
        argparse.Namespace(
            label="observe",
            out=None,
            max_long_side=1368,
            full_size=False,
            min_confidence=0.0,
            no_ocr=False,
            no_vlm=True,
            profile="builtin:mai-ui-2b-mlx-6bit@1",
            artifact_root=str(tmp_path),
            keep_artifacts=False,
            no_artifacts=False,
        )
    )

    assert result["ocr"]["enabled"] is True
    assert result["visual"] == {"enabled": False}


def test_observe_no_ocr_still_runs_visual_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_observe

    screenshot = {
        "artifactDir": str(tmp_path),
        "frame": {"path": str(tmp_path / "observe.png"), "widthPx": 100, "heightPx": 200},
    }
    monkeypatch.setattr(coretap.cli, "artifact_dir", lambda _root=None: tmp_path)
    monkeypatch.setattr(coretap.cli, "_screenshot_into", lambda _args, *, run_dir, label, out=None: screenshot)
    monkeypatch.setattr(coretap.cli, "run_ocr", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OCR should be disabled")))
    monkeypatch.setattr(
        coretap.cli,
        "run_visual_observe_model",
        lambda image, *, profile: {
            "schema": "coretap.visual.observe.v1",
            "enabled": True,
            "status": "ready",
            "profile": profile,
            "promptVersion": "visual-observe-v1",
            "summary": "Icon-only page",
            "elements": [
                {
                    "type": "visual",
                    "source": "vlm",
                    "label": "Text-shaped visual still present without OCR",
                    "role": "button",
                    "center": {"x": 0.6, "y": 0.625},
                    "bbox": {"x": 0.5, "y": 0.6, "width": 0.2, "height": 0.05},
                }
            ],
        },
    )

    result = command_observe(
        argparse.Namespace(
            label="observe",
            out=None,
            max_long_side=1368,
            full_size=False,
            min_confidence=0.0,
            no_ocr=True,
            no_vlm=False,
            profile="builtin:mai-ui-2b-mlx-6bit@1",
            artifact_root=str(tmp_path),
            keep_artifacts=False,
            no_artifacts=False,
        )
    )

    assert result["ocr"] == {"enabled": False}
    assert result["visual"]["enabled"] is True
    assert result["visual"]["summary"] == "Icon-only page"
    assert result["visual"]["elements"][0]["label"] == "Text-shaped visual still present without OCR"
    assert "ocrFilteredElementCount" not in result["visual"]


def test_observe_default_artifacts_are_temporary(monkeypatch: pytest.MonkeyPatch) -> None:
    import coretap.cli
    from coretap.cli import command_observe

    captured: dict[str, Path] = {}

    def fake_screenshot(_args: argparse.Namespace, *, run_dir: Path, label: str, out: Path | None = None) -> dict:
        captured["runDir"] = run_dir
        image = run_dir / f"{label}.png"
        image.write_bytes(b"fake")
        return {
            "frame": {"path": str(image), "widthPx": 100, "heightPx": 200},
            "sourceFrame": {"path": str(image), "widthPx": 100, "heightPx": 200},
        }

    monkeypatch.setattr(coretap.cli, "_screenshot_into", fake_screenshot)
    monkeypatch.setattr(coretap.cli, "run_ocr", lambda image: ([], {"engines": ["vision"], "visionJson": "[]", "errors": []}))

    result = command_observe(
        argparse.Namespace(
            label="observe",
            out=None,
            max_long_side=1368,
            full_size=False,
            min_confidence=0.0,
            no_ocr=False,
            no_vlm=True,
            profile="builtin:mai-ui-2b-mlx-6bit@1",
            artifact_root=None,
            keep_artifacts=False,
            no_artifacts=False,
            trace_id=None,
        )
    )

    assert "artifactDir" not in result
    assert "runDir" in captured
    assert not captured["runDir"].exists()


def test_observe_artifact_root_persists_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_observe

    def fake_screenshot(_args: argparse.Namespace, *, run_dir: Path, label: str, out: Path | None = None) -> dict:
        image = run_dir / f"{label}.png"
        image.write_bytes(b"fake")
        return {
            "frame": {"path": str(image), "widthPx": 100, "heightPx": 200},
            "sourceFrame": {"path": str(image), "widthPx": 100, "heightPx": 200},
        }

    monkeypatch.setattr(coretap.cli, "_screenshot_into", fake_screenshot)
    monkeypatch.setattr(coretap.cli, "run_ocr", lambda image: ([], {"engines": ["vision"], "visionJson": "[]", "errors": []}))

    result = command_observe(
        argparse.Namespace(
            label="observe",
            out=None,
            max_long_side=1368,
            full_size=False,
            min_confidence=0.0,
            no_ocr=False,
            no_vlm=True,
            profile="builtin:mai-ui-2b-mlx-6bit@1",
            artifact_root=str(tmp_path),
            keep_artifacts=False,
            no_artifacts=False,
            trace_id=None,
        )
    )

    artifact_dir = Path(result["artifactDir"])
    assert artifact_dir.parent == tmp_path
    assert artifact_dir.exists()
    assert (artifact_dir / "observe.result.json").exists()


def _observe_with_tokens(tokens: list[OcrToken], *, visual: dict | None = None) -> dict:
    return {
        "schema": "coretap.observe.result.v1",
        "frame": {"path": "screen.png", "widthPx": 632, "heightPx": 1368, "sha256": "before"},
        "ocr": {
            "enabled": True,
            "tokens": [
                {
                    "text": token.text,
                    "confidence": token.confidence,
                    "engine": token.engine,
                    "bboxPx": {"x": token.left, "y": token.top, "width": token.width, "height": token.height},
                    "normalized": {"x": token.center[0] / 632, "y": token.center[1] / 1368},
                }
                for token in tokens
            ],
        },
        "visual": visual if visual is not None else {"enabled": False},
    }


def _parse_xy(value: str) -> tuple[float, float]:
    x, y = value.split(",", 1)
    return float(x), float(y)


def test_paste_menu_candidate_clicks_substring_center_for_combined_ocr_token() -> None:
    from coretap.cli import _paste_menu_ocr_candidates

    observation = _observe_with_tokens([OcrToken("粘贴准自动填充", 30.0, 50, 189, 214, 38, "vision")])

    candidates = _paste_menu_ocr_candidates(observation, {"x": 0.32, "y": 0.091})

    assert candidates
    assert candidates[0]["label"] == "粘贴"
    assert candidates[0]["match"]["matchedKind"] == "token_contains"
    assert candidates[0]["point"]["x"] == pytest.approx(0.1275, abs=0.005)


def test_paste_menu_candidate_rejects_autofill_text_without_paste_label() -> None:
    from coretap.cli import _paste_menu_ocr_candidates

    observation = _observe_with_tokens([OcrToken("自动填充", 30.0, 50, 189, 214, 38, "vision")])

    assert _paste_menu_ocr_candidates(observation, {"x": 0.32, "y": 0.091}) == []


def test_paste_menu_candidate_accepts_vision_misread_paste_autofill_token() -> None:
    from coretap.cli import _paste_menu_ocr_candidates

    observation = _observe_with_tokens([OcrToken("大粒贴准若自动填充", 30.0, 38, 188, 226, 38, "vision")])

    candidates = _paste_menu_ocr_candidates(observation, {"x": 0.5, "y": 0.09})

    assert candidates
    assert candidates[0]["source"] == "ocr_fuzzy"
    assert candidates[0]["label"] == "粘贴"


def test_locate_paste_menu_uses_vlm_fallback_when_ocr_has_no_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import coretap.cli
    from coretap.cli import _locate_paste_menu

    observation = _observe_with_tokens([OcrToken("文梅购佳老自动填充", 30.0, 38, 188, 226, 38, "vision")])
    expected = {
        "schema": "coretap.paste-menu.candidate.v1",
        "source": "vlm",
        "label": "Paste/粘贴",
        "point": {"x": 0.14, "y": 0.15},
    }

    monkeypatch.setattr(coretap.cli, "_locate_paste_menu_with_vlm", lambda *_args, **_kwargs: expected)

    result = _locate_paste_menu(
        argparse.Namespace(profile="builtin:mai-ui-2b-mlx-6bit@1"),
        observation,
        anchor={"x": 0.5, "y": 0.09},
        run_dir=tmp_path,
        label="paste-menu",
        allow_vlm=True,
    )

    assert result == {**expected, "candidates": []}


def test_paste_menu_candidate_infers_left_paste_segment_from_fuzzy_menu_token() -> None:
    from coretap.cli import _paste_menu_ocr_candidates

    observation = _observe_with_tokens([OcrToken("5粒贴准程県砂填売", 30.0, 46, 189, 222, 49, "vision")])

    candidates = _paste_menu_ocr_candidates(observation, {"x": 0.32, "y": 0.091})

    assert candidates
    assert candidates[0]["source"] == "ocr_fuzzy"
    assert candidates[0]["match"]["matchedKind"] == "fuzzy_menu_token"
    assert candidates[0]["point"]["x"] == pytest.approx(0.136, abs=0.005)


def test_step_type_text_executes_ascii_without_ocr_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    monkeypatch.setattr(coretap.cli, "_run_observe_ocr", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OCR should not infer text entry context")))
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "hello",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("搜索", 90.0, 96, 1252, 58, 30, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["source"] == "vlm-first"
    assert result["textEntryContext"]["anchor"] is None
    assert result["focusResult"] is None
    assert captured["pasteAt"] is None


def test_step_type_text_refocuses_recent_text_field_anchor_for_ascii(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _remember_text_entry_anchor

    captured = {}

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        captured["focus"] = {"point": point, "reason": reason}
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        captured["replace"] = args.replace
        return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)
    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    _remember_text_entry_anchor(
        argparse.Namespace(backend="device", device="device-udid", dry_run=False),
        {"normalized": {"x": 0.27, "y": 0.154}},
        source="last-tap",
        action_type="tap",
        target="the App Store search text field",
    )
    args = argparse.Namespace(backend="device", device="device-udid", dry_run=False)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "xiaohongshu",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": True,
    }

    before = _observe_with_tokens([OcrToken("oldquery", 90.0, 150, 195, 100, 30, "vision")])
    result = _execute_step_action(args, action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "last-tap"
    assert result["textEntryContext"]["replaceDecision"]["status"] == "text"
    assert result["focusResult"]["reason"] == "focus-text-field-before-keyboard-input"
    assert captured["focus"] == {"point": {"x": 0.27, "y": 0.154}, "reason": "focus-text-field-before-keyboard-input"}
    assert captured["pasteAt"] is None
    assert captured["replace"] is True


def test_step_type_text_skips_replace_for_placeholder_text_near_anchor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _remember_text_entry_anchor

    captured = {}

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["replace"] = args.replace
        return {"attempted": True, "dryRun": False, "clearExisting": args.replace}

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)
    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    args = argparse.Namespace(backend="device", device="device-udid", dry_run=False)
    _remember_text_entry_anchor(
        args,
        {"normalized": {"x": 0.27, "y": 0.154}},
        source="last-tap",
        action_type="tap",
        target="the App Store search text field",
    )
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "xiaohongshu",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": True,
    }
    before = _observe_with_tokens(
        [
            OcrToken("游戏、App、故事等", 90.0, 94, 108, 250, 34, "vision"),
            OcrToken("为你推荐＞", 90.0, 32, 198, 180, 38, "vision"),
        ]
    )

    result = _execute_step_action(args, action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["replaceDecision"]["status"] == "placeholder"
    assert result["textEntryContext"]["replaceDecision"]["shouldClear"] is False
    assert captured["replace"] is False


def test_non_text_tap_target_does_not_become_text_entry_anchor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _last_text_entry_anchor, _remember_text_entry_anchor

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    args = argparse.Namespace(backend="device", device="device-udid", dry_run=False)

    anchor = _remember_text_entry_anchor(
        args,
        {"normalized": {"x": 0.9, "y": 0.2}},
        source="last-tap",
        action_type="tap",
        target="the blue cloud download button",
    )

    assert anchor is None
    assert _last_text_entry_anchor(args) is None


def test_step_type_text_uses_recent_tap_anchor_for_non_ascii(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _remember_text_entry_anchor

    captured: dict[str, object] = {}

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    def fake_visual(args: argparse.Namespace, action: dict, before: dict, run_dir: Path, *, context: dict, paste_at: object) -> dict:
        captured["action"] = action
        captured["context"] = context
        captured["pasteAt"] = paste_at
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "typeText",
            "strategy": "visual_paste_verified",
            "typeResult": {"inputMethod": "coredevice-pasteboard-visual-menu"},
        }

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    monkeypatch.setattr(coretap.cli, "_execute_step_type_text_visual", fake_visual)
    args = argparse.Namespace(
        backend="device",
        device="device-udid",
        dry_run=False,
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        no_ocr=False,
        profile="builtin:mai-ui-2b-mlx-6bit@1",
        max_long_side=1368,
    )
    _remember_text_entry_anchor(
        args,
        {"normalized": {"x": 0.25, "y": 0.09}},
        source="last-tap",
        action_type="tap",
        target="搜索框",
    )
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "测试文本",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "visual_paste_verified"
    assert captured["pasteAt"] == {"x": 0.25, "y": 0.09, "source": "last-tap"}
    assert captured["action"] == {**action, "replace": False}
    assert captured["context"]["anchor"]["source"] == "last-tap"


def test_step_type_text_uses_recent_tap_anchor_for_shifted_ascii(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _is_unshifted_virtual_keyboard_text, _remember_text_entry_anchor

    assert _is_unshifted_virtual_keyboard_text("example.com/path") is True
    assert _is_unshifted_virtual_keyboard_text("openai") is True
    assert _is_unshifted_virtual_keyboard_text("Safari") is True
    assert _is_unshifted_virtual_keyboard_text("OpenAI") is True
    assert _is_unshifted_virtual_keyboard_text("openai codex") is False
    assert _is_unshifted_virtual_keyboard_text("https://example.com") is False

    captured: dict[str, object] = {}

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    def fake_visual(args: argparse.Namespace, action: dict, before: dict, run_dir: Path, *, context: dict, paste_at: object) -> dict:
        captured["action"] = action
        captured["context"] = context
        captured["pasteAt"] = paste_at
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "typeText",
            "strategy": "visual_paste_verified",
            "typeResult": {"inputMethod": "coredevice-pasteboard-visual-menu"},
        }

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    monkeypatch.setattr(coretap.cli, "_execute_step_type_text_visual", fake_visual)
    args = argparse.Namespace(
        backend="device",
        device="device-udid",
        dry_run=False,
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        no_ocr=False,
        profile="builtin:mai-ui-2b-mlx-6bit@1",
        max_long_side=1368,
    )
    _remember_text_entry_anchor(
        args,
        {"normalized": {"x": 0.5, "y": 0.54}},
        source="last-tap",
        action_type="tap",
        target="the Safari bottom address or search field",
    )
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "https://example.com",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "visual_paste_verified"
    assert captured["pasteAt"] == {"x": 0.5, "y": 0.54, "source": "last-tap"}
    assert captured["action"] == {**action, "replace": False}
    assert captured["context"]["anchor"]["source"] == "last-tap"


def test_step_type_text_uses_keyboard_for_titlecase_ascii(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _remember_text_entry_anchor

    captured: dict[str, object] = {}

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        captured["focus"] = {"point": point, "reason": reason}
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["text"] = args.text
        captured["replace"] = args.replace
        captured["pasteAt"] = args.paste_at
        return {"inputMethod": "coredevice-virtual-keyboard", "pasteboardSet": False}

    def fail_visual(*_args: object, **_kwargs: object) -> dict:
        raise AssertionError("titlecase ASCII should use CoreDevice keyboard, not visual paste")

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)
    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    monkeypatch.setattr(coretap.cli, "_execute_step_type_text_visual", fail_visual)
    args = argparse.Namespace(
        backend="device",
        device="device-udid",
        dry_run=False,
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        no_ocr=False,
        profile="builtin:mai-ui-2b-mlx-6bit@1",
        max_long_side=1368,
    )
    _remember_text_entry_anchor(
        args,
        {"normalized": {"x": 0.45, "y": 0.931}},
        source="last-tap",
        action_type="tap",
        target="the Settings search field labeled 搜索",
    )
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "Safari",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": True,
    }

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "coredevice_hid_keyboard"
    assert result["textEntryContext"]["anchor"]["source"] == "last-tap"
    assert captured["focus"] == {"point": {"x": 0.45, "y": 0.931}, "reason": "focus-text-field-before-keyboard-input"}
    assert captured["text"] == "Safari"
    assert captured["replace"] is False
    assert captured["pasteAt"] is None


def test_step_type_text_fails_for_non_ascii_without_anchor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    def fake_ensure_state() -> dict[str, Path]:
        state = tmp_path / "state"
        state.mkdir(parents=True, exist_ok=True)
        return {"state": state}

    monkeypatch.setattr(coretap.cli, "ensure_state", fake_ensure_state)
    args = argparse.Namespace(backend="device", device="device-udid", dry_run=False)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "测试文本",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }

    with pytest.raises(CoretapError) as exc:
        _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert exc.value.code == "TEXT_INPUT_TARGET_UNKNOWN"


def test_step_type_text_preserves_explicit_paste_at_without_ocr_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        captured["focusPoint"] = point
        captured["focusReason"] = reason
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "hello",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": "0.5,0.535",
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("最佳搜索结果", 50.0, 10, 40, 58, 10, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["source"] == "vlm-first"
    assert captured["pasteAt"] == "0.5,0.535"
    assert captured["focusPoint"] == {"x": 0.5, "y": 0.535}
    assert captured["focusReason"] == "focus-text-field-before-keyboard-input"


def test_step_type_text_preserves_structured_explicit_paste_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        captured["focusPoint"] = point
        captured["focusReason"] = reason
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)

    action = _normalize_step_action(
        {
            "schema": "coretap.action.v2",
            "type": "typeText",
            "text": "hello",
            "pasteAt": {"x": 0.5, "y": 0.535, "source": "explicit"},
        }
    )
    result = _execute_step_action(argparse.Namespace(dry_run=False), action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert captured["pasteAt"] == {"x": 0.5, "y": 0.535, "source": "explicit"}
    assert captured["focusPoint"] == {"x": 0.5, "y": 0.535}
    assert captured["focusReason"] == "focus-text-field-before-keyboard-input"


def test_step_tap_uses_vlm_for_search_field_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    captured = {}
    calls = {"ground": 0}

    class FakeBackend:
        def tap_normalized(self, device: str, x: float, y: float, **kwargs: object) -> dict:
            captured["tap"] = {"device": device, "x": x, "y": y, "kwargs": kwargs}
            return {"attempted": True, "dryRun": False}

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        calls["ground"] += 1
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 316, "y": 684}, "normalized": {"x": 0.5, "y": 0.5}},
            "frame": {"widthPx": 632, "heightPx": 1368},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "prepare_grounding_image", lambda source_image, *, output_dir, max_long_side=1368: {"path": str(source_image), "widthPx": 632, "heightPx": 1368, "resized": False, "maxLongSidePx": max_long_side, "scale": 1.0})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = {"frame": {"path": str(source), "widthPx": 632, "heightPx": 1368}, "sourceFrame": {"path": str(source), "widthPx": 632, "heightPx": 1368}}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=512,
        no_refine=True,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "the App Store search field"}, before, tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "vlm_grounding"
    assert captured["tap"]["x"] == pytest.approx(0.5)
    assert calls["ground"] == 1


def test_step_tap_defaults_to_two_step_refinement(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE
    from PIL import Image

    captured = {}
    calls: list[str] = []

    class FakeBackend:
        def tap_normalized(self, device: str, x: float, y: float, **kwargs: object) -> dict:
            captured["tap"] = {"device": device, "x": x, "y": y, "kwargs": kwargs}
            return {"attempted": True, "dryRun": False}

    def fake_ground_target(image: Path, *_args: object, **_kwargs: object) -> dict:
        calls.append(Path(image).name)
        if len(calls) == 1:
            return {
                "schema": "coretap.ground.result.v1",
                "status": "found",
                "point": {"framePx": {"x": 500, "y": 600}, "normalized": {"x": 0.5, "y": 0.5}},
                "frame": {"widthPx": 1000, "heightPx": 1200},
            }
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 114, "y": 228}, "normalized": {"x": 0.25, "y": 0.5}},
            "frame": {"widthPx": 456, "heightPx": 456},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    Image.new("RGB", (1000, 1200), color=(255, 255, 255)).save(source)
    before = {"frame": {"path": str(source), "widthPx": 1000, "heightPx": 1200}, "sourceFrame": {"path": str(source), "widthPx": 1000, "heightPx": 1200}}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=1200,
        no_refine=False,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "Search"}, before, tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "vlm_grounding_refined"
    assert result["grounding"]["source"] == "refined"
    assert result["point"]["normalized"]["x"] == pytest.approx(0.386)
    assert result["point"]["normalized"]["y"] == pytest.approx(0.5)
    assert captured["tap"]["x"] == pytest.approx(0.386)
    assert calls == ["source.model-input.png", "step-grounding-refine-model-input.png"]
    assert (tmp_path / "step-grounding-coarse.json").exists()
    assert (tmp_path / "step-grounding-refined.json").exists()
    assert (tmp_path / "step-grounding-final.json").exists()


def test_step_tap_refinement_falls_back_to_coarse_when_refined_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE
    from PIL import Image

    captured = {}
    calls = {"ground": 0}

    class FakeBackend:
        def tap_normalized(self, _device: str, x: float, y: float, **_kwargs: object) -> dict:
            captured["tap"] = {"x": x, "y": y}
            return {"attempted": True, "dryRun": False}

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        calls["ground"] += 1
        if calls["ground"] == 1:
            return {
                "schema": "coretap.ground.result.v1",
                "status": "found",
                "point": {"framePx": {"x": 500, "y": 600}, "normalized": {"x": 0.5, "y": 0.5}},
                "frame": {"widthPx": 1000, "heightPx": 1200},
            }
        return {"schema": "coretap.ground.result.v1", "status": "not_found", "target": {"description": "Search"}}

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    Image.new("RGB", (1000, 1200), color=(255, 255, 255)).save(source)
    before = {"frame": {"path": str(source), "widthPx": 1000, "heightPx": 1200}, "sourceFrame": {"path": str(source), "widthPx": 1000, "heightPx": 1200}}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=1200,
        no_refine=False,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "Search"}, before, tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "vlm_grounding_coarse_fallback"
    assert result["grounding"]["source"] == "coarse_fallback"
    assert captured["tap"] == pytest.approx({"x": 0.5, "y": 0.5})
    assert calls["ground"] == 2


def test_step_tap_uses_requested_max_long_side_for_model_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    captured = {}

    class FakeBackend:
        def tap_normalized(self, *_args: object, **_kwargs: object) -> dict:
            return {"attempted": False, "dryRun": True}

    def fake_prepare_grounding_image(source_image: Path, *, output_dir: Path, max_long_side: int = 1368) -> dict:
        captured["maxLongSide"] = max_long_side
        return {
            "path": str(source_image),
            "widthPx": 632,
            "heightPx": 1368,
            "resized": True,
            "maxLongSidePx": max_long_side,
            "scale": 0.5,
        }

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 316, "y": 684}, "normalized": {"x": 0.5, "y": 0.5}},
            "frame": {"widthPx": 632, "heightPx": 1368},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "prepare_grounding_image", fake_prepare_grounding_image)
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())

    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = {
        "frame": {"path": str(source), "widthPx": 632, "heightPx": 1368, "sha256": "before"},
        "sourceFrame": {"path": str(source), "widthPx": 1125, "heightPx": 2436, "sha256": "source"},
    }
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=512,
        no_refine=True,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "Search"}, before, tmp_path)

    assert result["status"] == "executed"
    assert captured["maxLongSide"] == 512
    coarse = json.loads((tmp_path / "step-grounding-coarse.json").read_text(encoding="utf-8"))
    assert coarse["modelInput"]["maxLongSidePx"] == 512


def test_step_tap_top_search_field_target_records_active_text_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    class FakeBackend:
        def tap_normalized(self, _device: str, _x: float, _y: float, **_kwargs: object) -> dict:
            return {"attempted": True, "dryRun": True}

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 284, "y": 211}, "normalized": {"x": 0.45, "y": 0.154}},
            "frame": {"widthPx": 632, "heightPx": 1368},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "prepare_grounding_image", lambda source_image, *, output_dir, max_long_side=1368: {"path": str(source_image), "widthPx": 632, "heightPx": 1368, "resized": False, "maxLongSidePx": max_long_side, "scale": 1.0})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = _observe_with_tokens([])
    before["frame"]["path"] = str(source)
    before["sourceFrame"] = {"path": str(source), "widthPx": 632, "heightPx": 1368}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=1368,
        no_refine=True,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(
        args,
        {
            "schema": "coretap.action.v2",
            "type": "tap",
            "target": "App Store search field at the top",
        },
        before,
        tmp_path,
    )

    assert result["point"]["normalized"] == pytest.approx({"x": 0.45, "y": 0.154})
    assert result["textEntryAnchor"]["point"] == pytest.approx({"x": 0.5, "y": 0.09})


def test_step_tap_search_suggestion_row_does_not_record_text_entry_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    class FakeBackend:
        def tap_normalized(self, _device: str, _x: float, _y: float, **_kwargs: object) -> dict:
            return {"attempted": True, "dryRun": True}

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 115, "y": 216}, "normalized": {"x": 0.182, "y": 0.158}},
            "frame": {"widthPx": 632, "heightPx": 1368},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "prepare_grounding_image", lambda source_image, *, output_dir, max_long_side=1368: {"path": str(source_image), "widthPx": 632, "heightPx": 1368, "resized": False, "maxLongSidePx": max_long_side, "scale": 1.0})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = _observe_with_tokens([])
    before["frame"]["path"] = str(source)
    before["sourceFrame"] = {"path": str(source), "widthPx": 632, "heightPx": 1368}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=1368,
        no_refine=True,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(
        args,
        {
            "schema": "coretap.action.v2",
            "type": "tap",
            "target": "the first App Store search suggestion row for 小红书, not the search text field",
        },
        before,
        tmp_path,
    )

    assert result["status"] == "executed"
    assert result["point"]["normalized"] == pytest.approx({"x": 0.182, "y": 0.158})
    assert result["textEntryAnchor"] is None


def test_step_tap_bottom_visible_search_field_records_keyboard_adjacent_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    class FakeBackend:
        def tap_normalized(self, _device: str, _x: float, _y: float, **_kwargs: object) -> dict:
            return {"attempted": True, "dryRun": True}

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 284, "y": 1275}, "normalized": {"x": 0.45, "y": 0.932}},
            "frame": {"widthPx": 632, "heightPx": 1368},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(coretap.cli, "prepare_grounding_image", lambda source_image, *, output_dir, max_long_side=1368: {"path": str(source_image), "widthPx": 632, "heightPx": 1368, "resized": False, "maxLongSidePx": max_long_side, "scale": 1.0})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = _observe_with_tokens([])
    before["frame"]["path"] = str(source)
    before["sourceFrame"] = {"path": str(source), "widthPx": 632, "heightPx": 1368}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=1368,
        no_refine=True,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(
        args,
        {
            "schema": "coretap.action.v2",
            "type": "tap",
            "target": "the search field at the top of the iOS Settings app",
        },
        before,
        tmp_path,
    )

    assert result["point"]["normalized"] == pytest.approx({"x": 0.45, "y": 0.932})
    assert result["textEntryAnchor"]["point"] == pytest.approx({"x": 0.5, "y": 0.54})


def test_step_tap_recovers_once_when_model_worker_crashes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    calls = {"ground": 0, "warm": 0}

    class FakeBackend:
        def tap_normalized(self, *_args: object, **_kwargs: object) -> dict:
            return {"attempted": True, "dryRun": True}

    def fake_warm_model(*_args: object, **_kwargs: object) -> dict:
        calls["warm"] += 1
        return {}

    def fake_ground_target(*_args: object, **_kwargs: object) -> dict:
        calls["ground"] += 1
        if calls["ground"] == 1:
            raise CoretapError(
                "MODEL_WORKER_CRASHED",
                "worker exited",
                category="model",
                stage="model-worker",
                retryable=True,
            )
        return {
            "schema": "coretap.ground.result.v1",
            "status": "found",
            "point": {"framePx": {"x": 316, "y": 684}, "normalized": {"x": 0.5, "y": 0.5}},
            "frame": {"widthPx": 632, "heightPx": 1368},
        }

    monkeypatch.setattr(coretap.cli, "warm_model", fake_warm_model)
    monkeypatch.setattr(coretap.cli, "prepare_grounding_image", lambda source_image, *, output_dir, max_long_side=1368: {"path": str(source_image), "widthPx": 632, "heightPx": 1368, "resized": False, "maxLongSidePx": max_long_side, "scale": 1.0})
    monkeypatch.setattr(coretap.cli, "ground_target", fake_ground_target)
    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = {"frame": {"path": str(source), "widthPx": 632, "heightPx": 1368}, "sourceFrame": {"path": str(source), "widthPx": 632, "heightPx": 1368}}
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=True,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=512,
        no_refine=True,
        refine_crop_ratio=0.38,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "Search"}, before, tmp_path)

    assert result["status"] == "executed"
    assert calls == {"ground": 2, "warm": 2}
    assert result["grounding"]["modelRecovery"]["recoveredFrom"] == "MODEL_WORKER_CRASHED"


def test_step_tap_point_executes_explicit_coordinate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    class FakeBackend:
        def tap_normalized(self, device: str, x: float, y: float, **kwargs: object) -> dict:
            captured["tap"] = {"device": device, "x": x, "y": y, "kwargs": kwargs}
            return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    before = _observe_with_tokens([])
    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "tapPoint", "x": 0.25, "y": 0.5})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "explicit_point"
    assert captured["tap"]["x"] == pytest.approx(0.25)
    assert captured["tap"]["kwargs"]["hid_u16"] == {"x": 16384, "y": 32768}


def test_step_long_press_uses_same_point_drag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    class FakeBackend:
        def drag_normalized(self, device: str, start_x: float, start_y: float, end_x: float, end_y: float, **kwargs: object) -> dict:
            captured["drag"] = {"device": device, "from": (start_x, start_y), "to": (end_x, end_y), "kwargs": kwargs}
            return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    before = _observe_with_tokens([])
    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "longPress", "x": 0.4, "y": 0.6, "durationMs": 1500, "steps": 16})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "explicit_point_hold"
    assert captured["drag"]["from"] == pytest.approx((0.4, 0.6))
    assert captured["drag"]["to"] == pytest.approx((0.4, 0.6))
    assert captured["drag"]["kwargs"]["duration_ms"] == 1500
    assert captured["drag"]["kwargs"]["steps"] == 16


def test_step_terminate_app_executes_bundle_signal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    class FakeBackend:
        def terminate_app(self, device: str, bundle_id: str, **kwargs: object) -> dict:
            captured["terminate"] = {"device": device, "bundleId": bundle_id, "kwargs": kwargs}
            return {"attempted": True, "dryRun": False, "status": "terminated"}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "terminateApp", "bundleId": "com.apple.AppStore"})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "bundle_process_signal"
    assert captured["terminate"] == {"device": "device-udid", "bundleId": "com.apple.AppStore", "kwargs": {"signal": 9, "dry_run": False}}


def test_step_uninstall_app_executes_bundle_uninstall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    class FakeBackend:
        def uninstall_app(self, device: str, bundle_id: str, **kwargs: object) -> dict:
            captured["uninstall"] = {"device": device, "bundleId": bundle_id, "kwargs": kwargs}
            return {"attempted": True, "dryRun": False, "status": "uninstalled"}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "uninstallApp", "name": "小红书"})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "bundle_uninstall"
    assert result["bundleId"] == "com.xingin.discover"
    assert captured["uninstall"] == {
        "device": "device-udid",
        "bundleId": "com.xingin.discover",
        "kwargs": {"ignore_missing": True, "dry_run": False},
    }


def test_step_open_url_executes_web_navigation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    class FakeBackend:
        def open_url(self, device: str, url: str, **kwargs: object) -> dict:
            captured["openUrl"] = {"device": device, "url": url, "kwargs": kwargs}
            return {"attempted": True, "dryRun": False, "strategy": "webinspector-launch", "url": url}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "openUrl", "url": "https://example.com", "timeoutSec": 5})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "webinspector-launch"
    assert result["url"] == "https://example.com"
    assert captured["openUrl"] == {
        "device": "device-udid",
        "url": "https://example.com",
        "kwargs": {"timeout_sec": 5.0, "dry_run": False},
    }


def test_command_step_skips_before_observation_for_uninstall(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import coretap.cli
    from coretap.cli import command_step

    class FakeBackend:
        def uninstall_app(self, *_args: object, **_kwargs: object) -> dict:
            return {"attempted": True, "dryRun": False, "status": "uninstalled"}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    monkeypatch.setattr(coretap.cli, "_observe_into", lambda *_args, **_kwargs: pytest.fail("uninstallApp should not capture a screenshot"))
    args = argparse.Namespace(
        artifact_root=str(tmp_path),
        action='{"type":"uninstallApp","bundleId":"com.xingin.discover"}',
        action_file=None,
        expect_text=[],
        expect_no_text=[],
        expect_change=False,
        no_ocr=False,
        after_policy="never",
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        dry_run=False,
    )

    result = command_step(args)

    assert result["status"] == "executed"
    assert result["before"]["skipped"] is True
    assert result["execution"]["actionType"] == "uninstallApp"


def test_step_app_switcher_executes_named_drag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    captured = {}

    class FakeBackend:
        def drag_normalized(self, device: str, start_x: float, start_y: float, end_x: float, end_y: float, **kwargs: object) -> dict:
            captured["drag"] = {"device": device, "from": (start_x, start_y), "to": (end_x, end_y), "kwargs": kwargs}
            return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "appSwitcher"})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "home_indicator_up_and_hold"
    assert captured["drag"]["from"] == pytest.approx((0.5, 0.98))
    assert captured["drag"]["to"] == pytest.approx((0.5, 0.45))


def test_step_open_app_uses_spotlight_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    calls: list[tuple[str, object]] = []

    def fake_command_press(args: argparse.Namespace) -> dict:
        calls.append(("press", args.button))
        return {"button": args.button}

    def fake_command_wait(args: argparse.Namespace) -> dict:
        calls.append(("wait", args.ms))
        return {"waitedMs": args.ms}

    def fake_observe_into(args: argparse.Namespace, *, run_dir: Path, label: str, no_ocr: bool = False, no_vlm: bool = False) -> dict:
        calls.append(("observe", label))
        return _observe_with_tokens([OcrToken("搜索", 90.0, 100, 100, 80, 20, "vision")])

    def fake_execute_step_tap(args: argparse.Namespace, action: dict, before: dict, run_dir: Path) -> dict:
        calls.append(("tap", action["target"]))
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": "tap", "target": action["target"]}

    def fake_command_type(args: argparse.Namespace) -> dict:
        calls.append(("type", (args.text, args.replace, args.paste_at)))
        return {"attempted": True, "dryRun": False}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        calls.append(("focus", (point, reason)))
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    monkeypatch.setattr(coretap.cli, "command_press", fake_command_press)
    monkeypatch.setattr(coretap.cli, "command_wait", fake_command_wait)
    monkeypatch.setattr(coretap.cli, "_observe_into", fake_observe_into)
    monkeypatch.setattr(coretap.cli, "_execute_step_tap", fake_execute_step_tap)
    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)

    action = {
        "schema": "coretap.action.v2",
        "type": "openApp",
        "name": "sampleapp",
        "searchTarget": "the Search button at the bottom center of the iOS home screen",
    }
    before = _observe_with_tokens([])

    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)
    result = _execute_step_action(args, action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["actionType"] == "openApp"
    assert ("press", "home") in calls
    assert ("tap", "the Search button at the bottom center of the iOS home screen") in calls
    assert ("focus", ({"x": 0.5, "y": 0.925}, "focus-text-field-before-keyboard-input")) in calls
    assert ("type", ("sampleapp", False, "0.5,0.925")) in calls
    assert ("tap", "the sampleapp app icon in Spotlight search results") in calls
    assert ("observe", "open-app-after-launch") in calls


def test_step_open_app_prefers_builtin_bundle_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def launch_app(self, device: str, bundle_id: str, **kwargs: object) -> dict:
            calls.append(("launch", (device, bundle_id, kwargs)))
            return {"attempted": True, "dryRun": False, "pid": 123}

    def fake_observe_into(args: argparse.Namespace, *, run_dir: Path, label: str, no_ocr: bool = False, no_vlm: bool = False) -> dict:
        calls.append(("observe", label))
        return _observe_with_tokens([])

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    monkeypatch.setattr(coretap.cli, "command_wait", lambda args: calls.append(("wait", args.ms)) or {"waitedMs": args.ms})
    monkeypatch.setattr(coretap.cli, "_observe_into", fake_observe_into)

    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "openApp", "name": "App Store"})
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert action["bundleId"] == "com.apple.AppStore"
    assert result["status"] == "executed"
    assert result["strategy"] == "bundle-launch"
    assert calls == [
        ("launch", ("device-udid", "com.apple.AppStore", {"kill_existing": True, "dry_run": False})),
        ("wait", 800),
        ("observe", "open-app-after-launch"),
    ]


def test_step_open_app_bundle_strategy_blocks_when_coredevice_launch_has_no_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action, _normalize_step_action

    class FakeBackend:
        def launch_app(self, device: str, bundle_id: str, **kwargs: object) -> dict:
            return {
                "attempted": True,
                "dryRun": False,
                "backend": "device",
                "strategy": "coredevice-dvt-launch",
                "pid": None,
                "stdout": "",
            }

    monkeypatch.setattr(coretap.cli, "backend_for", lambda *_args, **_kwargs: FakeBackend())
    monkeypatch.setattr(coretap.cli, "command_wait", lambda _args: pytest.fail("pid-less bundle launch should not wait as success"))
    monkeypatch.setattr(coretap.cli, "_observe_into", lambda *_args, **_kwargs: pytest.fail("pid-less bundle launch should not observe as success"))

    action = _normalize_step_action(
        {"schema": "coretap.action.v2", "type": "openApp", "name": "小红书", "bundleId": "com.xingin.discover", "strategy": "bundle"}
    )
    args = argparse.Namespace(backend="device", device="device-udid", developer_dir=None, coredevice_tunnel_mode="userspace", dry_run=False)

    result = _execute_step_action(args, action, _observe_with_tokens([]), tmp_path)

    assert result["status"] == "blocked"
    assert result["code"] == "APP_LAUNCH_NOT_CONFIRMED"
    assert result["details"]["launch"]["pid"] is None


@pytest.mark.parametrize(
    ("name", "bundle_id"),
    [
        ("Safari", "com.apple.mobilesafari"),
        ("Safari浏览器", "com.apple.mobilesafari"),
        ("Settings", "com.apple.Preferences"),
        ("设置", "com.apple.Preferences"),
        ("小红书", "com.xingin.discover"),
    ],
)
def test_step_open_app_resolves_builtin_app_bundle_aliases(name: str, bundle_id: str) -> None:
    from coretap.cli import _normalize_step_action

    action = _normalize_step_action({"schema": "coretap.action.v2", "type": "openApp", "name": name})

    assert action["bundleId"] == bundle_id


def _step_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "action": '{"type":"wait","ms":1}',
        "action_file": None,
        "artifact_root": None,
        "keep_artifacts": False,
        "no_artifacts": False,
        "trace_id": None,
        "no_ocr": False,
        "no_vlm": False,
        "dry_run": False,
        "page_wait_ms": 0,
        "no_page": True,
        "min_confidence": 0.0,
        "max_long_side": 512,
        "no_refine": False,
        "refine_crop_ratio": 0.38,
        "full_size": False,
        "backend": "device",
        "device": "device-udid",
        "developer_dir": None,
        "coredevice_tunnel_mode": "userspace",
        "profile": "internal:test-fixture-grounder",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_step_wait_skips_page_observation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_step

    calls: list[tuple[str, object]] = []

    def fake_observe_into(args: argparse.Namespace, *, run_dir: Path, label: str, no_ocr: bool = False, no_vlm: bool = False) -> dict:
        calls.append(("observe", (label, no_ocr)))
        return _observe_with_tokens([])

    def fake_execute(args: argparse.Namespace, action: dict, before: dict, run_dir: Path) -> dict:
        calls.append(("execute", action["type"]))
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": action["type"]}

    monkeypatch.setattr(coretap.cli, "artifact_dir", lambda _root=None: tmp_path)
    monkeypatch.setattr(coretap.cli, "_observe_into", fake_observe_into)
    monkeypatch.setattr(coretap.cli, "_execute_step_action", fake_execute)
    monkeypatch.setattr(coretap.cli, "command_wait", lambda args: calls.append(("wait", args.ms)) or {"waitedMs": args.ms})

    result = command_step(_step_args())

    assert result["status"] == "executed"
    assert result["before"]["skipped"] is True
    assert calls == [("execute", "wait")]


def test_step_ui_action_attaches_default_page_observation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import command_step

    calls: list[tuple[str, bool, bool]] = []

    def fake_observe_into(args: argparse.Namespace, *, run_dir: Path, label: str, no_ocr: bool = False, no_vlm: bool = False) -> dict:
        calls.append((label, no_ocr, no_vlm))
        if label == "step-page":
            return _observe_with_tokens([OcrToken("搜索", 90.0, 100, 100, 80, 30, "vision")])
        return _observe_with_tokens([])

    monkeypatch.setattr(coretap.cli, "artifact_dir", lambda _root=None: tmp_path)
    monkeypatch.setattr(coretap.cli, "_observe_into", fake_observe_into)
    monkeypatch.setattr(coretap.cli, "command_press", lambda args: {"button": args.button, "attempted": True})
    monkeypatch.setattr(coretap.cli, "command_wait", lambda args: {"waitedMs": args.ms})

    result = command_step(
        _step_args(
            action='{"type":"press","button":"home"}',
            no_page=False,
            page_wait_ms=0,
        )
    )

    assert result["status"] == "executed"
    assert result["observation"]["status"] == "ready"
    assert result["observation"]["elements"][0]["label"] == "搜索"
    assert calls == [("step-before", True, True), ("step-page", False, False)]


def test_page_observation_summary_groups_generic_ui_candidates() -> None:
    from coretap.cli import page_observation_summary

    observation = _observe_with_tokens(
        [
            OcrToken("Q 小红书", 90.0, 50, 106, 130, 32, "vision"),
            OcrToken("小红书", 90.0, 100, 210, 80, 30, "vision"),
            OcrToken("Xingin", 90.0, 100, 252, 90, 24, "vision"),
            OcrToken("获取", 90.0, 520, 225, 60, 30, "vision"),
        ]
    )

    summary = page_observation_summary(observation)

    assert summary["status"] == "ready"
    assert summary["groups"]["inputs"][0]["label"] == "Q 小红书"
    assert summary["groups"]["buttons"][0]["label"] == "获取"
    assert any("小红书" in row["label"] and "Xingin" in row["label"] for row in summary["groups"]["rows"])
    assert summary["groups"]["textLines"][0]["type"] == "textLine"


def test_page_observation_summary_merges_visual_inventory() -> None:
    from coretap.cli import page_observation_summary

    observation = _observe_with_tokens(
        [OcrToken("搜索", 90.0, 100, 100, 80, 30, "vision")],
        visual={
            "enabled": True,
            "summary": "Home screen with icon-only controls",
            "elements": [
                {
                    "type": "visual",
                    "source": "vlm",
                    "label": "ChatGPT app icon",
                    "role": "appIcon",
                    "confidence": 0.8,
                    "center": {"x": 0.8, "y": 0.5},
                    "bbox": {"x": 0.72, "y": 0.44, "width": 0.16, "height": 0.12},
                },
                {
                    "type": "visual",
                    "source": "vlm",
                    "label": "Search tab icon",
                    "role": "tab",
                    "confidence": 0.7,
                    "center": {"x": 0.5, "y": 0.92},
                    "bbox": {"x": 0.45, "y": 0.88, "width": 0.1, "height": 0.08},
                },
                {
                    "type": "visual",
                    "source": "vlm",
                    "label": "Cloud download button",
                    "role": "button",
                    "confidence": 0.74,
                    "center": {"x": 0.88, "y": 0.31},
                    "bbox": {"x": 0.83, "y": 0.28, "width": 0.1, "height": 0.06},
                },
            ],
        },
    )

    summary = page_observation_summary(observation)

    assert summary["status"] == "ready"
    assert summary["visualCount"] == 3
    assert summary["textCount"] == 1
    assert summary["elements"][0]["source"] == "ocr"
    assert summary["elements"][1]["source"] == "vlm"
    assert summary["groups"]["appIcons"][0]["label"] == "ChatGPT app icon"
    assert summary["groups"]["tabs"][0]["label"] == "Search tab icon"
    assert summary["groups"]["visualButtons"][0]["label"] == "Cloud download button"


def test_type_text_verification_accepts_wrapped_url_near_safari_overlay() -> None:
    from coretap.cli import _verify_text_input_near_anchor

    observation = _observe_with_tokens(
        [
            OcrToken("https://www.bing.com/search？", 50.0, 108, 148, 420, 34, "vision"),
            OcrToken("q=openai%20codex", 50.0, 108, 188, 260, 34, "vision"),
        ]
    )

    result = _verify_text_input_near_anchor(
        observation,
        "https://www.bing.com/search?q=openai%20codex",
        {"x": 0.499, "y": 0.927},
    )

    assert result["status"] == "verified"
    assert result["match"]["matchedKind"] == "compact"


def test_type_text_verification_accepts_bottom_field_text_in_top_overlay() -> None:
    from coretap.cli import _verify_text_input_near_anchor

    observation = _observe_with_tokens([OcrToken("Q openai codex", 50.0, 60, 148, 222, 32, "vision")])

    result = _verify_text_input_near_anchor(observation, "openai codex", {"x": 0.325, "y": 0.93})

    assert result["status"] == "verified"
    assert result["match"]["matchedText"] == "Q openai codex"


def test_type_text_verification_prefers_ui_prefix_match_near_anchor_over_exact_elsewhere() -> None:
    from coretap.cli import _verify_text_input_near_anchor

    observation = _observe_with_tokens(
        [
            OcrToken("蓝牙", 100.0, 120, 172, 60, 32, "vision"),
            OcrToken("Q蓝牙", 50.0, 34, 714, 108, 34, "vision"),
        ]
    )

    result = _verify_text_input_near_anchor(observation, "蓝牙", {"x": 0.5, "y": 0.54})

    assert result["status"] == "verified"
    assert result["match"]["matchedText"] == "Q蓝牙"
    assert result["match"]["exactMatchStrategy"] == "ui-prefix-stripped"


def test_search_result_row_target_does_not_create_text_entry_anchor() -> None:
    from coretap.cli import _target_suggests_text_entry

    assert not _target_suggests_text_entry("the first Settings search result row labeled 蓝牙 below the search field")
    assert not _target_suggests_text_entry("the first App Store search suggestion row exactly 小红书 below the search field")
    assert not _target_suggests_text_entry("the first App Store search suggestion row for 小红书, not the search text field")
    assert _target_suggests_text_entry("the search field at the top of the iOS Settings app")
