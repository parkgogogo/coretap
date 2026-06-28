from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from coretap.daemon import handle_argv
from coretap.ocr import DEFAULT_OCR_LANG, OcrToken
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

    assert data["schema"] == "coretap.response.v1"
    assert data["ok"] is True
    assert data["requestId"].startswith("req_")
    assert data["result"]["version"] == "0.1.0"


def test_json_is_default_output_format() -> None:
    from coretap.cli import build_parser

    args = build_parser().parse_args(["status"])

    assert args.format == "json"


def test_model_status_json_envelope() -> None:
    data = run_coretap("model", "status")

    assert data["ok"] is True
    assert data["result"]["profile"] == "builtin:mai-ui-2b-mlx-6bit@1"


def test_internal_fixture_profile_is_not_default() -> None:
    data = run_coretap("--profile", "internal:test-fixture-grounder", "model", "status")

    assert data["ok"] is True
    assert data["result"]["implementation"] == "internal-ocr-fixture-grounder"


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
        main(["status"])

    assert exc.value.code == 0
    assert starts == [{"socket_path": None, "timeout": 5.0}]
    assert [call["argv"] for call in calls] == [["status"], ["status"]]
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


def test_text_ocr_commands_default_to_chinese_and_english() -> None:
    from coretap.cli import build_parser, normalize_global_args

    parser = build_parser()

    assert_text = parser.parse_args(normalize_global_args(["assert", "text", "--text", "搜索"]))
    assert assert_text.lang == DEFAULT_OCR_LANG

    wait_text = parser.parse_args(normalize_global_args(["wait", "text", "--text", "搜索"]))
    assert wait_text.lang == DEFAULT_OCR_LANG


def test_step_parser_accepts_single_action_runtime_options() -> None:
    from coretap.cli import build_parser, normalize_global_args

    action = '{"schema":"coretap.action.v2","type":"tap","target":"Search"}'
    args = build_parser().parse_args(
        normalize_global_args(
            [
                "step",
                "--action",
                action,
                "--post-wait-ms",
                "500",
                "--post-timeout-ms",
                "1500",
                "--expect-text",
                "搜索",
                "--expect-change",
            ]
        )
    )

    assert args.command == "step"
    assert args.action == action
    assert args.post_wait_ms == 500
    assert args.post_timeout_ms == 1500
    assert args.expect_text == ["搜索"]
    assert args.expect_change is True
    assert args.max_long_side == 512


def test_step_text_postconditions_get_default_retry_timeout() -> None:
    from coretap.cli import _effective_step_post_timeout_ms

    text_condition = [{"type": "textVisible", "text": "小红书"}]
    change_condition = [{"type": "screenChanged"}]

    assert _effective_step_post_timeout_ms(argparse.Namespace(post_timeout_ms=0), text_condition) == 3000
    assert _effective_step_post_timeout_ms(argparse.Namespace(post_timeout_ms=1500), text_condition) == 1500
    assert _effective_step_post_timeout_ms(argparse.Namespace(post_timeout_ms=0), change_condition) == 0


def test_step_action_schema_accepts_only_mobile_use_actions() -> None:
    from coretap.cli import _load_step_action, _normalize_step_action

    assert _normalize_step_action({"schema": "coretap.action.v2", "type": "tap", "target": "Search"}) == {
        "schema": "coretap.action.v2",
        "type": "tap",
        "target": "Search",
    }
    with pytest.raises(CoretapError) as exc:
        _normalize_step_action({"schema": "coretap.action.v2", "type": "tapPoint", "x": 0.5, "y": 0.5})

    assert exc.value.code == "ACTION_UNSUPPORTED"
    with pytest.raises(CoretapError) as alias_exc:
        _normalize_step_action({"schema": "coretap.action.v2", "type": "type", "text": "hello"})

    assert alias_exc.value.code == "ACTION_UNSUPPORTED"
    with pytest.raises(CoretapError) as schema_exc:
        _load_step_action(argparse.Namespace(action='{"type":"tap","target":"Search"}', action_file=None))

    assert schema_exc.value.code == "ACTION_SCHEMA_INVALID"


@pytest.mark.parametrize(
    "argv",
    [
        ["screenshot"],
        ["tap", "target", "--target", "Search"],
        ["tap", "text", "Search"],
        ["locate", "--target", "Search"],
        ["act", "--goal", "download Xiaohongshu"],
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
        "run_vision_ocr",
        lambda image: ([OcrToken("◎ 搜索", 30.0, 50, 120, 20, 10, "vision")], "[]"),
    )

    result = command_observe(
        argparse.Namespace(
            label="observe",
            out=None,
            max_long_side=1368,
            full_size=False,
            lang=DEFAULT_OCR_LANG,
            psm=11,
            ocr_engine="auto",
            min_confidence=0.0,
            no_ocr=False,
            artifact_root=None,
        )
    )

    assert result["schema"] == "coretap.observe.result.v1"
    assert result["ocr"]["schema"] == "coretap.ocr.page.v1"
    assert result["ocr"]["engineMode"] == "auto"
    assert result["ocr"]["selectedEngine"] == "vision"
    assert result["ocr"]["engines"] == ["vision"]
    assert result["ocr"]["plainText"] == "◎ 搜索"
    assert result["ocr"]["tokens"][0]["text"] == "◎ 搜索"
    assert result["ocr"]["tokens"][0]["normalizedText"] == "◎ 搜索"
    assert result["ocr"]["tokens"][0]["centerPx"] == {"x": 60.0, "y": 125.0}
    assert result["ocr"]["tokens"][0]["normalized"] == {"x": 0.6, "y": 0.625}
    assert (tmp_path / "observe.result.json").exists()


def _observe_with_tokens(tokens: list[OcrToken]) -> dict:
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
    }


def _parse_xy(value: str) -> tuple[float, float]:
    x, y = value.split(",", 1)
    return float(x), float(y)


def test_step_type_text_blocks_when_no_text_entry_context(tmp_path: Path) -> None:
    from coretap.cli import _execute_step_action

    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "こんにちは",
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

    assert result["status"] == "blocked"
    assert result["code"] == "TEXT_INPUT_TARGET_UNKNOWN"
    assert result["details"]["textEntryContext"]["ready"] is False


def test_step_type_text_uses_search_field_anchor_for_non_ascii_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("游戏、App、故事等", 95.0, 88, 108, 256, 34, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "search-placeholder"
    assert captured["pasteAt"] is None
    x, y = captured["focusPoint"]["x"], captured["focusPoint"]["y"]
    assert captured["focusReason"] == "focus-search-placeholder"
    assert 0.3 <= x <= 0.4
    assert 0.08 <= y <= 0.1


def test_step_type_text_falls_back_to_source_ocr_for_text_entry_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False, "inputMethod": "coredevice-pinyin-keyboard"}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        captured["focusPoint"] = point
        captured["focusReason"] = reason
        return {"attempted": True, "reason": reason, "point": {"normalized": point}}

    def fake_source_ocr(_image: Path, _args: argparse.Namespace):
        return [OcrToken("Q 小红书", 30.0, 88, 190, 256, 58, "vision")], {"engines": ["vision"], "visionJson": "[]"}, "vision"

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)
    monkeypatch.setattr(coretap.cli, "_run_observe_ocr", fake_source_ocr)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    before = _observe_with_tokens([OcrToken("Q", 30.0, 186, 468, 16, 14, "vision")])
    before["sourceFrame"] = {"path": str(source), "widthPx": 1125, "heightPx": 2436, "sha256": "source"}

    result = _execute_step_action(
        argparse.Namespace(
            dry_run=False,
            lang="chi_sim+eng",
            psm=11,
            ocr_engine="auto",
            min_confidence=0.0,
        ),
        action,
        before,
        tmp_path,
    )

    assert result["status"] == "executed"
    assert result["textEntryContext"]["source"] == "source-ocr"
    assert result["textEntryContext"]["anchor"]["source"] == "active-search-field"
    assert captured["focusReason"] == "focus-active-search-field"
    assert captured["pasteAt"] is None


def test_text_entry_context_detects_active_app_store_search_field() -> None:
    from coretap.cli import _text_entry_context

    before = _observe_with_tokens(
        [
            OcrToken("xiaohongshu小红书", 95.0, 92, 108, 258, 32, "vision"),
            OcrToken("Today", 95.0, 83, 1296, 54, 21, "vision"),
            OcrToken("游戏", 95.0, 194, 1296, 36, 18, "vision"),
            OcrToken("App", 95.0, 298, 1298, 36, 18, "vision"),
            OcrToken("Arcade", 95.0, 386, 1298, 64, 16, "vision"),
            OcrToken("搜索", 95.0, 502, 1296, 36, 20, "vision"),
        ]
    )

    context = _text_entry_context(before)

    assert context["ready"] is True
    assert context["anchor"]["source"] == "active-search-field"
    assert context["anchor"]["pointKind"] == "inferred-active-search-field"
    assert 0.55 <= context["anchor"]["point"]["x"] <= 0.65
    assert 0.08 <= context["anchor"]["point"]["y"] <= 0.1


def test_text_entry_context_detects_top_q_search_query_without_bottom_tabs() -> None:
    from coretap.cli import _text_entry_context

    before = _observe_with_tokens([OcrToken("Q 小红书", 30.0, 48, 106, 132, 33, "vision")])

    context = _text_entry_context(before)

    assert context["ready"] is True
    assert context["anchor"]["source"] == "active-search-field"
    assert context["anchor"]["pointKind"] == "inferred-active-search-field"


def test_text_entry_context_detects_empty_top_q_search_field() -> None:
    from coretap.cli import _text_entry_context

    before = _observe_with_tokens([OcrToken("Q", 30.0, 48, 106, 32, 33, "vision")])

    context = _text_entry_context(before)

    assert context["ready"] is True
    assert context["anchor"]["source"] == "active-search-field"


def test_step_tap_uses_ocr_search_field_fast_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_tap
    from coretap.model_pack import PUBLIC_MODEL_PROFILE

    captured = {}

    def fake_focus(args: argparse.Namespace, point: dict, *, reason: str) -> dict:
        captured["point"] = point
        captured["reason"] = reason
        return {"point": {"normalized": point}, "tap": {"attempted": True, "dryRun": False}}

    monkeypatch.setattr(coretap.cli, "_tap_normalized_for_step", fake_focus)
    monkeypatch.setattr(coretap.cli, "warm_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("VLM should not be used")))
    before = _observe_with_tokens(
        [
            OcrToken("xiaohongshu小红书", 95.0, 92, 108, 258, 32, "vision"),
            OcrToken("Today", 95.0, 83, 1296, 54, 21, "vision"),
            OcrToken("游戏", 95.0, 194, 1296, 36, 18, "vision"),
            OcrToken("App", 95.0, 298, 1298, 36, 18, "vision"),
            OcrToken("Arcade", 95.0, 386, 1298, 64, 16, "vision"),
            OcrToken("搜索", 95.0, 502, 1296, 36, 20, "vision"),
        ]
    )
    args = argparse.Namespace(
        profile=PUBLIC_MODEL_PROFILE,
        dry_run=False,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=512,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "the App Store search field"}, before, tmp_path)

    assert result["status"] == "executed"
    assert result["strategy"] == "ocr_search_field"
    assert result["anchor"]["source"] == "active-search-field"
    assert captured["reason"] == "tap-active-search-field"


def test_step_type_text_infers_spotlight_search_anchor_from_vision_misread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("Q製索", 30.0, 38, 716, 102, 32, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "search-field"
    assert result["textEntryContext"]["anchor"]["pointKind"] == "inferred-search-field-center"
    assert captured["pasteAt"] is None
    x, y = captured["focusPoint"]["x"], captured["focusPoint"]["y"]
    assert captured["focusReason"] == "focus-search-field"
    assert 0.45 <= x <= 0.47
    assert 0.53 <= y <= 0.54


def test_step_type_text_infers_search_anchor_from_q_super_misread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("Q超索", 30.0, 32, 728, 108, 32, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "search-field"
    assert captured["pasteAt"] is None
    x, y = captured["focusPoint"]["x"], captured["focusPoint"]["y"]
    assert captured["focusReason"] == "focus-search-field"
    assert 0.45 <= x <= 0.47
    assert 0.54 <= y <= 0.55


def test_step_type_text_recognizes_bottom_spotlight_search_misread(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("Q 學索", 30.0, 43, 1251, 118, 43, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "search-field"
    assert captured["pasteAt"] is None
    x, y = captured["focusPoint"]["x"], captured["focusPoint"]["y"]
    assert captured["focusReason"] == "focus-search-field"
    assert 0.49 <= x <= 0.5
    assert 0.92 <= y <= 0.94


def test_step_type_text_infers_bottom_spotlight_search_when_ocr_misses_placeholder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens(
        [
            OcrToken("Siri建议", 92.0, 41, 170, 96, 30, "vision"),
            OcrToken("相机", 91.0, 543, 543, 71, 26, "vision"),
            OcrToken("Safari浏処器", 83.0, 289, 539, 142, 28, "vision"),
            OcrToken("更少内容", 90.0, 485, 903, 92, 27, "vision"),
            OcrToken("App Store", 96.0, 66, 1032, 126, 32, "vision"),
        ]
    )

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "spotlight-bottom-search"
    assert result["textEntryContext"]["anchor"]["pointKind"] == "inferred-spotlight-bottom-search"
    assert captured["pasteAt"] is None
    x, y = captured["focusPoint"]["x"], captured["focusPoint"]["y"]
    assert captured["focusReason"] == "focus-spotlight-bottom-search"
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.925)


def test_step_type_text_allows_explicit_paste_at_without_visible_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "こんにちは",
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
    assert result["textEntryContext"]["ready"] is False
    assert captured["pasteAt"] == "0.5,0.535"


def test_text_entry_context_uses_left_biased_visible_paste_menu_point() -> None:
    from coretap.cli import _text_entry_context

    before = _observe_with_tokens([OcrToken("粘贴自动填充", 95.0, 100, 110, 260, 34, "vision")])

    context = _text_entry_context(before)

    assert context["ready"] is True
    assert context["anchor"]["source"] == "edit-menu"
    assert context["anchor"]["pointKind"] == "visible-paste-menu"
    assert context["anchor"]["point"]["x"] < 0.3


def test_text_entry_context_accepts_traditional_paste_label() -> None:
    from coretap.cli import _text_entry_context

    before = _observe_with_tokens([OcrToken("粘貼自动填充", 95.0, 100, 110, 260, 34, "vision")])

    context = _text_entry_context(before)

    assert context["ready"] is True
    assert context["anchor"]["source"] == "edit-menu"
    assert context["anchor"]["point"]["x"] < 0.3


def test_text_entry_context_left_biases_misread_paste_autofill_menu() -> None:
    from coretap.cli import _text_entry_context

    before = _observe_with_tokens([OcrToken("右點：一 动填充", 30.0, 64, 642, 210, 28, "vision")])

    context = _text_entry_context(before)

    assert context["ready"] is True
    assert context["anchor"]["source"] == "edit-menu"
    assert 0.15 <= context["anchor"]["point"]["x"] <= 0.2


def test_step_type_text_taps_visible_paste_menu_directly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "こんにちは",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("粘贴自动填充", 95.0, 100, 110, 260, 34, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "edit-menu"
    assert captured["pasteAt"]["mode"] == "menu"
    assert captured["pasteAt"]["x"] < 0.3


def test_step_type_text_cjk_prefers_visible_paste_menu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import coretap.cli
    from coretap.cli import _execute_step_action

    captured = {}

    def fake_command_type(args: argparse.Namespace) -> dict:
        captured["pasteAt"] = args.paste_at
        return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)
    action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": "小红书",
        "charDelayMs": 40,
        "interDelayMs": 20,
        "pasteAt": None,
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": False,
    }
    before = _observe_with_tokens([OcrToken("粘貼", 50.0, 22, 438, 20, 10, "vision")])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["textEntryContext"]["anchor"]["source"] == "edit-menu"
    assert captured["pasteAt"]["mode"] == "menu"


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
        dry_run=True,
        backend="device",
        device="device-udid",
        developer_dir=None,
        coredevice_tunnel_mode="userspace",
        max_long_side=512,
    )

    result = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": "Search"}, before, tmp_path)

    assert result["status"] == "executed"
    assert captured["maxLongSide"] == 512
    assert result["modelInput"]["maxLongSidePx"] == 512


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

    def fake_observe_into(args: argparse.Namespace, *, run_dir: Path, label: str, no_ocr: bool = False) -> dict:
        calls.append(("observe", label))
        return _observe_with_tokens([OcrToken("搜索", 90.0, 100, 100, 80, 20, "vision")])

    def fake_execute_step_tap(args: argparse.Namespace, action: dict, before: dict, run_dir: Path) -> dict:
        calls.append(("tap", action["target"]))
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": "tap", "target": action["target"]}

    def fake_command_type(args: argparse.Namespace) -> dict:
        calls.append(("type", (args.text, args.replace, args.paste_at)))
        return {"attempted": True, "dryRun": False}

    monkeypatch.setattr(coretap.cli, "command_press", fake_command_press)
    monkeypatch.setattr(coretap.cli, "command_wait", fake_command_wait)
    monkeypatch.setattr(coretap.cli, "_observe_into", fake_observe_into)
    monkeypatch.setattr(coretap.cli, "_execute_step_tap", fake_execute_step_tap)
    monkeypatch.setattr(coretap.cli, "command_type", fake_command_type)

    action = {
        "schema": "coretap.action.v2",
        "type": "openApp",
        "name": "App Store",
        "searchTarget": "the Search button at the bottom center of the iOS home screen",
    }
    before = _observe_with_tokens([])

    result = _execute_step_action(argparse.Namespace(dry_run=False), action, before, tmp_path)

    assert result["status"] == "executed"
    assert result["actionType"] == "openApp"
    assert ("press", "home") in calls
    assert ("tap", "the Search button at the bottom center of the iOS home screen") in calls
    assert ("type", ("App Store", True, "0.5,0.925")) in calls
    assert ("tap", "the large App Store app icon on the left side of the Best Search Result card in Spotlight search results") in calls
    assert ("observe", "open-app-after-launch") in calls


def test_open_app_launch_check_detects_spotlight_results() -> None:
    from coretap.cli import _looks_like_spotlight_results

    observation = _observe_with_tokens(
        [
            OcrToken("最佳搜索结果", 100.0, 8, 40, 62, 10, "vision"),
            OcrToken("App Store", 50.0, 14, 112, 40, 10, "vision"),
        ]
    )

    assert _looks_like_spotlight_results(observation) is True


def test_visible_app_label_anchor_targets_icon_above_label() -> None:
    from coretap.cli import _visible_app_label_anchor

    observation = _observe_with_tokens([OcrToken("App Store", 90.0, 72, 168, 40, 8, "vision")])

    anchor = _visible_app_label_anchor(observation, "App Store")

    assert anchor is not None
    assert anchor["source"] == "ocr-label"
    assert anchor["point"]["x"] == pytest.approx(92 / 632)
    assert anchor["point"]["y"] < anchor["label"]["normalized"]["y"]


def test_step_expect_text_requires_exact_match() -> None:
    from coretap.cli import _evaluate_postconditions

    before = _observe_with_tokens([])
    after = _observe_with_tokens([OcrToken("TestFlight", 90.0, 250, 110, 134, 29, "vision")])

    result = _evaluate_postconditions(
        before,
        after,
        [{"type": "textVisible", "text": "test", "caseSensitive": False, "minConfidence": 0.0, "matchMode": "exact"}],
    )

    assert result["status"] == "failed"
    assert result["checks"][0]["match"] is None


def test_step_expect_text_treats_common_chinese_ocr_variants_as_exact() -> None:
    from coretap.cli import _evaluate_postconditions

    before = _observe_with_tokens([])
    after = _observe_with_tokens([OcrToken("小紅书", 30.0, 12, 66, 56, 15, "vision")])

    result = _evaluate_postconditions(
        before,
        after,
        [{"type": "textVisible", "text": "小红书", "caseSensitive": False, "minConfidence": 0.0, "matchMode": "exact"}],
    )

    assert result["status"] == "satisfied"
    assert result["checks"][0]["match"]["matchedText"] == "小紅书"
