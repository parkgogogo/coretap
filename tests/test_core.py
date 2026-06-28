from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from coretap.backends import (
    DeviceBackend,
    SimulatorBackend,
    _check_coredevice_result,
    _coredevice_blank_screenshot,
    _coredevice_screenshot_rotation,
    parse_usbmux_devices,
)
from coretap.cli import point_to_hid
from coretap.device_buttons import resolve_button
from coretap.device_worker import (
    CoreDeviceWorkerPool,
    is_recoverable_coredevice_display_error,
    is_recoverable_userspace_tunnel_error,
    recover_coredevice_display_service,
    set_default_device_worker_pool,
)
from coretap.grounding import (
    assess_grounding_tap_safety,
    prepare_grounding_image,
    prepare_image_long_side,
    remap_grounding_to_source_frame,
    target_text_terms,
)
from coretap.model_pack import parse_grounding_output
from coretap.ocr import (
    DEFAULT_OCR_LANG,
    OcrToken,
    find_exact_text_candidates,
    find_text,
    missing_tesseract_languages,
    parse_tesseract_languages,
    parse_tsv,
    parse_vision_json,
)
from coretap.runtime import Completed, CoretapError, png_size
from coretap.text_input import validate_hid_text


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x03"
    b"\x00\x00\x00\x05"
    b"\x08\x02\x00\x00\x00"
    b"\x00\x00\x00\x00"
)


def test_point_to_hid_from_normalized() -> None:
    point = point_to_hid(0.5, 0.25, width=1000, height=2000, space="normalized")

    assert point["hidU16"] == {"x": 32768, "y": 16384}
    assert point["screenshotPx"] == {"x": 500.0, "y": 500.0}
    assert point["frame"] == {"known": True, "widthPx": 1000, "heightPx": 2000}


def test_point_to_hid_omits_screenshot_pixels_when_frame_unknown() -> None:
    point = point_to_hid(0.5, 0.25, width=1, height=1, space="normalized", frame_known=False)

    assert point["hidU16"] == {"x": 32768, "y": 16384}
    assert point["screenshotPx"] is None
    assert point["frame"] == {"known": False, "widthPx": None, "heightPx": None}


def test_point_to_hid_from_pixels() -> None:
    point = point_to_hid(250, 100, width=1000, height=400, space="px")

    assert point["normalized"] == {"x": 0.25, "y": 0.25}
    assert point["hidU16"] == {"x": 16384, "y": 16384}


def test_point_rejects_out_of_range_normalized() -> None:
    with pytest.raises(CoretapError):
        point_to_hid(1.1, 0.2, width=100, height=100, space="normalized")


def test_point_rejects_out_of_range_hid() -> None:
    with pytest.raises(CoretapError):
        point_to_hid(70000, 0, width=100, height=100, space="hid")


def test_parse_grounding_output_json_coordinate() -> None:
    result = parse_grounding_output('{"coordinate":[250, 500]}', width=100, height=200)

    assert result["status"] == "found"
    assert result["point"]["model1000"] == {"x": 250.0, "y": 500.0}
    assert result["point"]["framePx"] == {"x": 25.0, "y": 100.0}
    assert result["point"]["normalized"] == {"x": 0.25, "y": 0.5}


def test_parse_grounding_output_rejects_out_of_bounds() -> None:
    result = parse_grounding_output('{"coordinate":[1200, 50]}', width=100, height=200)

    assert result["status"] == "invalid"
    assert result["reason"] == "coordinate outside model-1000 space"


def test_prepare_grounding_image_downscales_to_default_long_side(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (1260, 2736), color=(255, 255, 255)).save(source)

    result = prepare_grounding_image(source, output_dir=tmp_path)

    assert result["resized"] is True
    assert result["widthPx"] == 630
    assert result["heightPx"] == 1368
    assert result["sourceWidthPx"] == 1260
    assert result["sourceHeightPx"] == 2736
    assert Path(result["path"]).exists()
    assert png_size(Path(result["path"])) == (630, 1368)


def test_prepare_image_long_side_writes_requested_output_path(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    output = tmp_path / "preview.png"
    Image.new("RGB", (1260, 2736), color=(255, 255, 255)).save(source)

    result = prepare_image_long_side(source, output_path=output, max_long_side=1368)

    assert result["path"] == str(output)
    assert result["resized"] is True
    assert result["widthPx"] == 630
    assert result["heightPx"] == 1368
    assert result["sourceWidthPx"] == 1260
    assert result["sourceHeightPx"] == 2736
    assert png_size(output) == (630, 1368)


def test_prepare_image_long_side_can_resize_in_place(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (1260, 2736), color=(255, 255, 255)).save(source)

    result = prepare_image_long_side(source, output_path=source, max_long_side=1368)

    assert result["path"] == str(source)
    assert result["resized"] is True
    assert png_size(source) == (630, 1368)


def test_remap_grounding_to_source_frame_preserves_normalized_coordinates() -> None:
    grounded = {
        "status": "found",
        "point": {
            "framePx": {"x": 534.24, "y": 664.848},
            "normalized": {"x": 0.848, "y": 0.486},
        },
        "frame": {"widthPx": 630, "heightPx": 1368},
    }

    result = remap_grounding_to_source_frame(grounded, source_width=1260, source_height=2736)

    assert result["point"]["modelInputFramePx"] == {"x": 534.24, "y": 664.848}
    assert result["point"]["framePx"] == {"x": 1068.48, "y": 1329.696}
    assert result["point"]["normalized"] == {"x": 0.848, "y": 0.486}
    assert result["frame"] == {"widthPx": 1260, "heightPx": 2736}


def test_parse_tsv_and_find_text() -> None:
    tsv = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95\tGeneral
5\t1\t1\t1\t1\t2\t50\t20\t40\t12\t90\tAbout
"""

    tokens = parse_tsv(tsv)
    match = find_text(tokens, "general")

    assert len(tokens) == 2
    assert match is not None
    assert match["matchedText"] == "General"
    assert match["matchedEngines"] == ["tesseract"]
    assert match["matchedBoxPx"] == {"x": 10, "y": 20, "width": 30, "height": 12}


def test_parse_vision_json_and_find_text() -> None:
    tokens = parse_vision_json('[{"text":"◎ 搜索","confidence":30,"left":502,"top":2022,"width":120,"height":42}]')
    match = find_text(tokens, "搜索")

    assert len(tokens) == 1
    assert tokens[0].engine == "vision"
    assert match is not None
    assert match["matchedText"] == "◎ 搜索"
    assert match["matchedEngines"] == ["vision"]


def test_default_ocr_language_requires_chinese_and_english() -> None:
    languages = parse_tesseract_languages(
        """List of available languages in "/opt/homebrew/share/tessdata/" (3):
eng
chi_sim
osd
"""
    )

    assert DEFAULT_OCR_LANG == "chi_sim+eng"
    assert missing_tesseract_languages(languages) == []
    assert missing_tesseract_languages(["eng"]) == ["chi_sim"]


def test_find_exact_text_candidates_requires_exact_normalized_match() -> None:
    tsv = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95\tChatGPT
5\t1\t1\t1\t1\t2\t50\t20\t40\t12\t95\tChatGPTX
5\t1\t1\t1\t1\t3\t100\t20\t30\t12\t40\tChatGPT
"""

    tokens = parse_tsv(tsv)
    matches = find_exact_text_candidates(tokens, "chatgpt", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "ChatGPT"
    assert matches[0]["matchedBoxPx"] == {"x": 10, "y": 20, "width": 30, "height": 12}


def test_find_exact_text_candidates_can_match_phrase() -> None:
    tsv = """level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext
5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t95\tApp
5\t1\t1\t1\t1\t2\t50\t20\t40\t12\t90\tStore
"""

    tokens = parse_tsv(tsv)
    matches = find_exact_text_candidates(tokens, "App Store")

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "App Store"
    assert matches[0]["matchedBoxPx"] == {"x": 10, "y": 20, "width": 80, "height": 12}


def test_find_exact_text_candidates_accepts_vision_token_contains() -> None:
    tokens = parse_vision_json('[{"text":"◎ 搜索","confidence":30,"left":502,"top":2022,"width":120,"height":42}]')
    matches = find_exact_text_candidates(tokens, "搜索", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "◎ 搜索"
    assert matches[0]["matchedKind"] == "token_contains"
    assert matches[0]["matchedEngines"] == ["vision"]


def test_find_exact_text_candidates_accepts_search_ui_prefix() -> None:
    tokens = parse_vision_json('[{"text":"Q 小红书","confidence":30,"left":10,"top":20,"width":100,"height":30}]')
    matches = find_exact_text_candidates(tokens, "小红书", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "Q 小红书"
    assert matches[0]["matchedKind"] == "exact"
    assert matches[0]["exactMatchStrategy"] == "ui-prefix-stripped"


def test_find_exact_text_candidates_accepts_badge_ui_prefix() -> None:
    tokens = parse_vision_json('[{"text":"⑧ 小红书","confidence":30,"left":10,"top":20,"width":100,"height":30}]')
    matches = find_exact_text_candidates(tokens, "小红书", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "⑧ 小红书"
    assert matches[0]["matchedKind"] == "exact"


def test_parse_usbmux_json_devices() -> None:
    devices = parse_usbmux_devices(
        """[
  {
    "Identifier": "00008110-001234",
    "UniqueDeviceID": "00008110-001234",
    "ProductType": "iPhone13,1",
    "ProductVersion": "27.0",
    "DeviceName": "Park iPhone"
  }
]"""
    )

    assert len(devices) == 1
    assert devices[0].udid == "00008110-001234"
    assert devices[0].name == "Park iPhone"
    assert devices[0].runtime == "27.0"
    assert devices[0].details["ProductType"] == "iPhone13,1"


def test_parse_usbmux_simple_lines() -> None:
    devices = parse_usbmux_devices("udid-one\nudid-two\n")

    assert [d.udid for d in devices] == ["udid-one", "udid-two"]


def test_coredevice_tunneld_error_is_detected_on_zero_exit() -> None:
    done = Completed(
        argv=["pymobiledevice3", "developer", "core-device", "screen-capture", "screenshot"],
        returncode=0,
        stdout="",
        stderr="ERROR Unable to connect to Tunneld. You can start one using: sudo python3 -m pymobiledevice3 remote tunneld",
        duration_ms=10,
    )

    with pytest.raises(CoretapError) as exc:
        _check_coredevice_result(done, code="COREDEVICE_SCREENSHOT_FAILED", stage="screenshot")

    assert exc.value.code == "COREDEVICE_TUNNELD_UNAVAILABLE"
    assert exc.value.retryable is True
    assert exc.value.details["suggestedCommand"] == "sudo pymobiledevice3 remote tunneld --daemonize"


def test_coredevice_default_tunnel_mode_uses_userspace() -> None:
    backend = DeviceBackend()

    assert backend.coredevice_tunnel_mode == "userspace"
    assert backend.coredevice_device_options("device-udid") == ["--userspace"]
    assert backend.coredevice_env("device-udid")["PYMOBILEDEVICE3_UDID"] == "device-udid"


def test_userspace_tunnel_singleton_error_is_recoverable() -> None:
    error = CoretapError(
        "COREDEVICE_SCREENSHOT_FAILED",
        "a userspace tunnel is already active in this process (PyTCP's stack is a process-global singleton)",
        stage="screenshot",
        retryable=True,
    )

    assert is_recoverable_userspace_tunnel_error(error) is True


def test_userspace_tunnel_singleton_error_is_not_display_service_error() -> None:
    error = CoretapError(
        "COREDEVICE_TAP_FAILED",
        "Persistent CoreDevice HID dispatch failed: a userspace tunnel is already active in this process",
        stage="tap",
        retryable=True,
        details={"previousError": "bounded touch session failed during universal-hid-enter: TimeoutError"},
    )

    assert is_recoverable_userspace_tunnel_error(error) is True
    assert is_recoverable_coredevice_display_error(error) is False


def test_coredevice_display_service_error_is_recoverable() -> None:
    error = CoretapError(
        "COREDEVICE_DISPLAY_SERVICE_FAILED",
        "bounded touch session failed during display-enter: TimeoutError:",
        stage="touch-session",
        retryable=True,
    )

    assert is_recoverable_coredevice_display_error(error) is True


def test_recover_coredevice_display_service_signals_dtremotedisplayd(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_command(argv: list[str], **kwargs: object) -> Completed:
        calls.append({"argv": argv, "env": kwargs.get("env")})
        if "list-processes" in argv:
            return Completed(
                argv=argv,
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "executableURL": {
                                "relative": "file:///System/Developer/usr/libexec/dtremotedisplayd"
                            },
                            "processIdentifier": 4037,
                        }
                    ]
                ),
                stderr="",
                duration_ms=3,
            )
        if "send-signal-to-process" in argv:
            return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=4)
        raise AssertionError(argv)

    monkeypatch.setattr("coretap.device_worker.run_command", fake_run_command)

    result = recover_coredevice_display_service("device-udid")

    assert result["status"] == "signaled"
    assert result["targets"] == [
        {"pid": 4037, "executable": "file:///System/Developer/usr/libexec/dtremotedisplayd", "signal": 15}
    ]
    assert calls[0]["argv"] == ["pymobiledevice3", "developer", "core-device", "list-processes", "--userspace"]
    assert calls[1]["argv"] == [
        "pymobiledevice3",
        "developer",
        "core-device",
        "send-signal-to-process",
        "--userspace",
        "4037",
        "15",
    ]
    assert calls[0]["env"]["PYMOBILEDEVICE3_UDID"] == "device-udid"  # type: ignore[index]


def test_recover_coredevice_display_service_restarts_dtuhidd_when_display_daemon_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        if "list-processes" in argv:
            return Completed(
                argv=argv,
                returncode=0,
                stdout=json.dumps(
                    [
                        {
                            "executableURL": {"relative": "file:///System/Developer/usr/libexec/dtuhidd"},
                            "processIdentifier": 3218,
                        }
                    ]
                ),
                stderr="",
                duration_ms=3,
            )
        if "send-signal-to-process" in argv:
            return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=4)
        raise AssertionError(argv)

    monkeypatch.setattr("coretap.device_worker.run_command", fake_run_command)

    result = recover_coredevice_display_service("device-udid")

    assert result["status"] == "signaled"
    assert result["targets"] == [
        {"pid": 3218, "executable": "file:///System/Developer/usr/libexec/dtuhidd", "signal": 9}
    ]
    assert calls[1] == [
        "pymobiledevice3",
        "developer",
        "core-device",
        "send-signal-to-process",
        "--userspace",
        "3218",
        "9",
    ]


def test_device_backend_tap_falls_back_after_worker_tunnel_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPool:
        def tap_userspace(self, *_args: object) -> dict[str, object]:
            raise CoretapError(
                "COREDEVICE_TAP_FAILED",
                "a userspace tunnel is already active in this process",
                stage="tap",
                retryable=True,
            )

    backend = DeviceBackend()
    set_default_device_worker_pool(FailingPool())  # type: ignore[arg-type]
    monkeypatch.setattr(
        backend,
        "_tap_userspace_helper",
        lambda device, x, y, hx, hy: {
            "attempted": True,
            "dryRun": False,
            "normalized": {"x": x, "y": y},
            "hidU16": {"x": hx, "y": hy},
            "dispatchStatus": "sent",
        },
    )
    try:
        result = backend.tap_normalized("device-udid", 0.25, 0.5, dry_run=False)
    finally:
        set_default_device_worker_pool(None)

    assert result["dispatchStatus"] == "sent"
    assert result["workerFallback"] == "coretap-device-hid-helper"
    assert result["previousError"]["code"] == "COREDEVICE_TAP_FAILED"


def test_device_backend_tap_does_not_fallback_after_display_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPool:
        def tap_userspace(self, *_args: object) -> dict[str, object]:
            raise CoretapError(
                "COREDEVICE_DISPLAY_SERVICE_FAILED",
                "bounded touch session failed during display-enter: TimeoutError:",
                stage="touch-session",
                retryable=True,
            )

    backend = DeviceBackend()
    set_default_device_worker_pool(FailingPool())  # type: ignore[arg-type]
    monkeypatch.setattr(
        backend,
        "_tap_userspace_helper",
        lambda *_args: pytest.fail("DisplayService errors should not fall back to one-shot HID helper"),
    )
    try:
        with pytest.raises(CoretapError) as exc:
            backend.tap_normalized("device-udid", 0.25, 0.5, dry_run=False)
    finally:
        set_default_device_worker_pool(None)

    assert exc.value.code == "COREDEVICE_DISPLAY_SERVICE_FAILED"


def test_device_backend_display_info_falls_back_after_worker_tunnel_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingPool:
        def display_info_userspace(self, *_args: object) -> dict[str, object]:
            raise CoretapError(
                "COREDEVICE_DISPLAY_INFO_FAILED",
                "Persistent CoreDevice display-info failed: a userspace tunnel is already active in this process "
                "(PyTCP's stack is a process-global singleton; only one userspace tunnel per process is supported)",
                stage="display-info",
                retryable=True,
                details={"workerRecovered": True},
            )

    calls: list[Completed] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        done = Completed(
            argv=argv,
            returncode=0,
            stdout=json.dumps(
                {
                    "displays": [
                        {
                            "primary": True,
                            "external": False,
                            "currentMode": {"size": [1125, 2436]},
                        }
                    ],
                    "orientation": {"currentDeviceNonFlatOrientation": "portrait"},
                }
            ),
            stderr="",
            duration_ms=12,
        )
        calls.append(done)
        return done

    backend = DeviceBackend()
    set_default_device_worker_pool(FailingPool())  # type: ignore[arg-type]
    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)
    try:
        result = backend.display_info("device-udid")
    finally:
        set_default_device_worker_pool(None)

    assert result["displays"][0]["currentMode"]["size"] == [1125, 2436]
    assert result["_coretap"]["fallback"] == "pymobiledevice3-cli-userspace"
    assert result["_coretap"]["previousError"]["code"] == "COREDEVICE_DISPLAY_INFO_FAILED"
    assert calls
    assert calls[0].argv == [
        "pymobiledevice3",
        "developer",
        "core-device",
        "get-display-info",
        "--userspace",
    ]


def test_target_text_terms_extracts_specific_app_name() -> None:
    assert target_text_terms("the ChatGPT app icon") == ["chatgpt"]
    assert target_text_terms("点击 搜索") == ["搜索"]
    assert target_text_terms("the App Store search tab") == ["store", "search", "搜索"]
    assert target_text_terms("the App Store search field") == ["store", "search", "搜索", "游戏"]
    assert target_text_terms("the center of the bottom iOS home screen search field") == ["search", "搜索", "游戏", "故事等"]
    assert target_text_terms("the retry button") == ["retry", "重试"]
    assert target_text_terms("the search field labeled 游戏、App、故事等 at the top") == ["search", "搜索", "游戏", "故事等"]


def test_grounding_tap_safety_rejects_missing_target_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(TINY_PNG)
    grounded = {
        "status": "found",
        "point": {"framePx": {"x": 120, "y": 160}, "normalized": {"x": 0.5, "y": 0.5}},
        "frame": {"widthPx": 240, "heightPx": 320},
    }
    monkeypatch.setattr("coretap.grounding.run_tesseract", lambda *_args, **_kwargs: ([], ""))
    monkeypatch.setattr("coretap.grounding.run_vision_ocr", lambda *_args, **_kwargs: ([], "[]"))

    safety = assess_grounding_tap_safety(image, "the ChatGPT app icon", grounded)

    assert safety["status"] == "unsafe"
    assert safety["safeToTap"] is False
    assert safety["checks"][1]["reason"] == "target text was not visible in the current screenshot"


def test_grounding_tap_safety_accepts_visible_nearby_target_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(TINY_PNG)
    grounded = {
        "status": "found",
        "point": {"framePx": {"x": 120, "y": 150}, "normalized": {"x": 0.5, "y": 0.5}},
        "frame": {"widthPx": 240, "heightPx": 320},
    }
    tokens = [OcrToken("ChatGPT", 95.0, 92, 178, 80, 20)]
    monkeypatch.setattr("coretap.grounding.run_tesseract", lambda *_args, **_kwargs: (tokens, ""))

    safety = assess_grounding_tap_safety(image, "the ChatGPT app icon", grounded)

    assert safety["status"] == "safe"
    assert safety["safeToTap"] is True


def test_grounding_tap_safety_uses_nearby_vision_match_when_tesseract_match_is_far(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(TINY_PNG)
    grounded = {
        "status": "found",
        "point": {"framePx": {"x": 120, "y": 150}, "normalized": {"x": 0.5, "y": 0.5}},
        "frame": {"widthPx": 240, "heightPx": 320},
    }
    tesseract_tokens = [OcrToken("搜索", 95.0, 4, 4, 40, 20)]
    vision_tokens = [OcrToken("搜索", 95.0, 100, 140, 48, 24, engine="vision")]
    monkeypatch.setattr("coretap.grounding.run_tesseract", lambda *_args, **_kwargs: (tesseract_tokens, ""))
    monkeypatch.setattr("coretap.grounding.run_vision_ocr", lambda *_args, **_kwargs: (vision_tokens, "[]"))

    safety = assess_grounding_tap_safety(image, "the App Store search tab", grounded)

    assert safety["status"] == "safe"
    assert safety["safeToTap"] is True
    evidence = safety["checks"][1]
    assert evidence["engines"] == ["tesseract", "vision"]
    assert evidence["nearestMatch"]["matchedEngines"] == ["vision"]


def test_grounding_tap_safety_accepts_app_store_search_placeholder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(TINY_PNG)
    grounded = {
        "status": "found",
        "point": {"framePx": {"x": 130, "y": 30}, "normalized": {"x": 0.54, "y": 0.09}},
        "frame": {"widthPx": 240, "heightPx": 320},
    }
    tokens = [OcrToken("游戏、App、故事等", 95.0, 40, 20, 130, 18, engine="vision")]
    monkeypatch.setattr("coretap.grounding.run_tesseract", lambda *_args, **_kwargs: ([], ""))
    monkeypatch.setattr("coretap.grounding.run_vision_ocr", lambda *_args, **_kwargs: (tokens, "[]"))

    safety = assess_grounding_tap_safety(image, "the App Store search field", grounded)

    assert safety["status"] == "safe"
    assert safety["safeToTap"] is True
    assert safety["checks"][1]["nearestMatch"]["matchedText"] == "游戏、App、故事等"


def test_grounding_tap_safety_accepts_search_field_with_existing_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(TINY_PNG)
    grounded = {
        "status": "found",
        "point": {"framePx": {"x": 110, "y": 44}, "normalized": {"x": 0.46, "y": 0.14}},
        "frame": {"widthPx": 240, "heightPx": 320},
    }
    tokens = [OcrToken("xiaohongshu小红书", 95.0, 40, 36, 130, 18, engine="vision")]
    monkeypatch.setattr("coretap.grounding.run_tesseract", lambda *_args, **_kwargs: ([], ""))
    monkeypatch.setattr("coretap.grounding.run_vision_ocr", lambda *_args, **_kwargs: (tokens, "[]"))

    safety = assess_grounding_tap_safety(image, "the App Store search field", grounded)

    assert safety["status"] == "safe"
    assert safety["safeToTap"] is True
    assert safety["checks"][1]["semanticEvidence"] == "search-field-content"
    assert safety["checks"][1]["nearestMatch"]["matchedText"] == "xiaohongshu小红书"


def test_paste_menu_point_uses_below_anchor_for_top_text_fields() -> None:
    from coretap.device_worker import _paste_menu_point_for_anchor

    _, top_y = _paste_menu_point_for_anchor(0.34, 0.09)
    _, middle_y = _paste_menu_point_for_anchor(0.2, 0.54)

    assert top_y > 0.09
    assert middle_y < 0.54


def test_coredevice_button_alias_resolves_to_canonical_lock() -> None:
    button = resolve_button("power")

    assert button is not None
    assert button.name == "lock"
    assert button.usage_page == 0x0C
    assert button.usage_code == 0x30
    assert button.hold_ms == 500


def test_device_backend_press_button_dry_run_uses_resolved_metadata() -> None:
    result = DeviceBackend().press_button("device-udid", "power", dry_run=True)

    assert result["button"] == "lock"
    assert result["requestedButton"] == "power"
    assert result["state"] == "press"
    assert result["holdMs"] == 500
    assert result["hidButton"] == {"usagePage": 0x0C, "usageCode": 0x30}
    assert result["attempted"] is False


def test_device_backend_type_text_dry_run_validates_ascii() -> None:
    result = DeviceBackend().type_text("device-udid", "Hello, iOS 123!", dry_run=True)

    assert result["attempted"] is False
    assert result["dryRun"] is True
    assert result["text"]["length"] == len("Hello, iOS 123!")
    assert result["text"]["asciiOnly"] is True
    assert result["inputMethod"] == "coredevice-pasteboard-edit-menu"
    assert validate_hid_text("Hello\tWorld\n").ok is True


def test_device_backend_type_text_dry_run_supports_unicode() -> None:
    result = DeviceBackend().type_text("device-udid", "搜索", dry_run=True)

    assert result["attempted"] is False
    assert result["text"]["length"] == 2
    assert result["text"]["asciiOnly"] is False
    assert result["inputMethod"] == "coredevice-pasteboard-edit-menu"


def test_simulator_backend_rejects_coredevice_press() -> None:
    with pytest.raises(CoretapError) as exc:
        SimulatorBackend().press_button("booted", "home")

    assert exc.value.code == "SIMULATOR_PRESS_UNSUPPORTED"


def test_coredevice_tunneld_mode_omits_userspace() -> None:
    backend = DeviceBackend(coredevice_tunnel_mode="tunneld")

    assert backend.coredevice_device_options("device-udid") == ["--tunnel", "device-udid"]
    assert backend.coredevice_env("device-udid").get("PYMOBILEDEVICE3_UDID") is None


def test_coredevice_screenshot_rotation_matches_display_orientation() -> None:
    assert _coredevice_screenshot_rotation((2736, 1260), (1260, 2736), "landscapeRight") == 270
    assert _coredevice_screenshot_rotation((2736, 1260), (1260, 2736), "landscapeLeft") == 90
    assert _coredevice_screenshot_rotation((1260, 2736), (1260, 2736), "portraitUpsideDown") == 180
    assert _coredevice_screenshot_rotation((1260, 2736), (1260, 2736), "portrait") is None


def test_coredevice_blank_screenshot_detects_all_black_png(tmp_path: Path) -> None:
    from PIL import Image

    black = tmp_path / "black.png"
    visible = tmp_path / "visible.png"
    Image.new("RGB", (3, 5), color=(0, 0, 0)).save(black)
    Image.new("RGB", (3, 5), color=(0, 0, 3)).save(visible)

    blank = _coredevice_blank_screenshot(black)

    assert blank is not None
    assert blank["reason"] == "all_black"
    assert _coredevice_blank_screenshot(visible) is None


def test_device_backend_uses_registered_persistent_worker_for_userspace_tap() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.calls = []

        def tap_userspace(self, device: str, x: float, y: float, hx: int, hy: int) -> dict:
            self.calls.append((device, x, y, hx, hy))
            return {"workerKind": "fake-persistent-worker", "dispatchStatus": "sent"}

    pool = FakePool()
    set_default_device_worker_pool(pool)  # type: ignore[arg-type]
    try:
        result = DeviceBackend().tap_normalized(
            "device-udid",
            0.25,
            0.5,
            dry_run=False,
            hid_u16={"x": 100, "y": 200},
        )
    finally:
        set_default_device_worker_pool(None)

    assert pool.calls == [("device-udid", 0.25, 0.5, 100, 200)]
    assert result == {"workerKind": "fake-persistent-worker", "dispatchStatus": "sent"}


def test_device_backend_uses_registered_persistent_worker_for_button_press() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.calls = []

        def press_button_userspace(
            self,
            device: str,
            *,
            button: str,
            state: str,
            usage_page: int,
            usage_code: int,
            hold_ms: int,
        ) -> dict:
            self.calls.append((device, button, state, usage_page, usage_code, hold_ms))
            return {"workerKind": "fake-persistent-worker", "dispatchStatus": "sent"}

    pool = FakePool()
    set_default_device_worker_pool(pool)  # type: ignore[arg-type]
    try:
        result = DeviceBackend().press_button("device-udid", "home")
    finally:
        set_default_device_worker_pool(None)

    assert pool.calls == [("device-udid", "home", "press", 0x0C, 0x40, 50)]
    assert result["workerKind"] == "fake-persistent-worker"
    assert result["dispatchStatus"] == "sent"
    assert result["attempted"] is True
    assert result["dryRun"] is False


def test_device_backend_uses_registered_persistent_worker_for_type_text() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.calls = []

        def type_text_userspace(
            self,
            device: str,
            *,
            text: str,
            char_delay_ms: int,
            inter_delay_ms: int,
            paste_at: dict | None,
            paste_hold_ms: int,
            clear_existing: bool,
        ) -> dict:
            self.calls.append((device, text, char_delay_ms, inter_delay_ms, paste_at, paste_hold_ms, clear_existing))
            return {"workerKind": "fake-persistent-worker", "dispatchStatus": "sent"}

    pool = FakePool()
    set_default_device_worker_pool(pool)  # type: ignore[arg-type]
    try:
        result = DeviceBackend().type_text(
            "device-udid",
            "搜索",
            char_delay_ms=1,
            inter_delay_ms=2,
            paste_at={"x": 0.2, "y": 0.925},
            paste_hold_ms=1300,
        )
    finally:
        set_default_device_worker_pool(None)

    assert pool.calls == [("device-udid", "搜索", 1, 2, {"x": 0.2, "y": 0.925}, 1300, False)]
    assert result["workerKind"] == "fake-persistent-worker"
    assert result["dispatchStatus"] == "sent"
    assert result["attempted"] is True
    assert result["dryRun"] is False


def test_device_backend_drag_does_not_fallback_after_display_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingPool:
        def drag_userspace(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise CoretapError(
                "COREDEVICE_DISPLAY_SERVICE_FAILED",
                "CoreDevice DisplayService touch session failed to open: display-enter TimeoutError",
                stage="touch-session",
                retryable=True,
            )

    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=1)

    set_default_device_worker_pool(FailingPool())  # type: ignore[arg-type]
    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)
    try:
        with pytest.raises(CoretapError) as exc:
            DeviceBackend().drag_normalized("device-udid", 0.2, 0.2, 0.8, 0.8, dry_run=False)
    finally:
        set_default_device_worker_pool(None)

    assert exc.value.code == "COREDEVICE_DISPLAY_SERVICE_FAILED"
    assert calls == []


def test_coredevice_worker_types_ascii_with_virtual_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        session = SimpleNamespace(
            device="device-udid",
            keyboard_service_id=None,
            typed_characters=0,
            last_used_at=0,
        )
        calls: list[tuple[str, object]] = []

        async def fake_get_or_open_keyboard_session(device: str) -> tuple[object, str]:
            calls.append(("open", device))
            return session, "created"

        async def fake_send_text(
            target: object,
            *,
            text: str,
            char_delay_ms: int,
            inter_delay_ms: int,
        ) -> None:
            calls.append(("send-text", (target, text, char_delay_ms, inter_delay_ms)))

        async def fake_clear_focused_text(target: object) -> int:
            calls.append(("clear", target))
            return 80

        async def fake_paste_text(*_args: object, **_kwargs: object) -> dict:
            raise AssertionError("ASCII input should not use pasteboard")

        monkeypatch.setattr(pool, "_get_or_open_keyboard_session", fake_get_or_open_keyboard_session)
        monkeypatch.setattr(pool, "_send_text", fake_send_text)
        monkeypatch.setattr(pool, "_clear_focused_text", fake_clear_focused_text)
        monkeypatch.setattr(pool, "_paste_text", fake_paste_text)

        result = await pool._type_text_userspace(
            "device-udid",
            text="xiaohongshu",
            char_delay_ms=1,
            inter_delay_ms=2,
            paste_at=None,
            paste_hold_ms=1300,
            clear_existing=True,
        )
        result["calls"] = calls
        return result

    result = asyncio.run(run())

    assert result["inputMethod"] == "coredevice-virtual-keyboard"
    assert result["pasteboardSet"] is False
    assert result["clearKeypresses"] == 80
    assert result["sessionTypedCharacterCount"] == len("xiaohongshu")
    assert result["calls"][0] == ("open", "device-udid")
    assert result["calls"][1][0] == "clear"
    assert result["calls"][2][0] == "send-text"


def test_coredevice_worker_types_cjk_with_pinyin_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        calls: list[tuple[str, object]] = []

        class FakeService:
            async def create_keyboard_service(self) -> int:
                calls.append(("create-keyboard", 123))
                return 123

            async def send_keyboard(self, keyboard_service_id: int, usages: tuple[int, ...]) -> None:
                calls.append(("send-keyboard", (keyboard_service_id, usages)))

            async def send_touchscreen(self, state: int, x: int, y: int) -> None:
                calls.append(("send-touchscreen", (state, x, y)))

        session = SimpleNamespace(
            device="device-udid",
            keyboard_service_id=None,
            service=FakeService(),
            typed_characters=0,
            last_used_at=0,
        )

        async def fake_get_or_open_keyboard_session(device: str) -> tuple[object, str]:
            calls.append(("open", device))
            return session, "created"

        async def fake_send_text(
            target: object,
            *,
            text: str,
            char_delay_ms: int,
            inter_delay_ms: int,
        ) -> None:
            calls.append(("send-text", (target, text, char_delay_ms, inter_delay_ms)))

        monkeypatch.setattr(pool, "_get_or_open_keyboard_session", fake_get_or_open_keyboard_session)
        monkeypatch.setattr(pool, "_send_text", fake_send_text)

        result = await pool._type_text_userspace(
            "device-udid",
            text="小红书",
            char_delay_ms=1,
            inter_delay_ms=2,
            paste_at=None,
            paste_hold_ms=1300,
            clear_existing=False,
        )
        result["calls"] = calls
        return result

    result = asyncio.run(run())

    assert result["inputMethod"] == "coredevice-pinyin-keyboard"
    assert result["convertedText"] == "xiaohongshu"
    assert result["candidateCommitAction"] == "tap-first-candidate"
    assert result["candidateSettleMs"] == 500
    assert result["candidateCommitPoint"]["normalized"] == {"x": 0.2, "y": 0.878}
    assert result["sessionTypedCharacterCount"] == len("小红书")
    assert result["calls"][0] == ("open", "device-udid")
    assert result["calls"][1][0] == "send-text"
    assert result["calls"][1][1][1] == "xiaohongshu"
    assert any(call[0] == "send-touchscreen" for call in result["calls"])


def test_coredevice_worker_rejects_non_cjk_without_paste_anchor() -> None:
    async def run() -> None:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        await pool._type_text_userspace(
            "device-udid",
            text="こんにちは",
            char_delay_ms=1,
            inter_delay_ms=2,
            paste_at=None,
            paste_hold_ms=1300,
            clear_existing=False,
        )

    with pytest.raises(CoretapError) as exc:
        asyncio.run(run())

    assert exc.value.code == "TEXT_INPUT_TARGET_UNKNOWN"


def test_coredevice_worker_touch_retry_keeps_existing_rsd(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        session = SimpleNamespace(
            device="device-udid",
            taps=0,
            last_used_at=0,
            last_tap_normalized=None,
        )
        calls: list[tuple[str, object]] = []

        async def fake_get_or_open_session(device: str) -> tuple[object, str]:
            calls.append(("open", device))
            if sum(call[0] == "open" for call in calls) == 1:
                raise CoretapError(
                    "COREDEVICE_DISPLAY_SERVICE_FAILED",
                    "CoreDevice DisplayService touch session failed to open: display-enter TimeoutError",
                    stage="touch-session",
                    retryable=True,
                )
            return session, "created"

        async def fake_send_tap(target: object, hx: int, hy: int) -> None:
            calls.append(("tap", (target, hx, hy)))

        async def fake_close_session(device: str) -> None:
            calls.append(("close-session", device))

        async def fake_close_rsd(_device: str) -> None:
            raise AssertionError("touch retry must not close the userspace RSD")

        async def fake_recover(device: str, *, stage: str, error: BaseException) -> dict:
            calls.append(("recover-display", (device, stage, type(error).__name__)))
            return {"status": "not-running"}

        monkeypatch.setattr(pool, "_get_or_open_session", fake_get_or_open_session)
        monkeypatch.setattr(pool, "_send_tap", fake_send_tap)
        monkeypatch.setattr(pool, "_close_session", fake_close_session)
        monkeypatch.setattr(pool, "_close_rsd", fake_close_rsd)
        monkeypatch.setattr(pool, "_recover_display_service_for_retry", fake_recover)

        result = await pool._tap_userspace("device-udid", 0.25, 0.5, 100, 200)
        result["calls"] = calls
        return result

    result = asyncio.run(run())

    assert result["dispatchStatus"] == "sent"
    assert result["sessionStatus"] == "recreated_after_display_service_recovery"
    assert [call[0] for call in result["calls"]] == ["open", "close-session", "recover-display", "open", "tap"]


def test_coredevice_worker_touch_retry_closes_device_for_userspace_tunnel_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        session = SimpleNamespace(
            device="device-udid",
            taps=0,
            last_used_at=0,
            last_tap_normalized=None,
        )
        calls: list[tuple[str, object]] = []

        async def fake_get_or_open_session(device: str) -> tuple[object, str]:
            calls.append(("open", device))
            if sum(call[0] == "open" for call in calls) == 1:
                raise CoretapError(
                    "COREDEVICE_TAP_FAILED",
                    "a userspace tunnel is already active in this process (PyTCP's stack is a process-global singleton)",
                    stage="tap",
                    retryable=True,
                )
            return session, "created"

        async def fake_send_tap(target: object, hx: int, hy: int) -> None:
            calls.append(("tap", (target, hx, hy)))

        async def fake_close_device(device: str) -> None:
            calls.append(("close-device", device))

        monkeypatch.setattr(pool, "_get_or_open_session", fake_get_or_open_session)
        monkeypatch.setattr(pool, "_send_tap", fake_send_tap)
        monkeypatch.setattr(pool, "_close_device", fake_close_device)

        result = await pool._tap_userspace("device-udid", 0.25, 0.5, 100, 200)
        result["calls"] = calls
        return result

    result = asyncio.run(run())

    assert result["dispatchStatus"] == "sent"
    assert result["sessionStatus"] == "recreated_after_error"
    assert result["retryCloseScope"] == "device"
    assert [call[0] for call in result["calls"]] == ["open", "close-device", "open", "tap"]


def test_device_backend_uses_helper_for_userspace_type_without_registered_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_helper(
        self: DeviceBackend,
        device: str,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        paste_at: dict | None,
        paste_hold_ms: int,
        clear_existing: bool,
    ) -> dict:
        calls.append((device, text, char_delay_ms, inter_delay_ms, paste_at, paste_hold_ms, clear_existing))
        return {"inputMethod": "coredevice-virtual-keyboard", "dispatchStatus": "sent"}

    set_default_device_worker_pool(None)
    monkeypatch.setattr(DeviceBackend, "_type_text_userspace_helper", fake_helper)

    result = DeviceBackend().type_text(
        "device-udid",
        "App Store",
        char_delay_ms=3,
        inter_delay_ms=4,
        paste_at=None,
        paste_hold_ms=1600,
        clear_existing=True,
    )

    assert calls == [("device-udid", "App Store", 3, 4, None, 1600, True)]
    assert result["inputMethod"] == "coredevice-virtual-keyboard"
    assert result["dispatchStatus"] == "sent"
    assert result["attempted"] is True
    assert result["dryRun"] is False


def test_device_backend_uses_registered_persistent_worker_for_screenshot(tmp_path: Path) -> None:
    class FakePool:
        def __init__(self) -> None:
            self.calls = []

        def capture_screenshot_userspace(self, device: str) -> dict:
            self.calls.append(("screenshot", device))
            return {"image": TINY_PNG, "workerKind": "fake-persistent-worker"}

        def display_info_userspace(self, device: str) -> dict:
            self.calls.append(("display-info", device))
            return {
                "displays": [
                    {
                        "primary": True,
                        "currentMode": {"size": [3, 5]},
                        "deviceOrientation": "portrait",
                    }
                ]
            }

    pool = FakePool()
    out = tmp_path / "screen.png"
    set_default_device_worker_pool(pool)  # type: ignore[arg-type]
    try:
        frame = DeviceBackend().screenshot("device-udid", out)
    finally:
        set_default_device_worker_pool(None)

    assert pool.calls == [("screenshot", "device-udid"), ("display-info", "device-udid")]
    assert frame.width == 3
    assert frame.height == 5
    assert frame.path == out
    assert out.read_bytes() == TINY_PNG


def test_device_backend_retries_all_black_coredevice_screenshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        if "screen-capture" in argv:
            output = Path(argv[-1])
            color = (0, 0, 0) if sum("screen-capture" in call for call in calls) == 1 else (255, 0, 0)
            Image.new("RGB", (3, 5), color=color).save(output)
            return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=1)
        if "get-display-info" in argv:
            return Completed(
                argv=argv,
                returncode=0,
                stdout=json.dumps({"displays": [{"primary": True, "currentMode": {"size": [3, 5]}}]}),
                stderr="",
                duration_ms=1,
            )
        raise AssertionError(argv)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)
    out = tmp_path / "screen.png"

    frame = DeviceBackend().screenshot("device-udid", out)

    assert frame.width == 3
    assert frame.height == 5
    assert sum("screen-capture" in call for call in calls) == 2
    assert _coredevice_blank_screenshot(out) is None


def test_device_backend_rejects_repeated_all_black_coredevice_screenshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        if "screen-capture" in argv:
            Image.new("RGB", (3, 5), color=(0, 0, 0)).save(Path(argv[-1]))
            return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=1)
        raise AssertionError(argv)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    with pytest.raises(CoretapError) as excinfo:
        DeviceBackend().screenshot("device-udid", tmp_path / "screen.png")

    assert excinfo.value.code == "COREDEVICE_SCREENSHOT_BLANK"
    assert excinfo.value.retryable is True
    assert excinfo.value.details["attempts"] == 2
    assert sum("screen-capture" in call for call in calls) == 2


def test_png_size(tmp_path: Path) -> None:
    png = tmp_path / "tiny.png"
    png.write_bytes(TINY_PNG)

    assert png_size(png) == (3, 5)
