from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import coretap.model_pack as model_pack
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
    _supports_unshifted_virtual_keyboard_text,
    is_recoverable_coredevice_display_error,
    is_recoverable_userspace_tunnel_error,
    recover_coredevice_display_service,
    set_default_device_worker_pool,
)
from coretap.grounding import (
    compute_refinement_crop,
    prepare_grounding_image,
    prepare_image_long_side,
    prepare_refinement_crop,
    remap_crop_grounding_to_source_frame,
    remap_grounding_to_source_frame,
)
from coretap.model_pack import (
    DEFAULT_VISUAL_OBSERVE_MAX_ELEMENTS,
    DEFAULT_VISUAL_OBSERVE_MAX_TOKENS,
    PUBLIC_MODEL_PROFILE,
    parse_grounding_output,
    parse_visual_observe_output,
    visual_observe_prompt,
)
from coretap.ocr import (
    DEFAULT_OCR_LANG,
    OcrToken,
    find_exact_text_candidates,
    find_text,
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


def test_parse_visual_observe_output_accepts_center_and_bbox() -> None:
    raw = json.dumps(
        {
            "summary": "Home screen with app icons",
            "elements": [
                {
                    "label": "ChatGPT app icon",
                    "role": "appIcon",
                    "center": [820, 530],
                    "bbox": [740, 470, 900, 610],
                    "confidence": 0.84,
                },
                {
                    "label": "Search icon button",
                    "role": "button",
                    "bbox": [440, 880, 560, 940],
                    "confidence": 1.2,
                },
            ],
        }
    )

    result = parse_visual_observe_output(raw, width=1000, height=2000)

    assert result["status"] == "ready"
    assert result["promptVersion"] == "visual-observe-v1"
    assert result["summary"] == "Home screen with app icons"
    assert result["elements"][0]["source"] == "vlm"
    assert result["elements"][0]["role"] == "appIcon"
    assert result["elements"][0]["center"] == {"x": 0.82, "y": 0.53}
    assert result["elements"][0]["bbox"] == {"x": 0.74, "y": 0.47, "width": 0.16, "height": 0.14}
    assert result["elements"][1]["center"] == {"x": 0.5, "y": 0.91}
    assert result["elements"][1]["confidence"] == 1.0


def test_visual_observe_prompt_limits_scope_to_non_text_elements() -> None:
    prompt = visual_observe_prompt(1000, 2000)

    assert f"at most {DEFAULT_VISUAL_OBSERVE_MAX_ELEMENTS} items" in prompt
    assert "only non-text visual UI elements" in prompt
    assert "Do not return plain text" in prompt
    assert "OCR-readable text buttons" in prompt
    assert "exclude the text label" in prompt


def test_parse_visual_observe_output_default_limit_is_twelve_elements() -> None:
    raw = json.dumps(
        {
            "summary": "Many icons",
            "elements": [
                {"label": f"Icon {index}", "role": "button", "center": [100 + index, 100], "confidence": 0.7}
                for index in range(20)
            ],
        }
    )

    result = parse_visual_observe_output(raw, width=1000, height=2000)

    assert len(result["elements"]) == DEFAULT_VISUAL_OBSERVE_MAX_ELEMENTS == 12
    assert result["rawElementCount"] == 20


def test_visual_observe_model_uses_short_generation_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_model_prompt(image: Path, prompt: str, *, profile: str, max_tokens: int) -> str:
        captured["image"] = image
        captured["prompt"] = prompt
        captured["profile"] = profile
        captured["maxTokens"] = max_tokens
        return '{"summary":"","elements":[]}'

    monkeypatch.setattr(model_pack, "_require_public_profile", lambda _profile: None)
    monkeypatch.setattr(model_pack, "png_size", lambda _image: (1000, 2000))
    monkeypatch.setattr(model_pack, "_run_model_prompt", fake_run_model_prompt)

    result = model_pack._run_visual_observe_model_inprocess(tmp_path / "screen.png", profile=PUBLIC_MODEL_PROFILE)

    assert result["status"] == "ready"
    assert captured["maxTokens"] == DEFAULT_VISUAL_OBSERVE_MAX_TOKENS == 256
    assert f"at most {DEFAULT_VISUAL_OBSERVE_MAX_ELEMENTS} items" in str(captured["prompt"])


def test_parse_visual_observe_output_drops_invalid_and_duplicate_elements() -> None:
    raw = json.dumps(
        {
            "summary": "Toolbar",
            "elements": [
                {"label": "Close button", "role": "button", "center": [100, 100], "confidence": 0.7},
                {"label": "Close button", "role": "button", "center": [100, 100], "confidence": 0.7},
                {"label": "Offscreen", "role": "button", "center": [1200, 100], "confidence": 0.7},
                {"label": "Missing coordinate", "role": "button", "confidence": 0.7},
                {"label": "Odd role", "role": "fancy", "center": {"x": 500, "y": 500}},
            ],
        }
    )

    result = parse_visual_observe_output(raw, width=1000, height=2000)

    assert [item["label"] for item in result["elements"]] == ["Close button", "Odd role"]
    assert result["elements"][1]["role"] == "unknown"


def test_parse_visual_observe_output_handles_invalid_json() -> None:
    result = parse_visual_observe_output("not json", width=1000, height=2000)

    assert result["status"] == "invalid"
    assert result["elements"] == []


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
    assert result["pointSource"] == "point"
    assert result["point"]["model1000"] == {"x": 250.0, "y": 500.0}
    assert result["point"]["framePx"] == {"x": 25.0, "y": 100.0}
    assert result["point"]["normalized"] == {"x": 0.25, "y": 0.5}


def test_parse_grounding_output_accepts_point_array_under_any_key() -> None:
    result = parse_grounding_output('{"any_key":[503,675]}', width=1000, height=2000)

    assert result["status"] == "found"
    assert result["pointSource"] == "point"
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}
    assert result["point"]["normalized"] == {"x": 0.503, "y": 0.675}


def test_parse_grounding_output_accepts_rect_array_under_any_key() -> None:
    result = parse_grounding_output('{"foo":[458,669,548,681]}', width=1000, height=2000)

    assert result["status"] == "found"
    assert result["pointSource"] == "rect_center"
    assert result["rectModel1000"] == {"x1": 458.0, "y1": 669.0, "x2": 548.0, "y2": 681.0}
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}
    assert result["point"]["normalized"] == {"x": 0.503, "y": 0.675}


def test_parse_grounding_output_accepts_bare_point_array() -> None:
    result = parse_grounding_output("[503,675]", width=1000, height=2000)

    assert result["status"] == "found"
    assert result["pointSource"] == "point"
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}


def test_parse_grounding_output_accepts_bare_rect_array() -> None:
    result = parse_grounding_output("[458,669,548,681]", width=1000, height=2000)

    assert result["status"] == "found"
    assert result["pointSource"] == "rect_center"
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}


def test_parse_grounding_output_accepts_truncated_bare_rect_array() -> None:
    result = parse_grounding_output("[458,669,548,681", width=1000, height=2000)

    assert result["status"] == "found"
    assert result["pointSource"] == "rect_center"
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}


def test_parse_grounding_output_does_not_parse_scattered_numbers() -> None:
    result = parse_grounding_output("target is around 458 669 548 681", width=1000, height=2000)

    assert result["status"] == "invalid"
    assert result["reason"] == "no coordinate"


def test_parse_grounding_output_prefers_point_over_rect_candidates() -> None:
    result = parse_grounding_output('{"rect":[458,669,548,681],"final":[503,675]}', width=1000, height=2000)

    assert result["status"] == "found"
    assert result["pointSource"] == "point"
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}


def test_parse_grounding_output_uses_last_candidate_in_same_priority_group() -> None:
    result = parse_grounding_output('{"first":[111,222],"final":[503,675]}', width=1000, height=2000)

    assert result["status"] == "found"
    assert result["point"]["model1000"] == {"x": 503.0, "y": 675.0}


def test_parse_grounding_output_rejects_out_of_bounds() -> None:
    result = parse_grounding_output('{"coordinate":[1200, 50]}', width=100, height=200)

    assert result["status"] == "invalid"
    assert result["reason"] == "coordinate outside model-1000 space"


def test_parse_grounding_output_rejects_out_of_bounds_rect() -> None:
    result = parse_grounding_output("[458,669,1201,681]", width=100, height=200)

    assert result["status"] == "invalid"
    assert result["reason"] == "coordinate outside model-1000 space"
    assert result["pointSource"] == "rect_center"


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


def test_compute_refinement_crop_centers_and_clamps() -> None:
    centered = compute_refinement_crop(source_width=1125, source_height=2436, center_x=560, center_y=1200, crop_ratio=0.38)
    near_edge = compute_refinement_crop(source_width=1125, source_height=2436, center_x=20, center_y=30, crop_ratio=0.38)

    assert centered["width"] == 900
    assert centered["height"] == 900
    assert centered["x"] == 110
    assert centered["y"] == 750
    assert near_edge["x"] == 0
    assert near_edge["y"] == 0


def test_compute_refinement_crop_honors_min_and_max_side() -> None:
    minimum = compute_refinement_crop(source_width=632, source_height=1368, center_x=300, center_y=600, crop_ratio=0.1)
    maximum = compute_refinement_crop(source_width=2000, source_height=3000, center_x=1000, center_y=1500, crop_ratio=0.8)

    assert minimum["width"] == 360
    assert minimum["height"] == 360
    assert maximum["width"] == 900
    assert maximum["height"] == 900


def test_prepare_refinement_crop_writes_crop_and_region(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (1125, 2436), color=(255, 255, 255)).save(source)

    crop = prepare_refinement_crop(source, center={"x": 560, "y": 1200}, output_dir=tmp_path, crop_ratio=0.38)

    assert crop["width"] == 900
    assert png_size(Path(crop["path"])) == (900, 900)
    assert Path(crop["regionPath"]).exists()


def test_remap_crop_grounding_to_source_frame() -> None:
    grounded = {
        "status": "found",
        "point": {"framePx": {"x": 100.0, "y": 200.0}, "normalized": {"x": 0.2, "y": 0.4}},
        "frame": {"widthPx": 500, "heightPx": 500},
    }
    crop = {"x": 50, "y": 60, "width": 500, "height": 500, "sourceWidthPx": 1000, "sourceHeightPx": 1200}

    result = remap_crop_grounding_to_source_frame(grounded, crop=crop)

    assert result["point"]["cropFramePx"] == {"x": 100.0, "y": 200.0}
    assert result["point"]["framePx"] == {"x": 150.0, "y": 260.0}
    assert result["point"]["normalized"] == {"x": 0.15, "y": 260.0 / 1200}
    assert result["frame"] == {"widthPx": 1000, "heightPx": 1200}


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


def test_find_text_uses_structured_ocr_tokens() -> None:
    tokens = [
        OcrToken("General", 95, 10, 20, 30, 12),
        OcrToken("About", 90, 50, 20, 40, 12),
    ]
    match = find_text(tokens, "general")

    assert len(tokens) == 2
    assert match is not None
    assert match["matchedText"] == "General"
    assert match["matchedEngines"] == ["vision"]
    assert match["matchedBoxPx"] == {"x": 10, "y": 20, "width": 30, "height": 12}


def test_parse_vision_json_and_find_text() -> None:
    tokens = parse_vision_json('[{"text":"◎ 搜索","confidence":30,"left":502,"top":2022,"width":120,"height":42}]')
    match = find_text(tokens, "搜索")

    assert len(tokens) == 1
    assert tokens[0].engine == "vision"
    assert match is not None
    assert match["matchedText"] == "◎ 搜索"
    assert match["matchedEngines"] == ["vision"]


def test_default_ocr_language_is_macos_vision_chinese_and_english() -> None:
    assert DEFAULT_OCR_LANG == "zh-Hans+en-US"


def test_find_exact_text_candidates_requires_exact_normalized_match() -> None:
    tokens = [
        OcrToken("ChatGPT", 95, 10, 20, 30, 12),
        OcrToken("ChatGPTX", 95, 50, 20, 40, 12),
        OcrToken("ChatGPT", 20, 100, 20, 30, 12),
    ]
    matches = find_exact_text_candidates(tokens, "chatgpt", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "ChatGPT"
    assert matches[0]["matchedBoxPx"] == {"x": 10, "y": 20, "width": 30, "height": 12}


def test_find_exact_text_candidates_can_match_phrase() -> None:
    tokens = [
        OcrToken("App", 95, 10, 20, 30, 12),
        OcrToken("Store", 90, 50, 20, 40, 12),
    ]
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
    tokens = parse_vision_json('[{"text":"Q 示例应用","confidence":30,"left":10,"top":20,"width":100,"height":30}]')
    matches = find_exact_text_candidates(tokens, "示例应用", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "Q 示例应用"
    assert matches[0]["matchedKind"] == "exact"
    assert matches[0]["exactMatchStrategy"] == "ui-prefix-stripped"


def test_find_exact_text_candidates_accepts_vision_single_letter_ui_prefix() -> None:
    tokens = parse_vision_json('[{"text":"g Xingin","confidence":30,"left":222,"top":978,"width":89,"height":21}]')
    matches = find_exact_text_candidates(tokens, "Xingin", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "g Xingin"
    assert matches[0]["matchedKind"] == "exact"
    assert matches[0]["exactMatchStrategy"] == "ui-prefix-stripped"


def test_find_exact_text_candidates_accepts_badge_ui_prefix() -> None:
    tokens = parse_vision_json('[{"text":"⑧ 示例应用","confidence":30,"left":10,"top":20,"width":100,"height":30}]')
    matches = find_exact_text_candidates(tokens, "示例应用", min_confidence=50)

    assert len(matches) == 1
    assert matches[0]["matchedText"] == "⑧ 示例应用"
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


def test_coredevice_failed_service_is_detected_on_zero_exit() -> None:
    done = Completed(
        argv=["pymobiledevice3", "developer", "core-device", "screen-capture", "screenshot", "--userspace"],
        returncode=0,
        stdout="",
        stderr="ERROR Failed to start service. Possible reasons are: Make sure the DeveloperDiskImage is mounted",
        duration_ms=10,
    )

    with pytest.raises(CoretapError) as exc:
        _check_coredevice_result(done, code="COREDEVICE_SCREENSHOT_FAILED", stage="screenshot")

    assert exc.value.code == "COREDEVICE_SCREENSHOT_FAILED"
    assert exc.value.category == "environment"
    assert exc.value.retryable is True
    assert "mounter auto-mount --userspace" in exc.value.details["suggestedCommands"][0]


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


def test_device_backend_type_text_dry_run_reports_shortcut_paste_mode() -> None:
    result = DeviceBackend().type_text(
        "device-udid",
        "搜索",
        paste_at={"x": 0.25, "y": 0.15, "mode": "shortcut"},
        dry_run=True,
    )

    assert result["attempted"] is False
    assert result["inputMethod"] == "coredevice-pasteboard-keyboard-shortcut"


def test_device_backend_set_pasteboard_text_cli_fallback_verifies_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        if "paste" in argv:
            return Completed(argv=argv, returncode=0, stdout="小红书", stderr="", duration_ms=2)
        return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=3)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().set_pasteboard_text("device-udid", "小红书", verify=True)

    assert result["pasteboardSet"] is True
    assert result["pasteboardVerified"] is True
    assert result["readbackMatches"] is True
    assert calls[0][:4] == ["pymobiledevice3", "developer", "core-device", "copy"]
    assert calls[1][:4] == ["pymobiledevice3", "developer", "core-device", "paste"]


def test_device_backend_set_pasteboard_text_rejects_readback_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        stdout = "xiaohongshu" if "paste" in argv else ""
        return Completed(argv=argv, returncode=0, stdout=stdout, stderr="", duration_ms=2)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    with pytest.raises(CoretapError) as exc:
        DeviceBackend().set_pasteboard_text("device-udid", "小红书", verify=True)

    assert exc.value.code == "PASTEBOARD_SET_FAILED"


def test_device_backend_terminate_app_sends_signal_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    process_lookup_count = 0

    def fake_run_command(argv: list[str], **kwargs: object) -> Completed:
        nonlocal process_lookup_count
        calls.append(argv)
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert env["PYMOBILEDEVICE3_UDID"] == "device-udid"
        if "process-id-for-bundle-id" in argv:
            process_lookup_count += 1
            stdout = "2372\n" if process_lookup_count == 1 else "0\n"
            return Completed(argv=argv, returncode=0, stdout=stdout, stderr="", duration_ms=2)
        if "send-signal-to-process" in argv:
            return Completed(argv=argv, returncode=0, stdout='{"pid":2372}\n', stderr="", duration_ms=3)
        if "is-running-pid" in argv:
            return Completed(argv=argv, returncode=0, stdout="false\n", stderr="", duration_ms=1)
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().terminate_app("device-udid", "com.apple.AppStore")

    assert result["status"] == "terminated"
    assert result["pidBefore"] == 2372
    assert result["pidAfter"] == 0
    assert result["runningAfter"] is False
    assert calls[0][:5] == ["pymobiledevice3", "developer", "dvt", "process-id-for-bundle-id", "--userspace"]
    assert calls[1][:5] == ["pymobiledevice3", "developer", "core-device", "send-signal-to-process", "--userspace"]
    assert calls[1][-2:] == ["2372", "9"]


def test_device_backend_terminate_app_does_not_treat_stale_pid_as_running(monkeypatch: pytest.MonkeyPatch) -> None:
    process_lookup_count = 0
    checked_pids: list[str] = []

    def fake_run_command(argv: list[str], **kwargs: object) -> Completed:
        nonlocal process_lookup_count
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert env["PYMOBILEDEVICE3_UDID"] == "device-udid"
        if "process-id-for-bundle-id" in argv:
            process_lookup_count += 1
            stdout = "2372\n" if process_lookup_count == 1 else "4287\n"
            return Completed(argv=argv, returncode=0, stdout=stdout, stderr="", duration_ms=2)
        if "send-signal-to-process" in argv:
            return Completed(argv=argv, returncode=0, stdout='{"pid":2372}\n', stderr="", duration_ms=3)
        if "is-running-pid" in argv:
            checked_pids.append(argv[-1])
            return Completed(argv=argv, returncode=0, stdout="false\n", stderr="", duration_ms=1)
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().terminate_app("device-udid", "com.xingin.discover")

    assert result["status"] == "terminated"
    assert result["pidBefore"] == 2372
    assert result["pidAfter"] == 4287
    assert result["runningAfter"] is False
    assert result["pidAfterRunning"] is False
    assert checked_pids == ["2372", "4287"]


def test_device_backend_terminate_app_is_idempotent_when_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        return Completed(argv=argv, returncode=0, stdout="0\n", stderr="", duration_ms=2)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().terminate_app("device-udid", "com.apple.AppStore")

    assert result["status"] == "not_running"
    assert result["attempted"] is False
    assert calls == [["pymobiledevice3", "developer", "dvt", "process-id-for-bundle-id", "--userspace", "com.apple.AppStore"]]


def test_device_backend_uninstall_app_uses_pymobiledevice3_apps_uninstall(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **kwargs: object) -> Completed:
        calls.append(argv)
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert env["PYMOBILEDEVICE3_UDID"] == "device-udid"
        return Completed(argv=argv, returncode=0, stdout="Uninstalled\n", stderr="", duration_ms=17)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().uninstall_app("device-udid", "com.xingin.discover")

    assert calls == [["pymobiledevice3", "apps", "uninstall", "--userspace", "com.xingin.discover"]]
    assert result["status"] == "uninstalled"
    assert result["attempted"] is True
    assert result["stdout"] == "Uninstalled"


def test_device_backend_uninstall_app_is_idempotent_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        return Completed(argv=argv, returncode=1, stdout="", stderr="Application is not installed", duration_ms=9)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().uninstall_app("device-udid", "com.xingin.discover")

    assert result["status"] == "not_installed"
    assert result["attempted"] is False


def test_simulator_backend_open_url_uses_simctl_openurl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        calls.append(argv)
        return Completed(argv=argv, returncode=0, stdout="", stderr="", duration_ms=11)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = SimulatorBackend().open_url("booted", "https://example.com/search?q=openai", timeout_sec=4)

    assert calls == [["xcrun", "simctl", "openurl", "booted", "https://example.com/search?q=openai"]]
    assert result["strategy"] == "simctl-openurl"
    assert result["attempted"] is True


def test_device_backend_open_url_uses_webinspector_launch_userspace(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **kwargs: object) -> Completed:
        calls.append(argv)
        env = kwargs.get("env")
        assert isinstance(env, dict)
        assert env["PYMOBILEDEVICE3_UDID"] == "device-udid"
        return Completed(argv=argv, returncode=0, stdout="launched\n", stderr="", duration_ms=23)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    result = DeviceBackend().open_url("device-udid", "https://www.google.com/search?q=openai", timeout_sec=7)

    assert calls == [
        [
            "pymobiledevice3",
            "webinspector",
            "launch",
            "--userspace",
            "--timeout",
            "7",
            "https://www.google.com/search?q=openai",
        ]
    ]
    assert result["strategy"] == "webinspector-launch"
    assert result["coredeviceTunnelMode"] == "userspace"
    assert result["stdout"] == "launched"


def test_device_backend_open_url_reports_webinspector_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(argv: list[str], **_kwargs: object) -> Completed:
        return Completed(argv=argv, returncode=1, stdout="", stderr="Remote Automation is disabled", duration_ms=19)

    monkeypatch.setattr("coretap.backends.run_command", fake_run_command)

    with pytest.raises(CoretapError) as exc:
        DeviceBackend().open_url("device-udid", "https://example.com")

    assert exc.value.code == "WEBINSPECTOR_OPEN_URL_FAILED"
    assert exc.value.category == "environment"
    assert "Remote Automation" in "\n".join(exc.value.details["requiredSettings"])


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


def test_device_backend_drag_falls_back_to_helper_after_worker_tunnel_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingPool:
        def drag_userspace(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise CoretapError(
                "COREDEVICE_DRAG_FAILED",
                "a userspace tunnel is already active in this process",
                stage="drag",
                retryable=True,
            )

    helper_calls: list[dict[str, object]] = []

    def fake_helper(self: DeviceBackend, device: str, **kwargs: object) -> dict[str, object]:
        helper_calls.append({"device": device, **kwargs})
        return {
            "attempted": True,
            "dryRun": False,
            "dispatchStatus": "sent",
            "confirmationStatus": "not_requested",
        }

    set_default_device_worker_pool(FailingPool())  # type: ignore[arg-type]
    monkeypatch.setattr(DeviceBackend, "_drag_userspace_helper", fake_helper)
    try:
        result = DeviceBackend().drag_normalized(
            "device-udid",
            0.2,
            0.3,
            0.4,
            0.5,
            dry_run=False,
            steps=12,
            duration_ms=1600,
        )
    finally:
        set_default_device_worker_pool(None)

    assert result["dispatchStatus"] == "sent"
    assert result["workerFallback"] == "coretap-device-hid-helper"
    assert result["previousError"]["code"] == "COREDEVICE_DRAG_FAILED"
    assert helper_calls == [
        {
            "device": "device-udid",
            "start_x": 0.2,
            "start_y": 0.3,
            "end_x": 0.4,
            "end_y": 0.5,
            "start_hx": round(0.2 * 65535),
            "start_hy": round(0.3 * 65535),
            "end_hx": round(0.4 * 65535),
            "end_hy": round(0.5 * 65535),
            "steps": 12,
            "duration_ms": 1600,
        }
    ]


def test_device_backend_drag_uses_helper_without_registered_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    helper_calls: list[dict[str, object]] = []

    def fake_helper(self: DeviceBackend, device: str, **kwargs: object) -> dict[str, object]:
        helper_calls.append({"device": device, **kwargs})
        return {"dispatchStatus": "sent", "attempted": True, "dryRun": False}

    set_default_device_worker_pool(None)
    monkeypatch.setattr(DeviceBackend, "_drag_userspace_helper", fake_helper)

    result = DeviceBackend().drag_normalized("device-udid", 0.1, 0.2, 0.3, 0.4, dry_run=False)

    assert result["dispatchStatus"] == "sent"
    assert helper_calls[0]["device"] == "device-udid"


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

        async def fake_clear_focused_text(target: object, *, count: int = 80) -> int:
            calls.append(("clear", (target, count)))
            return count

        async def fake_select_all_focused_text(_target: object) -> dict:
            raise AssertionError("replace=true should not use command+a on iOS")

        async def fake_clear_focused_text_for_replace(target: object) -> dict:
            clear_count = await fake_clear_focused_text(target, count=80)
            return {
                "replaceStrategy": "backspace-clear",
                "clearKeypresses": clear_count,
            }

        async def fake_paste_text(*_args: object, **_kwargs: object) -> dict:
            raise AssertionError("ASCII input should not use pasteboard")

        monkeypatch.setattr(pool, "_get_or_open_keyboard_session", fake_get_or_open_keyboard_session)
        monkeypatch.setattr(pool, "_send_text", fake_send_text)
        monkeypatch.setattr(pool, "_select_all_focused_text", fake_select_all_focused_text)
        monkeypatch.setattr(pool, "_clear_focused_text_for_replace", fake_clear_focused_text_for_replace)
        monkeypatch.setattr(pool, "_paste_text", fake_paste_text)

        result = await pool._type_text_userspace(
            "device-udid",
            text="sampleapp",
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
    assert result["replaceStrategy"] == "backspace-clear"
    assert result["sessionTypedCharacterCount"] == len("sampleapp")
    assert result["calls"][0] == ("open", "device-udid")
    assert result["calls"][1][0] == "clear"
    assert result["calls"][2][0] == "send-text"


def test_coredevice_worker_treats_shifted_ascii_as_paste_text() -> None:
    assert _supports_unshifted_virtual_keyboard_text("sampleapp") is True
    assert _supports_unshifted_virtual_keyboard_text("example.com/path") is True
    assert _supports_unshifted_virtual_keyboard_text("openai") is True
    assert _supports_unshifted_virtual_keyboard_text("Safari") is True
    assert _supports_unshifted_virtual_keyboard_text("OpenAI") is True
    assert _supports_unshifted_virtual_keyboard_text("openai codex") is False
    assert _supports_unshifted_virtual_keyboard_text("https://example.com") is False
    assert _supports_unshifted_virtual_keyboard_text("Hello!") is False


def test_coredevice_worker_pastes_shifted_ascii_with_last_tap_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        calls: list[tuple[str, object]] = []
        touch_session = SimpleNamespace(
            device="device-udid",
            typed_characters=0,
            keyboard_service_id=None,
            last_used_at=0,
            last_tap_normalized={"x": 0.5, "y": 0.54},
        )
        pool._sessions["device-udid"] = touch_session

        async def fake_get_or_open_session(device: str) -> tuple[object, str]:
            calls.append(("open-touch", device))
            return touch_session, "reused"

        async def fake_paste_text(
            target: object,
            *,
            text: str,
            char_delay_ms: int,
            inter_delay_ms: int,
            paste_at: dict[str, float] | None,
            paste_hold_ms: int,
            clear_existing: bool,
        ) -> dict:
            calls.append(("paste", (target, text, paste_at, paste_hold_ms, clear_existing)))
            return {"inputMethod": "coredevice-pasteboard-edit-menu", "pasteboardSet": True, "pasteAnchor": paste_at}

        async def fake_get_or_open_keyboard_session(_device: str) -> tuple[object, str]:
            raise AssertionError("shifted ASCII should not use virtual keyboard")

        monkeypatch.setattr(pool, "_get_or_open_session", fake_get_or_open_session)
        monkeypatch.setattr(pool, "_paste_text", fake_paste_text)
        monkeypatch.setattr(pool, "_get_or_open_keyboard_session", fake_get_or_open_keyboard_session)

        result = await pool._type_text_userspace(
            "device-udid",
            text="https://example.com",
            char_delay_ms=1,
            inter_delay_ms=2,
            paste_at=None,
            paste_hold_ms=1300,
            clear_existing=False,
        )
        result["calls"] = calls
        return result

    result = asyncio.run(run())

    assert result["inputMethod"] == "coredevice-pasteboard-edit-menu"
    assert result["pasteboardSet"] is True
    assert result["pasteAnchor"] == {"x": 0.5, "y": 0.54, "source": "last-tap"}
    assert result["sessionTypedCharacterCount"] == len("https://example.com")
    assert result["calls"][0] == ("open-touch", "device-udid")
    assert result["calls"][1][0] == "paste"


def test_coredevice_worker_types_cjk_with_last_tap_paste_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        calls: list[tuple[str, object]] = []
        touch_session = SimpleNamespace(
            device="device-udid",
            typed_characters=0,
            keyboard_service_id=None,
            last_used_at=0,
            last_tap_normalized={"x": 0.25, "y": 0.09},
        )
        pool._sessions["device-udid"] = touch_session

        async def fake_get_or_open_session(device: str) -> tuple[object, str]:
            calls.append(("open-touch", device))
            return touch_session, "reused"

        async def fake_paste_text(
            target: object,
            *,
            text: str,
            char_delay_ms: int,
            inter_delay_ms: int,
            paste_at: dict[str, float] | None,
            paste_hold_ms: int,
            clear_existing: bool,
        ) -> dict:
            calls.append(("paste", (target, text, paste_at, paste_hold_ms, clear_existing)))
            return {
                "inputMethod": "coredevice-pasteboard-edit-menu",
                "pasteboardSet": True,
                "pasteAnchor": paste_at,
                "clearExisting": clear_existing,
            }

        async def fake_get_or_open_keyboard_session(_device: str) -> tuple[object, str]:
            raise AssertionError("CJK step input with an anchor should prefer pasteboard over pinyin")

        monkeypatch.setattr(pool, "_get_or_open_session", fake_get_or_open_session)
        monkeypatch.setattr(pool, "_paste_text", fake_paste_text)
        monkeypatch.setattr(pool, "_get_or_open_keyboard_session", fake_get_or_open_keyboard_session)

        result = await pool._type_text_userspace(
            "device-udid",
            text="测试",
            char_delay_ms=1,
            inter_delay_ms=2,
            paste_at=None,
            paste_hold_ms=1300,
            clear_existing=False,
        )
        result["calls"] = calls
        return result

    result = asyncio.run(run())

    assert result["inputMethod"] == "coredevice-pasteboard-edit-menu"
    assert result["pasteboardSet"] is True
    assert result["pasteAnchor"] == {"x": 0.25, "y": 0.09, "source": "last-tap"}
    assert result["sessionTypedCharacterCount"] == len("测试")
    assert result["calls"][0] == ("open-touch", "device-udid")
    assert result["calls"][1][0] == "paste"


def test_coredevice_worker_types_cjk_without_paste_anchor_using_current_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run() -> dict:
        pool = CoreDeviceWorkerPool()
        pool._lock = asyncio.Lock()
        calls: list[tuple[str, object]] = []
        session = SimpleNamespace(
            device="device-udid",
            typed_characters=0,
            keyboard_service_id=None,
            last_used_at=0,
        )

        async def fake_get_or_open_keyboard_session(device: str) -> tuple[object, str]:
            calls.append(("open-keyboard", device))
            return session, "created"

        async def fake_input_pinyin_keyboard_text(
            target: object,
            *,
            text: str,
            char_delay_ms: int,
            inter_delay_ms: int,
            clear_existing: bool,
        ) -> dict:
            calls.append(("input-pinyin", (target, text, char_delay_ms, inter_delay_ms, clear_existing)))
            return {"inputMethod": "coredevice-pinyin-keyboard", "pasteboardSet": False, "convertedText": "ceshi"}

        monkeypatch.setattr(pool, "_get_or_open_keyboard_session", fake_get_or_open_keyboard_session)
        monkeypatch.setattr(pool, "_input_pinyin_keyboard_text", fake_input_pinyin_keyboard_text)

        result = await pool._type_text_userspace(
            "device-udid",
            text="测试",
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
    assert result["pasteboardSet"] is False
    assert result["calls"][0] == ("open-keyboard", "device-udid")
    assert result["calls"][1][0] == "input-pinyin"


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
