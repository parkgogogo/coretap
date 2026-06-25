from __future__ import annotations

from pathlib import Path

import pytest

from coretap.backends import DeviceBackend, _check_coredevice_result, _coredevice_screenshot_rotation, parse_usbmux_devices
from coretap.cli import point_to_hid
from coretap.model_pack import parse_grounding_output
from coretap.ocr import find_exact_text_candidates, find_text, parse_tsv
from coretap.runtime import Completed, CoretapError, png_size


def test_point_to_hid_from_normalized() -> None:
    point = point_to_hid(0.5, 0.25, width=1000, height=2000, space="normalized")

    assert point["hidU16"] == {"x": 32768, "y": 16384}
    assert point["screenshotPx"] == {"x": 500.0, "y": 500.0}


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
    assert match["matchedBoxPx"] == {"x": 10, "y": 20, "width": 30, "height": 12}


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


def test_coredevice_tunneld_mode_omits_userspace() -> None:
    backend = DeviceBackend(coredevice_tunnel_mode="tunneld")

    assert backend.coredevice_device_options("device-udid") == ["--tunnel", "device-udid"]
    assert backend.coredevice_env("device-udid").get("PYMOBILEDEVICE3_UDID") is None


def test_coredevice_screenshot_rotation_matches_display_orientation() -> None:
    assert _coredevice_screenshot_rotation((2736, 1260), (1260, 2736), "landscapeRight") == 270
    assert _coredevice_screenshot_rotation((2736, 1260), (1260, 2736), "landscapeLeft") == 90
    assert _coredevice_screenshot_rotation((1260, 2736), (1260, 2736), "portrait") is None


def test_png_size(tmp_path: Path) -> None:
    png = tmp_path / "tiny.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x03"
        b"\x00\x00\x00\x05"
        b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
    )

    assert png_size(png) == (3, 5)
