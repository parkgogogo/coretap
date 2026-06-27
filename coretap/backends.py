from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coretap.device_buttons import BUTTON_STATES, resolve_button
from coretap.runtime import (
    Completed,
    CoretapError,
    command_env,
    png_size,
    require_success,
    run_command,
)
from coretap.text_input import text_input_summary


@dataclass(frozen=True)
class Device:
    udid: str
    name: str | None
    backend: str
    state: str | None = None
    runtime: str | None = None
    eligible: bool = True
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class Frame:
    frame_id: str
    path: Path
    width: int
    height: int
    backend: str
    device: str


class SimulatorBackend:
    name = "simulator"

    def __init__(self, *, developer_dir: str | None = None) -> None:
        self.env = command_env(developer_dir)

    def discover(self) -> list[Device]:
        done = run_command(
            ["xcrun", "simctl", "list", "devices", "available", "--json"],
            env=self.env,
            timeout=10,
        )
        require_success(done, code="SIMCTL_LIST_FAILED", stage="discover")
        data = json.loads(done.stdout)
        devices: list[Device] = []
        for runtime, entries in data.get("devices", {}).items():
            if "iOS" not in runtime:
                continue
            for entry in entries:
                name = entry.get("name")
                state = entry.get("state")
                devices.append(
                    Device(
                        udid=entry["udid"],
                        name=name,
                        backend=self.name,
                        state=state,
                        runtime=runtime,
                        eligible=entry.get("isAvailable", False),
                        details=entry,
                    )
                )
        return devices

    def boot(self, device: str) -> None:
        done = run_command(["xcrun", "simctl", "boot", device], env=self.env, timeout=30)
        if done.returncode != 0 and "Unable to boot device in current state: Booted" not in done.stderr:
            require_success(done, code="SIMCTL_BOOT_FAILED", stage="boot")
        require_success(
            run_command(["xcrun", "simctl", "bootstatus", device, "-b"], env=self.env, timeout=60),
            code="SIMCTL_BOOTSTATUS_FAILED",
            stage="boot",
        )

    def screenshot(self, device: str, out: Path) -> Frame:
        out.parent.mkdir(parents=True, exist_ok=True)
        require_success(
            run_command(
                ["xcrun", "simctl", "io", device, "screenshot", "--type=png", str(out)],
                env=self.env,
                timeout=30,
            ),
            code="SIMCTL_SCREENSHOT_FAILED",
            stage="screenshot",
        )
        width, height = png_size(out)
        return Frame(f"frame_{out.stem}", out, width, height, self.name, device)

    def tap_hid(self, device: str, x: int, y: int, *, dry_run: bool) -> dict[str, Any]:
        return self.tap_normalized(device, x / 65535, y / 65535, dry_run=dry_run, hid_u16={"x": x, "y": y})

    def tap_normalized(
        self,
        device: str,
        x: float,
        y: float,
        *,
        dry_run: bool,
        hid_u16: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if dry_run:
            return {
                "attempted": False,
                "dryRun": True,
                "reason": "dry-run requested",
                "normalized": {"x": x, "y": y},
                "hidU16": hid_u16,
            }
        width, height = self._idb_screen_size(device)
        point_x = round(x * width)
        point_y = round(y * height)
        done = require_success(
            run_command(
                [
                    *self._idb_base_command(),
                    "ui",
                    "tap",
                    "--udid",
                    device,
                    str(point_x),
                    str(point_y),
                ],
                env=self.env,
                timeout=20,
            ),
            code="SIMULATOR_TAP_FAILED",
            stage="tap",
        )
        return {
            "attempted": True,
            "dryRun": False,
            "normalized": {"x": x, "y": y},
            "hidU16": hid_u16,
            "idbPoint": {"x": point_x, "y": point_y},
            "idbScreen": {"width": width, "height": height},
            "durationMs": done.duration_ms,
        }

    def press_button(
        self,
        device: str,
        button: str,
        *,
        state: str = "press",
        hold_ms: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        raise CoretapError(
            "SIMULATOR_PRESS_UNSUPPORTED",
            "CoreDevice hardware button events are only supported for real devices",
            category="usage",
            stage="press",
            details={"backend": self.name, "button": button, "state": state},
        )

    def type_text(
        self,
        device: str,
        text: str,
        *,
        char_delay_ms: int = 40,
        inter_delay_ms: int = 20,
        paste_at: dict[str, float] | None = None,
        paste_hold_ms: int = 1600,
        clear_existing: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        raise CoretapError(
            "SIMULATOR_TYPE_UNSUPPORTED",
            "CoreDevice virtual keyboard text input is only supported for real devices",
            category="usage",
            stage="type",
            details={"backend": self.name},
        )

    def drag_normalized(
        self,
        device: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        *,
        dry_run: bool,
        start_hid_u16: dict[str, int] | None = None,
        end_hid_u16: dict[str, int] | None = None,
        steps: int = 30,
        duration_ms: int = 600,
    ) -> dict[str, Any]:
        raise CoretapError(
            "SIMULATOR_DRAG_UNSUPPORTED",
            "CoreDevice drag events are only supported for real devices",
            category="usage",
            stage="drag",
            details={"backend": self.name},
        )

    def _idb_base_command(self) -> list[str]:
        companion = self._idb_companion_path()
        base = ["uvx", "--from", "fb-idb", "idb"]
        if companion:
            base.extend(["--companion-path", str(companion)])
        return base

    def _idb_companion_path(self) -> Path | None:
        configured = os.environ.get("CORETAP_IDB_COMPANION_PATH")
        if configured:
            path = Path(configured)
            if path.exists():
                return path
        managed = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Coretap"
            / "tools"
            / "idb-companion"
            / "1.1.8"
            / "idb-companion.universal"
            / "bin"
            / "idb_companion"
        )
        if managed.exists():
            return managed
        local = Path(__file__).resolve().parents[1] / ".tools" / "idb-companion.universal" / "bin" / "idb_companion"
        if local.exists():
            return local
        return None

    def _idb_screen_size(self, device: str) -> tuple[float, float]:
        done = require_success(
            run_command(
                [*self._idb_base_command(), "ui", "describe-all", "--udid", device, "--json"],
                env=self.env,
                timeout=20,
                max_output=10_000_000,
            ),
            code="SIMULATOR_DESCRIBE_FAILED",
            stage="tap",
        )
        try:
            elements = json.loads(done.stdout)
            root = elements[0]["frame"]
            return float(root["width"]), float(root["height"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CoretapError(
                "SIMULATOR_DESCRIBE_FAILED",
                "Could not parse simulator screen size from idb describe-all",
                stage="tap",
                details={"stdout": done.stdout[:1000], "stderr": done.stderr[:1000]},
            ) from exc


class DeviceBackend:
    name = "device"

    def __init__(self, *, coredevice_tunnel_mode: str | None = None) -> None:
        self.coredevice_tunnel_mode = self._resolve_coredevice_tunnel_mode(coredevice_tunnel_mode)

    @staticmethod
    def _resolve_coredevice_tunnel_mode(mode: str | None) -> str:
        resolved = mode or os.environ.get("CORETAP_COREDEVICE_TUNNEL_MODE") or "userspace"
        if resolved not in {"userspace", "tunneld"}:
            raise CoretapError(
                "INVALID_ARGUMENT",
                f"Unsupported CoreDevice tunnel mode: {resolved}",
                category="usage",
                stage="config",
                details={"validModes": ["userspace", "tunneld"]},
            )
        return resolved

    def coredevice_device_options(self, device: str) -> list[str]:
        if self.coredevice_tunnel_mode == "userspace":
            return ["--userspace"]
        return ["--tunnel", device]

    def coredevice_env(self, device: str) -> dict[str, str]:
        env = os.environ.copy()
        if self.coredevice_tunnel_mode == "userspace":
            env["PYMOBILEDEVICE3_UDID"] = device
        return env

    def discover(self) -> list[Device]:
        done = run_command(["pymobiledevice3", "usbmux", "list"], timeout=10)
        if done.returncode != 0:
            raise CoretapError(
                "PYMOBILEDEVICE3_DISCOVER_FAILED",
                "Failed to list usbmux devices",
                stage="discover",
                details={"stdout": done.stdout, "stderr": done.stderr},
            )
        return parse_usbmux_devices(done.stdout)

    def screenshot(self, device: str, out: Path) -> Frame:
        out.parent.mkdir(parents=True, exist_ok=True)
        capture: dict[str, Any] | None = None
        if self.coredevice_tunnel_mode == "userspace":
            from coretap.device_worker import get_default_device_worker_pool

            pool = get_default_device_worker_pool()
            if pool is not None:
                try:
                    capture = pool.capture_screenshot_userspace(device)
                    out.write_bytes(capture["image"])
                except CoretapError as exc:
                    if not exc.retryable:
                        raise
                    capture = self._screenshot_cli_userspace_fallback(device, out, previous_error=exc)
        if capture is None:
            done = _check_coredevice_result(
                run_command(
                    [
                        "pymobiledevice3",
                        "developer",
                        "core-device",
                        "screen-capture",
                        "screenshot",
                        *self.coredevice_device_options(device),
                        str(out),
                    ],
                    env=self.coredevice_env(device),
                    timeout=20,
                ),
                code="COREDEVICE_SCREENSHOT_FAILED",
                stage="screenshot",
            )
            require_success(done, code="COREDEVICE_SCREENSHOT_FAILED", stage="screenshot")
        if not out.exists() or out.stat().st_size == 0:
            raise CoretapError(
                "COREDEVICE_SCREENSHOT_EMPTY",
                "CoreDevice screenshot did not produce a valid PNG",
                stage="screenshot",
                details={"path": str(out), "capture": _public_capture_metadata(capture)},
            )
        self._normalize_screenshot_orientation(device, out)
        width, height = png_size(out)
        return Frame(f"frame_{out.stem}", out, width, height, self.name, device)

    def _screenshot_cli_userspace_fallback(self, device: str, out: Path, *, previous_error: CoretapError) -> dict[str, Any]:
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "screen-capture",
                    "screenshot",
                    "--userspace",
                    str(out),
                ],
                env=self.coredevice_env(device),
                timeout=30,
            ),
            code="COREDEVICE_SCREENSHOT_FAILED",
            stage="screenshot",
        )
        require_success(done, code="COREDEVICE_SCREENSHOT_FAILED", stage="screenshot")
        return {
            "fallback": "pymobiledevice3-cli-userspace",
            "previousError": {
                "code": previous_error.code,
                "message": str(previous_error),
                "details": previous_error.details,
            },
            "durationMs": done.duration_ms,
        }

    def display_info(self, device: str) -> dict[str, Any]:
        previous_error: CoretapError | None = None
        if self.coredevice_tunnel_mode == "userspace":
            from coretap.device_worker import get_default_device_worker_pool, is_recoverable_userspace_tunnel_error

            pool = get_default_device_worker_pool()
            if pool is not None:
                try:
                    return pool.display_info_userspace(device)
                except CoretapError as exc:
                    if not (exc.retryable or is_recoverable_userspace_tunnel_error(exc)):
                        raise
                    previous_error = exc
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "get-display-info",
                    *self.coredevice_device_options(device),
                ],
                env=self.coredevice_env(device),
                timeout=20,
                max_output=10_000_000,
            ),
            code="COREDEVICE_DISPLAY_INFO_FAILED",
            stage="display-info",
        )
        require_success(done, code="COREDEVICE_DISPLAY_INFO_FAILED", stage="display-info")
        try:
            data = json.loads(done.stdout)
        except json.JSONDecodeError as exc:
            raise CoretapError(
                "COREDEVICE_DISPLAY_INFO_INVALID",
                "Could not parse CoreDevice display info JSON",
                stage="display-info",
                details={"stdout": done.stdout[:1000], "stderr": done.stderr[:1000]},
            ) from exc
        if not isinstance(data, dict):
            raise CoretapError(
                "COREDEVICE_DISPLAY_INFO_INVALID",
                "CoreDevice display info was not a JSON object",
                stage="display-info",
                details={"stdout": done.stdout[:1000]},
            )
        if previous_error is not None:
            data["_coretap"] = {
                "fallback": "pymobiledevice3-cli-userspace",
                "previousError": _previous_error_metadata(previous_error),
                "durationMs": done.duration_ms,
            }
        return data

    def _normalize_screenshot_orientation(self, device: str, out: Path) -> None:
        image_size = png_size(out)
        info = self.display_info(device)
        display_size = _primary_display_size(info)
        orientation = _device_non_flat_orientation(info)
        rotation = _coredevice_screenshot_rotation(image_size, display_size, orientation)
        if rotation is None:
            return

        tmp = out.with_name(f"{out.stem}.normalized{out.suffix}")
        require_success(
            run_command(["sips", "-r", str(rotation), str(out), "--out", str(tmp)], timeout=20),
            code="SCREENSHOT_ORIENTATION_NORMALIZE_FAILED",
            stage="screenshot",
        )
        tmp.replace(out)
        normalized_size = png_size(out)
        if normalized_size != display_size:
            raise CoretapError(
                "SCREENSHOT_ORIENTATION_NORMALIZE_FAILED",
                "Normalized screenshot size did not match the primary display size",
                stage="screenshot",
                details={
                    "path": str(out),
                    "imageSize": list(image_size),
                    "displaySize": list(display_size),
                    "normalizedSize": list(normalized_size),
                    "rotation": rotation,
                    "orientation": orientation,
                },
            )

    def tap_hid(self, device: str, x: int, y: int, *, dry_run: bool) -> dict[str, Any]:
        return self.tap_normalized(device, x / 65535, y / 65535, dry_run=dry_run, hid_u16={"x": x, "y": y})

    def tap_normalized(
        self,
        device: str,
        x: float,
        y: float,
        *,
        dry_run: bool,
        hid_u16: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        hx = int(round(x * 65535)) if hid_u16 is None else hid_u16["x"]
        hy = int(round(y * 65535)) if hid_u16 is None else hid_u16["y"]
        if dry_run:
            return {
                "attempted": False,
                "dryRun": True,
                "normalized": {"x": x, "y": y},
                "hidU16": {"x": hx, "y": hy},
                "coredeviceTunnelMode": self.coredevice_tunnel_mode,
            }
        if self.coredevice_tunnel_mode == "userspace":
            return self._tap_userspace_direct(device, x, y, hx, hy)
        return self._tap_cli(device, x, y, hx, hy)

    def _tap_userspace_direct(self, device: str, x: float, y: float, hx: int, hy: int) -> dict[str, Any]:
        from coretap.device_worker import get_default_device_worker_pool, is_recoverable_userspace_tunnel_error

        pool = get_default_device_worker_pool()
        if pool is not None:
            try:
                return pool.tap_userspace(device, x, y, hx, hy)
            except CoretapError as exc:
                if not (exc.retryable or is_recoverable_userspace_tunnel_error(exc)):
                    raise
                result = self._tap_userspace_helper(device, x, y, hx, hy)
                return {
                    **result,
                    "workerFallback": "coretap-device-hid-helper",
                    "previousError": _previous_error_metadata(exc),
                }

        return self._tap_userspace_helper(device, x, y, hx, hy)

    def _tap_userspace_helper(self, device: str, x: float, y: float, hx: int, hy: int) -> dict[str, Any]:
        timeout = 15.0
        cleanup_grace = 1.0
        started = time.monotonic()
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "coretap.device_hid_helper",
                "--mode",
                "userspace",
                "--action",
                "tap",
                "--device",
                device,
                "--x",
                str(hx),
                "--y",
                str(hy),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.coredevice_env(device),
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        dispatch: dict[str, Any] | None = None
        try:
            while time.monotonic() - started < timeout:
                remaining = max(0.0, timeout - (time.monotonic() - started))
                events = selector.select(timeout=remaining)
                if not events:
                    break
                line = proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "dispatch_sent":
                    dispatch = event
                    break
                if event.get("event") == "error":
                    stderr = _terminate_process(proc)
                    raise CoretapError(
                        "COREDEVICE_TAP_FAILED",
                        "Direct CoreDevice HID helper failed before dispatch",
                        stage="tap",
                        details={"helperEvent": event, "stderr": stderr},
                    )
            if dispatch is None:
                stderr = _terminate_process(proc)
                raise CoretapError(
                    "COREDEVICE_TAP_FAILED",
                    "Timed out waiting for direct CoreDevice HID dispatch",
                    stage="tap",
                    retryable=True,
                    details={"timeoutMs": round(timeout * 1000), "stderr": stderr},
                )
            session_status = "exited"
            try:
                proc.wait(timeout=cleanup_grace)
            except subprocess.TimeoutExpired:
                _terminate_process(proc)
                session_status = "terminated_after_dispatch"
            return {
                "attempted": True,
                "dryRun": False,
                "normalized": {"x": x, "y": y},
                "hidU16": {"x": hx, "y": hy},
                "coredeviceTunnelMode": self.coredevice_tunnel_mode,
                "dispatchStatus": "sent",
                "confirmationStatus": "not_requested",
                "sessionStatus": session_status,
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        finally:
            selector.close()

    def _tap_cli(self, device: str, x: float, y: float, hx: int, hy: int) -> dict[str, Any]:
        timeout = 20
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "universal-hid-service",
                    "tap",
                    *self.coredevice_device_options(device),
                    str(hx),
                    str(hy),
                ],
                env=self.coredevice_env(device),
                timeout=timeout,
            ),
            code="COREDEVICE_TAP_FAILED",
            stage="tap",
        )
        require_success(done, code="COREDEVICE_TAP_FAILED", stage="tap")
        return {
            "attempted": True,
            "dryRun": False,
            "normalized": {"x": x, "y": y},
            "hidU16": {"x": hx, "y": hy},
            "coredeviceTunnelMode": self.coredevice_tunnel_mode,
            "completionStatus": "exited",
            "deliveryStatus": "sent",
            "durationMs": done.duration_ms,
        }

    def press_button(
        self,
        device: str,
        button: str,
        *,
        state: str = "press",
        hold_ms: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        spec = resolve_button(button)
        if spec is None:
            raise CoretapError(
                "INVALID_ARGUMENT",
                f"Unsupported CoreDevice button: {button}",
                category="usage",
                stage="press",
            )
        if state not in BUTTON_STATES:
            raise CoretapError(
                "INVALID_ARGUMENT",
                f"Unsupported CoreDevice button state: {state}",
                category="usage",
                stage="press",
                details={"validStates": list(BUTTON_STATES)},
            )
        resolved_hold_ms = spec.hold_ms if hold_ms is None else hold_ms
        base = {
            "button": spec.name,
            "requestedButton": button,
            "state": state,
            "hidButton": {"usagePage": spec.usage_page, "usageCode": spec.usage_code},
            "holdMs": resolved_hold_ms if state == "press" else 0,
            "coredeviceTunnelMode": self.coredevice_tunnel_mode,
        }
        if dry_run:
            return {
                **base,
                "attempted": False,
                "dryRun": True,
                "reason": "dry-run requested",
            }
        if self.coredevice_tunnel_mode == "userspace":
            from coretap.device_worker import get_default_device_worker_pool, is_recoverable_userspace_tunnel_error

            pool = get_default_device_worker_pool()
            if pool is not None:
                try:
                    result = pool.press_button_userspace(
                        device,
                        button=spec.name,
                        state=state,
                        usage_page=spec.usage_page,
                        usage_code=spec.usage_code,
                        hold_ms=resolved_hold_ms,
                    )
                    return {**base, **result, "attempted": True, "dryRun": False}
                except CoretapError as exc:
                    if not (exc.retryable or is_recoverable_userspace_tunnel_error(exc)):
                        raise
                    previous_error = exc
            else:
                previous_error = None
        else:
            previous_error = None
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "hid",
                    "button",
                    *self.coredevice_device_options(device),
                    spec.name,
                    state,
                ],
                env=self.coredevice_env(device),
                timeout=max(10, int((resolved_hold_ms / 1000) + 5)),
            ),
            code="COREDEVICE_PRESS_FAILED",
            stage="press",
        )
        require_success(done, code="COREDEVICE_PRESS_FAILED", stage="press")
        return {
            **base,
            "attempted": True,
            "dryRun": False,
            "dispatchStatus": "sent",
            "confirmationStatus": "not_requested",
            "completionStatus": "exited",
            "durationMs": done.duration_ms,
            **(
                {
                    "workerFallback": "pymobiledevice3-cli-userspace",
                    "previousError": _previous_error_metadata(previous_error),
                }
                if previous_error is not None
                else {}
            ),
        }

    def type_text(
        self,
        device: str,
        text: str,
        *,
        char_delay_ms: int = 40,
        inter_delay_ms: int = 20,
        paste_at: dict[str, float] | None = None,
        paste_hold_ms: int = 1600,
        clear_existing: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if char_delay_ms < 0:
            raise CoretapError("INVALID_ARGUMENT", "type --char-delay-ms must be >= 0", category="usage", stage="type")
        if inter_delay_ms < 0:
            raise CoretapError("INVALID_ARGUMENT", "type --inter-delay-ms must be >= 0", category="usage", stage="type")
        if paste_hold_ms < 300:
            raise CoretapError("INVALID_ARGUMENT", "type --paste-hold-ms must be >= 300", category="usage", stage="type")
        base = {
            "text": text_input_summary(text),
            "charDelayMs": char_delay_ms,
            "interDelayMs": inter_delay_ms,
            "pasteHoldMs": paste_hold_ms,
            "pasteAt": paste_at,
            "clearExisting": clear_existing,
            "inputMethod": "coredevice-pasteboard-edit-menu",
            "coredeviceTunnelMode": self.coredevice_tunnel_mode,
        }
        if dry_run:
            return {
                **base,
                "attempted": False,
                "dryRun": True,
                "reason": "dry-run requested",
            }
        if self.coredevice_tunnel_mode == "userspace":
            from coretap.device_worker import CoreDeviceWorkerPool, get_default_device_worker_pool, is_recoverable_userspace_tunnel_error

            pool = get_default_device_worker_pool()
            if pool is not None:
                try:
                    result = pool.type_text_userspace(
                        device,
                        text=text,
                        char_delay_ms=char_delay_ms,
                        inter_delay_ms=inter_delay_ms,
                        paste_at=paste_at,
                        paste_hold_ms=paste_hold_ms,
                        clear_existing=clear_existing,
                    )
                    return {**base, **result, "attempted": True, "dryRun": False}
                except CoretapError as exc:
                    if not (exc.retryable or is_recoverable_userspace_tunnel_error(exc)):
                        raise
                    result = self._type_text_userspace_helper(
                        device,
                        text=text,
                        char_delay_ms=char_delay_ms,
                        inter_delay_ms=inter_delay_ms,
                        paste_at=paste_at,
                        paste_hold_ms=paste_hold_ms,
                        clear_existing=clear_existing,
                    )
                    return {
                        **base,
                        **result,
                        "attempted": True,
                        "dryRun": False,
                        "workerFallback": "coretap-device-hid-helper",
                        "previousError": _previous_error_metadata(exc),
                    }
            temporary_pool = CoreDeviceWorkerPool()
            try:
                result = temporary_pool.type_text_userspace(
                    device,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    paste_at=paste_at,
                    paste_hold_ms=paste_hold_ms,
                    clear_existing=clear_existing,
                )
                return {**base, **result, "attempted": True, "dryRun": False}
            finally:
                temporary_pool.close()
        raise CoretapError(
            "COREDEVICE_TYPE_FAILED",
            "Pasteboard text input requires CoreDevice userspace mode",
            category="usage",
            stage="type",
            details={"coredeviceTunnelMode": self.coredevice_tunnel_mode},
        )

    def drag_normalized(
        self,
        device: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        *,
        dry_run: bool,
        start_hid_u16: dict[str, int] | None = None,
        end_hid_u16: dict[str, int] | None = None,
        steps: int = 30,
        duration_ms: int = 600,
    ) -> dict[str, Any]:
        start_hx = int(round(start_x * 65535)) if start_hid_u16 is None else start_hid_u16["x"]
        start_hy = int(round(start_y * 65535)) if start_hid_u16 is None else start_hid_u16["y"]
        end_hx = int(round(end_x * 65535)) if end_hid_u16 is None else end_hid_u16["x"]
        end_hy = int(round(end_y * 65535)) if end_hid_u16 is None else end_hid_u16["y"]
        if steps < 1:
            raise CoretapError("INVALID_ARGUMENT", "drag --steps must be >= 1", category="usage", stage="drag")
        if duration_ms < 0:
            raise CoretapError("INVALID_ARGUMENT", "drag --duration-ms must be >= 0", category="usage", stage="drag")
        base = {
            "from": {"normalized": {"x": start_x, "y": start_y}, "hidU16": {"x": start_hx, "y": start_hy}},
            "to": {"normalized": {"x": end_x, "y": end_y}, "hidU16": {"x": end_hx, "y": end_hy}},
            "steps": steps,
            "requestedDurationMs": duration_ms,
            "coredeviceTunnelMode": self.coredevice_tunnel_mode,
        }
        if dry_run:
            return {
                **base,
                "attempted": False,
                "dryRun": True,
                "reason": "dry-run requested",
            }
        if self.coredevice_tunnel_mode == "userspace":
            from coretap.device_worker import get_default_device_worker_pool, is_recoverable_userspace_tunnel_error

            pool = get_default_device_worker_pool()
            if pool is not None:
                try:
                    return pool.drag_userspace(
                        device,
                        start_x=start_x,
                        start_y=start_y,
                        end_x=end_x,
                        end_y=end_y,
                        start_hx=start_hx,
                        start_hy=start_hy,
                        end_hx=end_hx,
                        end_hy=end_hy,
                        steps=steps,
                        duration_ms=duration_ms,
                    )
                except CoretapError as exc:
                    if not (exc.retryable or is_recoverable_userspace_tunnel_error(exc)):
                        raise
                    previous_error = exc
            else:
                previous_error = None
        else:
            previous_error = None
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "universal-hid-service",
                    "drag",
                    *self.coredevice_device_options(device),
                    str(start_hx),
                    str(start_hy),
                    str(end_hx),
                    str(end_hy),
                    "--steps",
                    str(steps),
                    "--duration",
                    f"{duration_ms / 1000:.3f}",
                ],
                env=self.coredevice_env(device),
                timeout=max(10, int((duration_ms / 1000) + 5)),
            ),
            code="COREDEVICE_DRAG_FAILED",
            stage="drag",
        )
        require_success(done, code="COREDEVICE_DRAG_FAILED", stage="drag")
        return {
            **base,
            "attempted": True,
            "dryRun": False,
            "dispatchStatus": "sent",
            "confirmationStatus": "not_requested",
            "completionStatus": "exited",
            "durationMs": done.duration_ms,
            **(
                {
                    "workerFallback": "pymobiledevice3-cli-userspace",
                    "previousError": _previous_error_metadata(previous_error),
                }
                if previous_error is not None
                else {}
            ),
        }

    def _type_text_userspace_helper(
        self,
        device: str,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        paste_at: dict[str, float] | None,
        paste_hold_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        timeout = max(20.0, (paste_hold_ms / 1000) + (len(text) * max(char_delay_ms, 0) / 1000) + 10.0)
        started = time.monotonic()
        argv = [
            sys.executable,
            "-m",
            "coretap.device_hid_helper",
            "--mode",
            "userspace",
            "--action",
            "type",
            "--device",
            device,
            "--text",
            text,
            "--char-delay-ms",
            str(char_delay_ms),
            "--inter-delay-ms",
            str(inter_delay_ms),
            "--paste-hold-ms",
            str(paste_hold_ms),
        ]
        if paste_at is not None:
            argv.extend(["--paste-at", f"{paste_at['x']},{paste_at['y']}"])
        if clear_existing:
            argv.append("--clear-existing")
        proc = subprocess.Popen(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.coredevice_env(device),
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        result: dict[str, Any] | None = None
        try:
            while time.monotonic() - started < timeout:
                remaining = max(0.0, timeout - (time.monotonic() - started))
                events = selector.select(timeout=remaining)
                if not events:
                    break
                line = proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "result" and isinstance(event.get("result"), dict):
                    result = event["result"]
                    break
                if event.get("event") == "error":
                    stderr = _terminate_process(proc)
                    raise CoretapError(
                        "COREDEVICE_TYPE_FAILED",
                        "CoreDevice text input helper failed before dispatch",
                        stage="type",
                        retryable=True,
                        details={"helperEvent": event, "stderr": stderr},
                    )
            if result is None:
                stderr = _terminate_process(proc)
                raise CoretapError(
                    "COREDEVICE_TYPE_FAILED",
                    "Timed out waiting for CoreDevice text input helper",
                    stage="type",
                    retryable=True,
                    details={"timeoutMs": round(timeout * 1000), "stderr": stderr},
                )
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _terminate_process(proc)
            result["durationMs"] = round((time.monotonic() - started) * 1000)
            return result
        finally:
            selector.close()


def parse_usbmux_devices(stdout: str) -> list[Device]:
    try:
        data = json.loads(stdout or "[]")
    except json.JSONDecodeError:
        return _parse_simple_usbmux_lines(stdout)

    entries: list[Any]
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("DeviceList") or data.get("devices") or [data]
    else:
        entries = []

    devices: list[Device] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        udid = (
            entry.get("Identifier")
            or entry.get("UniqueDeviceID")
            or entry.get("UDID")
            or entry.get("SerialNumber")
            or entry.get("udid")
        )
        if not udid:
            continue
        product_type = entry.get("ProductType")
        product_version = entry.get("ProductVersion")
        name = entry.get("DeviceName") or entry.get("Name") or product_type or str(udid)
        devices.append(
            Device(
                udid=str(udid),
                name=str(name) if name is not None else None,
                backend=DeviceBackend.name,
                runtime=str(product_version) if product_version is not None else None,
                eligible=True,
                details=entry,
            )
        )
    return devices


def _primary_display_size(display_info: dict[str, Any]) -> tuple[int, int]:
    displays = display_info.get("displays")
    if not isinstance(displays, list):
        raise CoretapError(
            "COREDEVICE_DISPLAY_INFO_INVALID",
            "CoreDevice display info did not include displays",
            stage="display-info",
            details={"displayInfo": display_info},
        )
    primary = next((d for d in displays if isinstance(d, dict) and d.get("primary") is True), None)
    if primary is None:
        primary = next((d for d in displays if isinstance(d, dict) and d.get("external") is False), None)
    if primary is None and displays and isinstance(displays[0], dict):
        primary = displays[0]
    mode = primary.get("currentMode") if isinstance(primary, dict) else None
    size = mode.get("size") if isinstance(mode, dict) else None
    if not isinstance(size, list | tuple) or len(size) != 2:
        raise CoretapError(
            "COREDEVICE_DISPLAY_INFO_INVALID",
            "CoreDevice primary display did not include currentMode.size",
            stage="display-info",
            details={"displayInfo": display_info},
        )
    try:
        width = int(round(float(size[0])))
        height = int(round(float(size[1])))
    except (TypeError, ValueError) as exc:
        raise CoretapError(
            "COREDEVICE_DISPLAY_INFO_INVALID",
            "CoreDevice primary display size was not numeric",
            stage="display-info",
            details={"size": size},
        ) from exc
    if width <= 0 or height <= 0:
        raise CoretapError(
            "COREDEVICE_DISPLAY_INFO_INVALID",
            "CoreDevice primary display size was empty",
            stage="display-info",
            details={"size": size},
        )
    return width, height


def _device_non_flat_orientation(display_info: dict[str, Any]) -> str | None:
    orientation = display_info.get("orientation")
    if not isinstance(orientation, dict):
        return None
    value = orientation.get("currentDeviceNonFlatOrientation") or orientation.get("currentDeviceOrientation")
    return str(value) if value is not None else None


def _coredevice_screenshot_rotation(
    image_size: tuple[int, int],
    display_size: tuple[int, int],
    orientation: str | None,
) -> int | None:
    if orientation == "portraitUpsideDown":
        return 180
    if image_size == display_size:
        return None
    if image_size != (display_size[1], display_size[0]):
        return None
    if orientation == "landscapeLeft":
        return 90
    return 270


def _parse_simple_usbmux_lines(stdout: str) -> list[Device]:
    devices: list[Device] = []
    for line in stdout.splitlines():
        udid = line.strip()
        if not udid:
            continue
        devices.append(Device(udid=udid, name=udid, backend=DeviceBackend.name, eligible=True, details={"line": line}))
    return devices


def _terminate_process(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)
    stderr = ""
    if proc.stderr is not None:
        try:
            stderr = proc.stderr.read()
        except ValueError:
            stderr = ""
    return stderr


def _public_capture_metadata(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if capture is None:
        return None
    return {key: value for key, value in capture.items() if key != "image"}


def _previous_error_metadata(error: CoretapError) -> dict[str, Any]:
    return {
        "code": error.code,
        "stage": error.stage,
        "message": str(error),
        "retryable": error.retryable,
        "details": error.details,
    }


def _check_coredevice_result(done: Completed, *, code: str, stage: str) -> Completed:
    combined = f"{done.stdout}\n{done.stderr}"
    if "Unable to connect to Tunneld" in combined or "start one using" in combined and "tunneld" in combined:
        raise CoretapError(
            "COREDEVICE_TUNNELD_UNAVAILABLE",
            "pymobiledevice3 could not connect to tunneld for CoreDevice access",
            category="environment",
            stage=stage,
            retryable=True,
            details={
                "argv": done.argv,
                "stdout": done.stdout,
                "stderr": done.stderr,
                "suggestedCommand": "sudo pymobiledevice3 remote tunneld --daemonize",
            },
        )
    return done


def backend_for(
    name: str,
    *,
    developer_dir: str | None = None,
    coredevice_tunnel_mode: str | None = None,
) -> SimulatorBackend | DeviceBackend:
    if name == "simulator":
        return SimulatorBackend(developer_dir=developer_dir)
    if name == "device":
        return DeviceBackend(coredevice_tunnel_mode=coredevice_tunnel_mode)
    raise CoretapError("UNKNOWN_BACKEND", f"Unsupported backend: {name}", category="usage", stage="config")
