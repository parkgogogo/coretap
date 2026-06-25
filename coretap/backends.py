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

from coretap.runtime import (
    Completed,
    CoretapError,
    command_env,
    png_size,
    require_success,
    run_command,
)


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
                capture = pool.capture_screenshot_userspace(device)
                out.write_bytes(capture["image"])
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

    def display_info(self, device: str) -> dict[str, Any]:
        if self.coredevice_tunnel_mode == "userspace":
            from coretap.device_worker import get_default_device_worker_pool

            pool = get_default_device_worker_pool()
            if pool is not None:
                return pool.display_info_userspace(device)
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
        from coretap.device_worker import get_default_device_worker_pool

        pool = get_default_device_worker_pool()
        if pool is not None:
            return pool.tap_userspace(device, x, y, hx, hy)

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
    if image_size == display_size:
        return None
    if image_size != (display_size[1], display_size[0]):
        return None
    if orientation == "landscapeLeft":
        return 90
    if orientation == "portraitUpsideDown":
        return 180
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
