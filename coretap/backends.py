from __future__ import annotations

import json
import os
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
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "screen-capture",
                    "screenshot",
                    "--tunnel",
                    device,
                    str(out),
                ],
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
                details={"path": str(out), "stdout": done.stdout, "stderr": done.stderr},
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
        hx = int(round(x * 65535)) if hid_u16 is None else hid_u16["x"]
        hy = int(round(y * 65535)) if hid_u16 is None else hid_u16["y"]
        if dry_run:
            return {"attempted": False, "dryRun": True, "normalized": {"x": x, "y": y}, "hidU16": {"x": hx, "y": hy}}
        done = _check_coredevice_result(
            run_command(
                [
                    "pymobiledevice3",
                    "developer",
                    "core-device",
                    "universal-hid-service",
                    "tap",
                    "--tunnel",
                    device,
                    str(hx),
                    str(hy),
                ],
                timeout=10,
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


def _parse_simple_usbmux_lines(stdout: str) -> list[Device]:
    devices: list[Device] = []
    for line in stdout.splitlines():
        udid = line.strip()
        if not udid:
            continue
        devices.append(Device(udid=udid, name=udid, backend=DeviceBackend.name, eligible=True, details={"line": line}))
    return devices


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


def backend_for(name: str, *, developer_dir: str | None = None) -> SimulatorBackend | DeviceBackend:
    if name == "simulator":
        return SimulatorBackend(developer_dir=developer_dir)
    if name == "device":
        return DeviceBackend()
    raise CoretapError("UNKNOWN_BACKEND", f"Unsupported backend: {name}", category="usage", stage="config")
