from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from coretap.runtime import CoretapError, run_command


_DEFAULT_POOL: "CoreDeviceWorkerPool | None" = None

_USERSPACE_TUNNEL_RECOVERY_MARKERS = (
    "userspace tunnel is already active",
    "pytcp",
    "process-global singleton",
    "only one userspace tunnel",
    "remote service discovery",
    "rsd",
    "connection lost",
    "connection reset",
    "broken pipe",
    "socket is closed",
)

_DISPLAY_SERVICE_RECOVERY_MARKERS = (
    "displayservice",
    "display service",
    "display-enter",
    "start-video-stream",
    "start video stream",
    "stop-media-stream",
    "stop_media_stream",
    "media stream",
    "touch session",
    "dtremotedisplayd",
    "timeouterror",
    "timed out",
    "timeout",
)


KEYBOARD_KEY_CHOICES = ("backspace", "delete", "enter", "return", "tab", "escape", "esc", "space", "left", "right", "up", "down")


def _normalize_keyboard_key(key: str) -> str:
    normalized = key.strip().casefold().replace("_", "-")
    aliases = {
        "delete": "backspace",
        "return": "enter",
        "esc": "escape",
        "arrow-left": "left",
        "arrow-right": "right",
        "arrow-up": "up",
        "arrow-down": "down",
    }
    return aliases.get(normalized, normalized)


def _keyboard_key_usage(key: str) -> int:
    normalized = _normalize_keyboard_key(key)
    from pymobiledevice3.remote.core_device import hid_service

    usages = {
        "backspace": hid_service.KEY_BACKSPACE,
        "enter": hid_service.KEY_ENTER,
        "tab": hid_service.KEY_TAB,
        "escape": hid_service.KEY_ESC,
        "space": hid_service.KEY_SPACE,
        "left": hid_service.KEY_LEFT,
        "right": hid_service.KEY_RIGHT,
        "up": hid_service.KEY_UP,
        "down": hid_service.KEY_DOWN,
    }
    try:
        return usages[normalized]
    except KeyError as exc:
        raise CoretapError(
            "INVALID_ARGUMENT",
            f"Unsupported keyboard key: {key}",
            category="usage",
            stage="key",
            details={"key": key, "supportedKeys": list(KEYBOARD_KEY_CHOICES)},
        ) from exc


def set_default_device_worker_pool(pool: "CoreDeviceWorkerPool | None") -> None:
    global _DEFAULT_POOL
    _DEFAULT_POOL = pool


def get_default_device_worker_pool() -> "CoreDeviceWorkerPool | None":
    return _DEFAULT_POOL


def is_recoverable_userspace_tunnel_error(exc: BaseException) -> bool:
    text = _error_text(exc).casefold()
    if not text:
        return False
    if "userspace" not in text and "pytcp" not in text and "rsd" not in text and "coredevice" not in text:
        return False
    return any(marker in text for marker in _USERSPACE_TUNNEL_RECOVERY_MARKERS)


def is_recoverable_coredevice_display_error(exc: BaseException) -> bool:
    if is_recoverable_userspace_tunnel_error(exc):
        return False
    if isinstance(exc, TimeoutError):
        return True
    text = _error_text(exc).casefold()
    if not text:
        return False
    if "coredevice" not in text and "display" not in text and "touch" not in text and "hid" not in text:
        return False
    return any(marker in text for marker in _DISPLAY_SERVICE_RECOVERY_MARKERS)


def _error_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, CoretapError):
        parts.extend([value.code, value.stage, str(value)])
        parts.append(_error_text(value.details))
    elif isinstance(value, dict):
        for key, item in value.items():
            parts.append(str(key))
            parts.append(_error_text(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            parts.append(_error_text(item))
    elif value is not None:
        parts.append(str(value))
    return " ".join(part for part in parts if part)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _pinyin_keyboard_text(text: str) -> str | None:
    if not _contains_cjk(text):
        return None
    from pypinyin import Style, lazy_pinyin

    parts = lazy_pinyin(text, style=Style.NORMAL, errors="default", strict=False)
    keyboard_text = "".join(parts)
    if not keyboard_text or not keyboard_text.isascii():
        return None
    return keyboard_text


def _paste_menu_point_for_anchor(anchor_x: float, anchor_y: float) -> tuple[float, float]:
    paste_x = _clamp(anchor_x - 0.07, 0.08, 0.92)
    vertical_offset = 0.059 if anchor_y < 0.18 else -0.059
    paste_y = _clamp(anchor_y + vertical_offset, 0.05, 0.95)
    return paste_x, paste_y


def recover_coredevice_display_service(device: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYMOBILEDEVICE3_UDID"] = device
    started = time.monotonic()
    list_done = run_command(
        ["pymobiledevice3", "developer", "core-device", "list-processes", "--userspace"],
        env=env,
        timeout=10,
        max_output=20_000_000,
    )
    if list_done.returncode != 0:
        return {
            "attempted": True,
            "status": "list-processes-failed",
            "device": device,
            "returncode": list_done.returncode,
            "stderr": list_done.stderr[-2000:],
            "durationMs": round((time.monotonic() - started) * 1000),
        }
    try:
        process_data = json.loads(list_done.stdout or "[]")
    except json.JSONDecodeError as exc:
        return {
            "attempted": True,
            "status": "list-processes-invalid-json",
            "device": device,
            "errorType": type(exc).__name__,
            "stdout": list_done.stdout[:1000],
            "durationMs": round((time.monotonic() - started) * 1000),
        }

    display_targets: list[dict[str, Any]] = []
    hid_targets: list[dict[str, Any]] = []
    for process in _iter_coredevice_processes(process_data):
        executable = _coredevice_process_executable(process)
        pid = _coredevice_process_pid(process)
        if pid is None:
            continue
        if "dtremotedisplayd" in executable:
            display_targets.append({"pid": pid, "executable": executable, "signal": 15})
        elif "dtuhidd" in executable:
            hid_targets.append({"pid": pid, "executable": executable, "signal": 9})

    targets = display_targets or hid_targets
    signals: list[dict[str, Any]] = []
    for target in targets:
        done = run_command(
            [
                "pymobiledevice3",
                "developer",
                "core-device",
                "send-signal-to-process",
                "--userspace",
                str(target["pid"]),
                str(target["signal"]),
            ],
            env=env,
            timeout=10,
            max_output=200_000,
        )
        signals.append(
            {
                "pid": target["pid"],
                "executable": target["executable"],
                "signal": target["signal"],
                "returncode": done.returncode,
                "stderr": done.stderr[-1000:],
                "durationMs": done.duration_ms,
            }
        )

    status = "not-running"
    if signals:
        status = "signaled" if any(signal["returncode"] == 0 for signal in signals) else "signal-failed"
    return {
        "attempted": True,
        "status": status,
        "device": device,
        "targets": targets,
        "signals": signals,
        "durationMs": round((time.monotonic() - started) * 1000),
    }


def _iter_coredevice_processes(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("processes", "processList", "items", "result", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return []


def _coredevice_process_executable(process: dict[str, Any]) -> str:
    executable = process.get("executableURL") or process.get("executable") or process.get("path") or process.get("name")
    if isinstance(executable, dict):
        executable = executable.get("relative") or executable.get("absolute") or executable.get("path")
    return str(executable or "")


def _coredevice_process_pid(process: dict[str, Any]) -> int | None:
    value = process.get("processIdentifier") or process.get("pid") or process.get("processID")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _RsdSession:
    device: str
    rsd: Any
    created_at: float
    last_used_at: float
    requests: int = 0


@dataclass
class _TouchSession:
    device: str
    context: Any
    service: Any
    created_at: float
    last_used_at: float
    taps: int = 0
    drags: int = 0
    typed_characters: int = 0
    keyboard_service_id: int | None = None
    last_tap_normalized: dict[str, float] | None = None


@dataclass
class _ButtonSession:
    device: str
    context: Any
    service: Any
    created_at: float
    last_used_at: float
    presses: int = 0


@dataclass
class _KeyboardSession:
    device: str
    context: Any
    service: Any
    created_at: float
    last_used_at: float
    typed_characters: int = 0
    keypresses: int = 0
    keyboard_service_id: int | None = None


class CoreDeviceWorkerPool:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._rsd_sessions: dict[str, _RsdSession] = {}
        self._sessions: dict[str, _TouchSession] = {}
        self._button_sessions: dict[str, _ButtonSession] = {}
        self._keyboard_sessions: dict[str, _KeyboardSession] = {}
        self._lock: asyncio.Lock | None = None
        self._recovery_count = 0
        self._last_recovery: dict[str, Any] | None = None

    def tap_userspace(self, device: str, x: float, y: float, hx: int, hy: int) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(self._tap_userspace(device, x, y, hx, hy), stage="tap", device=device)
        result.update(
            {
                "attempted": True,
                "dryRun": False,
                "normalized": {"x": x, "y": y},
                "hidU16": {"x": hx, "y": hy},
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        )
        return result

    def drag_userspace(
        self,
        device: str,
        *,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        start_hx: int,
        start_hy: int,
        end_hx: int,
        end_hy: int,
        steps: int,
        duration_ms: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(
            self._drag_userspace(device, start_hx, start_hy, end_hx, end_hy, steps=steps, duration_ms=duration_ms),
            stage="drag",
            device=device,
        )
        result.update(
            {
                "attempted": True,
                "dryRun": False,
                "from": {"normalized": {"x": start_x, "y": start_y}, "hidU16": {"x": start_hx, "y": start_hy}},
                "to": {"normalized": {"x": end_x, "y": end_y}, "hidU16": {"x": end_hx, "y": end_hy}},
                "steps": steps,
                "durationMs": round((time.monotonic() - started) * 1000),
                "requestedDurationMs": duration_ms,
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
            }
        )
        return result

    def press_button_userspace(
        self,
        device: str,
        *,
        button: str,
        state: str,
        usage_page: int,
        usage_code: int,
        hold_ms: int,
    ) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(
            self._press_button_userspace(
                device,
                button=button,
                state=state,
                usage_page=usage_page,
                usage_code=usage_code,
                hold_ms=hold_ms,
            ),
            stage="press",
            device=device,
        )
        result.update(
            {
                "button": button,
                "state": state,
                "hidButton": {"usagePage": usage_page, "usageCode": usage_code},
                "holdMs": hold_ms if state == "press" else 0,
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        )
        return result

    def type_text_userspace(
        self,
        device: str,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        paste_at: dict[str, float] | None = None,
        paste_hold_ms: int = 1600,
        clear_existing: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(
            self._type_text_userspace(
                device,
                text=text,
                char_delay_ms=char_delay_ms,
                inter_delay_ms=inter_delay_ms,
                paste_at=paste_at,
                paste_hold_ms=paste_hold_ms,
                clear_existing=clear_existing,
            ),
            stage="type",
            device=device,
        )
        result.update(
            {
                "attempted": True,
                "dryRun": False,
                "typedCharacters": len(text),
                "charDelayMs": char_delay_ms,
                "interDelayMs": inter_delay_ms,
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        )
        return result

    def keyboard_key_userspace(
        self,
        device: str,
        *,
        key: str,
        count: int,
        inter_delay_ms: int = 20,
    ) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(
            self._keyboard_key_userspace(
                device,
                key=key,
                count=count,
                inter_delay_ms=inter_delay_ms,
            ),
            stage="key",
            device=device,
        )
        result.update(
            {
                "attempted": True,
                "dryRun": False,
                "key": key,
                "count": count,
                "interDelayMs": inter_delay_ms,
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        )
        return result

    def clear_text_userspace(self, device: str, *, count: int = 80, inter_delay_ms: int = 2) -> dict[str, Any]:
        result = self.keyboard_key_userspace(device, key="backspace", count=count, inter_delay_ms=inter_delay_ms)
        result.update({"operation": "clear", "clearKeypresses": count})
        return result

    def capture_screenshot_userspace(self, device: str) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(self._capture_screenshot_userspace(device), stage="screenshot", device=device)
        result.update(
            {
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        )
        return result

    def display_info_userspace(self, device: str) -> dict[str, Any]:
        return self._run(self._display_info_userspace(device), stage="display-info", device=device)

    def start(self) -> None:
        self._ensure_loop()

    def stats(self) -> dict[str, Any]:
        rsd_sessions = []
        touch_sessions = []
        button_sessions = []
        keyboard_sessions = []
        now = time.monotonic()
        for session in self._rsd_sessions.values():
            rsd_sessions.append(
                {
                    "device": session.device,
                    "ageMs": round((now - session.created_at) * 1000),
                    "idleMs": round((now - session.last_used_at) * 1000),
                    "requests": session.requests,
                }
            )
        for session in self._sessions.values():
            touch_sessions.append(
                {
                    "device": session.device,
                    "ageMs": round((now - session.created_at) * 1000),
                    "idleMs": round((now - session.last_used_at) * 1000),
                    "taps": session.taps,
                    "drags": session.drags,
                    "typedCharacters": session.typed_characters,
                    "keyboardServiceRegistered": session.keyboard_service_id is not None,
                    "lastTap": session.last_tap_normalized,
                }
            )
        for session in self._button_sessions.values():
            button_sessions.append(
                {
                    "device": session.device,
                    "ageMs": round((now - session.created_at) * 1000),
                    "idleMs": round((now - session.last_used_at) * 1000),
                    "presses": session.presses,
                }
            )
        for session in self._keyboard_sessions.values():
            keyboard_sessions.append(
                {
                    "device": session.device,
                    "ageMs": round((now - session.created_at) * 1000),
                    "idleMs": round((now - session.last_used_at) * 1000),
                    "typedCharacters": session.typed_characters,
                    "keypresses": session.keypresses,
                    "keyboardServiceRegistered": session.keyboard_service_id is not None,
                }
            )
        return {
            "kind": "coredevice-userspace-persistent",
            "running": self._thread is not None and self._thread.is_alive(),
            "rsdSessionCount": len(self._rsd_sessions),
            "touchSessionCount": len(self._sessions),
            "buttonSessionCount": len(self._button_sessions),
            "keyboardSessionCount": len(self._keyboard_sessions),
            "rsdSessions": rsd_sessions,
            "touchSessions": touch_sessions,
            "buttonSessions": button_sessions,
            "keyboardSessions": keyboard_sessions,
            "recoveryCount": self._recovery_count,
            "lastRecovery": self._last_recovery,
        }

    def close(self) -> None:
        self._shutdown_loop(close_sessions=True)

    def _run(self, coro: Any, *, stage: str, device: str | None = None) -> Any:
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            self._recover_worker_state(stage=stage, reason="timeout", error=exc, device=device)
            raise CoretapError(
                "COREDEVICE_WORKER_TIMEOUT",
                "CoreDevice worker request timed out",
                stage=stage,
                category="infrastructure",
                retryable=True,
                details={"workerRecovered": True},
            ) from exc
        except CoretapError as exc:
            if is_recoverable_userspace_tunnel_error(exc):
                self._recover_worker_state(stage=stage, reason="userspace-tunnel-error", error=exc, device=device)
                exc.details = {**exc.details, "workerRecovered": True}
            raise
        except Exception as exc:
            recovered = is_recoverable_userspace_tunnel_error(exc)
            if recovered:
                self._recover_worker_state(stage=stage, reason="userspace-tunnel-error", error=exc, device=device)
            raise CoretapError(
                "COREDEVICE_WORKER_FAILED",
                f"CoreDevice worker request failed: {exc}",
                stage=stage,
                category="infrastructure",
                retryable=True,
                details={
                    "errorType": type(exc).__name__,
                    "workerRecovered": recovered,
                },
            ) from exc

    def _recover_worker_state(
        self,
        *,
        stage: str,
        reason: str,
        error: BaseException,
        device: str | None = None,
    ) -> None:
        display_service_recovery = None
        if device is not None and (reason == "display-service-error" or reason == "timeout" and stage in {"tap", "drag", "type"}):
            with suppress(BaseException):
                display_service_recovery = recover_coredevice_display_service(device)
        self._recovery_count += 1
        self._last_recovery = {
            "stage": stage,
            "reason": reason,
            "errorType": type(error).__name__,
            "message": str(error),
            "atMonotonic": time.monotonic(),
            "displayService": display_service_recovery,
        }
        self._shutdown_loop(close_sessions=True)

    def _shutdown_loop(self, *, close_sessions: bool) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None and loop.is_running():
            if close_sessions:
                with suppress(BaseException):
                    future = asyncio.run_coroutine_threadsafe(self._close_all(), loop)
                    future.result(timeout=3)
            with suppress(BaseException):
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._loop = None
        self._thread = None
        self._lock = None
        self._ready.clear()
        self._rsd_sessions.clear()
        self._sessions.clear()
        self._button_sessions.clear()
        self._keyboard_sessions.clear()

    def _ensure_loop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._thread = threading.Thread(target=self._thread_main, name="coretap-coredevice-worker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise CoretapError(
                "COREDEVICE_WORKER_FAILED",
                "CoreDevice worker event loop did not start",
                stage="tap",
                category="infrastructure",
                retryable=True,
            )

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._lock = asyncio.Lock()
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        with suppress(BaseException):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    async def _tap_userspace(self, device: str, x: float, y: float, hx: int, hy: int) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            display_service_recovery = None
            retry_close_scope = None
            try:
                session, status = await self._get_or_open_session(device)
                await self._send_tap(session, hx, hy)
            except Exception as exc:
                retry_close_scope = await self._close_for_retry(device, exc, session_kind="touch")
                display_service_recovery = await self._recover_display_service_for_retry(device, stage="tap", error=exc)
                try:
                    session, _ = await self._get_or_open_session(device)
                    await self._send_tap(session, hx, hy)
                    status = "recreated_after_display_service_recovery" if display_service_recovery else "recreated_after_error"
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_TAP_FAILED",
                        f"Persistent CoreDevice HID dispatch failed: {retry_exc}",
                        stage="tap",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                            "retryCloseScope": retry_close_scope,
                            "displayServiceRecovery": display_service_recovery,
                        },
                    ) from retry_exc
            session.last_used_at = time.monotonic()
            session.taps += 1
            session.last_tap_normalized = {"x": x, "y": y}
            rsd_session = self._rsd_sessions.get(device)
            if rsd_session is not None:
                rsd_session.last_used_at = session.last_used_at
                rsd_session.requests += 1
            return {
                "dispatchStatus": "sent",
                "confirmationStatus": "not_requested",
                "sessionStatus": status,
                "sessionTapCount": session.taps,
                **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
                **({"displayServiceRecovery": display_service_recovery} if display_service_recovery else {}),
            }

    async def _drag_userspace(
        self,
        device: str,
        start_hx: int,
        start_hy: int,
        end_hx: int,
        end_hy: int,
        *,
        steps: int,
        duration_ms: int,
    ) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            display_service_recovery = None
            retry_close_scope = None
            try:
                session, status = await self._get_or_open_session(device)
                await self._send_drag(
                    session,
                    start_hx,
                    start_hy,
                    end_hx,
                    end_hy,
                    steps=steps,
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                retry_close_scope = await self._close_for_retry(device, exc, session_kind="touch")
                display_service_recovery = await self._recover_display_service_for_retry(device, stage="drag", error=exc)
                try:
                    session, _ = await self._get_or_open_session(device)
                    await self._send_drag(
                        session,
                        start_hx,
                        start_hy,
                        end_hx,
                        end_hy,
                        steps=steps,
                        duration_ms=duration_ms,
                    )
                    status = "recreated_after_display_service_recovery" if display_service_recovery else "recreated_after_error"
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_DRAG_FAILED",
                        f"Persistent CoreDevice drag dispatch failed: {retry_exc}",
                        stage="drag",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                            "retryCloseScope": retry_close_scope,
                            "displayServiceRecovery": display_service_recovery,
                        },
                    ) from retry_exc
            session.last_used_at = time.monotonic()
            session.drags += 1
            rsd_session = self._rsd_sessions.get(device)
            if rsd_session is not None:
                rsd_session.last_used_at = session.last_used_at
                rsd_session.requests += 1
            return {
                "dispatchStatus": "sent",
                "confirmationStatus": "not_requested",
                "sessionStatus": status,
                "sessionDragCount": session.drags,
                **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
                **({"displayServiceRecovery": display_service_recovery} if display_service_recovery else {}),
            }

    async def _press_button_userspace(
        self,
        device: str,
        *,
        button: str,
        state: str,
        usage_page: int,
        usage_code: int,
        hold_ms: int,
    ) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            retry_close_scope = None
            try:
                session, status = await self._get_or_open_button_session(device)
                await self._send_button(session, state=state, usage_page=usage_page, usage_code=usage_code, hold_ms=hold_ms)
            except Exception as exc:
                retry_close_scope = await self._close_for_retry(device, exc, session_kind="button")
                try:
                    session, _ = await self._get_or_open_button_session(device)
                    await self._send_button(session, state=state, usage_page=usage_page, usage_code=usage_code, hold_ms=hold_ms)
                    status = "recreated_after_error"
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_PRESS_FAILED",
                        f"Persistent CoreDevice button dispatch failed: {retry_exc}",
                        stage="press",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "button": button,
                            "state": state,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                            "retryCloseScope": retry_close_scope,
                        },
                    ) from retry_exc
            session.last_used_at = time.monotonic()
            session.presses += 1
            rsd_session = self._rsd_sessions.get(device)
            if rsd_session is not None:
                rsd_session.last_used_at = session.last_used_at
                rsd_session.requests += 1
            return {
                "dispatchStatus": "sent",
                "confirmationStatus": "not_requested",
                "sessionStatus": status,
                "sessionPressCount": session.presses,
                **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
            }

    async def _type_text_userspace(
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
        assert self._lock is not None
        async with self._lock:
            if self._supports_virtual_keyboard_text(text):
                return await self._type_text_with_keyboard_session(
                    device,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    clear_existing=clear_existing,
                )
            if paste_at is None and _pinyin_keyboard_text(text):
                return await self._type_text_with_pinyin_keyboard_session(
                    device,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    clear_existing=clear_existing,
                )
            if paste_at is None:
                raise CoretapError(
                    "TEXT_INPUT_TARGET_UNKNOWN",
                    "Non-ASCII text input requires a paste anchor resolved from a visible text field",
                    category="usage",
                    stage="type",
                    details={"device": device, "textLength": len(text)},
                )

            display_service_recovery = None
            retry_close_scope = None
            try:
                session, status = await self._get_or_open_session(device)
                input_result = await self._input_text(
                    session,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    paste_at=paste_at,
                    paste_hold_ms=paste_hold_ms,
                    clear_existing=clear_existing,
                )
            except Exception as exc:
                retry_close_scope = await self._close_for_retry(device, exc, session_kind="touch")
                display_service_recovery = await self._recover_display_service_for_retry(device, stage="type", error=exc)
                try:
                    session, _ = await self._get_or_open_session(device)
                    input_result = await self._input_text(
                        session,
                        text=text,
                        char_delay_ms=char_delay_ms,
                        inter_delay_ms=inter_delay_ms,
                        paste_at=paste_at,
                        paste_hold_ms=paste_hold_ms,
                        clear_existing=clear_existing,
                    )
                    status = "recreated_after_display_service_recovery" if display_service_recovery else "recreated_after_error"
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_TYPE_FAILED",
                        f"Persistent CoreDevice text input failed: {retry_exc}",
                        stage="type",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                            "retryCloseScope": retry_close_scope,
                            "displayServiceRecovery": display_service_recovery,
                        },
                    ) from retry_exc
            session.last_used_at = time.monotonic()
            session.typed_characters += len(text)
            rsd_session = self._rsd_sessions.get(device)
            if rsd_session is not None:
                rsd_session.last_used_at = session.last_used_at
                rsd_session.requests += 1
            return {
                "dispatchStatus": "sent",
                "confirmationStatus": "dispatched_unverified",
                "sessionStatus": status,
                "sessionTypedCharacterCount": session.typed_characters,
                "keyboardServiceRegistered": session.keyboard_service_id is not None,
                "pasteboardSet": True,
                **input_result,
                **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
                **({"displayServiceRecovery": display_service_recovery} if display_service_recovery else {}),
            }

    async def _capture_screenshot_userspace(self, device: str) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            try:
                return await self._capture_screenshot_once(device)
            except Exception as exc:
                if is_recoverable_userspace_tunnel_error(exc):
                    await self._close_device(device)
                    await asyncio.sleep(0.25)
                try:
                    result = await self._capture_screenshot_once(device)
                    result["sessionStatus"] = "recreated_after_error"
                    return result
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_SCREENSHOT_FAILED",
                        f"Persistent CoreDevice screenshot failed: {retry_exc}",
                        stage="screenshot",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                        },
                    ) from retry_exc

    async def _display_info_userspace(self, device: str) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            try:
                return await self._display_info_once(device)
            except Exception as exc:
                if is_recoverable_userspace_tunnel_error(exc):
                    await self._close_device(device)
                    await asyncio.sleep(0.25)
                try:
                    return await self._display_info_once(device)
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_DISPLAY_INFO_FAILED",
                        f"Persistent CoreDevice display-info failed: {retry_exc}",
                        stage="display-info",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                        },
                    ) from retry_exc

    async def _capture_screenshot_once(self, device: str) -> dict[str, Any]:
        from pymobiledevice3.remote.core_device.screen_capture_service import ScreenCaptureService

        rsd_session, status = await self._get_or_open_rsd(device)
        async with ScreenCaptureService(rsd_session.rsd) as service:
            response = await service.capture_screenshot()
        rsd_session.last_used_at = time.monotonic()
        rsd_session.requests += 1
        image = response["image"]
        return {
            "image": bytes(image),
            "imageFormat": response.get("imageFormat"),
            "displayUniqueID": response.get("displayUniqueID"),
            "sessionStatus": status,
        }

    async def _display_info_once(self, device: str) -> dict[str, Any]:
        from pymobiledevice3.remote.core_device.device_info import DeviceInfoService

        rsd_session, _ = await self._get_or_open_rsd(device)
        async with DeviceInfoService(rsd_session.rsd) as service:
            info = await service.get_display_info()
        rsd_session.last_used_at = time.monotonic()
        rsd_session.requests += 1
        return info

    async def _get_or_open_rsd(self, device: str) -> tuple[_RsdSession, str]:
        session = self._rsd_sessions.get(device)
        if session is not None:
            return session, "reused"

        from pymobiledevice3.remote import userspace_tunnel

        rsd = await userspace_tunnel.establish_userspace_rsd(serial=device)
        now = time.monotonic()
        session = _RsdSession(device=device, rsd=rsd, created_at=now, last_used_at=now)
        self._rsd_sessions[device] = session
        return session, "created"

    async def _get_or_open_session(self, device: str) -> tuple[_TouchSession, str]:
        session = self._sessions.get(device)
        if session is not None:
            return session, "reused"

        from coretap.device_hid_helper import bounded_touch_session

        rsd_session, _ = await self._get_or_open_rsd(device)
        context = bounded_touch_session(rsd_session.rsd)
        try:
            service = await asyncio.wait_for(context.__aenter__(), timeout=18.0)
        except Exception as exc:
            with suppress(BaseException):
                await asyncio.wait_for(context.__aexit__(None, None, None), timeout=2.0)
            raise CoretapError(
                "COREDEVICE_HID_SERVICE_FAILED",
                "CoreDevice UniversalHID touch session failed to open",
                stage="touch-session",
                category="infrastructure",
                retryable=True,
                details={"device": device, "errorType": type(exc).__name__, "message": str(exc)},
            ) from exc
        now = time.monotonic()
        session = _TouchSession(device=device, context=context, service=service, created_at=now, last_used_at=now)
        self._sessions[device] = session
        return session, "created"

    async def _get_or_open_button_session(self, device: str) -> tuple[_ButtonSession, str]:
        session = self._button_sessions.get(device)
        if session is not None:
            return session, "reused"

        from pymobiledevice3.remote.core_device.hid_service import IndigoHIDService

        rsd_session, _ = await self._get_or_open_rsd(device)
        context = IndigoHIDService(rsd_session.rsd)
        service = await context.__aenter__()
        now = time.monotonic()
        session = _ButtonSession(device=device, context=context, service=service, created_at=now, last_used_at=now)
        self._button_sessions[device] = session
        return session, "created"

    async def _get_or_open_keyboard_session(self, device: str) -> tuple[_KeyboardSession, str]:
        session = self._keyboard_sessions.get(device)
        if session is not None:
            return session, "reused"

        from pymobiledevice3.remote.core_device.hid_service import UniversalHIDServiceService

        rsd_session, _ = await self._get_or_open_rsd(device)
        context = UniversalHIDServiceService(rsd_session.rsd)
        try:
            service = await asyncio.wait_for(context.__aenter__(), timeout=10.0)
        except Exception as exc:
            with suppress(BaseException):
                await asyncio.wait_for(context.__aexit__(None, None, None), timeout=2.0)
            raise CoretapError(
                "COREDEVICE_HID_SERVICE_FAILED",
                "CoreDevice UniversalHID keyboard session failed to open",
                stage="type",
                category="infrastructure",
                retryable=True,
                details={"device": device, "errorType": type(exc).__name__, "message": str(exc)},
            ) from exc
        now = time.monotonic()
        session = _KeyboardSession(device=device, context=context, service=service, created_at=now, last_used_at=now)
        self._keyboard_sessions[device] = session
        return session, "created"

    async def _type_text_with_keyboard_session(
        self,
        device: str,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        retry_close_scope = None
        try:
            session, status = await self._get_or_open_keyboard_session(device)
            input_result = await self._input_keyboard_text(
                session,
                text=text,
                char_delay_ms=char_delay_ms,
                inter_delay_ms=inter_delay_ms,
                clear_existing=clear_existing,
            )
        except Exception as exc:
            retry_close_scope = await self._close_for_retry(device, exc, session_kind="keyboard")
            try:
                session, _ = await self._get_or_open_keyboard_session(device)
                input_result = await self._input_keyboard_text(
                    session,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    clear_existing=clear_existing,
                )
                status = "recreated_after_error"
            except Exception as retry_exc:
                raise CoretapError(
                    "COREDEVICE_TYPE_FAILED",
                    f"Persistent CoreDevice keyboard input failed: {retry_exc}",
                    stage="type",
                    category="infrastructure",
                    retryable=True,
                    details={
                        "device": device,
                        "errorType": type(retry_exc).__name__,
                        "previousErrorType": type(exc).__name__,
                        "retryCloseScope": retry_close_scope,
                    },
                ) from retry_exc
        session.last_used_at = time.monotonic()
        session.typed_characters += len(text)
        rsd_session = self._rsd_sessions.get(device)
        if rsd_session is not None:
            rsd_session.last_used_at = session.last_used_at
            rsd_session.requests += 1
        return {
            "dispatchStatus": "sent",
            "confirmationStatus": "dispatched_unverified",
            "sessionStatus": status,
            "sessionTypedCharacterCount": session.typed_characters,
            "keyboardServiceRegistered": session.keyboard_service_id is not None,
            **input_result,
            **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
        }

    async def _type_text_with_pinyin_keyboard_session(
        self,
        device: str,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        retry_close_scope = None
        try:
            session, status = await self._get_or_open_keyboard_session(device)
            input_result = await self._input_pinyin_keyboard_text(
                session,
                text=text,
                char_delay_ms=char_delay_ms,
                inter_delay_ms=inter_delay_ms,
                clear_existing=clear_existing,
            )
        except Exception as exc:
            retry_close_scope = await self._close_for_retry(device, exc, session_kind="keyboard")
            try:
                session, _ = await self._get_or_open_keyboard_session(device)
                input_result = await self._input_pinyin_keyboard_text(
                    session,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    clear_existing=clear_existing,
                )
                status = "recreated_after_error"
            except Exception as retry_exc:
                raise CoretapError(
                    "COREDEVICE_TYPE_FAILED",
                    f"Persistent CoreDevice pinyin text input failed: {retry_exc}",
                    stage="type",
                    category="infrastructure",
                    retryable=True,
                    details={
                        "device": device,
                        "errorType": type(retry_exc).__name__,
                        "previousErrorType": type(exc).__name__,
                        "retryCloseScope": retry_close_scope,
                    },
                ) from retry_exc
        session.last_used_at = time.monotonic()
        session.typed_characters += len(text)
        rsd_session = self._rsd_sessions.get(device)
        if rsd_session is not None:
            rsd_session.last_used_at = session.last_used_at
            rsd_session.requests += 1
        return {
            "dispatchStatus": "sent",
            "confirmationStatus": "dispatched_unverified",
            "sessionStatus": status,
            "sessionTypedCharacterCount": session.typed_characters,
            "keyboardServiceRegistered": session.keyboard_service_id is not None,
            **input_result,
            **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
        }

    async def _keyboard_key_userspace(
        self,
        device: str,
        *,
        key: str,
        count: int,
        inter_delay_ms: int,
    ) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            retry_close_scope = None
            try:
                session, status = await self._get_or_open_keyboard_session(device)
                input_result = await self._send_keyboard_key(
                    session,
                    key=key,
                    count=count,
                    inter_delay_ms=inter_delay_ms,
                )
            except Exception as exc:
                retry_close_scope = await self._close_for_retry(device, exc, session_kind="keyboard")
                try:
                    session, _ = await self._get_or_open_keyboard_session(device)
                    input_result = await self._send_keyboard_key(
                        session,
                        key=key,
                        count=count,
                        inter_delay_ms=inter_delay_ms,
                    )
                    status = "recreated_after_error"
                except Exception as retry_exc:
                    raise CoretapError(
                        "COREDEVICE_KEY_FAILED",
                        f"Persistent CoreDevice keyboard key input failed: {retry_exc}",
                        stage="key",
                        category="infrastructure",
                        retryable=True,
                        details={
                            "device": device,
                            "key": key,
                            "count": count,
                            "errorType": type(retry_exc).__name__,
                            "previousErrorType": type(exc).__name__,
                            "retryCloseScope": retry_close_scope,
                        },
                    ) from retry_exc
            session.last_used_at = time.monotonic()
            session.keypresses += count
            rsd_session = self._rsd_sessions.get(device)
            if rsd_session is not None:
                rsd_session.last_used_at = session.last_used_at
                rsd_session.requests += 1
            return {
                "dispatchStatus": "sent",
                "confirmationStatus": "dispatched_unverified",
                "sessionStatus": status,
                "sessionKeypressCount": session.keypresses,
                "keyboardServiceRegistered": session.keyboard_service_id is not None,
                **input_result,
                **({"retryCloseScope": retry_close_scope} if retry_close_scope else {}),
            }

    async def _send_tap(self, session: _TouchSession, hx: int, hy: int) -> None:
        from pymobiledevice3.remote.core_device.hid_service import TOUCHSCREEN_STATE_CONTACT, TOUCHSCREEN_STATE_RELEASE

        await session.service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, hx, hy)
        await asyncio.sleep(0.05)
        await session.service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, hx, hy)

    async def _send_drag(
        self,
        session: _TouchSession,
        start_hx: int,
        start_hy: int,
        end_hx: int,
        end_hy: int,
        *,
        steps: int,
        duration_ms: int,
    ) -> None:
        from pymobiledevice3.remote.core_device.hid_service import TOUCHSCREEN_STATE_CONTACT, TOUCHSCREEN_STATE_RELEASE

        step_count = max(1, steps)
        sleep_s = max(0, duration_ms) / 1000 / step_count
        for index in range(step_count + 1):
            ratio = index / step_count
            hx = int(round(start_hx + (end_hx - start_hx) * ratio))
            hy = int(round(start_hy + (end_hy - start_hy) * ratio))
            await session.service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, hx, hy)
            if index < step_count and sleep_s:
                await asyncio.sleep(sleep_s)
        await session.service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, end_hx, end_hy)

    async def _send_button(
        self,
        session: _ButtonSession,
        *,
        state: str,
        usage_page: int,
        usage_code: int,
        hold_ms: int,
    ) -> None:
        from pymobiledevice3.remote.core_device.hid_service import (
            HID_BUTTON_STATE_CANCELED,
            HID_BUTTON_STATE_DOWN,
            HID_BUTTON_STATE_UP,
        )

        states = {
            "down": HID_BUTTON_STATE_DOWN,
            "up": HID_BUTTON_STATE_UP,
            "canceled": HID_BUTTON_STATE_CANCELED,
        }
        if state == "press":
            await session.service.send_button(usage_page, usage_code, HID_BUTTON_STATE_DOWN)
            await asyncio.sleep(hold_ms / 1000)
            await session.service.send_button(usage_page, usage_code, HID_BUTTON_STATE_UP)
        else:
            await session.service.send_button(usage_page, usage_code, states[state])
        await asyncio.sleep(0.1)

    async def _send_text(
        self,
        session: Any,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
    ) -> None:
        from pymobiledevice3.remote.core_device.hid_service import ASCII_TO_HID, KEY_LEFT_SHIFT

        if session.keyboard_service_id is None:
            session.keyboard_service_id = await session.service.create_keyboard_service()
        char_delay = char_delay_ms / 1000
        inter_delay = inter_delay_ms / 1000
        for ch in text:
            mapping = ASCII_TO_HID.get(ch)
            if mapping is None:
                raise ValueError(f"unsupported character: {ch!r}")
            usage, needs_shift = mapping
            usages = (KEY_LEFT_SHIFT, usage) if needs_shift else (usage,)
            await session.service.send_keyboard(session.keyboard_service_id, usages)
            if char_delay:
                await asyncio.sleep(char_delay)
            await session.service.send_keyboard(session.keyboard_service_id, ())
            if inter_delay:
                await asyncio.sleep(inter_delay)

    async def _send_keyboard_key(
        self,
        session: Any,
        *,
        key: str,
        count: int,
        inter_delay_ms: int,
    ) -> dict[str, Any]:
        usage = _keyboard_key_usage(key)
        if session.keyboard_service_id is None:
            session.keyboard_service_id = await session.service.create_keyboard_service()
        inter_delay = inter_delay_ms / 1000
        for _ in range(count):
            await session.service.send_keyboard(session.keyboard_service_id, (usage,))
            await session.service.send_keyboard(session.keyboard_service_id, ())
            if inter_delay:
                await asyncio.sleep(inter_delay)
        return {
            "inputMethod": "coredevice-virtual-keyboard",
            "resolvedKey": _normalize_keyboard_key(key),
            "hidKeyboardUsage": usage,
            "keypressesSent": count,
        }

    def _supports_virtual_keyboard_text(self, text: str) -> bool:
        from pymobiledevice3.remote.core_device.hid_service import ASCII_TO_HID

        return all(ch in ASCII_TO_HID for ch in text)

    async def _input_keyboard_text(
        self,
        session: Any,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        clear_count = 0
        if clear_existing:
            clear_count = await self._clear_focused_text(session)
        if inter_delay_ms:
            await asyncio.sleep(inter_delay_ms / 1000)
        await self._send_text(
            session,
            text=text,
            char_delay_ms=char_delay_ms,
            inter_delay_ms=inter_delay_ms,
        )
        return {
            "inputMethod": "coredevice-virtual-keyboard",
            "pasteboardSet": False,
            "clearExisting": clear_existing,
            "clearKeypresses": clear_count,
        }

    async def _input_pinyin_keyboard_text(
        self,
        session: Any,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        from pymobiledevice3.remote.core_device.hid_service import TOUCHSCREEN_STATE_CONTACT, TOUCHSCREEN_STATE_RELEASE

        keyboard_text = _pinyin_keyboard_text(text)
        if keyboard_text is None:
            raise ValueError(f"text cannot be converted to pinyin keyboard input: {text!r}")
        clear_count = 0
        if clear_existing:
            clear_count = await self._clear_focused_text(session)
        if inter_delay_ms:
            await asyncio.sleep(inter_delay_ms / 1000)
        await self._send_text(
            session,
            text=keyboard_text,
            char_delay_ms=char_delay_ms,
            inter_delay_ms=inter_delay_ms,
        )
        if session.keyboard_service_id is None:
            session.keyboard_service_id = await session.service.create_keyboard_service()
        candidate_settle_ms = 500
        await asyncio.sleep(candidate_settle_ms / 1000)
        candidate_x = 0.2
        candidate_y = 0.878
        candidate_hx = int(round(candidate_x * 65535))
        candidate_hy = int(round(candidate_y * 65535))
        await session.service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, candidate_hx, candidate_hy)
        await asyncio.sleep(0.05)
        await session.service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, candidate_hx, candidate_hy)
        await asyncio.sleep(0.25)
        return {
            "inputMethod": "coredevice-pinyin-keyboard",
            "pasteboardSet": False,
            "convertedText": keyboard_text,
            "candidateSettleMs": candidate_settle_ms,
            "candidateCommitAction": "tap-first-candidate",
            "candidateCommitPoint": {
                "normalized": {"x": candidate_x, "y": candidate_y},
                "hidU16": {"x": candidate_hx, "y": candidate_hy},
            },
            "clearExisting": clear_existing,
            "clearKeypresses": clear_count,
        }

    async def _input_text(
        self,
        session: _TouchSession,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        paste_at: dict[str, float] | None,
        paste_hold_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        from pymobiledevice3.remote.core_device.hid_service import ASCII_TO_HID

        if all(ch in ASCII_TO_HID for ch in text):
            clear_count = 0
            if clear_existing:
                clear_count = await self._clear_focused_text(session)
            if inter_delay_ms:
                await asyncio.sleep(inter_delay_ms / 1000)
            await self._send_text(
                session,
                text=text,
                char_delay_ms=char_delay_ms,
                inter_delay_ms=inter_delay_ms,
            )
            return {
                "inputMethod": "coredevice-virtual-keyboard",
                "pasteboardSet": False,
                "clearExisting": clear_existing,
                "clearKeypresses": clear_count,
            }
        return await self._paste_text(
            session,
            text=text,
            char_delay_ms=char_delay_ms,
            inter_delay_ms=inter_delay_ms,
            paste_at=paste_at,
            paste_hold_ms=paste_hold_ms,
            clear_existing=clear_existing,
        )

    async def _paste_text(
        self,
        session: _TouchSession,
        *,
        text: str,
        char_delay_ms: int,
        inter_delay_ms: int,
        paste_at: dict[str, float] | None,
        paste_hold_ms: int,
        clear_existing: bool,
    ) -> dict[str, Any]:
        from pymobiledevice3.remote.core_device.pasteboard_service import PasteboardService

        clear_count = 0
        if clear_existing:
            clear_count = await self._clear_focused_text(session)

        rsd_session = self._rsd_sessions[session.device]
        async with PasteboardService(rsd_session.rsd) as pasteboard:
            await pasteboard.set_text(text)

        anchor, anchor_source, paste_mode = self._resolve_paste_anchor(session, paste_at)
        await asyncio.sleep(inter_delay_ms / 1000)
        menu_point: dict[str, Any] | None = None
        if paste_mode == "menu":
            assert anchor is not None
            menu_point = await self._tap_visible_paste_menu(session, paste_x=anchor["x"], paste_y=anchor["y"])
        else:
            assert anchor is not None
            menu_point = await self._paste_via_edit_menu(
                session,
                anchor_x=anchor["x"],
                anchor_y=anchor["y"],
                hold_ms=paste_hold_ms,
            )
        if char_delay_ms:
            await asyncio.sleep(char_delay_ms / 1000)
        return {
            "inputMethod": "coredevice-pasteboard-edit-menu",
            "pasteAnchor": {"source": anchor_source, **anchor} if anchor is not None else {"source": anchor_source},
            "pasteMenuTap": menu_point,
            "pasteMode": paste_mode,
            "pasteHoldMs": paste_hold_ms,
            "clearExisting": clear_existing,
            "clearKeypresses": clear_count,
        }

    def _resolve_paste_anchor(
        self,
        session: _TouchSession,
        paste_at: dict[str, float] | None,
    ) -> tuple[dict[str, float] | None, str, str]:
        if paste_at is not None:
            mode = str(paste_at.get("mode") or "anchor")
            if mode == "shortcut":
                raise CoretapError(
                    "TEXT_INPUT_UNSUPPORTED",
                    "CoreDevice keyboard shortcut paste is not supported for iOS text input",
                    category="usage",
                    stage="type",
                    details={"mode": mode},
                )
            source = "visible-edit-menu" if mode == "menu" else "explicit"
            return {"x": float(paste_at["x"]), "y": float(paste_at["y"])}, source, mode
        raise CoretapError(
            "TEXT_INPUT_TARGET_UNKNOWN",
            "Non-ASCII text input requires a paste anchor resolved from a visible text field",
            category="usage",
            stage="type",
        )

    async def _tap_visible_paste_menu(
        self,
        session: _TouchSession,
        *,
        paste_x: float,
        paste_y: float,
    ) -> dict[str, Any]:
        paste_hx = int(round(paste_x * 65535))
        paste_hy = int(round(paste_y * 65535))
        await self._send_tap(session, paste_hx, paste_hy)
        await asyncio.sleep(0.25)
        return {
            "normalized": {"x": paste_x, "y": paste_y},
            "hidU16": {"x": paste_hx, "y": paste_hy},
            "strategy": "visible-edit-menu",
        }

    async def _clear_focused_text(self, session: Any, *, count: int = 80) -> int:
        from pymobiledevice3.remote.core_device.hid_service import KEY_BACKSPACE

        if session.keyboard_service_id is None:
            session.keyboard_service_id = await session.service.create_keyboard_service()
        for _ in range(count):
            await session.service.send_keyboard(session.keyboard_service_id, (KEY_BACKSPACE,))
            await session.service.send_keyboard(session.keyboard_service_id, ())
        await asyncio.sleep(0.15)
        return count

    async def _paste_via_edit_menu(
        self,
        session: _TouchSession,
        *,
        anchor_x: float,
        anchor_y: float,
        hold_ms: int,
    ) -> dict[str, Any]:
        hx = int(round(anchor_x * 65535))
        hy = int(round(anchor_y * 65535))
        await self._send_drag(session, hx, hy, hx, hy, steps=12, duration_ms=hold_ms)
        await asyncio.sleep(0.5)
        paste_x, paste_y = _paste_menu_point_for_anchor(anchor_x, anchor_y)
        paste_hx = int(round(paste_x * 65535))
        paste_hy = int(round(paste_y * 65535))
        await self._send_tap(session, paste_hx, paste_hy)
        await asyncio.sleep(0.25)
        return {
            "normalized": {"x": paste_x, "y": paste_y},
            "hidU16": {"x": paste_hx, "y": paste_hy},
        }

    async def _close_button_session(self, device: str) -> None:
        session = self._button_sessions.pop(device, None)
        if session is None:
            return
        with suppress(BaseException):
            await session.context.__aexit__(None, None, None)

    async def _close_keyboard_session(self, device: str) -> None:
        session = self._keyboard_sessions.pop(device, None)
        if session is None:
            return
        with suppress(BaseException):
            await session.context.__aexit__(None, None, None)

    async def _close_for_retry(self, device: str, error: BaseException, *, session_kind: str) -> str:
        if is_recoverable_userspace_tunnel_error(error):
            await self._close_device(device)
            await asyncio.sleep(0.25)
            return "device"
        if session_kind == "touch":
            await self._close_session(device)
        elif session_kind == "button":
            await self._close_button_session(device)
        elif session_kind == "keyboard":
            await self._close_keyboard_session(device)
        else:
            await self._close_device(device)
        return session_kind

    async def _close_session(self, device: str) -> None:
        session = self._sessions.pop(device, None)
        if session is None:
            return
        with suppress(BaseException):
            await session.context.__aexit__(None, None, None)

    async def _close_rsd(self, device: str) -> None:
        session = self._rsd_sessions.pop(device, None)
        if session is None:
            return
        close = getattr(session.rsd, "close", None)
        if close is not None:
            with suppress(BaseException):
                await close()

    async def _close_device(self, device: str) -> None:
        await self._close_session(device)
        await self._close_button_session(device)
        await self._close_keyboard_session(device)
        await self._close_rsd(device)

    async def _close_all(self) -> None:
        for device in list(self._sessions):
            await self._close_session(device)
        for device in list(self._button_sessions):
            await self._close_button_session(device)
        for device in list(self._keyboard_sessions):
            await self._close_keyboard_session(device)
        for device in list(self._rsd_sessions):
            await self._close_rsd(device)

    async def _recover_display_service_for_retry(
        self,
        device: str,
        *,
        stage: str,
        error: BaseException,
    ) -> dict[str, Any] | None:
        if not is_recoverable_coredevice_display_error(error):
            return None
        recovery = await asyncio.to_thread(recover_coredevice_display_service, device)
        self._recovery_count += 1
        self._last_recovery = {
            "stage": stage,
            "reason": "display-service-error",
            "errorType": type(error).__name__,
            "message": str(error),
            "atMonotonic": time.monotonic(),
            "displayService": recovery,
        }
        await asyncio.sleep(1.0 if recovery.get("status") == "signaled" else 0.25)
        return recovery
