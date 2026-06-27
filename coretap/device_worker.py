from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from coretap.runtime import CoretapError


_DEFAULT_POOL: "CoreDeviceWorkerPool | None" = None


def set_default_device_worker_pool(pool: "CoreDeviceWorkerPool | None") -> None:
    global _DEFAULT_POOL
    _DEFAULT_POOL = pool


def get_default_device_worker_pool() -> "CoreDeviceWorkerPool | None":
    return _DEFAULT_POOL


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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


class CoreDeviceWorkerPool:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._rsd_sessions: dict[str, _RsdSession] = {}
        self._sessions: dict[str, _TouchSession] = {}
        self._button_sessions: dict[str, _ButtonSession] = {}
        self._lock: asyncio.Lock | None = None

    def tap_userspace(self, device: str, x: float, y: float, hx: int, hy: int) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(self._tap_userspace(device, x, y, hx, hy), stage="tap")
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

    def capture_screenshot_userspace(self, device: str) -> dict[str, Any]:
        started = time.monotonic()
        result = self._run(self._capture_screenshot_userspace(device), stage="screenshot")
        result.update(
            {
                "coredeviceTunnelMode": "userspace",
                "workerKind": "coredevice-userspace-persistent",
                "durationMs": round((time.monotonic() - started) * 1000),
            }
        )
        return result

    def display_info_userspace(self, device: str) -> dict[str, Any]:
        return self._run(self._display_info_userspace(device), stage="display-info")

    def start(self) -> None:
        self._ensure_loop()

    def stats(self) -> dict[str, Any]:
        rsd_sessions = []
        touch_sessions = []
        button_sessions = []
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
        return {
            "kind": "coredevice-userspace-persistent",
            "running": self._thread is not None and self._thread.is_alive(),
            "rsdSessionCount": len(self._rsd_sessions),
            "touchSessionCount": len(self._sessions),
            "buttonSessionCount": len(self._button_sessions),
            "rsdSessions": rsd_sessions,
            "touchSessions": touch_sessions,
            "buttonSessions": button_sessions,
        }

    def close(self) -> None:
        if self._loop is None:
            return
        with suppress(BaseException):
            self._run(self._close_all())
        loop = self._loop
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._loop = None
        self._thread = None
        self._ready.clear()

    def _run(self, coro: Any, *, stage: str) -> Any:
        self._ensure_loop()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=30)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise CoretapError(
                "COREDEVICE_WORKER_TIMEOUT",
                "CoreDevice worker request timed out",
                stage=stage,
                category="infrastructure",
                retryable=True,
            ) from exc
        except CoretapError:
            raise
        except Exception as exc:
            raise CoretapError(
                "COREDEVICE_WORKER_FAILED",
                f"CoreDevice worker request failed: {exc}",
                stage=stage,
                category="infrastructure",
                retryable=True,
                details={"errorType": type(exc).__name__},
            ) from exc

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
            try:
                session, status = await self._get_or_open_session(device)
                await self._send_tap(session, hx, hy)
            except Exception as exc:
                await self._close_device(device)
                try:
                    session, _ = await self._get_or_open_session(device)
                    await self._send_tap(session, hx, hy)
                    status = "recreated_after_error"
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
                await self._close_device(device)
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
                    status = "recreated_after_error"
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
            try:
                session, status = await self._get_or_open_button_session(device)
                await self._send_button(session, state=state, usage_page=usage_page, usage_code=usage_code, hold_ms=hold_ms)
            except Exception as exc:
                await self._close_device(device)
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
            try:
                session, status = await self._get_or_open_session(device)
                paste_result = await self._paste_text(
                    session,
                    text=text,
                    char_delay_ms=char_delay_ms,
                    inter_delay_ms=inter_delay_ms,
                    paste_at=paste_at,
                    paste_hold_ms=paste_hold_ms,
                    clear_existing=clear_existing,
                )
            except Exception as exc:
                await self._close_device(device)
                try:
                    session, _ = await self._get_or_open_session(device)
                    paste_result = await self._paste_text(
                        session,
                        text=text,
                        char_delay_ms=char_delay_ms,
                        inter_delay_ms=inter_delay_ms,
                        paste_at=paste_at,
                        paste_hold_ms=paste_hold_ms,
                        clear_existing=clear_existing,
                    )
                    status = "recreated_after_error"
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
                **paste_result,
            }

    async def _capture_screenshot_userspace(self, device: str) -> dict[str, Any]:
        assert self._lock is not None
        async with self._lock:
            try:
                return await self._capture_screenshot_once(device)
            except Exception as exc:
                await self._close_device(device)
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
                await self._close_device(device)
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

        from pymobiledevice3.remote.core_device.hid_service import touch_session

        rsd_session, _ = await self._get_or_open_rsd(device)
        context = touch_session(rsd_session.rsd)
        service = await context.__aenter__()
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
        session: _TouchSession,
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

        anchor, anchor_source = self._resolve_paste_anchor(session, paste_at)
        await asyncio.sleep(inter_delay_ms / 1000)
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
            "pasteAnchor": {"source": anchor_source, **anchor},
            "pasteMenuTap": menu_point,
            "pasteHoldMs": paste_hold_ms,
            "clearExisting": clear_existing,
            "clearKeypresses": clear_count,
        }

    def _resolve_paste_anchor(
        self,
        session: _TouchSession,
        paste_at: dict[str, float] | None,
    ) -> tuple[dict[str, float], str]:
        if paste_at is not None:
            return paste_at, "explicit"
        last = session.last_tap_normalized
        if last is not None:
            y = last["y"]
            if y < 0.75 or y >= 0.9:
                return dict(last), "last-tap"
        return {"x": 0.2, "y": 0.54}, "ios-spotlight-search-field"

    async def _clear_focused_text(self, session: _TouchSession, *, count: int = 80) -> int:
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
        paste_x = _clamp(anchor_x - 0.07, 0.08, 0.92)
        paste_y = _clamp(anchor_y - 0.059, 0.05, 0.95)
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
        await self._close_rsd(device)

    async def _close_all(self) -> None:
        for device in list(self._sessions):
            await self._close_session(device)
        for device in list(self._button_sessions):
            await self._close_button_session(device)
        for device in list(self._rsd_sessions):
            await self._close_rsd(device)
