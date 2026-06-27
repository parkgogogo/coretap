from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import uuid
from contextlib import suppress
from collections.abc import AsyncIterator


def emit(data: dict[str, object]) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


@contextlib.asynccontextmanager
async def bounded_touch_session(rsd: object, *, display_id: int = 1) -> AsyncIterator[object]:
    from pymobiledevice3.remote.core_device.display_service import DisplayService
    from pymobiledevice3.remote.core_device.hid_service import UniversalHIDServiceService
    from pymobiledevice3.remote.core_device.screen_stream import open_media_receiver

    sender_ip = rsd.service.address[0]  # type: ignore[attr-defined]
    display = DisplayService(rsd)  # type: ignore[arg-type]
    transport = None
    drain_task: asyncio.Task[None] | None = None
    client_session_id: uuid.UUID | None = None
    phase = "display-enter"

    try:
        await display.__aenter__()
        phase = "open-media-receiver"
        transport, receiver_ip = open_media_receiver(display, (1 * 1024 * 1024,))

        async def drain() -> None:
            try:
                while True:
                    await transport.recv()  # type: ignore[union-attr]
            except (asyncio.CancelledError, OSError):
                pass

        phase = "start-video-stream"
        answer = await asyncio.wait_for(
            display.start_video_stream(
                receiver_ip=receiver_ip,
                receiver_port=transport.port,
                sender_ip=sender_ip,
                display_id=display_id,
            ),
            timeout=12.0,
        )
        raw_session_id = answer["connection"]["options"]["avcMediaStreamOptionClientSessionID"]["uuid"]
        client_session_id = raw_session_id if isinstance(raw_session_id, uuid.UUID) else uuid.UUID(raw_session_id)
        await asyncio.sleep(0.3)
        drain_task = asyncio.create_task(drain())
        phase = "universal-hid-enter"
        async with UniversalHIDServiceService(rsd) as hid:  # type: ignore[arg-type]
            phase = "yield-hid"
            yield hid
    except BaseException as exc:
        raise RuntimeError(f"bounded touch session failed during {phase}: {type(exc).__name__}: {exc}") from exc
    finally:
        if drain_task is not None:
            drain_task.cancel()
            with suppress(BaseException):
                await asyncio.wait_for(drain_task, timeout=1.0)
        if client_session_id is not None:
            with suppress(BaseException):
                await asyncio.wait_for(display.stop_media_stream(client_session_id), timeout=2.0)
        if transport is not None:
            with suppress(BaseException):
                transport.close()
        with suppress(BaseException):
            await asyncio.wait_for(display.__aexit__(None, None, None), timeout=2.0)


async def tap_userspace(device: str, x: int, y: int) -> None:
    from pymobiledevice3.remote import userspace_tunnel
    from pymobiledevice3.remote.core_device.hid_service import (
        TOUCHSCREEN_STATE_CONTACT,
        TOUCHSCREEN_STATE_RELEASE,
    )

    stage = "establish-rsd"
    rsd = await userspace_tunnel.establish_userspace_rsd(serial=device)
    try:
        stage = "touch-session"
        async with bounded_touch_session(rsd) as service:
            stage = "send-contact"
            await service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, x, y)
            await asyncio.sleep(0.05)
            stage = "send-release"
            await service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, x, y)
        stage = "emit-result"
        emit({"event": "dispatch_sent", "x": x, "y": y})
    except BaseException as exc:
        emit({"event": "error", "errorType": type(exc).__name__, "message": str(exc), "stage": stage})
        raise
    finally:
        close = getattr(rsd, "close", None)
        if close is not None:
            with suppress(BaseException):
                await close()


async def type_text_userspace(
    device: str,
    *,
    text: str,
    char_delay_ms: int,
    inter_delay_ms: int,
    paste_at: dict[str, float] | None,
    paste_hold_ms: int,
    clear_existing: bool,
) -> None:
    from pymobiledevice3.remote import userspace_tunnel
    from pymobiledevice3.remote.core_device.hid_service import (
        ASCII_TO_HID,
        KEY_BACKSPACE,
        KEY_LEFT_SHIFT,
        TOUCHSCREEN_STATE_CONTACT,
        TOUCHSCREEN_STATE_RELEASE,
        UniversalHIDServiceService,
    )
    from pymobiledevice3.remote.core_device.pasteboard_service import PasteboardService

    rsd = await userspace_tunnel.establish_userspace_rsd(serial=device)
    keyboard_service_id: int | None = None

    async def send_tap(service: object, x: float, y: float) -> dict[str, object]:
        hx = int(round(x * 65535))
        hy = int(round(y * 65535))
        await service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, hx, hy)  # type: ignore[attr-defined]
        await asyncio.sleep(0.05)
        await service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, hx, hy)  # type: ignore[attr-defined]
        return {"normalized": {"x": x, "y": y}, "hidU16": {"x": hx, "y": hy}}

    async def send_drag(service: object, x: float, y: float, hold_ms: int) -> None:
        hx = int(round(x * 65535))
        hy = int(round(y * 65535))
        steps = 12
        sleep_s = hold_ms / 1000 / steps
        for index in range(steps + 1):
            await service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, hx, hy)  # type: ignore[attr-defined]
            if index < steps and sleep_s:
                await asyncio.sleep(sleep_s)
        await service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, hx, hy)  # type: ignore[attr-defined]

    async def clear_text(service: object) -> int:
        nonlocal keyboard_service_id
        if keyboard_service_id is None:
            keyboard_service_id = await service.create_keyboard_service()  # type: ignore[attr-defined]
        for _ in range(80):
            await service.send_keyboard(keyboard_service_id, (KEY_BACKSPACE,))  # type: ignore[attr-defined]
            await service.send_keyboard(keyboard_service_id, ())  # type: ignore[attr-defined]
        await asyncio.sleep(0.15)
        return 80

    async def type_ascii(service: object) -> None:
        nonlocal keyboard_service_id
        if keyboard_service_id is None:
            keyboard_service_id = await service.create_keyboard_service()  # type: ignore[attr-defined]
        char_delay = char_delay_ms / 1000
        inter_delay = inter_delay_ms / 1000
        for ch in text:
            usage, needs_shift = ASCII_TO_HID[ch]
            usages = (KEY_LEFT_SHIFT, usage) if needs_shift else (usage,)
            await service.send_keyboard(keyboard_service_id, usages)  # type: ignore[attr-defined]
            if char_delay:
                await asyncio.sleep(char_delay)
            await service.send_keyboard(keyboard_service_id, ())  # type: ignore[attr-defined]
            if inter_delay:
                await asyncio.sleep(inter_delay)

    stage = "set-pasteboard"
    try:
        ascii_only = all(ch in ASCII_TO_HID for ch in text)
        if ascii_only:
            stage = "keyboard-session"
            result: dict[str, object]
            async with UniversalHIDServiceService(rsd) as service:
                clear_count = 0
                if clear_existing:
                    stage = "clear-existing"
                    clear_count = await clear_text(service)
                stage = "type-ascii"
                await type_ascii(service)
                stage = "emit-result"
                result = {
                    "dispatchStatus": "sent",
                    "confirmationStatus": "dispatched_unverified",
                    "sessionStatus": "helper-direct",
                    "sessionTypedCharacterCount": len(text),
                    "keyboardServiceRegistered": keyboard_service_id is not None,
                    "pasteboardSet": False,
                    "inputMethod": "coredevice-virtual-keyboard",
                    "clearExisting": clear_existing,
                    "clearKeypresses": clear_count,
                }
            emit({"event": "result", "result": result})
            return

        async with PasteboardService(rsd) as pasteboard:
            await pasteboard.set_text(text)

        stage = "touch-session"
        result = {}
        async with bounded_touch_session(rsd) as service:
            clear_count = 0
            if clear_existing:
                stage = "clear-existing"
                clear_count = await clear_text(service)

            if paste_at is None:
                anchor = {"x": 0.2, "y": 0.54}
                anchor_source = "ios-spotlight-search-field"
            else:
                anchor = paste_at
                anchor_source = "explicit"

            if inter_delay_ms:
                await asyncio.sleep(inter_delay_ms / 1000)
            stage = "open-edit-menu"
            await send_drag(service, anchor["x"], anchor["y"], paste_hold_ms)
            await asyncio.sleep(0.5)
            paste_x = min(0.92, max(0.08, anchor["x"] - 0.07))
            paste_y = min(0.95, max(0.05, anchor["y"] - 0.059))
            stage = "tap-paste-menu"
            paste_menu_tap = await send_tap(service, paste_x, paste_y)
            if char_delay_ms:
                await asyncio.sleep(char_delay_ms / 1000)
            stage = "emit-result"
            result = {
                "dispatchStatus": "sent",
                "confirmationStatus": "dispatched_unverified",
                "sessionStatus": "helper-direct",
                "sessionTypedCharacterCount": len(text),
                "keyboardServiceRegistered": keyboard_service_id is not None,
                "pasteboardSet": True,
                "inputMethod": "coredevice-pasteboard-edit-menu",
                "pasteAnchor": {"source": anchor_source, **anchor},
                "pasteMenuTap": paste_menu_tap,
                "pasteHoldMs": paste_hold_ms,
                "clearExisting": clear_existing,
                "clearKeypresses": clear_count,
            }
        emit({"event": "result", "result": result})
    except BaseException as exc:
        emit({"event": "error", "errorType": type(exc).__name__, "message": str(exc), "stage": stage})
        raise
    finally:
        close = getattr(rsd, "close", None)
        if close is not None:
            with suppress(BaseException):
                await close()


def parse_paste_at(raw: str | None) -> dict[str, float] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--paste-at must be formatted as x,y")
    x = float(parts[0])
    y = float(parts[1])
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("--paste-at values must be normalized values in [0,1]")
    return {"x": x, "y": y}


async def run(args: argparse.Namespace) -> None:
    if args.mode != "userspace":
        raise ValueError(f"unsupported mode: {args.mode}")
    if args.action == "tap":
        if args.x is None or args.y is None:
            raise ValueError("tap requires --x and --y")
        await tap_userspace(args.device, args.x, args.y)
        return
    if args.action == "type":
        await type_text_userspace(
            args.device,
            text=args.text or "",
            char_delay_ms=args.char_delay_ms,
            inter_delay_ms=args.inter_delay_ms,
            paste_at=parse_paste_at(args.paste_at),
            paste_hold_ms=args.paste_hold_ms,
            clear_existing=args.clear_existing,
        )
        return
    raise ValueError(f"unsupported action: {args.action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coretap.device_hid_helper")
    parser.add_argument("--mode", choices=["userspace"], required=True)
    parser.add_argument("--action", choices=["tap", "type"], default="tap")
    parser.add_argument("--device", required=True)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--char-delay-ms", type=int, default=40)
    parser.add_argument("--inter-delay-ms", type=int, default=20)
    parser.add_argument("--paste-at", default=None)
    parser.add_argument("--paste-hold-ms", type=int, default=1600)
    parser.add_argument("--clear-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args))
    except BaseException as exc:
        emit({"event": "error", "errorType": type(exc).__name__, "message": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
