from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import uuid
from contextlib import suppress
from collections.abc import AsyncIterator

from coretap.runtime import CoretapError


def emit(data: dict[str, object]) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def pinyin_keyboard_text(text: str) -> str | None:
    if not contains_cjk(text):
        return None
    from pypinyin import Style, lazy_pinyin

    parts = lazy_pinyin(text, style=Style.NORMAL, errors="default", strict=False)
    keyboard_text = "".join(parts)
    if not keyboard_text or not keyboard_text.isascii():
        return None
    return keyboard_text


def paste_menu_point_for_anchor(anchor_x: float, anchor_y: float) -> tuple[float, float]:
    paste_x = min(0.92, max(0.08, anchor_x - 0.07))
    vertical_offset = 0.059 if anchor_y < 0.18 else -0.059
    paste_y = min(0.95, max(0.05, anchor_y + vertical_offset))
    return paste_x, paste_y


@contextlib.asynccontextmanager
async def bounded_touch_session(rsd: object, *, display_id: int = 1) -> AsyncIterator[object]:
    from pymobiledevice3.remote.core_device.hid_service import UniversalHIDServiceService

    try:
        async with UniversalHIDServiceService(rsd) as hid:  # type: ignore[arg-type]
            yield hid
    except BaseException as exc:
        raise RuntimeError(f"bounded touch session failed during universal-hid-enter: {type(exc).__name__}: {exc}") from exc


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
    paste_mode: str,
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

    async def type_pinyin(service: object, keyboard_text: str) -> dict[str, object]:
        nonlocal keyboard_service_id
        await type_ascii_text(service, keyboard_text)
        if keyboard_service_id is None:
            keyboard_service_id = await service.create_keyboard_service()  # type: ignore[attr-defined]
        await asyncio.sleep(0.5)
        commit_point = await send_tap(service, 0.2, 0.878)
        await asyncio.sleep(0.25)
        return commit_point

    async def type_ascii_text(service: object, keyboard_text: str) -> None:
        nonlocal keyboard_service_id
        if keyboard_service_id is None:
            keyboard_service_id = await service.create_keyboard_service()  # type: ignore[attr-defined]
        char_delay = char_delay_ms / 1000
        inter_delay = inter_delay_ms / 1000
        for ch in keyboard_text:
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

        pinyin_text = pinyin_keyboard_text(text)
        if pinyin_text is not None and paste_at is None:
            stage = "pinyin-keyboard-session"
            result = {}
            async with UniversalHIDServiceService(rsd) as service:
                clear_count = 0
                if clear_existing:
                    stage = "clear-existing"
                    clear_count = await clear_text(service)
                stage = "type-pinyin"
                commit_point = await type_pinyin(service, pinyin_text)
                stage = "emit-result"
                result = {
                    "dispatchStatus": "sent",
                    "confirmationStatus": "dispatched_unverified",
                    "sessionStatus": "helper-direct",
                    "sessionTypedCharacterCount": len(text),
                    "keyboardServiceRegistered": keyboard_service_id is not None,
                    "pasteboardSet": False,
                    "inputMethod": "coredevice-pinyin-keyboard",
                    "convertedText": pinyin_text,
                    "candidateCommitAction": "tap-first-candidate",
                    "candidateCommitPoint": commit_point,
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
                raise CoretapError(
                    "TEXT_INPUT_TARGET_UNKNOWN",
                    "Non-ASCII text input requires a paste anchor resolved from a visible text field",
                    category="usage",
                    stage="type",
                )
            if paste_mode == "shortcut":
                raise CoretapError(
                    "TEXT_INPUT_UNSUPPORTED",
                    "CoreDevice keyboard shortcut paste is not supported for iOS text input",
                    category="usage",
                    stage="type",
                    details={"mode": paste_mode},
                )
            else:
                anchor = paste_at
                anchor_source = "visible-edit-menu" if paste_mode == "menu" else "explicit"

            if inter_delay_ms:
                await asyncio.sleep(inter_delay_ms / 1000)
            paste_menu_tap = None
            if paste_mode == "menu":
                assert anchor is not None
                paste_x, paste_y = anchor["x"], anchor["y"]
                stage = "tap-paste-menu"
                paste_menu_tap = await send_tap(service, paste_x, paste_y)
            else:
                assert anchor is not None
                stage = "open-edit-menu"
                await send_drag(service, anchor["x"], anchor["y"], paste_hold_ms)
                await asyncio.sleep(0.5)
                paste_x, paste_y = paste_menu_point_for_anchor(anchor["x"], anchor["y"])
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
                "pasteAnchor": {"source": anchor_source, **anchor} if anchor is not None else {"source": anchor_source},
                "pasteMenuTap": paste_menu_tap,
                "pasteMode": paste_mode,
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


def normalize_keyboard_key(key: str) -> str:
    aliases = {
        "delete": "backspace",
        "return": "enter",
        "esc": "escape",
        "arrow-left": "left",
        "arrow-right": "right",
        "arrow-up": "up",
        "arrow-down": "down",
    }
    return aliases.get(key.strip().casefold().replace("_", "-"), key.strip().casefold())


def keyboard_key_usage(key: str) -> int:
    from pymobiledevice3.remote.core_device import hid_service

    normalized = normalize_keyboard_key(key)
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
        raise ValueError(f"unsupported keyboard key: {key}") from exc


async def keyboard_key_userspace(
    device: str,
    *,
    key: str,
    count: int,
    inter_delay_ms: int,
) -> None:
    from pymobiledevice3.remote import userspace_tunnel
    from pymobiledevice3.remote.core_device.hid_service import UniversalHIDServiceService

    rsd = await userspace_tunnel.establish_userspace_rsd(serial=device)
    stage = "keyboard-session"
    keyboard_service_id: int | None = None
    try:
        usage = keyboard_key_usage(key)
        normalized = normalize_keyboard_key(key)
        async with UniversalHIDServiceService(rsd) as service:
            keyboard_service_id = await service.create_keyboard_service()  # type: ignore[attr-defined]
            inter_delay = inter_delay_ms / 1000
            for _ in range(count):
                stage = "send-key"
                await service.send_keyboard(keyboard_service_id, (usage,))  # type: ignore[attr-defined]
                await service.send_keyboard(keyboard_service_id, ())  # type: ignore[attr-defined]
                if inter_delay:
                    await asyncio.sleep(inter_delay)
        emit(
            {
                "event": "result",
                "result": {
                    "dispatchStatus": "sent",
                    "confirmationStatus": "dispatched_unverified",
                    "sessionStatus": "helper-direct",
                    "keyboardServiceRegistered": keyboard_service_id is not None,
                    "inputMethod": "coredevice-virtual-keyboard",
                    "resolvedKey": normalized,
                    "hidKeyboardUsage": usage,
                    "keypressesSent": count,
                },
            }
        )
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
            paste_mode=args.paste_mode,
            paste_hold_ms=args.paste_hold_ms,
            clear_existing=args.clear_existing,
        )
        return
    if args.action == "key":
        await keyboard_key_userspace(
            args.device,
            key=args.key,
            count=args.count,
            inter_delay_ms=args.inter_delay_ms,
        )
        return
    if args.action == "clear":
        await keyboard_key_userspace(
            args.device,
            key="backspace",
            count=args.count,
            inter_delay_ms=args.inter_delay_ms,
        )
        return
    raise ValueError(f"unsupported action: {args.action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coretap.device_hid_helper")
    parser.add_argument("--mode", choices=["userspace"], required=True)
    parser.add_argument("--action", choices=["tap", "type", "key", "clear"], default="tap")
    parser.add_argument("--device", required=True)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--text", default=None)
    parser.add_argument("--char-delay-ms", type=int, default=40)
    parser.add_argument("--inter-delay-ms", type=int, default=20)
    parser.add_argument("--paste-at", default=None)
    parser.add_argument("--paste-mode", choices=["anchor", "menu"], default="anchor")
    parser.add_argument("--paste-hold-ms", type=int, default=1600)
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--key", default="backspace")
    parser.add_argument("--count", type=int, default=1)
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
