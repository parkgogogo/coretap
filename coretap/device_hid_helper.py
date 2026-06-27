from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import suppress


def emit(data: dict[str, object]) -> None:
    print(json.dumps(data, ensure_ascii=False), flush=True)


async def tap_userspace(device: str, x: int, y: int) -> None:
    from pymobiledevice3.remote import userspace_tunnel
    from pymobiledevice3.remote.core_device.hid_service import (
        TOUCHSCREEN_STATE_CONTACT,
        TOUCHSCREEN_STATE_RELEASE,
        touch_session,
    )

    rsd = await userspace_tunnel.establish_userspace_rsd(serial=device)
    try:
        async with touch_session(rsd) as service:
            await service.send_touchscreen(TOUCHSCREEN_STATE_CONTACT, x, y)
            await asyncio.sleep(0.05)
            await service.send_touchscreen(TOUCHSCREEN_STATE_RELEASE, x, y)
            emit({"event": "dispatch_sent", "x": x, "y": y})
    finally:
        close = getattr(rsd, "close", None)
        if close is not None:
            with suppress(BaseException):
                await close()


def type_text_userspace(
    device: str,
    *,
    text: str,
    char_delay_ms: int,
    inter_delay_ms: int,
    paste_at: dict[str, float] | None,
    paste_hold_ms: int,
    clear_existing: bool,
) -> None:
    from coretap.device_worker import CoreDeviceWorkerPool

    pool = CoreDeviceWorkerPool()
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
        emit({"event": "result", "result": result})
    finally:
        pool.close()


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
        type_text_userspace(
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
