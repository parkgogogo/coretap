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


async def run(args: argparse.Namespace) -> None:
    if args.mode != "userspace":
        raise ValueError(f"unsupported mode: {args.mode}")
    await tap_userspace(args.device, args.x, args.y)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m coretap.device_hid_helper")
    parser.add_argument("--mode", choices=["userspace"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
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
