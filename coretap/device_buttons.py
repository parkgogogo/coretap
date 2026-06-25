from __future__ import annotations

from dataclasses import dataclass


BUTTON_STATES = ("press", "down", "up", "canceled")


@dataclass(frozen=True)
class CoreDeviceButton:
    name: str
    usage_page: int
    usage_code: int
    hold_ms: int


COREDEVICE_BUTTONS: dict[str, CoreDeviceButton] = {
    "home": CoreDeviceButton("home", 0x0C, 0x40, 50),
    "lock": CoreDeviceButton("lock", 0x0C, 0x30, 500),
    "volume-up": CoreDeviceButton("volume-up", 0x0C, 0xE9, 50),
    "volume-down": CoreDeviceButton("volume-down", 0x0C, 0xEA, 50),
    "mute": CoreDeviceButton("mute", 0x0C, 0xE2, 50),
    "siri": CoreDeviceButton("siri", 0x0C, 0xCF, 1000),
}


BUTTON_ALIASES = {
    "power": "lock",
    "vol-up": "volume-up",
    "vol-down": "volume-down",
    "volumeup": "volume-up",
    "volumedown": "volume-down",
}


def button_choices() -> list[str]:
    return sorted([*COREDEVICE_BUTTONS, *BUTTON_ALIASES])


def resolve_button(name: str) -> CoreDeviceButton | None:
    normalized = name.strip().lower().replace("_", "-")
    canonical = BUTTON_ALIASES.get(normalized, normalized)
    return COREDEVICE_BUTTONS.get(canonical)
