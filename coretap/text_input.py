from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_CONTROL_CHARS = {"\n", "\t"}


@dataclass(frozen=True)
class TextInputValidation:
    ok: bool
    unsupported: list[str]


def validate_hid_text(text: str) -> TextInputValidation:
    unsupported: list[str] = []
    seen: set[str] = set()
    for ch in text:
        if ch in SUPPORTED_CONTROL_CHARS or 32 <= ord(ch) <= 126:
            continue
        if ch not in seen:
            unsupported.append(ch)
            seen.add(ch)
    return TextInputValidation(ok=not unsupported, unsupported=unsupported)


def text_input_summary(text: str) -> dict[str, object]:
    return {
        "length": len(text),
        "byteLength": len(text.encode("utf-8")),
        "asciiOnly": validate_hid_text(text).ok,
    }
