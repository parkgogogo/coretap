from __future__ import annotations

from pathlib import Path
from typing import Any

from coretap.ocr import find_text, run_tesseract, tesseract_status
from coretap.model_pack import (
    INTERNAL_FIXTURE_PROFILE,
    PUBLIC_MODEL_PROFILE,
    cache_status,
    check_model,
    gc_model,
    install_model,
    internal_profiles,
    public_profiles,
    run_grounding_model,
    stop_model,
    warm_model as warm_public_model,
)
from coretap.runtime import CoretapError


GROUNDING_PROFILES = public_profiles()
ALL_GROUNDING_PROFILES = {**GROUNDING_PROFILES, **internal_profiles()}


def model_status(profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    entry = ALL_GROUNDING_PROFILES.get(profile)
    if not entry:
        return {"ready": False, "profile": profile, "state": "unknown-profile"}
    if profile == INTERNAL_FIXTURE_PROFILE:
        ocr = tesseract_status()
        return {
            "ready": bool(ocr["ready"]),
            "profile": profile,
            "state": "ready" if ocr["ready"] else "missing-ocr",
            "implementation": "internal-ocr-fixture-grounder",
            "ocr": ocr,
        }
    return check_model(profile)


def warm_model(profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    if profile == INTERNAL_FIXTURE_PROFILE:
        status = model_status(profile)
        if not status["ready"]:
            raise CoretapError(
                "CAPABILITY_UNAVAILABLE",
                f"Internal fixture profile is not ready: {profile}",
                stage="model",
                details=status,
            )
        return {**status, "warm": True}
    return warm_public_model(profile)


def ground_target(image: Path, target: str, *, profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    if profile == INTERNAL_FIXTURE_PROFILE:
        return ground_text_target_fixture(image, target)
    return run_grounding_model(image, target, profile=profile)


def ground_text_target_fixture(image: Path, target: str) -> dict[str, Any]:
    tokens, raw_tsv = run_tesseract(image)
    match = find_text(tokens, target)
    if not match:
        return {
            "schema": "coretap.ground.result.v1",
            "status": "not_found",
            "target": {"description": target},
            "rawOcrTokenCount": len(tokens),
            "rawTsv": raw_tsv,
        }
    box = match["matchedBoxPx"]
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    width, height = _image_size(image)
    return {
        "schema": "coretap.ground.result.v1",
        "status": "found",
        "target": {"description": target},
        "point": {
            "framePx": {"x": x, "y": y},
            "normalized": {"x": x / width, "y": y / height},
        },
        "frame": {"widthPx": width, "heightPx": height},
        "matchedText": match["matchedText"],
        "matchedBoxPx": box,
        "rawOcrTokenCount": len(tokens),
        "rawTsv": raw_tsv,
        "model": {"profile": INTERNAL_FIXTURE_PROFILE},
    }


def _image_size(image: Path) -> tuple[int, int]:
    from coretap.runtime import png_size

    return png_size(image)


def model_install(profile: str = PUBLIC_MODEL_PROFILE, *, force: bool = False) -> dict[str, Any]:
    return install_model(profile, force=force)


def model_check(profile: str = PUBLIC_MODEL_PROFILE, *, deep: bool = False) -> dict[str, Any]:
    return check_model(profile, deep=deep)


def model_cache() -> dict[str, Any]:
    return cache_status()


def model_gc(*, dry_run: bool = False) -> dict[str, Any]:
    return gc_model(dry_run=dry_run)


def model_stop() -> dict[str, Any]:
    return stop_model()
