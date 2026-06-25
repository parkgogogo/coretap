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
DEFAULT_GROUNDING_IMAGE_LONG_SIDE = 1368


def prepare_grounding_image(
    image: Path,
    *,
    output_dir: Path,
    max_long_side: int = DEFAULT_GROUNDING_IMAGE_LONG_SIDE,
) -> dict[str, Any]:
    width, height = _image_size(image)
    long_side = max(width, height)
    if max_long_side <= 0 or long_side <= max_long_side:
        return {
            "path": str(image),
            "widthPx": width,
            "heightPx": height,
            "sourceWidthPx": width,
            "sourceHeightPx": height,
            "resized": False,
            "maxLongSidePx": max_long_side,
            "scale": 1.0,
        }

    scale = max_long_side / long_side
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    out = output_dir / f"{image.stem}.model-input.png"
    try:
        from PIL import Image
    except ImportError as exc:
        raise CoretapError(
            "DEPENDENCY_MISSING",
            "Pillow is required to resize grounding screenshots",
            stage="grounding-preprocess",
            category="environment",
            details={"package": "pillow"},
        ) from exc
    with Image.open(image) as source:
        source.convert("RGB").resize((resized_width, resized_height), Image.Resampling.LANCZOS).save(out, optimize=True)
    return {
        "path": str(out),
        "widthPx": resized_width,
        "heightPx": resized_height,
        "sourceWidthPx": width,
        "sourceHeightPx": height,
        "resized": True,
        "maxLongSidePx": max_long_side,
        "scale": scale,
    }


def remap_grounding_to_source_frame(grounded: dict[str, Any], *, source_width: int, source_height: int) -> dict[str, Any]:
    point = grounded.get("point")
    if not isinstance(point, dict):
        return grounded
    normalized = point.get("normalized")
    if not isinstance(normalized, dict):
        return grounded
    try:
        x = float(normalized["x"])
        y = float(normalized["y"])
    except (KeyError, TypeError, ValueError):
        return grounded
    model_frame_px = point.get("framePx")
    if isinstance(model_frame_px, dict):
        point["modelInputFramePx"] = model_frame_px
    point["framePx"] = {"x": x * source_width, "y": y * source_height}
    grounded["frame"] = {"widthPx": source_width, "heightPx": source_height}
    return grounded


def model_status(profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    entry = ALL_GROUNDING_PROFILES.get(profile)
    if not entry:
        return {"ready": False, "profile": profile, "state": "unknown-profile"}
    if profile == INTERNAL_FIXTURE_PROFILE:
        ocr = tesseract_status()
        ready = bool(ocr["ready"] and ocr["defaultLangAvailable"])
        return {
            "ready": ready,
            "profile": profile,
            "state": "ready" if ready else "missing-ocr",
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
