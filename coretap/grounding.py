from __future__ import annotations

import shutil
import math
import re
from pathlib import Path
from typing import Any

from coretap.ocr import find_text, normalize_text, run_tesseract, run_vision_ocr, tesseract_status
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

_TARGET_TEXT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "app",
    "button",
    "click",
    "control",
    "field",
    "icon",
    "image",
    "ios",
    "label",
    "named",
    "open",
    "press",
    "screen",
    "tap",
    "target",
    "text",
    "the",
    "to",
    "ui",
    "with",
}
_TARGET_TEXT_CJK_STOP_WORDS = {"点击", "轻点", "按下", "打开", "图标", "按钮", "文本", "目标", "应用"}


def prepare_image_long_side(
    image: Path,
    *,
    output_path: Path,
    max_long_side: int = DEFAULT_GROUNDING_IMAGE_LONG_SIDE,
    stage: str = "image-preprocess",
) -> dict[str, Any]:
    width, height = _image_size(image)
    long_side = max(width, height)
    if max_long_side <= 0 or long_side <= max_long_side:
        if output_path != image:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, output_path)
        return {
            "path": str(output_path if output_path != image else image),
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
    try:
        from PIL import Image
    except ImportError as exc:
        raise CoretapError(
            "DEPENDENCY_MISSING",
            "Pillow is required to resize screenshots",
            stage=stage,
            category="environment",
            details={"package": "pillow"},
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized_output = output_path
    if output_path == image:
        resized_output = output_path.with_name(f"{output_path.stem}.resized.tmp{output_path.suffix}")
    with Image.open(image) as source:
        source.convert("RGB").resize((resized_width, resized_height), Image.Resampling.LANCZOS).save(resized_output, compress_level=1)
    if resized_output != output_path:
        resized_output.replace(output_path)
    return {
        "path": str(output_path),
        "widthPx": resized_width,
        "heightPx": resized_height,
        "sourceWidthPx": width,
        "sourceHeightPx": height,
        "resized": True,
        "maxLongSidePx": max_long_side,
        "scale": scale,
    }


def prepare_grounding_image(
    image: Path,
    *,
    output_dir: Path,
    max_long_side: int = DEFAULT_GROUNDING_IMAGE_LONG_SIDE,
) -> dict[str, Any]:
    return prepare_image_long_side(
        image,
        output_path=output_dir / f"{image.stem}.model-input.png",
        max_long_side=max_long_side,
        stage="grounding-preprocess",
    )


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


def grounding_safety_diagnostics(target: str, grounded: dict[str, Any]) -> dict[str, Any]:
    checks = [_coordinate_safety_check(grounded)]
    terms = target_text_terms(target)
    checks.append(
        {
            "id": "target-text-evidence",
            "status": "unchecked" if terms else "skipped",
            "terms": terms,
            "reason": "text-like target terms require OCR validation before real tap" if terms else "no specific target text term",
        }
    )
    return _safety_result(checks)


def assess_grounding_tap_safety(image: Path, target: str, grounded: dict[str, Any]) -> dict[str, Any]:
    checks = [_coordinate_safety_check(grounded)]
    terms = target_text_terms(target)
    if terms:
        checks.append(_target_text_safety_check(image, grounded, terms))
    else:
        checks.append(
            {
                "id": "target-text-evidence",
                "status": "skipped",
                "terms": [],
                "reason": "target does not include a specific text term",
            }
        )
    return _safety_result(checks)


def target_text_terms(target: str) -> list[str]:
    normalized = normalize_text(target)
    latin_terms = [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._+-]*", normalized)
        if len(token) >= 2 and token not in _TARGET_TEXT_STOP_WORDS
    ]
    cjk_terms = [term for term in re.findall(r"[\u3400-\u9fff]{1,}", target) if term not in _TARGET_TEXT_CJK_STOP_WORDS]
    terms: list[str] = []
    for term in [*latin_terms, *cjk_terms]:
        if term not in terms:
            terms.append(term)
    return terms[:4]


def _coordinate_safety_check(grounded: dict[str, Any]) -> dict[str, Any]:
    point = grounded.get("point") if isinstance(grounded, dict) else None
    normalized = point.get("normalized") if isinstance(point, dict) else None
    if not isinstance(normalized, dict):
        return {"id": "coordinate", "status": "unsafe", "reason": "grounding did not include normalized coordinates"}
    try:
        x = float(normalized["x"])
        y = float(normalized["y"])
    except (KeyError, TypeError, ValueError):
        return {"id": "coordinate", "status": "unsafe", "reason": "grounding coordinates were not numeric"}
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        return {
            "id": "coordinate",
            "status": "unsafe",
            "reason": "grounding coordinates were outside the screenshot",
            "normalized": {"x": x, "y": y},
        }
    edge_margin = 0.01
    if x < edge_margin or x > 1 - edge_margin or y < edge_margin or y > 1 - edge_margin:
        return {
            "id": "coordinate",
            "status": "low_confidence",
            "reason": "grounding point is very close to the screenshot edge",
            "normalized": {"x": x, "y": y},
        }
    return {"id": "coordinate", "status": "pass", "normalized": {"x": x, "y": y}}


def _target_text_safety_check(image: Path, grounded: dict[str, Any], terms: list[str]) -> dict[str, Any]:
    tokens: list[Any] = []
    engines: list[str] = []
    errors: list[dict[str, Any]] = []
    try:
        tesseract_tokens, _ = run_tesseract(image)
        tokens.extend(tesseract_tokens)
        engines.append("tesseract")
    except CoretapError as exc:
        errors.append({"engine": "tesseract", "code": exc.code, "message": str(exc), "details": exc.details})

    matches = _text_matches(tokens, terms)
    if not matches:
        try:
            vision_tokens, _ = run_vision_ocr(image)
            tokens.extend(vision_tokens)
            engines.append("vision")
            matches = _text_matches(tokens, terms)
        except CoretapError as exc:
            errors.append({"engine": "vision", "code": exc.code, "message": str(exc), "details": exc.details})

    if not engines:
        return {
            "id": "target-text-evidence",
            "status": "unsafe",
            "terms": terms,
            "reason": "OCR was unavailable, so text-like target visibility could not be verified",
            "errors": errors,
        }
    if not matches:
        return {
            "id": "target-text-evidence",
            "status": "unsafe",
            "terms": terms,
            "reason": "target text was not visible in the current screenshot",
            "engines": engines,
            "errors": errors,
            "tokenCount": len(tokens),
        }

    point = grounded.get("point", {}).get("framePx", {})
    try:
        point_x = float(point["x"])
        point_y = float(point["y"])
    except (KeyError, TypeError, ValueError):
        return {
            "id": "target-text-evidence",
            "status": "unsafe",
            "terms": terms,
            "reason": "grounding did not include frame pixel coordinates for text proximity validation",
            "matches": matches[:3],
        }

    frame = grounded.get("frame") if isinstance(grounded.get("frame"), dict) else {}
    width = float(frame.get("widthPx") or 0)
    height = float(frame.get("heightPx") or 0)
    max_distance = max(80.0, min(max(width, height) * 0.13, 360.0))
    nearest = min(matches, key=lambda match: _box_distance(point_x, point_y, match["matchedBoxPx"]))
    distance = _box_distance(point_x, point_y, nearest["matchedBoxPx"])
    status = "pass" if distance <= max_distance else "unsafe"
    reason = "target text is visible near the grounded point" if status == "pass" else "target text is visible but far from the grounded point"
    return {
        "id": "target-text-evidence",
        "status": status,
        "terms": terms,
        "reason": reason,
        "engines": engines,
        "tokenCount": len(tokens),
        "nearestDistancePx": distance,
        "maxDistancePx": max_distance,
        "nearestMatch": nearest,
        "matchCount": len(matches),
        "errors": errors,
    }


def _text_matches(tokens: list[Any], terms: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for term in terms:
        match = find_text(tokens, term)
        if not match:
            continue
        token_range = match.get("matchedTokenRange", {})
        key = (int(token_range.get("start", -1)), int(token_range.get("endExclusive", -1)))
        if key in seen:
            continue
        match["matchedTerm"] = term
        matches.append(match)
        seen.add(key)
    return matches


def _box_distance(x: float, y: float, box: dict[str, Any]) -> float:
    left = float(box["x"])
    top = float(box["y"])
    right = left + float(box["width"])
    bottom = top + float(box["height"])
    dx = 0.0 if left <= x <= right else min(abs(x - left), abs(x - right))
    dy = 0.0 if top <= y <= bottom else min(abs(y - top), abs(y - bottom))
    return math.hypot(dx, dy)


def _safety_result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [check.get("status") for check in checks]
    if "unsafe" in statuses:
        status = "unsafe"
    elif "low_confidence" in statuses or "unchecked" in statuses:
        status = "low_confidence"
    else:
        status = "safe"
    return {
        "schema": "coretap.grounding.safety.v1",
        "status": status,
        "safeToTap": status == "safe",
        "checks": checks,
    }


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


def model_install(profile: str = PUBLIC_MODEL_PROFILE, *, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    return install_model(profile, force=force, dry_run=dry_run)


def model_check(profile: str = PUBLIC_MODEL_PROFILE, *, deep: bool = False) -> dict[str, Any]:
    return check_model(profile, deep=deep)


def model_cache() -> dict[str, Any]:
    return cache_status()


def model_gc(*, dry_run: bool = False) -> dict[str, Any]:
    return gc_model(dry_run=dry_run)


def model_stop() -> dict[str, Any]:
    return stop_model()
