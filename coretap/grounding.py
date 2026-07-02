from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from coretap.ocr import find_text, run_ocr, vision_ocr_status
from coretap.model_pack import (
    INTERNAL_FIXTURE_PROFILE,
    PUBLIC_MODEL_PROFILE,
    check_model,
    install_model,
    internal_profiles,
    public_profiles,
    run_grounding_model,
    warm_model as warm_public_model,
)
from coretap.runtime import CoretapError


GROUNDING_PROFILES = public_profiles()
ALL_GROUNDING_PROFILES = {**GROUNDING_PROFILES, **internal_profiles()}
DEFAULT_GROUNDING_IMAGE_LONG_SIDE = 1368
DEFAULT_REFINEMENT_CROP_RATIO = 0.38
REFINEMENT_CROP_MIN_SIDE_PX = 360
REFINEMENT_CROP_MAX_SIDE_PX = 900


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


def compute_refinement_crop(
    *,
    source_width: int,
    source_height: int,
    center_x: float,
    center_y: float,
    crop_ratio: float = DEFAULT_REFINEMENT_CROP_RATIO,
    min_side_px: int = REFINEMENT_CROP_MIN_SIDE_PX,
    max_side_px: int = REFINEMENT_CROP_MAX_SIDE_PX,
) -> dict[str, Any]:
    if source_width <= 0 or source_height <= 0:
        raise CoretapError("INVALID_IMAGE_SIZE", "source image dimensions must be positive", category="internal", stage="refine-crop")
    if crop_ratio <= 0:
        raise CoretapError("INVALID_ARGUMENT", "refine crop ratio must be > 0", category="usage", stage="refine-crop")
    long_side = max(source_width, source_height)
    side = round(long_side * crop_ratio)
    side = max(min_side_px, min(max_side_px, side))
    side = max(1, min(side, source_width, source_height))
    clamped_center_x = max(0.0, min(float(source_width), float(center_x)))
    clamped_center_y = max(0.0, min(float(source_height), float(center_y)))
    left = round(clamped_center_x - side / 2)
    top = round(clamped_center_y - side / 2)
    left = max(0, min(left, source_width - side))
    top = max(0, min(top, source_height - side))
    return {
        "x": left,
        "y": top,
        "width": side,
        "height": side,
        "sourceWidthPx": source_width,
        "sourceHeightPx": source_height,
        "center": {"x": clamped_center_x, "y": clamped_center_y},
        "cropRatio": crop_ratio,
        "minSidePx": min_side_px,
        "maxSidePx": max_side_px,
    }


def prepare_refinement_crop(
    image: Path,
    *,
    center: dict[str, Any],
    output_dir: Path,
    crop_ratio: float = DEFAULT_REFINEMENT_CROP_RATIO,
    crop_name: str = "step-grounding-refine-crop.png",
    region_name: str = "step-grounding-refine-region.png",
) -> dict[str, Any]:
    source_width, source_height = _image_size(image)
    try:
        center_x = float(center["x"])
        center_y = float(center["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoretapError("INVALID_POINT", "refinement crop center must contain numeric x and y", category="internal", stage="refine-crop") from exc
    crop = compute_refinement_crop(
        source_width=source_width,
        source_height=source_height,
        center_x=center_x,
        center_y=center_y,
        crop_ratio=crop_ratio,
    )
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise CoretapError(
            "DEPENDENCY_MISSING",
            "Pillow is required to crop refinement screenshots",
            category="environment",
            stage="refine-crop",
            details={"package": "pillow"},
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / crop_name
    region_path = output_dir / region_name
    left = int(crop["x"])
    top = int(crop["y"])
    right = left + int(crop["width"])
    bottom = top + int(crop["height"])
    with Image.open(image) as source:
        rgb = source.convert("RGB")
        rgb.crop((left, top, right, bottom)).save(crop_path, compress_level=1)
        region = rgb.copy()
        draw = ImageDraw.Draw(region)
        width = max(3, round(max(source_width, source_height) * 0.002))
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 0, 0), width=width)
        region.save(region_path, compress_level=1)
    return {
        **crop,
        "path": str(crop_path),
        "regionPath": str(region_path),
    }


def remap_crop_grounding_to_source_frame(grounded: dict[str, Any], *, crop: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(grounded)
    point = result.get("point")
    if not isinstance(point, dict):
        return result
    crop_width = int(crop["width"])
    crop_height = int(crop["height"])
    source_width = int(crop["sourceWidthPx"])
    source_height = int(crop["sourceHeightPx"])
    crop_frame_px = point.get("framePx")
    normalized = point.get("normalized")
    try:
        if isinstance(crop_frame_px, dict):
            crop_x = float(crop_frame_px["x"])
            crop_y = float(crop_frame_px["y"])
        elif isinstance(normalized, dict):
            crop_x = float(normalized["x"]) * crop_width
            crop_y = float(normalized["y"]) * crop_height
            crop_frame_px = {"x": crop_x, "y": crop_y}
        else:
            return result
    except (KeyError, TypeError, ValueError):
        return result
    source_x = float(crop["x"]) + crop_x
    source_y = float(crop["y"]) + crop_y
    if isinstance(normalized, dict):
        point["cropNormalized"] = dict(normalized)
    point["cropFramePx"] = {"x": crop_x, "y": crop_y}
    point["framePx"] = {"x": source_x, "y": source_y}
    point["normalized"] = {"x": source_x / source_width, "y": source_y / source_height}
    result["frame"] = {"widthPx": source_width, "heightPx": source_height}
    return result


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
        ocr = vision_ocr_status()
        ready = bool(ocr["ready"])
        return {
            "ready": ready,
            "profile": profile,
            "state": "ready" if ready else "missing-ocr",
            "implementation": "internal-vision-fixture-grounder",
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
    tokens, raw = run_ocr(image)
    match = find_text(tokens, target)
    if not match:
        return {
            "schema": "coretap.ground.result.v1",
            "status": "not_found",
            "target": {"description": target},
            "rawOcrTokenCount": len(tokens),
            "rawOcr": raw,
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
