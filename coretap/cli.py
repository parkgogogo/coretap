from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from coretap import __version__
from coretap.backends import backend_for
from coretap.device_buttons import BUTTON_STATES, button_choices
from coretap.grounding import (
    DEFAULT_GROUNDING_IMAGE_LONG_SIDE,
    GROUNDING_PROFILES,
    assess_grounding_tap_safety,
    ground_target,
    grounding_safety_diagnostics,
    model_check,
    model_install,
    model_status,
    prepare_image_long_side,
    prepare_grounding_image,
    remap_grounding_to_source_frame,
    warm_model,
)
from coretap.model_pack import INTERNAL_FIXTURE_PROFILE, PUBLIC_MODEL_PROFILE
from coretap.ocr import (
    DEFAULT_OCR_LANG,
    OcrToken,
    find_exact_text_candidates,
    find_text,
    normalize_text,
    run_ocr,
    run_tesseract,
    run_vision_ocr,
    tesseract_status,
)
from coretap.runtime import (
    CoretapError,
    artifact_dir,
    ensure_state,
    png_size,
    response_error,
    response_ok,
    sha256_file,
    write_json,
)


EXIT_CODES = {
    "COMMAND_NOT_FOUND": 10,
    "COMMAND_TIMEOUT": 12,
    "INVALID_ARGUMENT": 2,
    "UNKNOWN_BACKEND": 2,
    "SIMCTL_LIST_FAILED": 10,
    "SIMCTL_BOOT_FAILED": 10,
    "SIMCTL_BOOTSTATUS_FAILED": 10,
    "SIMCTL_SCREENSHOT_FAILED": 21,
    "PYMOBILEDEVICE3_DISCOVER_FAILED": 20,
    "COREDEVICE_SCREENSHOT_FAILED": 21,
    "COREDEVICE_SCREENSHOT_EMPTY": 21,
    "COREDEVICE_DISPLAY_INFO_FAILED": 21,
    "COREDEVICE_DISPLAY_INFO_INVALID": 21,
    "COREDEVICE_TUNNELD_UNAVAILABLE": 10,
    "COREDEVICE_DRAG_FAILED": 32,
    "COREDEVICE_PRESS_FAILED": 32,
    "COREDEVICE_TAP_FAILED": 32,
    "COREDEVICE_TYPE_FAILED": 32,
    "COREDEVICE_KEY_FAILED": 32,
    "COREDEVICE_WORKER_FAILED": 32,
    "COREDEVICE_WORKER_TIMEOUT": 32,
    "SIMULATOR_DRAG_UNSUPPORTED": 32,
    "SIMULATOR_PRESS_UNSUPPORTED": 32,
    "SIMULATOR_TAP_UNSUPPORTED": 32,
    "SIMULATOR_TYPE_UNSUPPORTED": 32,
    "SIMULATOR_KEY_UNSUPPORTED": 32,
    "SIMULATOR_TAP_FAILED": 32,
    "SIMULATOR_DESCRIBE_FAILED": 32,
    "OCR_UNAVAILABLE": 10,
    "OCR_PROCESS_FAILED": 40,
    "VISION_OCR_UNAVAILABLE": 10,
    "VISION_OCR_FAILED": 40,
    "CAPABILITY_UNAVAILABLE": 10,
    "UNKNOWN_MODEL_PROFILE": 2,
    "MODEL_NOT_INSTALLED": 60,
    "MODEL_INCOMPATIBLE": 60,
    "MODEL_LOAD_FAILED": 60,
    "MODEL_RUN_FAILED": 60,
    "TARGET_ABSENT": 30,
    "TEXT_TARGET_NOT_FOUND": 30,
    "TEXT_TARGET_AMBIGUOUS": 30,
    "TEXT_INPUT_UNSUPPORTED": 30,
    "TEXT_INPUT_TARGET_UNKNOWN": 30,
    "TEXT_INPUT_VERIFICATION_FAILED": 30,
    "TEXT_ASSERTION_FAILED": 30,
    "GROUNDING_NOT_FOUND": 30,
    "GROUNDING_AMBIGUOUS": 30,
    "GROUNDING_SCHEMA_INVALID": 30,
    "GROUNDING_UNSAFE_TO_TAP": 30,
    "ACTION_SCHEMA_INVALID": 2,
    "ACTION_UNSUPPORTED": 2,
    "POSTCONDITION_FAILED": 30,
    "INVALID_POINT": 31,
    "FLOW_FAILED": 50,
    "DAEMON_UNAVAILABLE": 14,
    "DAEMON_START_FAILED": 14,
    "DAEMON_ALREADY_RUNNING": 14,
    "DAEMON_REQUEST_FAILED": 14,
}

DEFAULT_STEP_MODEL_INPUT_LONG_SIDE = 512
DEFAULT_TEXT_POST_TIMEOUT_MS = 3000


def emit(data: dict[str, Any], fmt: str) -> None:
    if fmt == "text":
        if data.get("ok"):
            print(json.dumps(data["result"], ensure_ascii=False, indent=2))
        else:
            err = data["error"]
            print(f"{err['code']}: {err['message']}", file=sys.stderr)
    else:
        print(json.dumps(data, ensure_ascii=False))


def point_to_hid(x: float, y: float, *, width: int, height: int, space: str, frame_known: bool = True) -> dict[str, Any]:
    if space == "hid":
        hx, hy = int(round(x)), int(round(y))
        if not (0 <= hx <= 65535 and 0 <= hy <= 65535):
            raise CoretapError("INVALID_POINT", "HID coordinates must be in [0,65535]", category="usage", stage="coordinate")
        normalized = {"x": hx / 65535, "y": hy / 65535}
        screenshot_px = {"x": normalized["x"] * width, "y": normalized["y"] * height} if frame_known else None
    elif space == "normalized":
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise CoretapError("INVALID_POINT", "Normalized coordinates must be in [0,1]", category="usage", stage="coordinate")
        hx, hy = int(round(x * 65535)), int(round(y * 65535))
        normalized = {"x": x, "y": y}
        screenshot_px = {"x": x * width, "y": y * height} if frame_known else None
    elif space == "px":
        if width <= 0 or height <= 0:
            raise CoretapError("INVALID_POINT", "Frame dimensions are required for pixel coordinates", category="usage", stage="coordinate")
        if not (0 <= x <= width and 0 <= y <= height):
            raise CoretapError("INVALID_POINT", "Pixel coordinates are outside the screenshot", category="usage", stage="coordinate")
        hx, hy = int(round((x / width) * 65535)), int(round((y / height) * 65535))
        normalized = {"x": x / width, "y": y / height}
        screenshot_px = {"x": x, "y": y}
    else:
        raise CoretapError("INVALID_POINT", f"Unknown coordinate space: {space}", category="usage", stage="coordinate")
    return {
        "input": {"space": space, "x": x, "y": y},
        "normalized": normalized,
        "screenshotPx": screenshot_px,
        "frame": {"known": frame_known, "widthPx": width if frame_known else None, "heightPx": height if frame_known else None},
        "hidU16": {"x": hx, "y": hy},
    }


def command_setup(args: argparse.Namespace) -> dict[str, Any]:
    roots = ensure_state()
    config_path = roots["state"] / "config.json"
    write_json(
        config_path,
        {
            "schema": "coretap.config.v1",
            "version": 1,
            "capabilities": {
                "grounding": {"profile": PUBLIC_MODEL_PROFILE},
                "ocr": {"profile": "builtin:vision-tesseract-chi-sim-eng@1", "lang": DEFAULT_OCR_LANG},
            },
            "storage": {name: str(path) for name, path in roots.items()},
        },
    )
    return {
        "version": __version__,
        "stateRoot": str(roots["state"]),
        "cacheRoot": str(roots["cache"]),
        "config": str(config_path),
        "profiles": list(GROUNDING_PROFILES),
    }


def command_status(args: argparse.Namespace) -> dict[str, Any]:
    roots = ensure_state()
    ocr = tesseract_status()
    model = model_status(args.profile)
    return {
        "version": __version__,
        "stateRoot": str(roots["state"]),
        "cacheRoot": str(roots["cache"]),
        "model": model,
        "ocr": ocr,
        "ready": {
            "grounding": bool(model.get("ready")),
            "textAssertions": bool(ocr.get("ready") and ocr.get("defaultLangAvailable")),
        },
    }


def command_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config_command != "check":
        raise CoretapError("UNKNOWN_CONFIG_COMMAND", args.config_command, category="usage", stage="config")
    roots = ensure_state()
    config_path = roots["state"] / "config.json"
    if not config_path.exists():
        command_setup(args)
    return {"valid": True, "config": str(config_path)}


def command_discover(args: argparse.Namespace) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    devices = backend.discover()
    return {
        "backend": args.backend,
        "devices": [
            {
                "udid": d.udid,
                "name": d.name,
                "backend": d.backend,
                "state": d.state,
                "runtime": d.runtime,
                "eligible": d.eligible,
                "details": d.details,
            }
            for d in devices
        ],
    }


def command_doctor(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append({"id": "state", "status": "pass", "details": command_setup(args)})
    model = model_status(args.profile)
    checks.append({"id": "grounding", "status": "pass" if model["ready"] else "fail", "details": model})
    ocr = tesseract_status()
    checks.append({"id": "ocr", "status": "pass" if ocr["ready"] and ocr["defaultLangAvailable"] else "warn", "details": ocr})
    try:
        devices = command_discover(args)["devices"]
        checks.append({"id": f"{args.backend}-discover", "status": "pass", "details": {"count": len(devices)}})
    except CoretapError as exc:
        checks.append({"id": f"{args.backend}-discover", "status": "fail", "details": exc.details, "message": str(exc)})
    ready = all(c["status"] != "fail" for c in checks)
    return {"ready": ready, "checks": checks}


def command_model(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_command == "status":
        return model_status(args.profile)
    if args.model_command == "check":
        return model_check(args.profile, deep=args.deep)
    if args.model_command == "warm":
        return warm_model(args.profile)
    if args.model_command == "install":
        return model_install(args.profile, force=args.force, dry_run=args.dry_run)
    raise CoretapError("UNKNOWN_MODEL_COMMAND", args.model_command, category="usage", stage="model")


def _frame_json(frame: Any) -> dict[str, Any]:
    return {
        "frameId": frame.frame_id,
        "path": str(frame.path),
        "widthPx": frame.width,
        "heightPx": frame.height,
        "backend": frame.backend,
        "device": frame.device,
        "sha256": sha256_file(frame.path),
        "capturedAt": _now_iso(),
    }


def _capture_to(args: argparse.Namespace, *, label: str, run_dir: Path, out: Path, write_frame: bool = True) -> Any:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    frame = backend.screenshot(args.device, out)
    if write_frame:
        write_json(run_dir / f"{label}.frame.json", _frame_json(frame))
    return frame


def capture(args: argparse.Namespace, *, label: str = "screenshot") -> tuple[Any, Path, Path]:
    run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    out = Path(args.out) if getattr(args, "out", None) else run_dir / f"{label}.png"
    frame = _capture_to(args, label=label, run_dir=run_dir, out=out)
    return frame, run_dir, out


def _preserve_source_image(source: Path, preserved: Path) -> None:
    if source == preserved:
        return
    preserved.parent.mkdir(parents=True, exist_ok=True)
    preserved.unlink(missing_ok=True)
    try:
        os.link(source, preserved)
    except OSError:
        shutil.copyfile(source, preserved)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _screenshot_into(args: argparse.Namespace, *, run_dir: Path, label: str, out: Path | None = None) -> dict[str, Any]:
    captured_at = _now_iso()
    output_path = out or (Path(args.out) if getattr(args, "out", None) else run_dir / f"{label}.png")
    if getattr(args, "full_size", False):
        frame = _capture_to(args, label=label, run_dir=run_dir, out=output_path)
        result = {
            "artifactDir": str(run_dir),
            "frame": {
                "frameId": frame.frame_id,
                "path": str(frame.path),
                "widthPx": frame.width,
                "heightPx": frame.height,
                "backend": frame.backend,
                "device": frame.device,
                "resized": False,
                "maxLongSidePx": None,
                "scale": 1.0,
                "sha256": sha256_file(frame.path),
                "capturedAt": captured_at,
            },
        }
        write_json(run_dir / f"{label}.result.json", result)
        return result

    frame = _capture_to(args, label=label, run_dir=run_dir, out=output_path, write_frame=False)
    max_long_side = getattr(args, "max_long_side", DEFAULT_GROUNDING_IMAGE_LONG_SIDE)
    source_image = output_path
    source_label = f"{label}.source"
    if max_long_side > 0 and max(frame.width, frame.height) > max_long_side:
        source_image = run_dir / f"{source_label}.png"
        _preserve_source_image(output_path, source_image)
    source_frame_json = {
        "frameId": f"frame_{source_label}" if source_image != output_path else frame.frame_id,
        "path": str(source_image),
        "widthPx": frame.width,
        "heightPx": frame.height,
        "backend": frame.backend,
        "device": frame.device,
        "sha256": sha256_file(source_image),
        "capturedAt": captured_at,
    }
    write_json(run_dir / f"{source_label}.frame.json", source_frame_json)
    preview = prepare_image_long_side(output_path, output_path=output_path, max_long_side=max_long_side)
    result = {
        "artifactDir": str(run_dir),
        "frame": {
            "frameId": f"frame_{label}",
            "path": preview["path"],
            "widthPx": preview["widthPx"],
            "heightPx": preview["heightPx"],
            "backend": frame.backend,
            "device": frame.device,
            "resized": preview["resized"],
            "maxLongSidePx": preview["maxLongSidePx"],
            "scale": preview["scale"],
            "sha256": sha256_file(Path(preview["path"])),
            "capturedAt": captured_at,
        },
        "sourceFrame": {
            "frameId": f"frame_{source_label}" if source_image != output_path else frame.frame_id,
            "path": str(source_image),
            "widthPx": frame.width,
            "heightPx": frame.height,
            "backend": frame.backend,
            "device": frame.device,
            "preserved": source_image != output_path,
            "sha256": sha256_file(source_image),
            "capturedAt": captured_at,
        },
    }
    write_json(run_dir / f"{label}.result.json", result)
    return {
        **result,
    }


def _ocr_token_json(token: Any, *, index: int, frame_width: int, frame_height: int) -> dict[str, Any]:
    center_x, center_y = token.center
    bbox = {"x": token.left, "y": token.top, "width": token.width, "height": token.height}
    return {
        "index": index,
        "text": token.text,
        "normalizedText": normalize_text(token.text),
        "confidence": token.confidence,
        "engine": token.engine,
        "bboxPx": bbox,
        "centerPx": {"x": center_x, "y": center_y},
        "normalized": {
            "x": center_x / frame_width if frame_width else None,
            "y": center_y / frame_height if frame_height else None,
        },
        "bboxNormalized": {
            "x": token.left / frame_width if frame_width else None,
            "y": token.top / frame_height if frame_height else None,
            "width": token.width / frame_width if frame_width else None,
            "height": token.height / frame_height if frame_height else None,
        },
    }


def _ocr_error_details(exc: CoretapError, *, engine: str) -> dict[str, Any]:
    return {"engine": engine, "code": exc.code, "message": str(exc), "details": exc.details}


def _raise_ocr_unavailable(image: Path, raw: dict[str, Any]) -> None:
    errors = raw.get("errors") or []
    first = errors[0] if errors else {}
    raise CoretapError(
        first.get("code") or "OCR_UNAVAILABLE",
        first.get("message") or "No OCR engine is available",
        stage="ocr",
        category="environment",
        details={"image": str(image), "errors": errors},
    )


def _run_observe_ocr(image: Path, args: argparse.Namespace) -> tuple[list[Any], dict[str, Any], str]:
    if args.ocr_engine == "all":
        tokens, raw = run_ocr(image, lang=args.lang, psm=args.psm)
        return tokens, raw, "all"

    raw: dict[str, Any] = {"engines": [], "errors": []}
    if args.ocr_engine in ("auto", "vision"):
        try:
            tokens, vision_stdout = run_vision_ocr(image)
            raw["engines"].append("vision")
            raw["visionJson"] = vision_stdout
            return tokens, raw, "vision"
        except CoretapError as exc:
            raw["errors"].append(_ocr_error_details(exc, engine="vision"))
            if args.ocr_engine == "vision":
                _raise_ocr_unavailable(image, raw)

    try:
        tokens, tsv = run_tesseract(image, lang=args.lang, psm=args.psm)
        raw["engines"].append("tesseract")
        raw["tesseractTsv"] = tsv
        return tokens, raw, "tesseract"
    except CoretapError as exc:
        raw["errors"].append(_ocr_error_details(exc, engine="tesseract"))

    _raise_ocr_unavailable(image, raw)


def _observe_into(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    label: str,
    no_ocr: bool | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    ns = argparse.Namespace(**vars(args))
    ns.label = label
    ns.out = str(out) if out else None
    screenshot = _screenshot_into(ns, run_dir=run_dir, label=label, out=out)
    frame = screenshot["frame"]
    result: dict[str, Any] = {
        "schema": "coretap.observe.result.v1",
        "artifactDir": screenshot["artifactDir"],
        "frame": frame,
        "sourceFrame": screenshot.get("sourceFrame"),
    }
    should_skip_ocr = bool(args.no_ocr if no_ocr is None else no_ocr)
    if should_skip_ocr:
        result["ocr"] = {"enabled": False}
        write_json(run_dir / f"{label}.observe.result.json", result)
        return result

    image = Path(frame["path"])
    tokens, raw, selected_engine = _run_observe_ocr(image, args)
    filtered = [token for token in tokens if token.confidence >= args.min_confidence]
    token_json = [
        _ocr_token_json(token, index=index, frame_width=frame["widthPx"], frame_height=frame["heightPx"])
        for index, token in enumerate(filtered)
    ]
    _write_ocr_artifacts(run_dir, label, raw)
    result["ocr"] = {
        "schema": "coretap.ocr.page.v1",
        "enabled": True,
        "lang": args.lang,
        "psm": args.psm,
        "engineMode": args.ocr_engine,
        "selectedEngine": selected_engine,
        "minConfidence": args.min_confidence,
        "tokenCount": len(token_json),
        "rawTokenCount": len(tokens),
        "plainText": "\n".join(token["text"] for token in token_json),
        **_ocr_summary(raw),
        "tokens": token_json,
    }
    write_json(run_dir / f"{label}.observe.result.json", result)
    return result


def command_observe(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    result = _observe_into(args, run_dir=run_dir, label=args.label)
    write_json(run_dir / "observe.result.json", result)
    return result


def _parse_xy_pair(raw: str, *, option: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise CoretapError("INVALID_ARGUMENT", f"{option} must be formatted as x,y", category="usage", stage="coordinate")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise CoretapError("INVALID_ARGUMENT", f"{option} must contain numeric coordinates", category="usage", stage="coordinate") from exc


def _parse_normalized_pair(raw: str, *, option: str, stage: str) -> dict[str, float]:
    x, y = _parse_xy_pair(raw, option=option)
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise CoretapError("INVALID_POINT", f"{option} coordinates must be normalized values in [0,1]", category="usage", stage=stage)
    return {"x": x, "y": y}


def _point_frame_dimensions(args: argparse.Namespace) -> tuple[int, int, bool]:
    if getattr(args, "frame", None):
        width, height = png_size(Path(args.frame))
        return width, height, True
    if args.width is not None and args.height is not None:
        return args.width, args.height, True
    return 1, 1, False


def command_drag(args: argparse.Namespace) -> dict[str, Any]:
    width, height, frame_known = _point_frame_dimensions(args)
    from_x, from_y = _parse_xy_pair(args.from_point, option="--from")
    to_x, to_y = _parse_xy_pair(args.to_point, option="--to")
    from_point = point_to_hid(from_x, from_y, width=width, height=height, space=args.space, frame_known=frame_known)
    to_point = point_to_hid(to_x, to_y, width=width, height=height, space=args.space, frame_known=frame_known)
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    drag = backend.drag_normalized(
        args.device,
        from_point["normalized"]["x"],
        from_point["normalized"]["y"],
        to_point["normalized"]["x"],
        to_point["normalized"]["y"],
        dry_run=args.dry_run,
        start_hid_u16=from_point["hidU16"],
        end_hid_u16=to_point["hidU16"],
        steps=args.steps,
        duration_ms=args.duration_ms,
    )
    return {"from": from_point, "to": to_point, "drag": drag}


def command_scroll(args: argparse.Namespace) -> dict[str, Any]:
    if not (0 < args.distance <= 0.9):
        raise CoretapError("INVALID_ARGUMENT", "scroll --distance must be in (0, 0.9]", category="usage", stage="scroll")

    half = args.distance / 2
    edge_margin = 0.05

    def clamp(value: float) -> float:
        return max(edge_margin, min(1 - edge_margin, value))

    if args.direction == "down":
        from_x, from_y = args.anchor_x, clamp(args.anchor_y + half)
        to_x, to_y = args.anchor_x, clamp(args.anchor_y - half)
    elif args.direction == "up":
        from_x, from_y = args.anchor_x, clamp(args.anchor_y - half)
        to_x, to_y = args.anchor_x, clamp(args.anchor_y + half)
    else:
        raise CoretapError("INVALID_ARGUMENT", f"Unsupported scroll direction: {args.direction}", category="usage", stage="scroll")

    ns = argparse.Namespace(**vars(args))
    ns.space = "normalized"
    ns.from_point = f"{from_x},{from_y}"
    ns.to_point = f"{to_x},{to_y}"
    ns.frame = None
    ns.width = 1
    ns.height = 1
    result = command_drag(ns)
    return {
        "direction": args.direction,
        "distance": args.distance,
        "anchor": {"x": args.anchor_x, "y": args.anchor_y},
        **result,
    }


def _grounding_error_code(status: str) -> str:
    if status == "not_found":
        return "GROUNDING_NOT_FOUND"
    if status == "ambiguous":
        return "GROUNDING_AMBIGUOUS"
    return "GROUNDING_SCHEMA_INVALID"


def _run_ocr_progressive(
    image: Path,
    *,
    lang: str,
    psm: int,
    is_match: Any,
) -> tuple[list[Any], dict[str, Any]]:
    tokens: list[Any] = []
    raw: dict[str, Any] = {"engines": [], "errors": []}
    try:
        tesseract_tokens, tsv = run_tesseract(image, lang=lang, psm=psm)
        tokens.extend(tesseract_tokens)
        raw["engines"].append("tesseract")
        raw["tesseractTsv"] = tsv
        if is_match(tokens):
            return tokens, raw
    except CoretapError as exc:
        raw["errors"].append({"engine": "tesseract", "code": exc.code, "message": str(exc), "details": exc.details})

    try:
        vision_tokens, vision_stdout = run_vision_ocr(image)
        tokens.extend(vision_tokens)
        raw["engines"].append("vision")
        raw["visionJson"] = vision_stdout
    except CoretapError as exc:
        raw["errors"].append({"engine": "vision", "code": exc.code, "message": str(exc), "details": exc.details})

    if not raw["engines"]:
        errors = raw.get("errors") or []
        first = errors[0] if errors else {}
        raise CoretapError(
            first.get("code") or "OCR_UNAVAILABLE",
            first.get("message") or "No OCR engine is available",
            stage="ocr",
            category="environment",
            details={"image": str(image), "errors": errors},
        )
    return tokens, raw


def command_press(args: argparse.Namespace) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    pressed = backend.press_button(
        args.device,
        args.button,
        state=args.state,
        hold_ms=args.hold_ms,
        dry_run=args.dry_run,
    )
    return pressed


def _type_text_query(args: argparse.Namespace) -> str:
    text = getattr(args, "text", None) or getattr(args, "text_query", None)
    if text is None:
        raise CoretapError("INVALID_ARGUMENT", "type requires text", category="usage", stage="type")
    return str(text)


def command_type(args: argparse.Namespace) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    text = _type_text_query(args)
    if args.verify_timeout_ms < 0:
        raise CoretapError("INVALID_ARGUMENT", "type --verify-timeout-ms must be >= 0", category="usage", stage="type")
    if isinstance(args.paste_at, dict):
        paste_at = args.paste_at
    else:
        paste_at = _parse_normalized_pair(args.paste_at, option="--paste-at", stage="type") if args.paste_at else None
    result = backend.type_text(
        args.device,
        text,
        char_delay_ms=args.char_delay_ms,
        inter_delay_ms=args.inter_delay_ms,
        paste_at=paste_at,
        paste_hold_ms=args.paste_hold_ms,
        clear_existing=args.replace,
        dry_run=args.dry_run,
    )
    if args.dry_run or args.no_verify or not text:
        return result
    return _verify_type_result(args, text, result)


def command_key(args: argparse.Namespace) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    return backend.keyboard_key(
        args.device,
        args.key,
        count=args.count,
        inter_delay_ms=args.inter_delay_ms,
        dry_run=args.dry_run,
    )


def command_clear(args: argparse.Namespace) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    return backend.clear_text(
        args.device,
        count=args.count,
        inter_delay_ms=args.inter_delay_ms,
        dry_run=args.dry_run,
    )


def _verify_type_result(args: argparse.Namespace, text: str, result: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + (args.verify_timeout_ms / 1000)
    attempts = 0
    last: dict[str, Any] | None = None
    while True:
        attempts += 1
        capture_args = argparse.Namespace(**vars(args))
        capture_args.out = None
        frame, run_dir, image = capture(capture_args, label="type-verify")
        tokens, raw = _run_ocr_progressive(
            image,
            lang=DEFAULT_OCR_LANG,
            psm=11,
            is_match=lambda current: bool(find_exact_text_candidates(current, text, min_confidence=25.0)),
        )
        _write_ocr_artifacts(run_dir, "type-verify", raw)
        candidates = find_exact_text_candidates(tokens, text, min_confidence=25.0)
        last = {
            "artifactDir": str(run_dir),
            "frame": {"path": str(image), "widthPx": frame.width, "heightPx": frame.height},
            "attempts": attempts,
            "tokenCount": len(tokens),
            "candidateCount": len(candidates),
            "ocr": _ocr_summary(raw),
        }
        if candidates:
            verification = {**last, "match": candidates[0]}
            return {
                **result,
                "confirmationStatus": "verified_text",
                "verification": verification,
            }
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    raise CoretapError(
        "TEXT_INPUT_VERIFICATION_FAILED",
        f"Typed text was not visible after input: {text}",
        stage="type",
        category="assertion",
        details={"text": text, "input": result, "lastVerification": last},
    )


def command_assert_text(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    if args.image:
        image = Path(args.image)
        frame_info: dict[str, Any] | None = None
    else:
        image = run_dir / "assert-000.png"
        frame_info = None
    deadline = time.monotonic() + (args.timeout_ms / 1000)
    attempts = 0
    last: dict[str, Any] | None = None
    while True:
        attempts += 1
        if not args.image:
            live_image = run_dir / f"assert-{attempts:03d}.png"
            frame = _capture_to(args, label=f"assert-{attempts:03d}", run_dir=run_dir, out=live_image)
            image = live_image
            frame_info = {
                "path": str(frame.path),
                "widthPx": frame.width,
                "heightPx": frame.height,
                "backend": frame.backend,
                "device": frame.device,
                "sha256": sha256_file(frame.path),
                "capturedAt": _now_iso(),
            }
        lang = getattr(args, "lang", DEFAULT_OCR_LANG)
        psm = int(getattr(args, "psm", 11))
        tokens, raw = _run_ocr_progressive(
            image,
            lang=lang,
            psm=psm,
            is_match=lambda current: bool(find_text(current, args.text, case_sensitive=args.case_sensitive)),
        )
        _write_ocr_artifacts(run_dir, f"assert-{attempts:03d}", raw)
        match = find_text(tokens, args.text, case_sensitive=args.case_sensitive)
        last = {
            "attempts": attempts,
            "image": str(image),
            "frame": frame_info,
            "tokenCount": len(tokens),
            "ocr": {"lang": lang, "psm": psm, **_ocr_summary(raw)},
            "match": match,
        }
        if match:
            result = {
                "artifactDir": str(run_dir),
                "expected": args.text,
                "matched": True,
                "frame": frame_info,
                "ocr": {"lang": lang, "psm": psm, "tokenCount": len(tokens), **_ocr_summary(raw)},
                **match,
                "attempts": attempts,
            }
            write_json(run_dir / "assert-text.result.json", result)
            return result
        if time.monotonic() >= deadline:
            break
        time.sleep(args.poll_interval_ms / 1000)
    raise CoretapError(
        "TEXT_ASSERTION_FAILED",
        f"Text was not visible before timeout: {args.text}",
        stage="assert-text",
        category="test",
        details=last or {},
    )


_STEP_ACTION_TYPES = {
    "tap",
    "openApp",
    "typeText",
    "key",
    "clear",
    "press",
    "scroll",
    "wait",
}


def _load_step_action(args: argparse.Namespace) -> dict[str, Any]:
    has_inline = bool(getattr(args, "action", None))
    has_file = bool(getattr(args, "action_file", None))
    if has_inline == has_file:
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            "step requires exactly one of --action or --action-file",
            category="usage",
            stage="step-action",
        )
    if has_file:
        source = Path(args.action_file)
        try:
            raw = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise CoretapError(
                "ACTION_SCHEMA_INVALID",
                f"Could not read step action file: {source}",
                category="usage",
                stage="step-action",
                details={"path": str(source), "error": str(exc)},
            ) from exc
    else:
        raw_arg = str(args.action)
        if raw_arg.startswith("@"):
            source = Path(raw_arg[1:])
            try:
                raw = source.read_text(encoding="utf-8")
            except OSError as exc:
                raise CoretapError(
                    "ACTION_SCHEMA_INVALID",
                    f"Could not read step action file: {source}",
                    category="usage",
                    stage="step-action",
                    details={"path": str(source), "error": str(exc)},
                ) from exc
        else:
            raw = raw_arg
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            f"step action is not valid JSON: {exc}",
            category="usage",
            stage="step-action",
        ) from exc
    if not isinstance(action, dict):
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            "step action must be a JSON object, not an array or scalar",
            category="usage",
            stage="step-action",
            details={"receivedType": type(action).__name__},
        )
    schema = action.get("schema")
    if schema != "coretap.action.v2":
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            f"Unsupported action schema: {schema}",
            category="usage",
            stage="step-action",
            details={"schema": schema},
        )
    return action


def _require_str(payload: dict[str, Any], key: str, *, stage: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action is missing {key}", category="usage", stage=stage)
    return value


def _number(value: Any, *, key: str, stage: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must be numeric", category="usage", stage=stage) from exc


def _integer(value: Any, *, key: str, stage: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must be an integer", category="usage", stage=stage) from exc


def _point_pair_from_action(value: Any, *, key: str, stage: str) -> str:
    if isinstance(value, str):
        _parse_xy_pair(value, option=key)
        return value
    if isinstance(value, dict):
        x = _number(value.get("x"), key=f"{key}.x", stage=stage)
        y = _number(value.get("y"), key=f"{key}.y", stage=stage)
        return f"{x},{y}"
    raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must be x,y or an object", category="usage", stage=stage)


def _normalize_step_action(raw: dict[str, Any]) -> dict[str, Any]:
    stage = "step-action"
    action_type = str(raw.get("type") or raw.get("action") or "").strip()
    if action_type not in _STEP_ACTION_TYPES:
        raise CoretapError(
            "ACTION_UNSUPPORTED",
            f"Unsupported step action type: {action_type or '<missing>'}",
            category="usage",
            stage=stage,
            details={"supported": sorted(_STEP_ACTION_TYPES), "action": raw},
        )

    action: dict[str, Any] = {"schema": "coretap.action.v2", "type": action_type}
    if "postconditions" in raw:
        action["postconditions"] = raw["postconditions"]

    if action_type == "tap":
        action["target"] = _require_str(raw, "target", stage=stage)
    elif action_type == "openApp":
        action["name"] = _require_str(raw, "name", stage=stage)
        action["searchTarget"] = str(raw.get("searchTarget") or "the Search button at the bottom center of the iOS home screen")
        action["resultTarget"] = str(raw.get("resultTarget") or f"the large {action['name']} app icon on the left side of the Best Search Result card in Spotlight search results")
    elif action_type == "typeText":
        action["text"] = _require_str(raw, "text", stage=stage)
        action["charDelayMs"] = _integer(raw.get("charDelayMs", 40), key="charDelayMs", stage=stage)
        action["interDelayMs"] = _integer(raw.get("interDelayMs", 20), key="interDelayMs", stage=stage)
        paste_at = raw.get("pasteAt")
        action["pasteAt"] = _point_pair_from_action(paste_at, key="pasteAt", stage=stage) if paste_at is not None else None
        action["pasteHoldMs"] = _integer(raw.get("pasteHoldMs", 1600), key="pasteHoldMs", stage=stage)
        action["verifyTimeoutMs"] = _integer(raw.get("verifyTimeoutMs", 0), key="verifyTimeoutMs", stage=stage)
        action["noVerify"] = bool(raw.get("noVerify", True))
        action["replace"] = bool(raw.get("replace", False))
    elif action_type == "key":
        action["key"] = _require_str(raw, "key", stage=stage)
        action["count"] = _integer(raw.get("count", 1), key="count", stage=stage)
        action["interDelayMs"] = _integer(raw.get("interDelayMs", 20), key="interDelayMs", stage=stage)
    elif action_type == "clear":
        action["count"] = _integer(raw.get("count", 80), key="count", stage=stage)
        action["interDelayMs"] = _integer(raw.get("interDelayMs", 2), key="interDelayMs", stage=stage)
    elif action_type == "press":
        action["button"] = _require_str(raw, "button", stage=stage)
        action["state"] = str(raw.get("state") or "press")
        action["holdMs"] = raw.get("holdMs")
    elif action_type == "scroll":
        direction = _require_str(raw, "direction", stage=stage)
        if direction not in {"down", "up"}:
            raise CoretapError("ACTION_SCHEMA_INVALID", "scroll direction must be down or up", category="usage", stage=stage)
        anchor = raw.get("anchor") if isinstance(raw.get("anchor"), dict) else {}
        action["direction"] = direction
        action["distance"] = _number(raw.get("distance", 0.5), key="distance", stage=stage)
        action["anchorX"] = _number(raw.get("anchorX", anchor.get("x", 0.5)), key="anchorX", stage=stage)
        action["anchorY"] = _number(raw.get("anchorY", anchor.get("y", 0.5)), key="anchorY", stage=stage)
        action["steps"] = _integer(raw.get("steps", 30), key="steps", stage=stage)
        action["durationMs"] = _integer(raw.get("durationMs", 600), key="durationMs", stage=stage)
    elif action_type == "wait":
        action["ms"] = _integer(raw.get("ms", 700), key="ms", stage=stage)
    return action


def _normalize_postcondition(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise CoretapError("ACTION_SCHEMA_INVALID", "postcondition must be a JSON object", category="usage", stage="step-postcondition")
    kind = str(raw.get("type") or raw.get("kind") or "").strip()
    if kind in {"expectText", "text"}:
        kind = "textVisible"
    elif kind in {"expectNoText", "noText"}:
        kind = "textAbsent"
    if kind not in {"textVisible", "textAbsent", "screenChanged"}:
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            f"Unsupported postcondition type: {kind or '<missing>'}",
            category="usage",
            stage="step-postcondition",
        )
    item = {"type": kind}
    if kind in {"textVisible", "textAbsent"}:
        item["text"] = _require_str(raw, "text", stage="step-postcondition")
        item["caseSensitive"] = bool(raw.get("caseSensitive", False))
        item["minConfidence"] = _number(raw.get("minConfidence", 0.0), key="minConfidence", stage="step-postcondition")
        match_mode = str(raw.get("matchMode") or raw.get("mode") or "contains").strip()
        if match_mode not in {"contains", "exact"}:
            raise CoretapError(
                "ACTION_SCHEMA_INVALID",
                "text postcondition matchMode must be contains or exact",
                category="usage",
                stage="step-postcondition",
            )
        item["matchMode"] = match_mode
    return item


def _step_postconditions(args: argparse.Namespace, action: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = action.get("postconditions") or []
    if not isinstance(raw_items, list):
        raise CoretapError("ACTION_SCHEMA_INVALID", "postconditions must be an array", category="usage", stage="step-postcondition")
    items = [_normalize_postcondition(item) for item in raw_items]
    for text in getattr(args, "expect_text", []) or []:
        items.append({"type": "textVisible", "text": text, "caseSensitive": False, "minConfidence": 0.0, "matchMode": "exact"})
    for text in getattr(args, "expect_no_text", []) or []:
        items.append({"type": "textAbsent", "text": text, "caseSensitive": False, "minConfidence": 0.0, "matchMode": "exact"})
    if getattr(args, "expect_change", False):
        items.append({"type": "screenChanged"})
    if getattr(args, "no_ocr", False) and any(item["type"] in {"textVisible", "textAbsent"} for item in items):
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            "text postconditions require OCR; remove --no-ocr",
            category="usage",
            stage="step-postcondition",
        )
    return items


def _observation_tokens(observation: dict[str, Any]) -> list[OcrToken]:
    ocr = observation.get("ocr") if isinstance(observation, dict) else None
    if not isinstance(ocr, dict) or not ocr.get("enabled"):
        return []
    tokens: list[OcrToken] = []
    for item in ocr.get("tokens") or []:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bboxPx") or {}
        try:
            tokens.append(
                OcrToken(
                    text=str(item.get("text") or ""),
                    confidence=float(item.get("confidence") or 0),
                    left=int(float(bbox.get("x") or 0)),
                    top=int(float(bbox.get("y") or 0)),
                    width=int(float(bbox.get("width") or 0)),
                    height=int(float(bbox.get("height") or 0)),
                    engine=str(item.get("engine") or "ocr"),
                )
            )
        except (TypeError, ValueError):
            continue
    return tokens


def _find_observation_text(
    observation: dict[str, Any],
    text: str,
    *,
    case_sensitive: bool,
    min_confidence: float,
    match_mode: str = "contains",
) -> dict[str, Any] | None:
    tokens = [token for token in _observation_tokens(observation) if token.confidence >= min_confidence]
    if match_mode == "exact":
        for candidate in find_exact_text_candidates(tokens, text, case_sensitive=case_sensitive, min_confidence=min_confidence):
            if candidate.get("matchedKind") == "exact":
                return candidate
        return None
    return find_text(tokens, text, case_sensitive=case_sensitive)


def _token_center_normalized(token: dict[str, Any]) -> dict[str, float] | None:
    center = token.get("normalized") if isinstance(token, dict) else None
    if not isinstance(center, dict):
        return None
    try:
        x = float(center["x"])
        y = float(center["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return {"x": x, "y": y}


def _token_bbox_normalized(token: dict[str, Any], observation: dict[str, Any]) -> dict[str, float] | None:
    bbox = token.get("bboxPx") if isinstance(token, dict) else None
    frame = observation.get("frame") if isinstance(observation, dict) else None
    if not isinstance(bbox, dict) or not isinstance(frame, dict):
        return None
    try:
        frame_width = float(frame["widthPx"])
        frame_height = float(frame["heightPx"])
        left = float(bbox["x"])
        top = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if frame_width <= 0 or frame_height <= 0 or width <= 0 or height <= 0:
        return None
    return {
        "x": min(1.0, max(0.0, left / frame_width)),
        "y": min(1.0, max(0.0, top / frame_height)),
        "width": min(1.0, max(0.0, width / frame_width)),
        "height": min(1.0, max(0.0, height / frame_height)),
    }


def _visible_paste_menu_point(token: dict[str, Any], observation: dict[str, Any]) -> dict[str, float] | None:
    center = _token_center_normalized(token)
    if center is None:
        return None
    text = str(token.get("text") or "")
    normalized = normalize_text(text)
    bbox = _token_bbox_normalized(token, observation)
    if bbox is None:
        return center

    compact = "".join(ch for ch in text if not ch.isspace())
    compact_casefold = compact.casefold()
    paste_index = compact.find("粘贴")
    paste_width = 2
    if paste_index < 0:
        paste_index = compact.find("粘貼")
    if paste_index < 0:
        paste_index = compact_casefold.find("paste")
        paste_width = 5
    if paste_index < 0 and "填充" in compact:
        x = bbox["x"] + bbox["width"] * 0.22
        return {"x": min(0.98, max(0.02, x)), "y": center["y"]}
    if paste_index < 0:
        return center

    has_neighbor = (
        "自动填充" in compact
        or "autofill" in normalized
        or len(compact) > paste_width + 1
    )
    if not has_neighbor:
        return center

    glyph_count = max(len(compact), paste_width)
    ratio = (paste_index + (paste_width / 2)) / glyph_count
    x = bbox["x"] + bbox["width"] * min(0.92, max(0.08, ratio))
    return {"x": min(0.98, max(0.02, x)), "y": center["y"]}


def _observation_ocr_tokens_json(observation: dict[str, Any]) -> list[dict[str, Any]]:
    ocr = observation.get("ocr") if isinstance(observation, dict) else None
    if not isinstance(ocr, dict) or not ocr.get("enabled"):
        return []
    return [item for item in ocr.get("tokens") or [] if isinstance(item, dict)]


def _text_entry_context(observation: dict[str, Any]) -> dict[str, Any]:
    tokens = _observation_ocr_tokens_json(observation)
    anchor_candidates: list[dict[str, Any]] = []
    keyboard_marker_count = 0
    has_edit_menu = False
    edit_menu_anchor: dict[str, Any] | None = None
    app_store_search_context = _looks_like_app_store_search_tokens(tokens)

    for token in tokens:
        text = str(token.get("text") or "")
        normalized = normalize_text(text)
        center = _token_center_normalized(token)
        if center is None:
            continue
        if "粘贴" in text or "粘貼" in text or "填充" in text or "paste" in normalized or "autofill" in normalized:
            has_edit_menu = True
            if edit_menu_anchor is None:
                menu_point = _visible_paste_menu_point(token, observation) or center
                edit_menu_anchor = {
                    "source": "edit-menu",
                    "point": menu_point,
                    "pointKind": "visible-paste-menu",
                    "token": token,
                }
        if _looks_like_top_search_content_token(token, observation, normalized) and (
            _looks_like_search_query_prefix(normalized) or app_store_search_context
        ):
            anchor_candidates.append(
                {
                    "priority": 0,
                    "source": "active-search-field",
                    "point": _search_field_anchor_point(token, observation) or center,
                    "pointKind": "inferred-active-search-field",
                    "token": token,
                }
            )
        elif ("游戏" in text and "故事" in text) or ("game" in normalized and "app" in normalized):
            anchor_candidates.append({"priority": 0, "source": "search-placeholder", "point": center, "token": token})
        elif _looks_like_search_field_token(text, normalized) and _search_field_token_y_is_plausible(token, normalized):
            anchor_candidates.append(
                {
                    "priority": 1,
                    "source": "search-field",
                    "point": _search_field_anchor_point(token, observation) or center,
                    "pointKind": "inferred-search-field-center",
                    "token": token,
                }
            )
        elif (normalized in {"空格", "space", "123", "换行", "return"} or (len(normalized) == 1 and normalized.isascii() and normalized.isalnum())) and center["y"] >= 0.6:
            keyboard_marker_count += 1

    anchor = None
    if edit_menu_anchor is not None:
        anchor = edit_menu_anchor
    elif anchor_candidates:
        anchor = sorted(anchor_candidates, key=lambda item: (item["priority"], item["point"]["y"]))[0]
    elif _looks_like_spotlight_suggestions(tokens):
        anchor = {
            "priority": 2,
            "source": "spotlight-bottom-search",
            "point": {"x": 0.5, "y": 0.925},
            "pointKind": "inferred-spotlight-bottom-search",
        }

    ready = bool(anchor or has_edit_menu or keyboard_marker_count >= 3)
    return {
        "schema": "coretap.text-entry-context.v1",
        "ready": ready,
        "reason": "text input context is visible" if ready else "no focused text input context was visible before typing",
        "anchor": anchor,
        "hasEditMenu": has_edit_menu,
        "keyboardMarkerCount": keyboard_marker_count,
        "tokenCount": len(tokens),
    }


def _source_ocr_text_entry_context(args: argparse.Namespace, before: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    source_frame = before.get("sourceFrame") if isinstance(before, dict) else None
    preview_frame = before.get("frame") if isinstance(before, dict) else None
    if not isinstance(source_frame, dict) or not source_frame.get("path"):
        return None
    if isinstance(preview_frame, dict) and source_frame.get("path") == preview_frame.get("path"):
        return None

    image = Path(str(source_frame["path"]))
    tokens, raw, selected_engine = _run_observe_ocr(image, args)
    filtered = [token for token in tokens if token.confidence >= args.min_confidence]
    token_json = [
        _ocr_token_json(token, index=index, frame_width=source_frame["widthPx"], frame_height=source_frame["heightPx"])
        for index, token in enumerate(filtered)
    ]
    label = "step-before-source-ocr"
    _write_ocr_artifacts(run_dir, label, raw)
    observation = {
        "schema": "coretap.observe.result.v1",
        "artifactDir": before.get("artifactDir"),
        "frame": source_frame,
        "sourceFrame": source_frame,
        "ocr": {
            "schema": "coretap.ocr.page.v1",
            "enabled": True,
            "lang": args.lang,
            "psm": args.psm,
            "engineMode": args.ocr_engine,
            "selectedEngine": selected_engine,
            "minConfidence": args.min_confidence,
            "tokenCount": len(token_json),
            "rawTokenCount": len(tokens),
            "plainText": "\n".join(token["text"] for token in token_json),
            **_ocr_summary(raw),
            "tokens": token_json,
        },
    }
    write_json(run_dir / f"{label}.observe.result.json", observation)
    context = _text_entry_context(observation)
    context["source"] = "source-ocr"
    return context


def _looks_like_spotlight_suggestions(tokens: list[dict[str, Any]]) -> bool:
    normalized_items = [normalize_text(str(token.get("text") or "")).replace(" ", "") for token in tokens]
    joined = " ".join(normalized_items)
    has_siri_suggestions = any("siri建议" in item or ("siri" in item and "建议" in item) for item in normalized_items)
    has_less_content = any("更少内容" in item for item in normalized_items)
    if has_siri_suggestions and has_less_content:
        return True

    suggestion_markers = 0
    for marker in ("appstore", "safari", "相机", "健康", "天气", "扫一扫", "邮件"):
        if marker in joined:
            suggestion_markers += 1
    return has_siri_suggestions and suggestion_markers >= 2


def _looks_like_app_store_search_tokens(tokens: list[dict[str, Any]]) -> bool:
    normalized_items = [normalize_text(str(token.get("text") or "")).replace(" ", "") for token in tokens]
    joined = " ".join(normalized_items)
    has_search_tab = "搜索" in joined or "search" in joined
    has_app_store_tabs = ("today" in joined or "游戏" in joined) and ("app" in joined or "arcade" in joined)
    return has_search_tab and has_app_store_tabs


def _looks_like_top_search_content_token(token: dict[str, Any], observation: dict[str, Any], normalized: str) -> bool:
    if not normalized or normalized.replace(":", "").isdigit():
        return False
    bbox = _token_bbox_normalized(token, observation)
    if bbox is None:
        return False
    top = bbox["y"]
    if not (0.055 <= top <= 0.14):
        return False
    compact = normalized.replace(" ", "")
    if compact in {"100", "wifi", "lte", "5g"}:
        return False
    return True


def _looks_like_search_query_prefix(normalized: str) -> bool:
    compact = normalized.replace(" ", "").replace(".", "")
    return compact == "q" or (compact.startswith("q") and len(compact) > 1)


def _looks_like_spotlight_results(observation: dict[str, Any]) -> bool:
    ocr = observation.get("ocr") if isinstance(observation, dict) else None
    if not isinstance(ocr, dict) or not ocr.get("enabled"):
        return False
    plain_text = str(ocr.get("plainText") or "")
    if not plain_text:
        plain_text = " ".join(str(token.get("text") or "") for token in ocr.get("tokens") or [] if isinstance(token, dict))
    plain = normalize_text(plain_text)
    compact = plain.replace(" ", "")
    return "最佳搜索结果" in compact or "在app中搜索" in compact


def _looks_like_spotlight_overlay(observation: dict[str, Any]) -> bool:
    return _looks_like_spotlight_results(observation) or _looks_like_spotlight_suggestions(_observation_ocr_tokens_json(observation))


def _visible_app_label_anchor(observation: dict[str, Any], app_name: str) -> dict[str, Any] | None:
    target = normalize_text(app_name).replace(" ", "")
    if not target:
        return None
    for token in _observation_ocr_tokens_json(observation):
        text = str(token.get("text") or "")
        normalized = normalize_text(text)
        if normalized.replace(" ", "") != target:
            continue
        center = _token_center_normalized(token)
        if center is None or center["y"] > 0.62:
            continue
        point = {"x": center["x"], "y": max(0.06, center["y"] - 0.07)}
        return {
            "schema": "coretap.visible-app-label-anchor.v1",
            "app": app_name,
            "source": "ocr-label",
            "label": token,
            "point": point,
        }
    return None


def _looks_like_search_field_token(text: str, normalized: str) -> bool:
    compact = normalized.replace(" ", "").replace(".", "")
    if normalized == "search" or compact in {"搜索", "q搜索", "、搜索"}:
        return True
    if compact.startswith("q") and 2 <= len(compact) <= 5:
        tail = compact[1:]
        if tail in {"搜素", "搜紫", "搜萦", "製索", "櫻索", "接索", "超索", "提察", "提索", "學索", "学索", "優索"}:
            return True
        if tail.endswith("索") and any(ch in tail for ch in "搜製櫻接超提學学優"):
            return True
    return False


def _search_field_token_y_is_plausible(token: dict[str, Any], normalized: str) -> bool:
    center = _token_center_normalized(token)
    if center is None:
        return False
    if 0.06 <= center["y"] <= 0.7:
        return True
    compact = normalized.replace(" ", "").replace(".", "")
    return compact.startswith("q") and 0.72 <= center["y"] <= 0.97


def _search_field_anchor_point(token: dict[str, Any], observation: dict[str, Any]) -> dict[str, float] | None:
    center = _token_center_normalized(token)
    if center is None:
        return None
    bbox = _token_bbox_normalized(token, observation)
    if bbox is None:
        return {"x": 0.5, "y": center["y"]}
    text_right = bbox["x"] + bbox["width"]
    x = max(0.35, min(0.62, text_right + 0.24))
    return {"x": x, "y": center["y"]}


def _is_ascii_text(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _resolve_type_text_paste_at(action: dict[str, Any], context: dict[str, Any]) -> str | dict[str, Any] | None:
    explicit = action.get("pasteAt")
    anchor = context.get("anchor") if isinstance(context, dict) else None
    point = anchor.get("point") if isinstance(anchor, dict) else None
    if not _is_ascii_text(action["text"]):
        if explicit is not None:
            return explicit
        if isinstance(anchor, dict) and isinstance(point, dict) and anchor.get("source") == "edit-menu":
            return {"x": float(point["x"]), "y": float(point["y"]), "mode": "menu"}
        return None
    return explicit


def _type_text_focus_anchor(action: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None:
    if _is_ascii_text(action["text"]) or action.get("pasteAt") is not None:
        return None
    anchor = context.get("anchor") if isinstance(context, dict) else None
    point = anchor.get("point") if isinstance(anchor, dict) else None
    if not isinstance(point, dict):
        return None
    if anchor.get("source") not in {"search-field", "search-placeholder", "active-search-field", "spotlight-bottom-search"}:
        return None
    return anchor


def _is_search_field_target(target: str) -> bool:
    normalized = normalize_text(target)
    return "search" in normalized and ("field" in normalized or "bar" in normalized)


def _search_field_tap_anchor(target: str, observation: dict[str, Any]) -> dict[str, Any] | None:
    if not _is_search_field_target(target):
        return None
    context = _text_entry_context(observation)
    anchor = context.get("anchor") if isinstance(context, dict) else None
    if not isinstance(anchor, dict) or anchor.get("source") not in {"search-field", "search-placeholder", "active-search-field"}:
        return None
    point = anchor.get("point")
    if not isinstance(point, dict):
        return None
    return anchor


def _tap_normalized_for_step(args: argparse.Namespace, point: dict[str, float], *, reason: str) -> dict[str, Any]:
    x = float(point["x"])
    y = float(point["y"])
    point_info = point_to_hid(x, y, width=1, height=1, space="normalized")
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    tap = backend.tap_normalized(
        args.device,
        point_info["normalized"]["x"],
        point_info["normalized"]["y"],
        dry_run=args.dry_run,
        hid_u16=point_info["hidU16"],
    )
    return {
        "schema": "coretap.step.focus.v1",
        "reason": reason,
        "point": point_info,
        "tap": tap,
    }


def _frame_digest(observation: dict[str, Any]) -> str | None:
    frame = observation.get("frame") if isinstance(observation, dict) else None
    return frame.get("sha256") if isinstance(frame, dict) else None


def _evaluate_postconditions(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    postconditions: list[dict[str, Any]],
) -> dict[str, Any]:
    if not postconditions:
        return {"schema": "coretap.step.postconditions.v1", "status": "unchecked", "checks": []}
    checks: list[dict[str, Any]] = []
    target_observation = after or before
    for item in postconditions:
        kind = item["type"]
        if kind == "textVisible":
            match = _find_observation_text(
                target_observation,
                item["text"],
                case_sensitive=bool(item.get("caseSensitive", False)),
                min_confidence=float(item.get("minConfidence", 0.0)),
                match_mode=str(item.get("matchMode") or "contains"),
            )
            checks.append({"type": kind, "text": item["text"], "passed": bool(match), "match": match, "matchMode": item.get("matchMode") or "contains"})
        elif kind == "textAbsent":
            match = _find_observation_text(
                target_observation,
                item["text"],
                case_sensitive=bool(item.get("caseSensitive", False)),
                min_confidence=float(item.get("minConfidence", 0.0)),
                match_mode=str(item.get("matchMode") or "contains"),
            )
            checks.append({"type": kind, "text": item["text"], "passed": match is None, "match": match, "matchMode": item.get("matchMode") or "contains"})
        elif kind == "screenChanged":
            before_sha = _frame_digest(before)
            after_sha = _frame_digest(after or {})
            checks.append({"type": kind, "passed": bool(before_sha and after_sha and before_sha != after_sha), "beforeSha256": before_sha, "afterSha256": after_sha})
    status = "satisfied" if all(check["passed"] for check in checks) else "failed"
    return {"schema": "coretap.step.postconditions.v1", "status": status, "checks": checks}


def _effective_step_post_timeout_ms(args: argparse.Namespace, postconditions: list[dict[str, Any]]) -> int:
    configured = int(getattr(args, "post_timeout_ms", 0) or 0)
    if configured > 0:
        return configured
    if any(item["type"] in {"textVisible", "textAbsent"} for item in postconditions):
        return DEFAULT_TEXT_POST_TIMEOUT_MS
    return 0


_PASTE_PERMISSION_ALLOW_TEXTS = ("允许粘贴", "Allow Paste")


def _dismiss_paste_permission_prompt(args: argparse.Namespace, observation: dict[str, Any]) -> dict[str, Any] | None:
    for text in _PASTE_PERMISSION_ALLOW_TEXTS:
        match = _find_observation_text(observation, text, case_sensitive=False, min_confidence=0.0)
        if not match:
            continue
        frame = observation.get("frame") if isinstance(observation, dict) else None
        if not isinstance(frame, dict):
            return None
        try:
            width = float(frame["widthPx"])
            height = float(frame["heightPx"])
            box = match["matchedBoxPx"]
            x = (float(box["x"]) + float(box["width"]) / 2) / width
            y = (float(box["y"]) + float(box["height"]) / 2) / height
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
        backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
        tap = backend.tap_normalized(args.device, x, y, dry_run=False)
        return {
            "schema": "coretap.step.recovery.v1",
            "type": "pastePermissionPrompt",
            "matchedText": text,
            "match": match,
            "point": {"normalized": {"x": x, "y": y}},
            "tap": tap,
        }
    return None


def _step_blocked(action: dict[str, Any], code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "coretap.step.execution.v1",
        "status": "blocked",
        "actionType": action["type"],
        "code": code,
        "reason": message,
        "details": details or {},
    }


def _observation_frame(observation: dict[str, Any], *, reference: str = "source") -> dict[str, Any]:
    if reference == "preview":
        return observation["frame"]
    return observation.get("sourceFrame") or observation["frame"]


def _write_grounding_artifacts(run_dir: Path, grounded: dict[str, Any], *, stem: str) -> None:
    raw_tsv = grounded.pop("rawTsv", None)
    raw_output = grounded.pop("rawOutput", None)
    if raw_tsv is not None:
        (run_dir / f"{stem}.tsv").write_text(raw_tsv, encoding="utf-8")
    if raw_output is not None:
        (run_dir / f"{stem}.raw.txt").write_text(raw_output, encoding="utf-8")
    write_json(run_dir / f"{stem}.json", grounded)


def _execute_step_tap(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    target = action["target"]
    search_anchor = _search_field_tap_anchor(target, before)
    if search_anchor is not None:
        focus = _tap_normalized_for_step(args, search_anchor["point"], reason=f"tap-{search_anchor['source']}")
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "tap",
            "strategy": "ocr_search_field",
            "target": target,
            "anchor": search_anchor,
            "point": focus["point"],
            "tap": focus["tap"],
        }
    warm_model(args.profile)
    source_frame = _observation_frame(before, reference="source")
    source_image = Path(source_frame["path"])
    if args.profile == INTERNAL_FIXTURE_PROFILE:
        model_input = {
            "path": str(source_image),
            "widthPx": source_frame["widthPx"],
            "heightPx": source_frame["heightPx"],
            "resized": False,
            "maxLongSidePx": None,
            "scale": 1.0,
        }
    else:
        model_input = prepare_grounding_image(source_image, output_dir=run_dir, max_long_side=args.max_long_side)
    grounded = ground_target(Path(model_input["path"]), target, profile=args.profile)
    grounded["modelInput"] = {
        "path": model_input["path"],
        "widthPx": model_input["widthPx"],
        "heightPx": model_input["heightPx"],
        "resized": model_input["resized"],
        "maxLongSidePx": model_input["maxLongSidePx"],
        "scale": model_input["scale"],
    }
    grounded = remap_grounding_to_source_frame(
        grounded,
        source_width=int(source_frame["widthPx"]),
        source_height=int(source_frame["heightPx"]),
    )
    if grounded.get("status") != "found":
        grounded["safety"] = grounding_safety_diagnostics(target, grounded)
        _write_grounding_artifacts(run_dir, grounded, stem="step-grounding")
        return _step_blocked(
            action,
            _grounding_error_code(str(grounded.get("status") or "invalid")),
            f"Target was not found: {target}",
            details={"grounding": grounded, "modelInput": model_input},
        )
    grounded["safety"] = grounding_safety_diagnostics(target, grounded) if args.dry_run else assess_grounding_tap_safety(source_image, target, grounded)
    _write_grounding_artifacts(run_dir, grounded, stem="step-grounding")
    safety = grounded.get("safety")
    if not args.dry_run and isinstance(safety, dict) and not safety.get("safeToTap", False):
        return _step_blocked(
            action,
            "GROUNDING_UNSAFE_TO_TAP",
            f"Grounding was not trusted enough for a real tap: {target}",
            details={"grounding": grounded, "safety": safety},
        )
    point_px = grounded["point"]["framePx"]
    point = point_to_hid(
        point_px["x"],
        point_px["y"],
        width=int(source_frame["widthPx"]),
        height=int(source_frame["heightPx"]),
        space="px",
    )
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    tap = backend.tap_normalized(
        args.device,
        point["normalized"]["x"],
        point["normalized"]["y"],
        dry_run=args.dry_run,
        hid_u16=point["hidU16"],
    )
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "tap",
        "strategy": "vlm_grounding",
        "target": target,
        "profile": args.profile,
        "modelInput": grounded.get("modelInput"),
        "grounding": grounded,
        "point": point,
        "tap": tap,
    }


def _execute_step_open_app(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    app_name = action["name"]
    search_target = str(action.get("searchTarget") or "the Search button at the bottom center of the iOS home screen")
    result_target = str(action.get("resultTarget") or f"the large {app_name} app icon on the left side of the Best Search Result card in Spotlight search results")
    if args.dry_run:
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "openApp",
            "strategy": "spotlight-search",
            "app": app_name,
            "attempted": False,
            "dryRun": True,
        }

    substeps: list[dict[str, Any]] = []

    press_args = argparse.Namespace(**vars(args))
    press_args.button = "home"
    press_args.state = "press"
    press_args.hold_ms = None
    press_result = command_press(press_args)
    substeps.append({"name": "press-home", "status": "executed", "result": press_result})
    command_wait(argparse.Namespace(ms=700, wait_command=None))

    home = _observe_into(args, run_dir=run_dir, label="open-app-home")
    search_tap = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": search_target}, home, run_dir)
    substeps.append({"name": "tap-spotlight-search", "status": search_tap.get("status"), "result": search_tap})
    if search_tap.get("status") == "blocked":
        return _step_blocked(
            action,
            str(search_tap.get("code") or "FLOW_FAILED"),
            f"Could not focus Spotlight search while opening app: {app_name}",
            details={"substeps": substeps},
        )
    command_wait(argparse.Namespace(ms=800, wait_command=None))

    search = _observe_into(args, run_dir=run_dir, label="open-app-search")
    visible_anchor = _visible_app_label_anchor(search, app_name)
    if visible_anchor is not None:
        visible_tap = _tap_normalized_for_step(args, visible_anchor["point"], reason="open-visible-app-label")
        substeps.append({"name": "tap-visible-app-label", "status": "executed", "result": {"anchor": visible_anchor, "tap": visible_tap}})
        command_wait(argparse.Namespace(ms=1500, wait_command=None))
        visible_launch = _observe_into(args, run_dir=run_dir, label="open-app-visible-after-launch")
        substeps.append({"name": "observe-visible-after-launch", "status": "observed", "result": visible_launch})
        if not _looks_like_spotlight_overlay(visible_launch):
            return {
                "schema": "coretap.step.execution.v1",
                "status": "executed",
                "actionType": "openApp",
                "strategy": "spotlight-visible-label",
                "app": app_name,
                "substeps": substeps,
            }

    type_action = {
        "schema": "coretap.action.v2",
        "type": "typeText",
        "text": app_name,
        "charDelayMs": 30,
        "interDelayMs": 15,
        "pasteAt": "0.5,0.925",
        "pasteHoldMs": 1600,
        "verifyTimeoutMs": 0,
        "noVerify": True,
        "replace": True,
    }
    type_exec = _execute_step_action(args, type_action, search, run_dir)
    substeps.append({"name": "type-app-name", "status": type_exec.get("status"), "result": type_exec})
    if type_exec.get("status") == "blocked":
        return _step_blocked(
            action,
            str(type_exec.get("code") or "TEXT_INPUT_TARGET_UNKNOWN"),
            f"Could not type app name in Spotlight search: {app_name}",
            details={"substeps": substeps},
        )
    command_wait(argparse.Namespace(ms=900, wait_command=None))

    results = _observe_into(args, run_dir=run_dir, label="open-app-results")
    result_tap = _execute_step_tap(args, {"schema": "coretap.action.v2", "type": "tap", "target": result_target}, results, run_dir)
    substeps.append({"name": "tap-app-result", "status": result_tap.get("status"), "result": result_tap})
    if result_tap.get("status") == "blocked":
        return _step_blocked(
            action,
            str(result_tap.get("code") or "FLOW_FAILED"),
            f"Could not tap app result while opening app: {app_name}",
            details={"substeps": substeps},
        )
    command_wait(argparse.Namespace(ms=1500, wait_command=None))

    launched = _observe_into(args, run_dir=run_dir, label="open-app-after-launch")
    substeps.append({"name": "observe-after-launch", "status": "observed", "result": launched})
    if _looks_like_spotlight_overlay(launched):
        return _step_blocked(
            action,
            "FLOW_FAILED",
            f"Spotlight result tap did not open app: {app_name}",
            details={"substeps": substeps},
        )

    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "openApp",
        "strategy": "spotlight-search",
        "app": app_name,
        "substeps": substeps,
    }


def _execute_step_action(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    action_type = action["type"]
    if action_type == "tap":
        return _execute_step_tap(args, action, before, run_dir)
    if action_type == "openApp":
        return _execute_step_open_app(args, action, before, run_dir)
    if action_type == "typeText":
        context = _text_entry_context(before)
        has_explicit_paste_at = action.get("pasteAt") is not None
        if not args.dry_run and not context["ready"] and not has_explicit_paste_at:
            source_context = _source_ocr_text_entry_context(args, before, run_dir)
            if source_context is not None and source_context["ready"]:
                context = source_context
        if not args.dry_run and not context["ready"] and not has_explicit_paste_at:
            return _step_blocked(
                action,
                "TEXT_INPUT_TARGET_UNKNOWN",
                "No focused text input context was visible before typing",
                details={"textEntryContext": context},
            )
        ns = argparse.Namespace(**vars(args))
        ns.text = action["text"]
        ns.text_query = None
        ns.char_delay_ms = action["charDelayMs"]
        ns.inter_delay_ms = action["interDelayMs"]
        ns.paste_at = _resolve_type_text_paste_at(action, context)
        ns.paste_hold_ms = action["pasteHoldMs"]
        ns.verify_timeout_ms = action["verifyTimeoutMs"]
        ns.no_verify = action["noVerify"]
        ns.replace = action["replace"]
        focus_anchor = _type_text_focus_anchor(action, context)
        focus_result = None
        if focus_anchor is not None and not args.dry_run:
            focus_result = _tap_normalized_for_step(args, focus_anchor["point"], reason=f"focus-{focus_anchor['source']}")
            command_wait(argparse.Namespace(ms=650, wait_command=None))
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": action_type,
            "textEntryContext": context,
            "focusResult": focus_result,
            "resolvedPasteAt": ns.paste_at,
            "typeResult": command_type(ns),
        }
    if action_type == "key":
        ns = argparse.Namespace(**vars(args))
        ns.key = action["key"]
        ns.count = action["count"]
        ns.inter_delay_ms = action["interDelayMs"]
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": action_type, "keyResult": command_key(ns)}
    if action_type == "clear":
        ns = argparse.Namespace(**vars(args))
        ns.count = action["count"]
        ns.inter_delay_ms = action["interDelayMs"]
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": action_type, "clearResult": command_clear(ns)}
    if action_type == "press":
        ns = argparse.Namespace(**vars(args))
        ns.button = action["button"]
        ns.state = action["state"]
        ns.hold_ms = action["holdMs"]
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": action_type, "pressResult": command_press(ns)}
    if action_type == "scroll":
        ns = argparse.Namespace(**vars(args))
        ns.direction = action["direction"]
        ns.distance = action["distance"]
        ns.anchor_x = action["anchorX"]
        ns.anchor_y = action["anchorY"]
        ns.steps = action["steps"]
        ns.duration_ms = action["durationMs"]
        return {"schema": "coretap.step.execution.v1", "status": "executed", "actionType": action_type, "scrollResult": command_scroll(ns)}
    if action_type == "wait":
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": action_type,
            "waitResult": command_wait(argparse.Namespace(ms=action["ms"], wait_command=None)),
        }
    raise CoretapError("ACTION_UNSUPPORTED", f"Unsupported step action type: {action_type}", category="usage", stage="step-action")


def command_step(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    raw_action = _load_step_action(args)
    action = _normalize_step_action(raw_action)
    postconditions = _step_postconditions(args, action)
    needs_ocr = any(item["type"] in {"textVisible", "textAbsent"} for item in postconditions)
    before = _observe_into(args, run_dir=run_dir, label="step-before", no_ocr=(args.no_ocr and not needs_ocr))
    result: dict[str, Any] = {
        "schema": "coretap.step.result.v1",
        "artifactDir": str(run_dir),
        "action": action,
        "before": before,
        "postconditions": {"schema": "coretap.step.postconditions.v1", "status": "not_evaluated", "checks": []},
    }
    execution = _execute_step_action(args, action, before, run_dir)
    result["execution"] = execution

    if execution.get("status") == "blocked":
        result["status"] = "blocked"
        result["postconditions"] = {"schema": "coretap.step.postconditions.v1", "status": "skipped", "reason": "action blocked", "checks": []}
        write_json(run_dir / "step.result.json", result)
        raise CoretapError(
            str(execution.get("code") or "FLOW_FAILED"),
            str(execution.get("reason") or "step action was blocked"),
            category="test",
            stage="step-action",
            details=result,
        )

    if args.dry_run:
        result["status"] = "dry_run"
        result["postconditions"] = {"schema": "coretap.step.postconditions.v1", "status": "skipped", "reason": "dry run", "checks": []}
        write_json(run_dir / "step.result.json", result)
        return result

    if action["type"] != "wait" and args.post_wait_ms > 0:
        command_wait(argparse.Namespace(ms=args.post_wait_ms, wait_command=None))

    after_attempts: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    poll_index = 0
    post_timeout_ms = _effective_step_post_timeout_ms(args, postconditions)
    deadline = time.monotonic() + (post_timeout_ms / 1000)
    while True:
        label = "step-after" if poll_index == 0 else f"step-after-retry-{poll_index:03d}"
        after_needs_ocr = any(item["type"] in {"textVisible", "textAbsent"} for item in postconditions)
        after = _observe_into(args, run_dir=run_dir, label=label, no_ocr=(args.no_ocr and not after_needs_ocr))
        after_attempts.append(after)
        postcondition_result = _evaluate_postconditions(before, after, postconditions)
        if postcondition_result["status"] == "failed" and action["type"] == "typeText" and not recoveries:
            recovery = _dismiss_paste_permission_prompt(args, after)
            if recovery is not None:
                recoveries.append(recovery)
                result["recoveries"] = recoveries
                command_wait(argparse.Namespace(ms=600, wait_command=None))
                poll_index += 1
                continue
        if postcondition_result["status"] != "failed" or post_timeout_ms <= 0 or time.monotonic() >= deadline:
            result["after"] = after
            result["afterAttempts"] = len(after_attempts)
            result["postTimeoutMs"] = post_timeout_ms
            result["postconditions"] = postcondition_result
            break
        poll_index += 1
        time.sleep(args.poll_interval_ms / 1000)

    if result["postconditions"]["status"] == "failed":
        result["status"] = "postcondition_failed"
        write_json(run_dir / "step.result.json", result)
        raise CoretapError(
            "POSTCONDITION_FAILED",
            "step action executed but postconditions were not satisfied",
            category="test",
            stage="step-postcondition",
            details=result,
        )
    result["status"] = "executed"
    write_json(run_dir / "step.result.json", result)
    return result


def command_wait(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "wait_command", None) == "text":
        if not args.text:
            raise CoretapError("INVALID_ARGUMENT", "wait text requires --text", category="usage", stage="wait")
        return command_assert_text(args)
    if args.ms is None:
        raise CoretapError("INVALID_ARGUMENT", "wait requires --ms", category="usage", stage="wait")
    time.sleep(args.ms / 1000)
    return {"waitedMs": args.ms}


def _write_ocr_artifacts(run_dir: Path, stem: str, raw: dict[str, Any]) -> None:
    if "tesseractTsv" in raw:
        (run_dir / f"{stem}.tsv").write_text(raw["tesseractTsv"], encoding="utf-8")
    if "visionJson" in raw:
        (run_dir / f"{stem}.vision.json").write_text(raw["visionJson"], encoding="utf-8")
    write_json(run_dir / f"{stem}.ocr.json", _ocr_summary(raw))


def _ocr_summary(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "engines": list(raw.get("engines") or []),
        "engineErrors": list(raw.get("errors") or []),
    }


def command_daemon(args: argparse.Namespace) -> dict[str, Any]:
    from coretap.daemon import default_socket_path, ping_daemon, start_daemon, stop_daemon

    socket_path = Path(args.socket).expanduser() if args.socket else None
    socket_text = str(socket_path or default_socket_path())
    if args.daemon_command == "start":
        return start_daemon(socket_path=socket_path, timeout=args.timeout_ms / 1000)
    if args.daemon_command == "status":
        try:
            data = ping_daemon(socket_path=socket_path, timeout=args.timeout_ms / 1000)
            return {"running": True, "socket": socket_text, "response": data.get("result")}
        except CoretapError as exc:
            if exc.code != "DAEMON_UNAVAILABLE":
                raise
            return {"running": False, "socket": socket_text, "error": exc.details}
    if args.daemon_command == "stop":
        try:
            data = stop_daemon(socket_path=socket_path, timeout=args.timeout_ms / 1000)
            return data.get("result", data)
        except CoretapError as exc:
            if exc.code != "DAEMON_UNAVAILABLE":
                raise
            return {"stopping": False, "alreadyStopped": True, "running": False, "socket": socket_text, "error": exc.details}
    raise CoretapError("UNKNOWN_COMMAND", args.daemon_command, category="usage", stage="daemon")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coretap")
    parser.add_argument("--format", choices=["text", "json", "ndjson"], default="json")
    parser.add_argument("--backend", choices=["simulator", "device"], default="simulator")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--developer-dir", default=None)
    parser.add_argument("--coredevice-tunnel-mode", choices=["userspace", "tunneld"], default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--profile", default=PUBLIC_MODEL_PROFILE)
    parser.add_argument("--daemon", choices=["off", "auto", "on"], default="auto")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    sub.add_parser("status")
    config = sub.add_parser("config")
    config.add_argument("config_command", choices=["check"])
    sub.add_parser("discover")
    sub.add_parser("doctor")

    daemon = sub.add_parser("daemon")
    daemon.add_argument("daemon_command", choices=["start", "status", "stop"])
    daemon.add_argument("--socket", default=None)
    daemon.add_argument("--timeout-ms", type=int, default=5000)

    model = sub.add_parser("model")
    model.add_argument("model_command", choices=["install", "check", "warm", "status"])
    model.add_argument("--force", action="store_true")
    model.add_argument("--deep", action="store_true")
    model.add_argument("--dry-run", action="store_true")

    observe = sub.add_parser("observe")
    observe.add_argument("--label", default="observe")
    observe.add_argument("--out", default=None)
    observe.add_argument("--max-long-side", type=int, default=DEFAULT_GROUNDING_IMAGE_LONG_SIDE)
    observe.add_argument("--full-size", action="store_true")
    observe.add_argument("--lang", default=DEFAULT_OCR_LANG)
    observe.add_argument("--psm", type=int, default=11)
    observe.add_argument("--ocr-engine", choices=["auto", "vision", "tesseract", "all"], default="auto")
    observe.add_argument("--min-confidence", type=float, default=0.0)
    observe.add_argument("--no-ocr", action="store_true")

    step = sub.add_parser("step")
    step.add_argument("--action", default=None, help="Single coretap.action.v2 JSON object")
    step.add_argument("--action-file", default=None, help="Path to a single coretap.action.v2 JSON object")
    step.add_argument("--post-wait-ms", type=int, default=700)
    step.add_argument("--post-timeout-ms", type=int, default=0)
    step.add_argument("--poll-interval-ms", type=int, default=300)
    step.add_argument("--expect-text", action="append", default=[])
    step.add_argument("--expect-no-text", action="append", default=[])
    step.add_argument("--expect-change", action="store_true")
    step.add_argument("--fail-on-postcondition", action="store_true")
    step.add_argument("--dry-run", action="store_true")
    step.add_argument("--lang", default=DEFAULT_OCR_LANG)
    step.add_argument("--psm", type=int, default=11)
    step.add_argument("--ocr-engine", choices=["auto", "vision", "tesseract", "all"], default="auto")
    step.add_argument("--min-confidence", type=float, default=0.0)
    step.add_argument("--max-long-side", type=int, default=DEFAULT_STEP_MODEL_INPUT_LONG_SIDE)
    step.add_argument("--full-size", action="store_true")
    step.add_argument("--no-ocr", action="store_true")

    assert_text = sub.add_parser("assert")
    assert_sub = assert_text.add_subparsers(dest="assert_command", required=True)
    text = assert_sub.add_parser("text")
    text.add_argument("--text", required=True)
    text.add_argument("--image", default=None)
    text.add_argument("--timeout-ms", type=int, default=3000)
    text.add_argument("--poll-interval-ms", type=int, default=300)
    text.add_argument("--lang", default=DEFAULT_OCR_LANG)
    text.add_argument("--psm", type=int, default=11)
    text.add_argument("--case-sensitive", action="store_true")

    wait = sub.add_parser("wait")
    wait.add_argument("wait_command", choices=["text"])
    wait.add_argument("--text", required=True)
    wait.add_argument("--image", default=None)
    wait.add_argument("--timeout-ms", type=int, default=3000)
    wait.add_argument("--poll-interval-ms", type=int, default=300)
    wait.add_argument("--lang", default=DEFAULT_OCR_LANG)
    wait.add_argument("--psm", type=int, default=11)
    wait.add_argument("--case-sensitive", action="store_true")
    return parser


COMMON_OPTIONS_WITH_VALUES = {
    "--format",
    "--backend",
    "--device",
    "--developer-dir",
    "--coredevice-tunnel-mode",
    "--artifact-root",
    "--profile",
    "--daemon",
}


def normalize_global_args(argv: list[str]) -> list[str]:
    if "--" in argv:
        split = argv.index("--")
        head, tail = argv[:split], argv[split:]
    else:
        head, tail = argv, []
    moved: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(head):
        token = head[i]
        if token in COMMON_OPTIONS_WITH_VALUES and i + 1 < len(head):
            moved.extend([token, head[i + 1]])
            i += 2
        else:
            rest.append(token)
            i += 1
    return moved + rest + tail


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "setup":
        return command_setup(args)
    if args.command == "status":
        return command_status(args)
    if args.command == "config":
        return command_config(args)
    if args.command == "discover":
        return command_discover(args)
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "daemon":
        return command_daemon(args)
    if args.command == "model":
        return command_model(args)
    if args.command == "observe":
        return command_observe(args)
    if args.command == "step":
        return command_step(args)
    if args.command == "assert" and args.assert_command == "text":
        return command_assert_text(args)
    if args.command == "wait":
        return command_wait(args)
    raise CoretapError("UNKNOWN_COMMAND", args.command, category="usage", stage="cli")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    normalized = normalize_global_args(list(argv if argv is not None else sys.argv[1:]))
    args = parser.parse_args(normalized)
    if args.command != "daemon" and args.daemon != "off":
        from coretap.daemon import request_daemon, start_daemon

        try:
            data = request_daemon(normalized, cwd=str(Path.cwd()))
            emit(data, args.format)
            raise SystemExit(int(data.get("exitCode", 0 if data.get("ok") else 70)))
        except CoretapError as exc:
            if args.daemon == "auto" and exc.code == "DAEMON_UNAVAILABLE":
                start_daemon()
                data = request_daemon(normalized, cwd=str(Path.cwd()))
                emit(data, args.format)
                raise SystemExit(int(data.get("exitCode", 0 if data.get("ok") else 70)))
            if args.daemon == "on":
                data = response_error(args.command, exc)
                emit(data, args.format)
                raise SystemExit(EXIT_CODES.get(exc.code, 70))
            data = response_error(args.command, exc)
            emit(data, args.format)
            raise SystemExit(EXIT_CODES.get(exc.code, 70))
    started = time.monotonic()
    try:
        result = dispatch(args)
        data = response_ok(args.command, result)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        emit(data, args.format)
        raise SystemExit(0)
    except CoretapError as exc:
        data = response_error(args.command, exc)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        emit(data, args.format)
        raise SystemExit(EXIT_CODES.get(exc.code, 70))


if __name__ == "__main__":
    main()
