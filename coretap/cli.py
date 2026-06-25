from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from coretap import __version__
from coretap.backends import backend_for
from coretap.grounding import (
    GROUNDING_PROFILES,
    ground_target,
    model_cache,
    model_check,
    model_gc,
    model_install,
    model_status,
    model_stop,
    warm_model,
)
from coretap.model_pack import INTERNAL_FIXTURE_PROFILE, PUBLIC_MODEL_PROFILE
from coretap.ocr import find_exact_text_candidates, find_text, run_tesseract, tesseract_status
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
    "COREDEVICE_TUNNELD_UNAVAILABLE": 10,
    "COREDEVICE_TAP_FAILED": 32,
    "SIMULATOR_TAP_UNSUPPORTED": 32,
    "SIMULATOR_TAP_FAILED": 32,
    "SIMULATOR_DESCRIBE_FAILED": 32,
    "OCR_UNAVAILABLE": 10,
    "OCR_PROCESS_FAILED": 40,
    "CAPABILITY_UNAVAILABLE": 10,
    "UNKNOWN_MODEL_PROFILE": 2,
    "MODEL_NOT_INSTALLED": 60,
    "MODEL_INCOMPATIBLE": 60,
    "MODEL_LOAD_FAILED": 60,
    "MODEL_RUN_FAILED": 60,
    "TARGET_ABSENT": 30,
    "TEXT_TARGET_NOT_FOUND": 30,
    "TEXT_TARGET_AMBIGUOUS": 30,
    "GROUNDING_NOT_FOUND": 30,
    "GROUNDING_AMBIGUOUS": 30,
    "GROUNDING_SCHEMA_INVALID": 30,
    "INVALID_POINT": 31,
    "FLOW_FAILED": 50,
    "DAEMON_UNAVAILABLE": 14,
    "DAEMON_START_FAILED": 14,
    "DAEMON_ALREADY_RUNNING": 14,
    "DAEMON_REQUEST_FAILED": 14,
}


def emit(data: dict[str, Any], fmt: str) -> None:
    if fmt == "text":
        if data.get("ok"):
            print(json.dumps(data["result"], ensure_ascii=False, indent=2))
        else:
            err = data["error"]
            print(f"{err['code']}: {err['message']}", file=sys.stderr)
    else:
        print(json.dumps(data, ensure_ascii=False))


def point_to_hid(x: float, y: float, *, width: int, height: int, space: str) -> dict[str, Any]:
    if space == "hid":
        hx, hy = int(round(x)), int(round(y))
        if not (0 <= hx <= 65535 and 0 <= hy <= 65535):
            raise CoretapError("INVALID_POINT", "HID coordinates must be in [0,65535]", category="usage", stage="coordinate")
        normalized = {"x": hx / 65535, "y": hy / 65535}
        screenshot_px = {"x": normalized["x"] * width, "y": normalized["y"] * height}
    elif space == "normalized":
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise CoretapError("INVALID_POINT", "Normalized coordinates must be in [0,1]", category="usage", stage="coordinate")
        hx, hy = int(round(x * 65535)), int(round(y * 65535))
        normalized = {"x": x, "y": y}
        screenshot_px = {"x": x * width, "y": y * height}
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
        "hidU16": {"x": hx, "y": hy},
    }


def command_setup(args: argparse.Namespace) -> dict[str, Any]:
    roots = ensure_state()
    config_path = roots["state"] / "config.json"
    if not config_path.exists():
        write_json(
            config_path,
            {
                "schema": "coretap.config.v1",
                "version": 1,
                "capabilities": {
                    "grounding": {"profile": PUBLIC_MODEL_PROFILE},
                    "ocr": {"profile": "builtin:tesseract-fast-eng@dev"},
                },
                "storage": {name: str(path) for name, path in roots.items()},
            },
        )
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = {}
        grounding = config.setdefault("capabilities", {}).setdefault("grounding", {})
        if grounding.get("profile") == "builtin:text-ocr-grounder@dev":
            grounding["profile"] = PUBLIC_MODEL_PROFILE
            write_json(config_path, config)
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
            "textAssertions": bool(ocr.get("ready")),
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
    checks.append({"id": "ocr", "status": "pass" if ocr["ready"] else "warn", "details": ocr})
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
        return model_install(args.profile, force=args.force)
    if args.model_command == "run":
        if not args.image or not args.target:
            raise CoretapError("INVALID_ARGUMENT", "model run requires --image and --target", category="usage", stage="model")
        image = Path(args.image)
        if not image.exists():
            raise CoretapError("INVALID_ARGUMENT", f"Image does not exist: {image}", category="usage", stage="model")
        return ground_target(image, args.target, profile=args.profile)
    if args.model_command == "stop":
        return model_stop()
    if args.model_command == "cache":
        return model_cache()
    if args.model_command == "gc":
        return model_gc(dry_run=args.dry_run)
    raise CoretapError("UNKNOWN_MODEL_COMMAND", args.model_command, category="usage", stage="model")


def command_ocr(args: argparse.Namespace) -> dict[str, Any]:
    status = tesseract_status()
    if args.ocr_command == "status":
        return status
    if args.ocr_command == "check":
        if not status["ready"]:
            raise CoretapError("OCR_UNAVAILABLE", "tesseract not found or not runnable", stage="ocr", details=status)
        return status
    raise CoretapError("UNKNOWN_OCR_COMMAND", args.ocr_command, category="usage", stage="ocr")


def capture(args: argparse.Namespace, *, label: str = "screenshot") -> tuple[Any, Path, Path]:
    run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    out = Path(args.out) if getattr(args, "out", None) else run_dir / f"{label}.png"
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    frame = backend.screenshot(args.device, out)
    frame_json = {
        "frameId": frame.frame_id,
        "path": str(frame.path),
        "widthPx": frame.width,
        "heightPx": frame.height,
        "backend": frame.backend,
        "device": frame.device,
        "sha256": sha256_file(frame.path),
    }
    write_json(run_dir / f"{label}.frame.json", frame_json)
    return frame, run_dir, out


def command_screenshot(args: argparse.Namespace) -> dict[str, Any]:
    frame, run_dir, _ = capture(args, label=args.label)
    return {
        "artifactDir": str(run_dir),
        "frame": {
            "frameId": frame.frame_id,
            "path": str(frame.path),
            "widthPx": frame.width,
            "heightPx": frame.height,
            "backend": frame.backend,
            "device": frame.device,
        },
    }


def command_tap_point(args: argparse.Namespace) -> dict[str, Any]:
    if args.frame:
        width, height = png_size(Path(args.frame))
    else:
        width = args.width or 1
        height = args.height or 1
    point = point_to_hid(args.x, args.y, width=width, height=height, space=args.space)
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    tap = backend.tap_normalized(
        args.device,
        point["normalized"]["x"],
        point["normalized"]["y"],
        dry_run=args.dry_run,
        hid_u16=point["hidU16"],
    )
    return {"point": point, "tap": tap}


def command_locate(args: argparse.Namespace) -> dict[str, Any]:
    warm_model(args.profile)
    frame, run_dir, _ = capture(args, label="source")
    grounded = ground_target(frame.path, args.target, profile=args.profile)
    raw_tsv = grounded.pop("rawTsv", None)
    raw_output = grounded.pop("rawOutput", None)
    if raw_tsv is not None:
        (run_dir / "grounding-raw.tsv").write_text(raw_tsv, encoding="utf-8")
    if raw_output is not None:
        (run_dir / "grounding.raw.txt").write_text(raw_output, encoding="utf-8")
    write_json(run_dir / "grounding.json", grounded)
    result = {
        "artifactDir": str(run_dir),
        "target": args.target,
        "profile": args.profile,
        "frame": {
            "path": str(frame.path),
            "widthPx": frame.width,
            "heightPx": frame.height,
        },
        "grounding": grounded,
    }
    write_json(run_dir / "locate.result.json", result)
    return result


def _grounding_error_code(status: str) -> str:
    if status == "not_found":
        return "GROUNDING_NOT_FOUND"
    if status == "ambiguous":
        return "GROUNDING_AMBIGUOUS"
    return "GROUNDING_SCHEMA_INVALID"


def command_tap_target(args: argparse.Namespace) -> dict[str, Any]:
    located = command_locate(args)
    grounded = located["grounding"]
    if grounded["status"] != "found":
        raise CoretapError(
            _grounding_error_code(grounded["status"]),
            f"Target was not found: {args.target}",
            stage="grounding",
            category="grounding",
            details={"artifactDir": located["artifactDir"], "grounding": grounded},
        )
    frame = located["frame"]
    p = grounded["point"]["framePx"]
    point = point_to_hid(p["x"], p["y"], width=frame["widthPx"], height=frame["heightPx"], space="px")
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    tap = backend.tap_normalized(
        args.device,
        point["normalized"]["x"],
        point["normalized"]["y"],
        dry_run=args.dry_run,
        hid_u16=point["hidU16"],
    )
    result = {
        "artifactDir": located["artifactDir"],
        "target": args.target,
        "profile": args.profile,
        "frame": frame,
        "grounding": grounded,
        "point": point,
        "tap": tap,
    }
    write_json(Path(located["artifactDir"]) / "tap-target.result.json", result)
    return result


def _text_query(args: argparse.Namespace) -> str:
    text = getattr(args, "text", None) or getattr(args, "text_query", None)
    if not text:
        raise CoretapError("INVALID_ARGUMENT", "tap text requires text", category="usage", stage="tap-text")
    return str(text)


def command_tap_text(args: argparse.Namespace) -> dict[str, Any]:
    text = _text_query(args)
    frame, run_dir, image = capture(args, label="text-source")
    tokens, raw = run_tesseract(image, lang=args.lang, psm=args.psm)
    (run_dir / "ocr.tsv").write_text(raw, encoding="utf-8")
    candidates = find_exact_text_candidates(
        tokens,
        text,
        case_sensitive=args.case_sensitive,
        min_confidence=args.min_confidence,
    )
    if not candidates:
        raise CoretapError(
            "TEXT_TARGET_NOT_FOUND",
            f"Text target was not found: {text}",
            stage="tap-text",
            category="grounding",
            details={"artifactDir": str(run_dir), "text": text, "tokenCount": len(tokens)},
        )
    if len(candidates) > 1:
        raise CoretapError(
            "TEXT_TARGET_AMBIGUOUS",
            f"Text target matched multiple regions: {text}",
            stage="tap-text",
            category="grounding",
            details={"artifactDir": str(run_dir), "text": text, "candidateCount": len(candidates), "candidates": candidates},
        )
    match = candidates[0]
    box = match["matchedBoxPx"]
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    point = point_to_hid(x, y, width=frame.width, height=frame.height, space="px")
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    tap = backend.tap_normalized(
        args.device,
        point["normalized"]["x"],
        point["normalized"]["y"],
        dry_run=args.dry_run,
        hid_u16=point["hidU16"],
    )
    result = {
        "artifactDir": str(run_dir),
        "text": text,
        "strategy": "ocr_exact",
        "frame": {"path": str(image), "widthPx": frame.width, "heightPx": frame.height},
        "ocr": {
            "tokenCount": len(tokens),
            "candidateCount": len(candidates),
            "lang": args.lang,
            "psm": args.psm,
            "minConfidence": args.min_confidence,
            "match": match,
        },
        "point": point,
        "tap": tap,
    }
    write_json(run_dir / "tap-text.result.json", result)
    return result


def command_assert_text(args: argparse.Namespace) -> dict[str, Any]:
    if args.image:
        image = Path(args.image)
        run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    else:
        frame, run_dir, image = capture(args, label="assert")
    deadline = time.monotonic() + (args.timeout_ms / 1000)
    attempts = 0
    last: dict[str, Any] | None = None
    while True:
        attempts += 1
        tokens, raw = run_tesseract(image)
        (run_dir / f"assert-{attempts:03d}.tsv").write_text(raw, encoding="utf-8")
        match = find_text(tokens, args.text, case_sensitive=args.case_sensitive)
        last = {
            "attempts": attempts,
            "image": str(image),
            "tokenCount": len(tokens),
            "match": match,
        }
        if match:
            result = {
                "artifactDir": str(run_dir),
                "expected": args.text,
                "matched": True,
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


def command_wait(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "wait_command", None) == "text":
        if not args.text:
            raise CoretapError("INVALID_ARGUMENT", "wait text requires --text", category="usage", stage="wait")
        return command_assert_text(args)
    if args.ms is None:
        raise CoretapError("INVALID_ARGUMENT", "wait requires --ms", category="usage", stage="wait")
    time.sleep(args.ms / 1000)
    return {"waitedMs": args.ms}


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    flow_path = Path(args.flow)
    if flow_path.suffix.lower() != ".json":
        raise CoretapError("FLOW_FAILED", "MVP flow runner supports JSON flows only", category="usage", stage="flow")
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    steps = flow.get("steps", [])
    results = []
    for step in steps:
        if "wait" in step:
            ns = argparse.Namespace(ms=int(step["wait"].get("ms", 0)))
            results.append({"wait": command_wait(ns)})
        elif "screenshot" in step:
            ns = argparse.Namespace(**vars(args))
            ns.label = step["screenshot"].get("label", "screenshot")
            ns.out = step["screenshot"].get("out")
            results.append({"screenshot": command_screenshot(ns)})
        elif "tapTarget" in step:
            ns = argparse.Namespace(**vars(args))
            ns.target = step["tapTarget"]["target"]
            ns.profile = step["tapTarget"].get("profile", args.profile)
            ns.dry_run = bool(step["tapTarget"].get("dryRun", args.dry_run))
            results.append({"tapTarget": command_tap_target(ns)})
        elif "tapText" in step:
            ns = argparse.Namespace(**vars(args))
            ns.text = step["tapText"]["text"]
            ns.lang = step["tapText"].get("lang", "eng")
            ns.psm = int(step["tapText"].get("psm", 11))
            ns.min_confidence = float(step["tapText"].get("minConfidence", 50.0))
            ns.case_sensitive = bool(step["tapText"].get("caseSensitive", False))
            ns.dry_run = bool(step["tapText"].get("dryRun", args.dry_run))
            results.append({"tapText": command_tap_text(ns)})
        elif "locate" in step:
            ns = argparse.Namespace(**vars(args))
            ns.target = step["locate"]["target"]
            ns.profile = step["locate"].get("profile", args.profile)
            results.append({"locate": command_locate(ns)})
        elif "assertText" in step:
            ns = argparse.Namespace(**vars(args))
            ns.text = step["assertText"]["text"]
            ns.image = step["assertText"].get("image")
            ns.timeout_ms = int(step["assertText"].get("timeoutMs", args.timeout_ms))
            ns.poll_interval_ms = int(step["assertText"].get("pollIntervalMs", args.poll_interval_ms))
            ns.case_sensitive = bool(step["assertText"].get("caseSensitive", False))
            results.append({"assertText": command_assert_text(ns)})
        else:
            raise CoretapError("FLOW_FAILED", f"Unknown flow step: {step}", category="usage", stage="flow")
    return {"name": flow.get("name", flow_path.stem), "stepCount": len(steps), "steps": results}


def command_replay(args: argparse.Namespace) -> dict[str, Any]:
    target = Path(args.path)
    if target.is_file() and target.suffix == ".json":
        data = json.loads(target.read_text(encoding="utf-8"))
        base = target.parent
    elif target.is_dir():
        base = target
        result_json = base / "tap-target.result.json"
        if result_json.exists():
            data = json.loads(result_json.read_text(encoding="utf-8"))
        else:
            raise CoretapError("REPLAY_UNSUPPORTED", f"No replayable result in {target}", category="usage", stage="replay")
    else:
        raise CoretapError("REPLAY_UNSUPPORTED", f"Replay path does not exist: {target}", category="usage", stage="replay")

    if "target" not in data or "frame" not in data:
        raise CoretapError("REPLAY_UNSUPPORTED", "Only tap-target result replay is implemented in this MVP", category="usage", stage="replay")
    image = Path(data["frame"]["path"])
    if not image.is_absolute():
        image = (base / image).resolve() if not image.exists() else image.resolve()
    profile = data.get("profile") or data.get("grounding", {}).get("model", {}).get("profile") or args.profile
    replayed = ground_target(image, data["target"], profile=profile)
    replayed.pop("rawTsv", None)
    replayed.pop("rawOutput", None)
    comparison = {
        "statusEqual": replayed.get("status") == data.get("grounding", {}).get("status"),
        "pointDeltaPx": None,
    }
    old_point = data.get("grounding", {}).get("point", {}).get("framePx")
    new_point = replayed.get("point", {}).get("framePx")
    if old_point and new_point:
        comparison["pointDeltaPx"] = {
            "x": new_point["x"] - old_point["x"],
            "y": new_point["y"] - old_point["y"],
        }
    return {
        "operation": "grounding",
        "source": str(target),
        "target": data["target"],
        "recorded": data.get("grounding"),
        "replayed": replayed,
        "comparison": comparison,
    }


def command_test(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = artifact_dir(Path(args.artifact_root) if args.artifact_root else None)
    env = dict(**__import__("os").environ)
    env["CORETAP_ARTIFACT_DIR"] = str(run_dir)
    env["CORETAP_BACKEND"] = args.backend
    env["CORETAP_DEVICE"] = args.device
    proc = subprocess.run(args.child, text=True, capture_output=True, env=env, check=False)
    (run_dir / "child.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "child.stderr.log").write_text(proc.stderr, encoding="utf-8")
    result = {
        "artifactDir": str(run_dir),
        "exitCode": proc.returncode,
        "stdout": "child.stdout.log",
        "stderr": "child.stderr.log",
    }
    write_json(run_dir / "test.result.json", result)
    if proc.returncode != 0:
        raise CoretapError("FLOW_FAILED", "Child test command failed", category="test", stage="test", details=result)
    return result


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
        data = stop_daemon(socket_path=socket_path, timeout=args.timeout_ms / 1000)
        return data.get("result", data)
    raise CoretapError("UNKNOWN_COMMAND", args.daemon_command, category="usage", stage="daemon")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coretap")
    parser.add_argument("--format", choices=["text", "json", "ndjson"], default="text")
    parser.add_argument("--backend", choices=["simulator", "device"], default="simulator")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--developer-dir", default=None)
    parser.add_argument("--coredevice-tunnel-mode", choices=["userspace", "tunneld"], default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--profile", default=PUBLIC_MODEL_PROFILE)
    parser.add_argument("--daemon", choices=["off", "auto", "on"], default="off")

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
    model.add_argument("model_command", choices=["install", "check", "warm", "run", "status", "stop", "cache", "gc"])
    model.add_argument("--force", action="store_true")
    model.add_argument("--deep", action="store_true")
    model.add_argument("--dry-run", action="store_true")
    model.add_argument("--image", default=None)
    model.add_argument("--target", default=None)

    ocr = sub.add_parser("ocr")
    ocr.add_argument("ocr_command", choices=["status", "check"])

    screenshot = sub.add_parser("screenshot")
    screenshot.add_argument("--label", default="screenshot")
    screenshot.add_argument("--out", default=None)

    locate = sub.add_parser("locate")
    locate.add_argument("--target", required=True)

    tap = sub.add_parser("tap")
    tap_sub = tap.add_subparsers(dest="tap_command", required=True)
    point = tap_sub.add_parser("point")
    point.add_argument("--space", choices=["px", "normalized", "hid"], required=True)
    point.add_argument("--x", type=float, required=True)
    point.add_argument("--y", type=float, required=True)
    point.add_argument("--frame", default=None)
    point.add_argument("--width", type=int, default=None)
    point.add_argument("--height", type=int, default=None)
    point.add_argument("--dry-run", action="store_true")

    target = tap_sub.add_parser("target")
    target.add_argument("--target", required=True)
    target.add_argument("--dry-run", action="store_true")

    tap_text = tap_sub.add_parser("text")
    tap_text.add_argument("text_query", nargs="?")
    tap_text.add_argument("--text", dest="text", default=None)
    tap_text.add_argument("--dry-run", action="store_true")
    tap_text.add_argument("--lang", default="eng")
    tap_text.add_argument("--psm", type=int, default=11)
    tap_text.add_argument("--min-confidence", type=float, default=50.0)
    tap_text.add_argument("--case-sensitive", action="store_true")

    assert_text = sub.add_parser("assert")
    assert_sub = assert_text.add_subparsers(dest="assert_command", required=True)
    text = assert_sub.add_parser("text")
    text.add_argument("--text", required=True)
    text.add_argument("--image", default=None)
    text.add_argument("--timeout-ms", type=int, default=3000)
    text.add_argument("--poll-interval-ms", type=int, default=300)
    text.add_argument("--case-sensitive", action="store_true")

    wait = sub.add_parser("wait")
    wait.add_argument("wait_command", nargs="?", choices=["text"])
    wait.add_argument("--ms", type=int, default=None)
    wait.add_argument("--text", default=None)
    wait.add_argument("--image", default=None)
    wait.add_argument("--timeout-ms", type=int, default=3000)
    wait.add_argument("--poll-interval-ms", type=int, default=300)
    wait.add_argument("--case-sensitive", action="store_true")

    run = sub.add_parser("run")
    run.add_argument("flow")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--timeout-ms", type=int, default=3000)
    run.add_argument("--poll-interval-ms", type=int, default=300)
    run.add_argument("--case-sensitive", action="store_true")

    test = sub.add_parser("test")
    test.add_argument("child", nargs=argparse.REMAINDER)

    replay = sub.add_parser("replay")
    replay.add_argument("path")
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
    if args.command == "ocr":
        return command_ocr(args)
    if args.command == "screenshot":
        return command_screenshot(args)
    if args.command == "locate":
        return command_locate(args)
    if args.command == "tap":
        if args.tap_command == "point":
            return command_tap_point(args)
        if args.tap_command == "target":
            return command_tap_target(args)
        if args.tap_command == "text":
            return command_tap_text(args)
    if args.command == "assert" and args.assert_command == "text":
        return command_assert_text(args)
    if args.command == "wait":
        return command_wait(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "test":
        if args.child and args.child[0] == "--":
            args.child = args.child[1:]
        if not args.child:
            raise CoretapError("FLOW_FAILED", "coretap test requires a child command after --", category="usage", stage="test")
        return command_test(args)
    if args.command == "replay":
        return command_replay(args)
    raise CoretapError("UNKNOWN_COMMAND", args.command, category="usage", stage="cli")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    normalized = normalize_global_args(list(argv if argv is not None else sys.argv[1:]))
    args = parser.parse_args(normalized)
    if args.command != "daemon" and args.daemon != "off":
        from coretap.daemon import request_daemon

        try:
            data = request_daemon(normalized, cwd=str(Path.cwd()))
            emit(data, args.format)
            raise SystemExit(int(data.get("exitCode", 0 if data.get("ok") else 70)))
        except CoretapError as exc:
            if args.daemon == "auto":
                pass
            else:
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
