from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, suppress
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from coretap import __version__
from coretap.backends import backend_for
from coretap.device_buttons import BUTTON_STATES, button_choices
from coretap.grounding import (
    DEFAULT_GROUNDING_IMAGE_LONG_SIDE,
    DEFAULT_REFINEMENT_CROP_RATIO,
    GROUNDING_PROFILES,
    ground_target,
    model_check,
    model_install,
    model_status,
    prepare_refinement_crop,
    prepare_image_long_side,
    prepare_grounding_image,
    remap_crop_grounding_to_source_frame,
    remap_grounding_to_source_frame,
    warm_model,
)
from coretap.model_pack import (
    INTERNAL_FIXTURE_PROFILE,
    PUBLIC_MODEL_PROFILE,
    PUBLIC_VISUAL_OBSERVE_PROMPT_VERSION,
    run_visual_observe_model,
)
from coretap.ocr import (
    DEFAULT_OCR_ENGINE,
    DEFAULT_OCR_LANG,
    OcrToken,
    find_exact_text_candidates,
    find_text,
    normalize_text,
    run_ocr,
    vision_ocr_status,
    token_match,
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
    "COREDEVICE_LAUNCH_APP_FAILED": 32,
    "COREDEVICE_UNINSTALL_APP_FAILED": 32,
    "WEBINSPECTOR_OPEN_URL_FAILED": 32,
    "COREDEVICE_WORKER_FAILED": 32,
    "COREDEVICE_WORKER_TIMEOUT": 32,
    "PASTEBOARD_SET_FAILED": 32,
    "PASTE_MENU_NOT_FOUND": 32,
    "TEXT_INPUT_VERIFICATION_FAILED": 30,
    "SIMULATOR_DRAG_UNSUPPORTED": 32,
    "SIMULATOR_PRESS_UNSUPPORTED": 32,
    "SIMULATOR_TAP_UNSUPPORTED": 32,
    "SIMULATOR_TYPE_UNSUPPORTED": 32,
    "SIMULATOR_KEY_UNSUPPORTED": 32,
    "SIMULATOR_TAP_FAILED": 32,
    "SIMCTL_LAUNCH_APP_FAILED": 32,
    "SIMCTL_UNINSTALL_APP_FAILED": 32,
    "SIMCTL_OPEN_URL_FAILED": 32,
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
    "ACTION_SCHEMA_INVALID": 2,
    "ACTION_UNSUPPORTED": 2,
    "INVALID_POINT": 31,
    "FLOW_FAILED": 50,
    "DAEMON_UNAVAILABLE": 14,
    "DAEMON_START_FAILED": 14,
    "DAEMON_ALREADY_RUNNING": 14,
    "DAEMON_STALE": 14,
    "DAEMON_REQUEST_FAILED": 14,
}

DEFAULT_STEP_MODEL_INPUT_LONG_SIDE = DEFAULT_GROUNDING_IMAGE_LONG_SIDE
DEFAULT_STEP_PAGE_WAIT_MS = 6000
TEXT_ENTRY_ANCHOR_MAX_AGE_MS = 5 * 60 * 1000
TRACE_SCHEMA = "coretap.trace.v1"
TRACE_EVENT_SCHEMA = "coretap.trace.event.v1"
TEXT_ENTRY_ANCHOR_CACHE_SCHEMA = "coretap.text-entry-anchor-cache.v1"
TEXT_ENTRY_ANCHOR_SCHEMA = "coretap.text-entry-anchor.v1"
TEXT_ENTRY_TARGET_MARKERS = (
    "text field",
    "textfield",
    "search field",
    "search text field",
    "search bar",
    "input field",
    "input",
    "field",
    "搜索框",
    "搜索栏",
    "搜索输入",
    "输入框",
    "文本框",
)
TOP_TEXT_ENTRY_TARGET_MARKERS = (
    "top",
    "at the top",
    "顶部",
    "上方",
    "顶端",
)
TEXT_ENTRY_PLACEHOLDER_MARKERS = (
    "placeholder",
    "search",
    "搜索",
    "输入",
    "游戏",
    "故事",
    "app",
    "应用",
)
NON_TEXT_ENTRY_CONTEXT_MARKERS = (
    "row",
    "result",
    "suggestion",
    "card",
    "list item",
    "结果",
    "建议",
    "行",
    "卡片",
)
BUILTIN_APP_BUNDLE_IDS = {
    "app store": "com.apple.AppStore",
    "appstore": "com.apple.AppStore",
    "apple store": "com.apple.AppStore",
    "safari": "com.apple.mobilesafari",
    "safari浏览器": "com.apple.mobilesafari",
    "settings": "com.apple.Preferences",
    "设置": "com.apple.Preferences",
    "messages": "com.apple.MobileSMS",
    "信息": "com.apple.MobileSMS",
    "phone": "com.apple.mobilephone",
    "电话": "com.apple.mobilephone",
    "photos": "com.apple.mobileslideshow",
    "照片": "com.apple.mobileslideshow",
    "camera": "com.apple.camera",
    "相机": "com.apple.camera",
    "maps": "com.apple.Maps",
    "地图": "com.apple.Maps",
    "notes": "com.apple.mobilenotes",
    "备忘录": "com.apple.mobilenotes",
    "calendar": "com.apple.mobilecal",
    "日历": "com.apple.mobilecal",
    "clock": "com.apple.mobiletimer",
    "时钟": "com.apple.mobiletimer",
    "files": "com.apple.DocumentsApp",
    "文件": "com.apple.DocumentsApp",
    "rednote": "com.xingin.discover",
    "xhs": "com.xingin.discover",
    "xiaohongshu": "com.xingin.discover",
    "小红书": "com.xingin.discover",
    "小紅書": "com.xingin.discover",
}


def emit(data: dict[str, Any]) -> None:
    print(json.dumps(compact_response(data), ensure_ascii=False))


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_trace_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    cleaned = cleaned.strip(".-_")
    if not cleaned:
        raise CoretapError("INVALID_ARGUMENT", "trace id must contain at least one alphanumeric character", category="usage", stage="trace")
    return cleaned[:120]


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _persistent_artifact_root(args: argparse.Namespace) -> Path | None:
    if getattr(args, "no_artifacts", False):
        return None
    raw_root = getattr(args, "artifact_root", None)
    if raw_root:
        return Path(str(raw_root)).expanduser()
    if getattr(args, "keep_artifacts", False) or getattr(args, "trace_id", None):
        return ensure_state()["artifacts"]
    return None


def _trace_artifact_root(args: argparse.Namespace) -> Path:
    raw_root = getattr(args, "artifact_root", None)
    if raw_root:
        return Path(str(raw_root)).expanduser()
    return ensure_state()["artifacts"]


def _artifacts_are_persistent(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "_artifacts_persistent", False))


def _artifact_result(args: argparse.Namespace, run_dir: Path) -> dict[str, str]:
    if not _artifacts_are_persistent(args):
        return {}
    return {"artifactDir": str(run_dir)}


@contextmanager
def _command_artifacts(args: argparse.Namespace):
    existing = getattr(args, "_artifact_run_dir", None)
    if existing:
        yield Path(str(existing))
        return

    root = _persistent_artifact_root(args)
    if root is not None:
        run_dir = artifact_dir(root)
        previous_dir = getattr(args, "_artifact_run_dir", None)
        previous_persistent = getattr(args, "_artifacts_persistent", None)
        args._artifact_run_dir = str(run_dir)
        args._artifacts_persistent = True
        try:
            yield run_dir
        finally:
            if previous_dir is None:
                with suppress(AttributeError):
                    delattr(args, "_artifact_run_dir")
            else:
                args._artifact_run_dir = previous_dir
            if previous_persistent is None:
                with suppress(AttributeError):
                    delattr(args, "_artifacts_persistent")
            else:
                args._artifacts_persistent = previous_persistent
        return

    previous_dir = getattr(args, "_artifact_run_dir", None)
    previous_persistent = getattr(args, "_artifacts_persistent", None)
    with tempfile.TemporaryDirectory(prefix="coretap-") as temp_root:
        run_dir = artifact_dir(Path(temp_root))
        args._artifact_run_dir = str(run_dir)
        args._artifacts_persistent = False
        try:
            yield run_dir
        finally:
            if previous_dir is None:
                with suppress(AttributeError):
                    delattr(args, "_artifact_run_dir")
            else:
                args._artifact_run_dir = previous_dir
            if previous_persistent is None:
                with suppress(AttributeError):
                    delattr(args, "_artifacts_persistent")
            else:
                args._artifacts_persistent = previous_persistent


def _trace_root(args: argparse.Namespace) -> Path:
    root = _trace_artifact_root(args)
    return root / "traces"


def _trace_paths(args: argparse.Namespace) -> dict[str, Path] | None:
    trace_id = getattr(args, "trace_id", None) or os.environ.get("CORETAP_TRACE_ID")
    if not trace_id:
        return None
    safe_id = _safe_trace_id(str(trace_id))
    trace_dir = _trace_root(args) / safe_id
    return {
        "dir": trace_dir,
        "trace": trace_dir / "trace.json",
        "events": trace_dir / "events.jsonl",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def _trace_title(args: argparse.Namespace) -> str | None:
    title = getattr(args, "trace_title", None) or os.environ.get("CORETAP_TRACE_TITLE")
    return str(title) if title else None


def _action_trace_summary(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    summary = {"type": action.get("type")}
    for key in ("target", "name", "text", "key", "button", "direction", "ms"):
        if key in action:
            summary[key] = action[key]
    return summary


def _observation_trace_summary(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(observation, dict):
        return None
    frame = observation.get("frame") if isinstance(observation.get("frame"), dict) else {}
    source = observation.get("sourceFrame") if isinstance(observation.get("sourceFrame"), dict) else {}
    ocr = observation.get("ocr") if isinstance(observation.get("ocr"), dict) else {}
    return {
        "frame": {
            "path": frame.get("path"),
            "widthPx": frame.get("widthPx"),
            "heightPx": frame.get("heightPx"),
            "sha256": frame.get("sha256"),
        },
        "sourceFrame": {
            "path": source.get("path"),
            "widthPx": source.get("widthPx"),
            "heightPx": source.get("heightPx"),
            "sha256": source.get("sha256"),
        }
        if source
        else None,
        "ocr": {
            "enabled": bool(ocr.get("enabled")),
            "selectedEngine": ocr.get("selectedEngine"),
            "tokenCount": ocr.get("tokenCount"),
        }
        if ocr
        else None,
        "visual": {
            "enabled": bool((observation.get("visual") or {}).get("enabled")),
            "status": (observation.get("visual") or {}).get("status"),
            "elementCount": len((observation.get("visual") or {}).get("elements") or []),
        }
        if isinstance(observation.get("visual"), dict)
        else None,
    }


def _execution_trace_summary(execution: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(execution, dict):
        return None
    summary: dict[str, Any] = {
        "status": execution.get("status"),
        "actionType": execution.get("actionType"),
        "strategy": execution.get("strategy"),
        "code": execution.get("code"),
        "reason": execution.get("reason"),
    }
    if isinstance(execution.get("modelInput"), dict):
        model_input = execution["modelInput"]
        summary["modelInput"] = {
            "path": model_input.get("path"),
            "widthPx": model_input.get("widthPx"),
            "heightPx": model_input.get("heightPx"),
            "resized": model_input.get("resized"),
            "maxLongSidePx": model_input.get("maxLongSidePx"),
            "scale": model_input.get("scale"),
        }
    grounding = execution.get("grounding") if isinstance(execution.get("grounding"), dict) else None
    if grounding:
        point = grounding.get("point") if isinstance(grounding.get("point"), dict) else {}
        summary["grounding"] = {
            "status": grounding.get("status"),
            "target": grounding.get("target"),
            "point": {
                "normalized": point.get("normalized"),
                "framePx": point.get("framePx"),
            },
        }
    if isinstance(execution.get("point"), dict):
        point = execution["point"]
        summary["point"] = {
            "normalized": point.get("normalized"),
            "screenshotPx": point.get("screenshotPx"),
            "hidU16": point.get("hidU16"),
        }
    for result_key in ("tap", "typeResult", "pressResult", "keyResult", "clearResult", "scrollResult", "openUrlResult", "waitResult"):
        value = execution.get(result_key)
        if isinstance(value, dict):
            summary[result_key] = {
                key: value.get(key)
                for key in ("attempted", "dryRun", "deliveryStatus", "completionStatus", "reason", "convertedText", "requestedButton", "button", "key", "direction", "url", "strategy", "waitedMs")
                if key in value
            }
    substeps = execution.get("substeps")
    if isinstance(substeps, list):
        summary["substeps"] = [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "result": _execution_trace_summary(item.get("result")) if isinstance(item, dict) and isinstance(item.get("result"), dict) else None,
            }
            for item in substeps
            if isinstance(item, dict)
        ]
    return {key: value for key, value in summary.items() if value is not None}


def _trace_result_summary(data: dict[str, Any]) -> dict[str, Any]:
    if not data.get("ok"):
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        return {
            "status": "error",
            "error": {
                "code": error.get("code"),
                "category": error.get("category"),
                "stage": error.get("stage"),
                "message": error.get("message"),
                "retryable": error.get("retryable"),
            },
        }
    result = data.get("result")
    if not isinstance(result, dict):
        return {"status": "ok", "resultType": type(result).__name__}
    summary: dict[str, Any] = {
        "status": result.get("status") or "ok",
        "schema": result.get("schema"),
        "artifactDir": result.get("artifactDir"),
    }
    if data.get("command") == "step":
        summary.update(
            {
                "action": _action_trace_summary(result.get("action")),
                "before": _observation_trace_summary(result.get("before")),
                "execution": _execution_trace_summary(result.get("execution")),
            }
        )
    elif data.get("command") == "observe":
        summary["observation"] = _observation_trace_summary(result)
    elif data.get("command") in {"assert", "wait"}:
        summary["assertion"] = {
            "text": result.get("text"),
            "matched": result.get("matched"),
            "attempts": result.get("attempts"),
            "artifactDir": result.get("artifactDir"),
        }
    return {key: value for key, value in summary.items() if value is not None}


def record_trace(args: argparse.Namespace, data: dict[str, Any], *, argv: list[str], cwd: str | None = None) -> dict[str, Any] | None:
    paths = _trace_paths(args)
    if paths is None:
        return None
    trace_dir = paths["dir"]
    trace_dir.mkdir(parents=True, exist_ok=True)
    events = _read_jsonl(paths["events"])
    sequence = len(events) + 1
    event_name = f"event-{sequence:06d}"
    response_path = trace_dir / f"{event_name}.response.json"
    trace_id = trace_dir.name
    now = _utc_now_iso()

    event = {
        "schema": TRACE_EVENT_SCHEMA,
        "traceId": trace_id,
        "sequence": sequence,
        "timestamp": now,
        "cwd": cwd or str(Path.cwd()),
        "argv": argv,
        "command": data.get("command"),
        "requestId": data.get("requestId"),
        "ok": bool(data.get("ok")),
        "durationMs": data.get("durationMs"),
        "summary": _trace_result_summary(data),
        "responsePath": str(response_path),
    }
    write_json(response_path, data)
    with paths["events"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    existing = {}
    if paths["trace"].exists():
        try:
            parsed = json.loads(paths["trace"].read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing = parsed
        except json.JSONDecodeError:
            existing = {}
    trace = {
        "schema": TRACE_SCHEMA,
        "id": trace_id,
        "title": _trace_title(args) or existing.get("title"),
        "createdAt": existing.get("createdAt") or now,
        "updatedAt": now,
        "artifactRoot": str(_trace_artifact_root(args)),
        "eventCount": sequence,
        "eventsPath": str(paths["events"]),
        "latestResponsePath": str(response_path),
    }
    write_json(paths["trace"], trace)
    return {
        "schema": "coretap.trace.ref.v1",
        "id": trace_id,
        "title": trace.get("title"),
        "dir": str(trace_dir),
        "tracePath": str(paths["trace"]),
        "eventsPath": str(paths["events"]),
        "eventSequence": sequence,
        "responsePath": str(response_path),
    }


def attach_trace(args: argparse.Namespace, data: dict[str, Any], *, argv: list[str], cwd: str | None = None) -> dict[str, Any]:
    try:
        trace = record_trace(args, data, argv=argv, cwd=cwd)
        if trace is not None:
            data["trace"] = trace
            write_json(Path(trace["responsePath"]), data)
    except Exception as exc:  # Trace logging must not break the device action.
        warnings = data.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append({"code": "TRACE_WRITE_FAILED", "message": str(exc), "stage": "trace"})
        return data
    return data


def _compact_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    compact = {key: value for key, value in action.items() if key != "schema" and value is not None}
    return compact


def _compact_frame(frame: Any) -> dict[str, Any] | None:
    if not isinstance(frame, dict):
        return None
    keys = ("path", "widthPx", "heightPx", "resized", "maxLongSidePx", "sha256")
    return {key: frame[key] for key in keys if key in frame}


def _compact_execution(execution: Any) -> dict[str, Any] | None:
    if not isinstance(execution, dict):
        return None
    compact: dict[str, Any] = {
        "status": execution.get("status"),
        "actionType": execution.get("actionType"),
    }
    if execution.get("strategy"):
        compact["strategy"] = execution.get("strategy")
    point = execution.get("point")
    if isinstance(point, dict):
        compact["point"] = {
            key: point.get(key)
            for key in ("normalized", "screenshotPx", "hidU16")
            if key in point
        }
    grounding = execution.get("grounding")
    if isinstance(grounding, dict):
        gpoint = grounding.get("point") if isinstance(grounding.get("point"), dict) else {}
        compact["grounding"] = {
            "status": grounding.get("status"),
            "source": grounding.get("source"),
            "point": {
                key: gpoint.get(key)
                for key in ("normalized", "framePx")
                if key in gpoint
            },
        }
    for result_key in ("tap", "scrollResult", "pressResult", "keyResult", "clearResult", "typeResult", "waitResult"):
        value = execution.get(result_key)
        if isinstance(value, dict):
            compact[result_key] = _compact_nested_result(value)
    return {key: value for key, value in compact.items() if value is not None}


def _compact_nested_result(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "status",
        "attempted",
        "dryRun",
        "dispatchStatus",
        "deliveryStatus",
        "confirmationStatus",
        "direction",
        "distance",
        "button",
        "key",
        "count",
        "waitedMs",
        "typedCharacters",
        "verified",
    }
    result = {key: value[key] for key in allowed if key in value}
    if "from" in value and isinstance(value["from"], dict):
        result["from"] = {"normalized": value["from"].get("normalized")}
    if "to" in value and isinstance(value["to"], dict):
        result["to"] = {"normalized": value["to"].get("normalized")}
    if "anchor" in value:
        result["anchor"] = value["anchor"]
    return result


def _debug_ref(data: dict[str, Any], result: Any) -> dict[str, Any] | None:
    debug: dict[str, Any] = {}
    if isinstance(result, dict):
        artifact_dir = result.get("artifactDir")
        if isinstance(artifact_dir, str):
            debug["artifactDir"] = artifact_dir
            if data.get("command") == "step":
                debug["resultPath"] = str(Path(artifact_dir) / "step.result.json")
            elif data.get("command") == "observe":
                debug["resultPath"] = str(Path(artifact_dir) / "observe.result.json")
            elif data.get("command") in {"assert", "wait"}:
                debug["resultPath"] = str(Path(artifact_dir) / "assert-text.result.json")
    trace = data.get("trace")
    if isinstance(trace, dict):
        debug["tracePath"] = trace.get("tracePath")
        debug["responsePath"] = trace.get("responsePath")
    return {key: value for key, value in debug.items() if value} or None


def _compact_status(result_status: Any, *, ok: bool) -> str:
    if not ok:
        return "error"
    if not isinstance(result_status, str):
        return "success"
    if result_status in (None, "ok", "executed", "satisfied"):
        return "success"
    return result_status


def compact_response(data: dict[str, Any]) -> dict[str, Any]:
    ok = bool(data.get("ok"))
    result = data.get("result")
    result_status = result.get("status") if isinstance(result, dict) else None
    compact: dict[str, Any] = {
        "ok": ok,
        "command": data.get("command"),
        "status": _compact_status(result_status, ok=ok),
        "requestId": data.get("requestId"),
        "durationMs": data.get("durationMs"),
    }
    if not ok:
        error = data.get("error") if isinstance(data.get("error"), dict) else {}
        compact["error"] = {
            key: error.get(key)
            for key in ("code", "category", "stage", "message", "retryable")
            if key in error
        }
        debug = _debug_ref(data, error.get("details") if isinstance(error, dict) else None)
        if debug:
            compact["debug"] = debug
        return {key: value for key, value in compact.items() if value is not None}

    if data.get("command") == "step" and isinstance(result, dict):
        compact.update(_compact_step_response(data, result))
    elif data.get("command") == "screenshot" and isinstance(result, dict):
        compact["frame"] = _compact_frame(result.get("frame"))
        compact["artifactDir"] = result.get("artifactDir")
    elif data.get("command") == "observe" and isinstance(result, dict):
        compact["observation"] = page_observation_summary(result)
        compact["artifactDir"] = result.get("artifactDir")
    elif data.get("command") in {"assert", "wait"} and isinstance(result, dict):
        compact["matched"] = result.get("matched")
        compact["expected"] = result.get("expected")
        compact["attempts"] = result.get("attempts")
        compact["match"] = {
            key: result.get(key)
            for key in ("matchedText", "matchedEngines", "matchedTokenMeanConfidence", "matchedBoxPx")
            if key in result
        }
        compact["artifactDir"] = result.get("artifactDir")
    else:
        compact["result"] = result

    debug = _debug_ref(data, result)
    if debug:
        compact["debug"] = debug
    warnings = data.get("warnings")
    if warnings:
        compact["warnings"] = warnings
    return {key: value for key, value in compact.items() if value is not None}


def _compact_step_response(data: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "artifactDir": result.get("artifactDir"),
        "action": _compact_action(result.get("action")),
        "execution": _compact_execution(result.get("execution")),
    }
    observation = result.get("observation")
    if isinstance(observation, dict):
        compact["observation"] = observation
    return compact


def _element_bbox(element: dict[str, Any]) -> dict[str, float] | None:
    bbox = element.get("bbox")
    if not isinstance(bbox, dict):
        return None
    try:
        x = float(bbox["x"])
        y = float(bbox["y"])
        width = float(bbox["width"])
        height = float(bbox["height"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _union_bbox(items: list[dict[str, Any]]) -> dict[str, float] | None:
    boxes = [box for box in (_element_bbox(item) for item in items) if box is not None]
    if not boxes:
        return None
    left = min(box["x"] for box in boxes)
    top = min(box["y"] for box in boxes)
    right = max(box["x"] + box["width"] for box in boxes)
    bottom = max(box["y"] + box["height"] for box in boxes)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def _bbox_center(bbox: dict[str, float] | None) -> dict[str, float] | None:
    if not bbox:
        return None
    return {"x": bbox["x"] + bbox["width"] / 2, "y": bbox["y"] + bbox["height"] / 2}


def _mean_confidence(items: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for item in items:
        try:
            values.append(float(item["confidence"]))
        except (KeyError, TypeError, ValueError):
            pass
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _text_group(label: str, items: list[dict[str, Any]], *, group_type: str) -> dict[str, Any]:
    bbox = _union_bbox(items)
    result = {
        "type": group_type,
        "label": label,
        "center": _bbox_center(bbox),
        "bbox": bbox,
        "tokenIndexes": [item.get("index") for item in items if item.get("index") is not None],
        "confidence": _mean_confidence(items),
    }
    return {key: value for key, value in result.items() if value is not None}


def _line_group_threshold(current: list[dict[str, Any]], candidate: dict[str, Any]) -> float:
    heights = []
    for item in [*current, candidate]:
        box = _element_bbox(item)
        if box:
            heights.append(box["height"])
    height = sorted(heights)[len(heights) // 2] if heights else 0.018
    return max(0.018, height * 0.9)


def _group_text_lines(elements: list[dict[str, Any]], *, max_lines: int = 60) -> list[dict[str, Any]]:
    positioned = [item for item in elements if _element_bbox(item) is not None]
    positioned.sort(key=lambda item: ((_element_bbox(item) or {}).get("y", 0), (_element_bbox(item) or {}).get("x", 0)))
    raw_lines: list[list[dict[str, Any]]] = []
    for item in positioned:
        center = item.get("center") if isinstance(item.get("center"), dict) else None
        try:
            y = float(center["y"]) if center else (_element_bbox(item) or {})["y"]
        except (KeyError, TypeError, ValueError):
            continue
        if not raw_lines:
            raw_lines.append([item])
            continue
        last = raw_lines[-1]
        last_centers = [float(member.get("center", {}).get("y")) for member in last if isinstance(member.get("center"), dict)]
        last_y = sum(last_centers) / len(last_centers) if last_centers else y
        if abs(y - last_y) <= _line_group_threshold(last, item):
            last.append(item)
        else:
            raw_lines.append([item])

    lines: list[dict[str, Any]] = []
    for raw in raw_lines[:max_lines]:
        raw.sort(key=lambda item: (_element_bbox(item) or {}).get("x", 0))
        label = " ".join(str(item.get("label") or "").strip() for item in raw if str(item.get("label") or "").strip())
        if label:
            lines.append(_text_group(label, raw, group_type="textLine"))
    return lines


_BUTTON_TEXTS = {
    "allow",
    "cancel",
    "close",
    "continue",
    "done",
    "download",
    "get",
    "install",
    "next",
    "ok",
    "open",
    "paste",
    "retry",
    "save",
    "search",
    "update",
    "允许",
    "取消",
    "关闭",
    "继续",
    "完成",
    "下载",
    "获取",
    "安装",
    "下一步",
    "好",
    "确定",
    "打开",
    "粘贴",
    "重试",
    "保存",
    "搜索",
    "更新",
}


def _label_suggests_button(label: str) -> bool:
    normalized = normalize_text(label)
    compact = re.sub(r"\s+", "", normalized)
    return normalized in _BUTTON_TEXTS or compact in _BUTTON_TEXTS


def _line_suggests_button(line: dict[str, Any]) -> bool:
    label = str(line.get("label") or "").strip()
    return _label_suggests_button(label)


def _line_suggests_input(line: dict[str, Any]) -> bool:
    label = str(line.get("label") or "").strip()
    normalized = normalize_text(label)
    compact = re.sub(r"\s+", "", normalized)
    if re.match(r"^(q|◎|🔍|⌕)\s*", label.casefold()):
        return True
    if any(marker in compact for marker in ("search", "搜索", "输入", "查找")):
        return True
    return "游戏" in compact and ("app" in compact or "应用" in compact or "故事" in compact)


def _line_vertical_gap(previous: dict[str, Any], current: dict[str, Any]) -> float | None:
    prev_box = previous.get("bbox") if isinstance(previous.get("bbox"), dict) else None
    cur_box = current.get("bbox") if isinstance(current.get("bbox"), dict) else None
    if not prev_box or not cur_box:
        return None
    try:
        return float(cur_box["y"]) - (float(prev_box["y"]) + float(prev_box["height"]))
    except (KeyError, TypeError, ValueError):
        return None


def _row_height(lines: list[dict[str, Any]]) -> float:
    boxes = [line.get("bbox") for line in lines if isinstance(line.get("bbox"), dict)]
    if not boxes:
        return 0.0
    top = min(float(box["y"]) for box in boxes)
    bottom = max(float(box["y"]) + float(box["height"]) for box in boxes)
    return bottom - top


def _group_list_rows(lines: list[dict[str, Any]], *, max_rows: int = 40) -> list[dict[str, Any]]:
    sorted_lines = [line for line in lines if isinstance(line.get("bbox"), dict)]
    sorted_lines.sort(key=lambda line: (float(line["bbox"]["y"]), float(line["bbox"]["x"])))
    rows: list[list[dict[str, Any]]] = []
    for line in sorted_lines:
        if not rows:
            rows.append([line])
            continue
        current = rows[-1]
        gap = _line_vertical_gap(current[-1], line)
        tentative = [*current, line]
        if gap is not None and gap <= 0.045 and _row_height(tentative) <= 0.14:
            current.append(line)
        else:
            rows.append([line])

    result: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        label = " / ".join(str(line.get("label") or "").strip() for line in row if str(line.get("label") or "").strip())
        if not label:
            continue
        pseudo_items = [
            {
                "index": token_index,
                "label": line.get("label"),
                "confidence": line.get("confidence"),
                "center": line.get("center"),
                "bbox": line.get("bbox"),
            }
            for line in row
            for token_index in (line.get("tokenIndexes") or [None])
        ]
        result.append(_text_group(label, pseudo_items, group_type="listRowCandidate"))
    return result


def _visual_group_items(elements: list[dict[str, Any]], roles: set[str], *, group_type: str, limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in elements:
        if item.get("source") != "vlm" or item.get("role") not in roles:
            continue
        center = item.get("center") if isinstance(item.get("center"), dict) else {}
        key = (str(item.get("label") or ""), f"{center.get('x')}:{center.get('y')}")
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item, type=group_type))
        if len(result) >= limit:
            break
    return result


def _observation_groups(elements: list[dict[str, Any]]) -> dict[str, Any]:
    text_elements = [item for item in elements if item.get("source") == "ocr" or item.get("type") == "text"]
    visual_elements = [item for item in elements if item.get("source") == "vlm"]
    lines = _group_text_lines(text_elements)
    buttons = [dict(item, type="buttonCandidate") for item in text_elements if _label_suggests_button(str(item.get("label") or ""))]
    buttons.extend(dict(line, type="buttonCandidate") for line in lines if _line_suggests_button(line))
    inputs = [dict(line, type="inputCandidate") for line in lines if _line_suggests_input(line)]
    rows = _group_list_rows(lines)
    deduped_buttons: list[dict[str, Any]] = []
    seen_buttons: set[tuple[str, str]] = set()
    for button in buttons:
        center = button.get("center") if isinstance(button.get("center"), dict) else {}
        key = (str(button.get("label") or ""), f"{center.get('x')}:{center.get('y')}")
        if key in seen_buttons:
            continue
        seen_buttons.add(key)
        deduped_buttons.append(button)
    return {
        "textLines": lines,
        "buttons": deduped_buttons[:30],
        "inputs": inputs[:20],
        "rows": rows[:40],
        "visualButtons": _visual_group_items(visual_elements, {"button", "toggle", "navigation"}, group_type="visualButton", limit=40),
        "appIcons": _visual_group_items(visual_elements, {"appIcon"}, group_type="appIcon", limit=40),
        "tabs": _visual_group_items(visual_elements, {"tab"}, group_type="tab", limit=20),
    }


def page_observation_summary(observation: dict[str, Any], *, max_elements: int = 80) -> dict[str, Any]:
    ocr = observation.get("ocr") if isinstance(observation, dict) else None
    visual = observation.get("visual") if isinstance(observation, dict) else None
    frame = observation.get("frame") if isinstance(observation, dict) else None
    ocr_enabled = isinstance(ocr, dict) and bool(ocr.get("enabled"))
    visual_enabled = isinstance(visual, dict) and bool(visual.get("enabled"))
    summary: dict[str, Any] = {
        "status": "ready" if ocr_enabled or visual_enabled else "no_observation",
        "frame": _compact_frame(frame),
    }
    elements: list[dict[str, Any]] = []
    visible_texts: list[str] = []
    seen_texts: set[str] = set()
    frame_width = float(frame.get("widthPx") or 0) if isinstance(frame, dict) else 0.0
    frame_height = float(frame.get("heightPx") or 0) if isinstance(frame, dict) else 0.0
    tokens = [item for item in ((ocr or {}).get("tokens") or []) if isinstance(item, dict)] if ocr_enabled else []
    if ocr_enabled:
        for fallback_index, token in enumerate(tokens[:max_elements]):
            text = str(token.get("text") or "").strip()
            if not text:
                continue
            normalized = token.get("normalized") if isinstance(token.get("normalized"), dict) else None
            bbox = token.get("bboxNormalized") if isinstance(token.get("bboxNormalized"), dict) else None
            if bbox is None and isinstance(token.get("bboxPx"), dict) and frame_width > 0 and frame_height > 0:
                bbox_px = token["bboxPx"]
                try:
                    bbox = {
                        "x": float(bbox_px["x"]) / frame_width,
                        "y": float(bbox_px["y"]) / frame_height,
                        "width": float(bbox_px["width"]) / frame_width,
                        "height": float(bbox_px["height"]) / frame_height,
                    }
                except (KeyError, TypeError, ValueError):
                    bbox = None
            elements.append(
                {
                    "index": token.get("index", fallback_index),
                    "type": "text",
                    "source": "ocr",
                    "label": text,
                    "confidence": token.get("confidence"),
                    "center": normalized,
                    "bbox": bbox,
                }
            )
            dedupe_key = normalize_text(text)
            if dedupe_key and dedupe_key not in seen_texts:
                seen_texts.add(dedupe_key)
                visible_texts.append(text)
    visual_elements = [item for item in ((visual or {}).get("elements") or []) if isinstance(item, dict)] if visual_enabled else []
    elements.extend(visual_elements[:max_elements])
    visual_summary = str((visual or {}).get("summary") or "").strip() if isinstance(visual, dict) else ""
    text_summary = "可见文本：" + " / ".join(visible_texts[:24]) if visible_texts else "未识别到可见文本"
    if visual_summary and visible_texts:
        summary["summary"] = f"{visual_summary}；{text_summary}"
    elif visual_summary:
        summary["summary"] = visual_summary
    else:
        summary["summary"] = text_summary
    summary["elements"] = elements
    summary["groups"] = _observation_groups(elements)
    summary["textCount"] = len(tokens)
    summary["visualCount"] = len(visual_elements)
    plain_text = (ocr or {}).get("plainText") if ocr_enabled else None
    if isinstance(plain_text, str) and plain_text:
        summary["plainText"] = plain_text
    return {key: value for key, value in summary.items() if value is not None}


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
                "ocr": {"profile": "builtin:macos-vision-zh-hans-en-us@1", "lang": DEFAULT_OCR_LANG},
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
    ocr = vision_ocr_status()
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
    ocr = vision_ocr_status()
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
    existing = getattr(args, "_artifact_run_dir", None)
    if existing:
        run_dir = Path(str(existing))
        out = Path(args.out) if getattr(args, "out", None) else run_dir / f"{label}.png"
        frame = _capture_to(args, label=label, run_dir=run_dir, out=out)
        return frame, run_dir, out
    root = _persistent_artifact_root(args)
    if root is None:
        raise CoretapError(
            "ARTIFACT_CONTEXT_REQUIRED",
            "capture requires an active artifact context or persistent artifacts",
            category="internal",
            stage="artifact",
        )
    run_dir = artifact_dir(root)
    args._artifact_run_dir = str(run_dir)
    args._artifacts_persistent = True
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


def _text_entry_anchor_cache_path() -> Path:
    return ensure_state()["state"] / "text-entry-anchors.json"


def _text_entry_anchor_key(args: argparse.Namespace) -> str:
    backend = str(getattr(args, "backend", "") or "device")
    device = str(getattr(args, "device", "") or "default")
    return f"{backend}:{device}"


def _load_text_entry_anchor_cache() -> dict[str, Any]:
    path = _text_entry_anchor_cache_path()
    if not path.exists():
        return {"schema": TEXT_ENTRY_ANCHOR_CACHE_SCHEMA, "anchors": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": TEXT_ENTRY_ANCHOR_CACHE_SCHEMA, "anchors": {}}
    if not isinstance(data, dict):
        return {"schema": TEXT_ENTRY_ANCHOR_CACHE_SCHEMA, "anchors": {}}
    anchors = data.get("anchors")
    if not isinstance(anchors, dict):
        data["anchors"] = {}
    data["schema"] = TEXT_ENTRY_ANCHOR_CACHE_SCHEMA
    return data


def _target_suggests_text_entry(target: str | None) -> bool:
    if not target:
        return False
    raw = str(target)
    haystack = f"{raw} {normalize_text(raw)}".casefold()
    if any(marker in haystack for marker in NON_TEXT_ENTRY_CONTEXT_MARKERS):
        return False
    return any(marker in haystack for marker in TEXT_ENTRY_TARGET_MARKERS)


def _target_suggests_top_text_entry(target: str | None, point: dict[str, Any] | None = None) -> bool:
    if not _target_suggests_text_entry(target):
        return False
    raw = str(target or "")
    haystack = f"{raw} {normalize_text(raw)}".casefold()
    normalized = point.get("normalized") if isinstance(point, dict) else None
    try:
        point_y = float(normalized["y"]) if isinstance(normalized, dict) else None
    except (KeyError, TypeError, ValueError):
        point_y = None
    if any(marker in haystack for marker in TOP_TEXT_ENTRY_TARGET_MARKERS):
        return point_y is None or point_y < 0.35
    if point_y is not None and point_y < 0.24 and any(marker in haystack for marker in ("search", "搜索")):
        return True
    return False


def _target_suggests_relocated_search_entry(target: str | None, point: dict[str, Any] | None = None) -> bool:
    if not _target_suggests_text_entry(target):
        return False
    raw = str(target or "")
    haystack = f"{raw} {normalize_text(raw)}".casefold()
    if not any(marker in haystack for marker in ("search", "搜索", "address", "地址")):
        return False
    normalized = point.get("normalized") if isinstance(point, dict) else None
    try:
        point_y = float(normalized["y"]) if isinstance(normalized, dict) else None
    except (KeyError, TypeError, ValueError):
        return False
    return point_y >= 0.70


def _remember_text_entry_anchor(
    args: argparse.Namespace,
    point: dict[str, Any],
    *,
    source: str,
    action_type: str,
    target: str | None = None,
) -> dict[str, Any] | None:
    if getattr(args, "dry_run", False):
        return None
    if action_type == "tap" and not _target_suggests_text_entry(target):
        return None
    normalized = point.get("normalized") if isinstance(point, dict) else None
    if not isinstance(normalized, dict):
        return None
    try:
        x = float(normalized["x"])
        y = float(normalized["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    anchor = {
        "schema": TEXT_ENTRY_ANCHOR_SCHEMA,
        "source": source,
        "actionType": action_type,
        "target": target,
        "backend": str(getattr(args, "backend", "") or "device"),
        "device": str(getattr(args, "device", "") or "default"),
        "point": {"x": x, "y": y},
        "updatedAt": _utc_now_iso(),
        "updatedAtEpochMs": round(time.time() * 1000),
    }
    cache = _load_text_entry_anchor_cache()
    anchors = cache.setdefault("anchors", {})
    anchors[_text_entry_anchor_key(args)] = anchor
    write_json(_text_entry_anchor_cache_path(), cache)
    return anchor


def _last_text_entry_anchor(args: argparse.Namespace) -> dict[str, Any] | None:
    cache = _load_text_entry_anchor_cache()
    anchors = cache.get("anchors")
    if not isinstance(anchors, dict):
        return None
    anchor = anchors.get(_text_entry_anchor_key(args))
    if not isinstance(anchor, dict):
        return None
    point = anchor.get("point")
    if not isinstance(point, dict):
        return None
    try:
        x = float(point["x"])
        y = float(point["y"])
    except (KeyError, TypeError, ValueError):
        return None
    updated_at = anchor.get("updatedAtEpochMs")
    try:
        age_ms = round(time.time() * 1000) - int(updated_at)
    except (TypeError, ValueError):
        age_ms = TEXT_ENTRY_ANCHOR_MAX_AGE_MS + 1
    if age_ms > TEXT_ENTRY_ANCHOR_MAX_AGE_MS:
        return None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return {**anchor, "point": {"x": x, "y": y}, "ageMs": age_ms}


def _is_non_ascii_text(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _is_unshifted_virtual_keyboard_text(text: str) -> bool:
    if any(ch.isspace() for ch in text):
        return False
    from pymobiledevice3.remote.core_device.hid_service import ASCII_TO_HID

    for ch in text:
        mapping = ASCII_TO_HID.get(ch)
        if mapping is None:
            return False
        _usage, needs_shift = mapping
        if needs_shift and not ch.isalpha():
            return False
    return True


def _screenshot_into(args: argparse.Namespace, *, run_dir: Path, label: str, out: Path | None = None) -> dict[str, Any]:
    captured_at = _now_iso()
    output_path = out or (Path(args.out) if getattr(args, "out", None) else run_dir / f"{label}.png")
    if getattr(args, "full_size", False):
        frame = _capture_to(args, label=label, run_dir=run_dir, out=output_path)
        result = {
            **_artifact_result(args, run_dir),
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
        **_artifact_result(args, run_dir),
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
    try:
        tokens, raw = run_ocr(image)
        return tokens, raw, DEFAULT_OCR_ENGINE
    except CoretapError as exc:
        raw = {"engines": [], "errors": [_ocr_error_details(exc, engine=DEFAULT_OCR_ENGINE)]}
        _raise_ocr_unavailable(image, raw)


def _run_observe_visual(image: Path, args: argparse.Namespace) -> dict[str, Any]:
    profile = getattr(args, "profile", PUBLIC_MODEL_PROFILE)
    if profile == INTERNAL_FIXTURE_PROFILE:
        return {
            "schema": "coretap.visual.observe.v1",
            "enabled": False,
            "status": "skipped",
            "reason": "profile does not support VLM visual observe",
            "profile": profile,
            "promptVersion": PUBLIC_VISUAL_OBSERVE_PROMPT_VERSION,
            "summary": "",
            "elements": [],
        }
    try:
        return run_visual_observe_model(image, profile=profile)
    except CoretapError as exc:
        if exc.code != "MODEL_WORKER_CRASHED" or not exc.retryable:
            raise
        warm_model(profile)
        result = run_visual_observe_model(image, profile=profile)
        result["modelRecovery"] = {
            "schema": "coretap.model-recovery.v1",
            "recoveredFrom": exc.code,
            "attempts": 2,
        }
        return result


VISUAL_OCR_TEXT_OVERLAP_DROP_RATIO = 0.45


def _as_rect_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_rect(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    x = _as_rect_number(value.get("x"))
    y = _as_rect_number(value.get("y"))
    width = _as_rect_number(value.get("width"))
    height = _as_rect_number(value.get("height"))
    if x is None or y is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    left = max(0.0, min(1.0, x))
    top = max(0.0, min(1.0, y))
    right = max(0.0, min(1.0, x + width))
    bottom = max(0.0, min(1.0, y + height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def _filter_visual_elements_against_ocr(visual: dict[str, Any], ocr_tokens: list[dict[str, Any]]) -> dict[str, Any]:
    elements = visual.get("elements")
    if not isinstance(elements, list) or not elements or not ocr_tokens:
        return visual
    text_rects = [
        rect
        for token in ocr_tokens
        if isinstance(token, dict)
        for rect in [_normalized_rect(token.get("bboxNormalized"))]
        if rect is not None
    ]
    if not text_rects:
        return visual

    kept: list[Any] = []
    removed_count = 0
    for element in elements:
        if not isinstance(element, dict):
            kept.append(element)
            continue
        rect = _normalized_rect(element.get("bbox"))
        if rect is None:
            kept.append(element)
            continue
        area = _rect_area(rect)
        overlap = min(area, sum(_rect_overlap_area(rect, text_rect) for text_rect in text_rects))
        if area > 0 and overlap / area >= VISUAL_OCR_TEXT_OVERLAP_DROP_RATIO:
            removed_count += 1
            continue
        kept.append(element)

    if removed_count == 0:
        return visual
    filtered = dict(visual)
    filtered["elements"] = kept
    filtered["ocrFilteredElementCount"] = removed_count
    return filtered


def _observe_into(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    label: str,
    no_ocr: bool | None = None,
    no_vlm: bool | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    ns = argparse.Namespace(**vars(args))
    ns.label = label
    ns.out = str(out) if out else None
    screenshot = _screenshot_into(ns, run_dir=run_dir, label=label, out=out)
    frame = screenshot["frame"]
    result: dict[str, Any] = {
        "schema": "coretap.observe.result.v1",
        "frame": frame,
        "sourceFrame": screenshot.get("sourceFrame"),
        **_artifact_result(args, run_dir),
    }
    should_skip_ocr = bool(getattr(args, "no_ocr", False) if no_ocr is None else no_ocr)
    token_json: list[dict[str, Any]] = []
    if should_skip_ocr:
        result["ocr"] = {"enabled": False}
    else:
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
            "lang": DEFAULT_OCR_LANG,
            "engineMode": DEFAULT_OCR_ENGINE,
            "selectedEngine": selected_engine,
            "minConfidence": args.min_confidence,
            "tokenCount": len(token_json),
            "rawTokenCount": len(tokens),
            "plainText": "\n".join(token["text"] for token in token_json),
            **_ocr_summary(raw),
            "tokens": token_json,
        }

    image = Path(frame["path"])
    should_skip_vlm = bool(getattr(args, "no_vlm", False) if no_vlm is None else no_vlm)
    if should_skip_vlm:
        result["visual"] = {"enabled": False}
    else:
        visual = _run_observe_visual(image, args)
        if not should_skip_ocr:
            visual = _filter_visual_elements_against_ocr(visual, token_json)
        result["visual"] = visual
        write_json(run_dir / f"{label}.visual.json", visual)
    write_json(run_dir / f"{label}.observe.result.json", result)
    return result


def command_observe(args: argparse.Namespace) -> dict[str, Any]:
    with _command_artifacts(args) as run_dir:
        result = _observe_into(args, run_dir=run_dir, label=args.label)
        write_json(run_dir / "observe.result.json", result)
        return result


@contextmanager
def _screenshot_artifacts(args: argparse.Namespace):
    if getattr(args, "out", None):
        with _command_artifacts(args) as run_dir:
            yield run_dir
        return
    if getattr(args, "no_artifacts", False):
        raise CoretapError(
            "SCREENSHOT_OUTPUT_REQUIRED",
            "screenshot requires --out when --no-artifacts is set",
            category="usage",
            stage="screenshot",
        )
    root = _persistent_artifact_root(args) or ensure_state()["artifacts"]
    run_dir = artifact_dir(root)
    previous_dir = getattr(args, "_artifact_run_dir", None)
    previous_persistent = getattr(args, "_artifacts_persistent", None)
    args._artifact_run_dir = str(run_dir)
    args._artifacts_persistent = True
    try:
        yield run_dir
    finally:
        if previous_dir is None:
            with suppress(AttributeError):
                delattr(args, "_artifact_run_dir")
        else:
            args._artifact_run_dir = previous_dir
        if previous_persistent is None:
            with suppress(AttributeError):
                delattr(args, "_artifacts_persistent")
        else:
            args._artifacts_persistent = previous_persistent


def command_screenshot(args: argparse.Namespace) -> dict[str, Any]:
    label = str(getattr(args, "label", None) or "screenshot")
    with _screenshot_artifacts(args) as run_dir:
        output_path = Path(args.out).expanduser() if getattr(args, "out", None) else run_dir / f"{label}.png"
        frame = _capture_to(args, label=label, run_dir=run_dir, out=output_path)
        frame_json = {
            **_frame_json(frame),
            "resized": False,
            "maxLongSidePx": None,
            "scale": 1.0,
        }
        result = {
            "schema": "coretap.screenshot.result.v1",
            **_artifact_result(args, run_dir),
            "frame": frame_json,
        }
        write_json(run_dir / f"{label}.screenshot.result.json", result)
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
    is_match: Any,
) -> tuple[list[Any], dict[str, Any]]:
    tokens, raw = run_ocr(image)
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
    with _command_artifacts(args) as run_dir:
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
        return _verify_type_result(args, text, result, run_dir)


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


def _verify_type_result(args: argparse.Namespace, text: str, result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    deadline = time.monotonic() + (args.verify_timeout_ms / 1000)
    attempts = 0
    last: dict[str, Any] | None = None
    while True:
        attempts += 1
        capture_args = argparse.Namespace(**vars(args))
        capture_args.out = None
        image = run_dir / "type-verify.png"
        frame = _capture_to(capture_args, label="type-verify", run_dir=run_dir, out=image)
        tokens, raw = _run_ocr_progressive(
            image,
            is_match=lambda current: bool(find_exact_text_candidates(current, text, min_confidence=25.0)),
        )
        _write_ocr_artifacts(run_dir, "type-verify", raw)
        candidates = find_exact_text_candidates(tokens, text, min_confidence=25.0)
        last = {
            "frame": {"path": str(image), "widthPx": frame.width, "heightPx": frame.height},
            "attempts": attempts,
            "tokenCount": len(tokens),
            "candidateCount": len(candidates),
            "ocr": _ocr_summary(raw),
            **_artifact_result(args, run_dir),
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
    with _command_artifacts(args) as run_dir:
        return _command_assert_text_into(args, run_dir)


def _command_assert_text_into(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
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
        tokens, raw = _run_ocr_progressive(
            image,
            is_match=lambda current: bool(find_text(current, args.text, case_sensitive=args.case_sensitive)),
        )
        min_confidence = float(getattr(args, "min_confidence", 0.0) or 0.0)
        if min_confidence > 0:
            tokens = [token for token in tokens if token.confidence >= min_confidence]
        _write_ocr_artifacts(run_dir, f"assert-{attempts:03d}", raw)
        match = find_text(tokens, args.text, case_sensitive=args.case_sensitive)
        last = {
            "attempts": attempts,
            "image": str(image),
            "frame": frame_info,
            "tokenCount": len(tokens),
            "ocr": {"lang": DEFAULT_OCR_LANG, "selectedEngine": DEFAULT_OCR_ENGINE, **_ocr_summary(raw)},
            "match": match,
        }
        if match:
            result = {
                **_artifact_result(args, run_dir),
                "expected": args.text,
                "matched": True,
                "frame": frame_info,
                "ocr": {"lang": DEFAULT_OCR_LANG, "selectedEngine": DEFAULT_OCR_ENGINE, "tokenCount": len(tokens), **_ocr_summary(raw)},
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
    "tapPoint",
    "longPress",
    "openApp",
    "openUrl",
    "typeText",
    "key",
    "clear",
    "press",
    "scroll",
    "appSwitcher",
    "terminateApp",
    "uninstallApp",
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
    if schema not in (None, "", "coretap.action.v2"):
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            f"Unsupported action schema: {schema}",
            category="usage",
            stage="step-action",
            details={"schema": schema, "supported": ["coretap.action.v2"], "note": "schema is optional for step actions"},
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


def _point_pair_from_action(value: Any, *, key: str, stage: str) -> str | dict[str, Any]:
    if isinstance(value, str):
        _parse_xy_pair(value, option=key)
        return value
    if isinstance(value, dict):
        x = _number(value.get("x"), key=f"{key}.x", stage=stage)
        y = _number(value.get("y"), key=f"{key}.y", stage=stage)
        point: dict[str, Any] = {"x": x, "y": y}
        source = str(value.get("source") or "").strip()
        if source:
            point["source"] = source
        return point
    raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must be x,y or an object", category="usage", stage=stage)


def _point_from_action(raw: dict[str, Any], *, stage: str, key: str = "point") -> dict[str, Any]:
    payload = raw.get(key)
    if payload is None and key == "point":
        payload = raw
    if isinstance(payload, str):
        x, y = _parse_xy_pair(payload, option=key)
        space = str(raw.get("space") or "normalized")
        reference = str(raw.get("reference") or "source")
    elif isinstance(payload, dict):
        x = _number(payload.get("x"), key=f"{key}.x", stage=stage)
        y = _number(payload.get("y"), key=f"{key}.y", stage=stage)
        space = str(raw.get("space") or payload.get("space") or "normalized")
        reference = str(raw.get("reference") or payload.get("reference") or "source")
    else:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must be x,y or an object", category="usage", stage=stage)
    if space not in {"normalized", "px", "hid"}:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key}.space must be normalized, px, or hid", category="usage", stage=stage)
    if reference not in {"source", "preview"}:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key}.reference must be source or preview", category="usage", stage=stage)
    return {"x": x, "y": y, "space": space, "reference": reference}


def _string_list(value: Any, *, key: str, stage: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must be a string or array", category="usage", stage=stage)
    result = [str(item).strip() for item in items if str(item).strip()]
    if not result:
        raise CoretapError("ACTION_SCHEMA_INVALID", f"action field {key} must not be empty", category="usage", stage=stage)
    return result


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
        raise CoretapError("ACTION_SCHEMA_INVALID", "step action postconditions were removed", category="usage", stage=stage)

    if action_type == "tap":
        action["target"] = _require_str(raw, "target", stage=stage)
        if "constraints" in raw:
            raise CoretapError("ACTION_SCHEMA_INVALID", "tap constraints were removed; use VLM target text directly", category="usage", stage=stage)
    elif action_type == "tapPoint":
        action["point"] = _point_from_action(raw, stage=stage)
    elif action_type == "longPress":
        action["point"] = _point_from_action(raw, stage=stage)
        action["durationMs"] = _integer(raw.get("durationMs", 1200), key="durationMs", stage=stage)
        action["steps"] = _integer(raw.get("steps", 12), key="steps", stage=stage)
    elif action_type == "openApp":
        action["name"] = _require_str(raw, "name", stage=stage)
        bundle_id = str(raw.get("bundleId") or "").strip()
        builtin_bundle_id = BUILTIN_APP_BUNDLE_IDS.get(normalize_text(action["name"]))
        if bundle_id or builtin_bundle_id:
            action["bundleId"] = bundle_id or builtin_bundle_id
        strategy = str(raw.get("strategy") or "auto").strip()
        if strategy not in {"auto", "bundle", "spotlight"}:
            raise CoretapError("ACTION_SCHEMA_INVALID", "openApp strategy must be auto, bundle, or spotlight", category="usage", stage=stage)
        action["strategy"] = strategy
        action["killExisting"] = bool(raw.get("killExisting", True))
        action["searchTarget"] = str(raw.get("searchTarget") or "the Search button at the bottom center of the iOS home screen")
        action["resultTarget"] = str(raw.get("resultTarget") or f"the {action['name']} app icon in Spotlight search results")
    elif action_type == "openUrl":
        action["url"] = _require_str(raw, "url", stage=stage)
        action["timeoutSec"] = _number(raw.get("timeoutSec", raw.get("timeout", 8.0)), key="timeoutSec", stage=stage)
        if action["timeoutSec"] <= 0:
            raise CoretapError("ACTION_SCHEMA_INVALID", "openUrl timeoutSec must be > 0", category="usage", stage=stage)
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
    elif action_type == "appSwitcher":
        start = raw.get("start") if isinstance(raw.get("start"), dict) else {}
        end = raw.get("end") if isinstance(raw.get("end"), dict) else {}
        action["start"] = {
            "x": _number(raw.get("startX", start.get("x", 0.5)), key="startX", stage=stage),
            "y": _number(raw.get("startY", start.get("y", 0.98)), key="startY", stage=stage),
            "space": "normalized",
            "reference": "source",
        }
        action["end"] = {
            "x": _number(raw.get("endX", end.get("x", 0.5)), key="endX", stage=stage),
            "y": _number(raw.get("endY", end.get("y", 0.45)), key="endY", stage=stage),
            "space": "normalized",
            "reference": "source",
        }
        action["steps"] = _integer(raw.get("steps", 40), key="steps", stage=stage)
        action["durationMs"] = _integer(raw.get("durationMs", 1200), key="durationMs", stage=stage)
    elif action_type == "terminateApp":
        action["bundleId"] = _require_str(raw, "bundleId", stage=stage)
        action["signal"] = _integer(raw.get("signal", 9), key="signal", stage=stage)
    elif action_type == "uninstallApp":
        name = str(raw.get("name") or "").strip()
        bundle_id = str(raw.get("bundleId") or "").strip()
        builtin_bundle_id = BUILTIN_APP_BUNDLE_IDS.get(normalize_text(name)) if name else None
        if not bundle_id and not builtin_bundle_id:
            field = "bundleId or a known app name" if name else "bundleId"
            raise CoretapError("ACTION_SCHEMA_INVALID", f"action is missing {field}", category="usage", stage=stage)
        if name:
            action["name"] = name
        action["bundleId"] = bundle_id or builtin_bundle_id
        action["ignoreMissing"] = bool(raw.get("ignoreMissing", True))
    elif action_type == "wait":
        action["ms"] = _integer(raw.get("ms", 700), key="ms", stage=stage)
    return action


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


def _text_near_anchor(observation: dict[str, Any], anchor: dict[str, float]) -> list[OcrToken]:
    tokens = _observation_tokens(observation)
    near: list[OcrToken] = []
    y_tolerance = 0.09 if anchor["y"] < 0.22 else 0.055
    for token in tokens:
        center_x = token.left + token.width / 2
        center_y = token.top + token.height / 2
        frame = observation.get("frame") if isinstance(observation, dict) else {}
        try:
            normalized_x = center_x / float(frame["widthPx"])
            normalized_y = center_y / float(frame["heightPx"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if abs(normalized_y - anchor["y"]) <= y_tolerance and abs(normalized_x - anchor["x"]) <= 0.42:
            near.append(token)
    return near


def _replace_decision_from_before_observation(observation: dict[str, Any], anchor: dict[str, float]) -> dict[str, Any]:
    ocr = observation.get("ocr") if isinstance(observation, dict) else None
    if not isinstance(ocr, dict) or not ocr.get("enabled"):
        return {
            "schema": "coretap.typeText.replace-decision.v1",
            "status": "unknown",
            "shouldClear": True,
            "reason": "before observation has no OCR",
        }
    near_tokens = [token for token in _text_near_anchor(observation, anchor) if token.text.strip()]
    texts = [token.text.strip() for token in near_tokens]
    if not texts:
        return {
            "schema": "coretap.typeText.replace-decision.v1",
            "status": "empty",
            "shouldClear": False,
            "reason": "no OCR text near text-entry anchor",
            "nearText": [],
        }
    joined = " ".join(texts)
    normalized = normalize_text(joined).casefold()
    placeholder_like = any(marker in normalized for marker in TEXT_ENTRY_PLACEHOLDER_MARKERS)
    return {
        "schema": "coretap.typeText.replace-decision.v1",
        "status": "placeholder" if placeholder_like else "text",
        "shouldClear": not placeholder_like,
        "reason": "placeholder text should not be cleared with keyboard shortcuts" if placeholder_like else "existing text is visible near text-entry anchor",
        "nearText": texts[:8],
    }


def _resolve_type_text_paste_at(
    args: argparse.Namespace,
    action: dict[str, Any],
    context: dict[str, Any],
) -> str | dict[str, Any] | None:
    explicit = action.get("pasteAt")
    if explicit is not None:
        context["anchor"] = {"source": "explicit", "point": explicit}
        return explicit
    text = str(action.get("text") or "")
    if _is_unshifted_virtual_keyboard_text(text):
        return None
    anchor = _last_text_entry_anchor(args)
    if anchor is None:
        context["ready"] = False
        context["reason"] = "paste-backed typeText requires a recent tap focus anchor or explicit pasteAt"
        raise CoretapError(
            "TEXT_INPUT_TARGET_UNKNOWN",
            "typeText requires a recent tap focus anchor for this text; tap the text field first or pass pasteAt",
            category="usage",
            stage="type",
            details={
                "textLength": len(text),
                "backend": getattr(args, "backend", None),
                "device": getattr(args, "device", None),
                "anchorMaxAgeMs": TEXT_ENTRY_ANCHOR_MAX_AGE_MS,
            },
        )
    point = anchor["point"]
    resolved = {"x": point["x"], "y": point["y"], "source": "last-tap"}
    context["anchor"] = {
        "source": "last-tap",
        "point": {"x": point["x"], "y": point["y"]},
        "ageMs": anchor.get("ageMs"),
        "target": anchor.get("target"),
        "actionType": anchor.get("actionType"),
    }
    return resolved


def _anchor_point_from_value(value: str | dict[str, Any], *, source: str) -> dict[str, Any]:
    if isinstance(value, str):
        x, y = _parse_xy_pair(value, option="pasteAt")
        return {"x": x, "y": y, "source": source}
    try:
        x = float(value["x"])
        y = float(value["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            "typeText pasteAt must contain normalized x and y",
            category="usage",
            stage="type",
            details={"pasteAt": value},
        ) from exc
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise CoretapError(
            "INVALID_POINT",
            "typeText pasteAt must be normalized coordinates between 0 and 1",
            category="usage",
            stage="type",
            details={"pasteAt": value},
        )
    return {"x": x, "y": y, "source": str(value.get("source") or source)}


def _resolve_keyboard_text_focus_anchor(
    args: argparse.Namespace,
    paste_at: str | dict[str, Any] | None,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if paste_at is not None:
        resolved = _anchor_point_from_value(paste_at, source="explicit")
        context["anchor"] = {"source": resolved["source"], "point": {"x": resolved["x"], "y": resolved["y"]}}
        return resolved
    anchor = _last_text_entry_anchor(args)
    if anchor is None:
        context["reason"] = "no recent text-entry anchor; keyboard input will rely on the currently focused field"
        return None
    point = anchor["point"]
    context["anchor"] = {
        "source": "last-tap",
        "point": {"x": point["x"], "y": point["y"]},
        "ageMs": anchor.get("ageMs"),
        "target": anchor.get("target"),
        "actionType": anchor.get("actionType"),
    }
    return {"x": point["x"], "y": point["y"], "source": "last-tap"}


_PASTE_MENU_LABELS = ("粘贴", "粘貼", "Paste")
_PASTE_MENU_FUZZY_MARKERS = ("粘", "贴", "貼")
_PASTE_MENU_REJECT_MARKERS = ("自动填充", "自動填充", "autofill")
_TEXT_INPUT_VISUAL_ATTEMPTS = 2


def _set_device_pasteboard_text(args: argparse.Namespace, text: str, *, verify: bool = True) -> dict[str, Any]:
    backend = backend_for(
        args.backend,
        developer_dir=getattr(args, "developer_dir", None),
        coredevice_tunnel_mode=getattr(args, "coredevice_tunnel_mode", None),
    )
    return backend.set_pasteboard_text(args.device, text, verify=verify, dry_run=getattr(args, "dry_run", False))


def _match_center_normalized(match: dict[str, Any], frame: dict[str, Any]) -> dict[str, float]:
    box = match["matchedBoxPx"]
    width = float(frame["widthPx"])
    height = float(frame["heightPx"])
    return {
        "x": (float(box["x"]) + float(box["width"]) / 2) / width,
        "y": (float(box["y"]) + float(box["height"]) / 2) / height,
    }


def _match_label_center_normalized(match: dict[str, Any], label: str, frame: dict[str, Any]) -> dict[str, float]:
    box = match["matchedBoxPx"]
    box_x = float(box["x"])
    box_y = float(box["y"])
    box_width = float(box["width"])
    box_height = float(box["height"])
    center_x = box_x + box_width / 2
    if match.get("matchedKind") == "token_contains":
        text = str(match.get("matchedText") or "")
        folded_text = text.casefold()
        folded_label = label.casefold()
        start = folded_text.find(folded_label)
        if start >= 0 and text:
            center_index = start + len(label) / 2
            center_x = box_x + box_width * (center_index / max(len(text), 1))
    width = float(frame["widthPx"])
    height = float(frame["heightPx"])
    return {
        "x": center_x / width,
        "y": (box_y + box_height / 2) / height,
    }


def _distance_from_anchor(point: dict[str, float], anchor: dict[str, float]) -> float:
    return ((point["x"] - anchor["x"]) ** 2 + (point["y"] - anchor["y"]) ** 2) ** 0.5


def _paste_menu_ocr_candidates(observation: dict[str, Any], anchor: dict[str, float]) -> list[dict[str, Any]]:
    tokens = _observation_tokens(observation)
    frame = observation["frame"]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for label in _PASTE_MENU_LABELS:
        for match in find_exact_text_candidates(tokens, label, case_sensitive=False, min_confidence=0.0):
            center = _match_label_center_normalized(match, label, frame)
            distance = _distance_from_anchor(center, anchor)
            if distance > 0.55:
                continue
            token_range = match.get("matchedTokenRange") or {}
            key = (int(token_range.get("start", -1)), int(token_range.get("endExclusive", -1)), label)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "schema": "coretap.paste-menu.candidate.v1",
                    "source": "ocr",
                    "label": label,
                    "point": center,
                    "distanceFromAnchor": distance,
                    "match": match,
                }
            )
    if candidates:
        return sorted(candidates, key=lambda item: (item["distanceFromAnchor"], -float(item["match"].get("matchedTokenMeanConfidence", 0.0))))

    frame_width = float(frame["widthPx"])
    frame_height = float(frame["heightPx"])
    for index, token in enumerate(tokens):
        text = token.text.strip()
        folded = text.casefold()
        if not text or not any(marker in folded for marker in _PASTE_MENU_FUZZY_MARKERS):
            continue
        box = {"x": token.left, "y": token.top, "width": token.width, "height": token.height}
        center_y = (float(token.top) + float(token.height) / 2) / frame_height
        token_center_x = (float(token.left) + float(token.width) / 2) / frame_width
        if anchor["y"] < 0.2:
            if not (anchor["y"] + 0.025 <= center_y <= anchor["y"] + 0.12):
                continue
        elif abs(center_y - anchor["y"]) > 0.13:
            continue
        if abs(token_center_x - anchor["x"]) > 0.28:
            continue
        point = {
            "x": (float(token.left) + float(token.width) * 0.18) / frame_width,
            "y": center_y,
        }
        distance = _distance_from_anchor(point, anchor)
        candidates.append(
            {
                "schema": "coretap.paste-menu.candidate.v1",
                "source": "ocr_fuzzy",
                "label": "粘贴",
                "point": point,
                "distanceFromAnchor": distance,
                "match": {
                    "matchedText": text,
                    "matchedEngines": [token.engine],
                    "matchedTokenRange": {"start": index, "endExclusive": index + 1},
                    "matchedTokenMeanConfidence": token.confidence,
                    "matchedTokenMinimumConfidence": token.confidence,
                    "matchedBoxPx": box,
                    "matchedKind": "fuzzy_menu_token",
                },
            }
        )
    return sorted(candidates, key=lambda item: (item["distanceFromAnchor"], -float(item["match"].get("matchedTokenMeanConfidence", 0.0))))


def _locate_paste_menu_with_vlm(
    args: argparse.Namespace,
    observation: dict[str, Any],
    *,
    run_dir: Path,
    label: str,
) -> dict[str, Any] | None:
    if args.profile == INTERNAL_FIXTURE_PROFILE:
        return None
    frame = observation["frame"]
    image = Path(frame["path"])
    model_input = prepare_grounding_image(image, output_dir=run_dir, max_long_side=args.max_long_side)
    target = "the visible Paste or 粘贴 item in the iOS edit menu"
    grounded = _ground_target_with_recovery(Path(model_input["path"]), target, profile=args.profile)
    grounded["modelInput"] = {
        "path": model_input["path"],
        "widthPx": model_input["widthPx"],
        "heightPx": model_input["heightPx"],
        "resized": model_input["resized"],
        "maxLongSidePx": model_input["maxLongSidePx"],
        "scale": model_input["scale"],
    }
    grounded = remap_grounding_to_source_frame(grounded, source_width=int(frame["widthPx"]), source_height=int(frame["heightPx"]))
    _write_grounding_artifacts(run_dir, grounded, stem=f"{label}-vlm")
    if grounded.get("status") != "found":
        return None
    normalized = dict(grounded["point"]["normalized"])
    return {
        "schema": "coretap.paste-menu.candidate.v1",
        "source": "vlm",
        "label": "Paste/粘贴",
        "point": {"x": float(normalized["x"]), "y": float(normalized["y"])},
        "grounding": grounded,
    }


def _locate_paste_menu(
    args: argparse.Namespace,
    observation: dict[str, Any],
    *,
    anchor: dict[str, float],
    run_dir: Path,
    label: str,
    allow_vlm: bool = False,
) -> dict[str, Any] | None:
    candidates = _paste_menu_ocr_candidates(observation, anchor)
    if candidates:
        return {**candidates[0], "candidates": candidates}
    if not allow_vlm:
        return None
    vlm_candidate = _locate_paste_menu_with_vlm(args, observation, run_dir=run_dir, label=label)
    if vlm_candidate is not None:
        return {**vlm_candidate, "candidates": candidates}
    return None


def _text_match_is_near_anchor(match: dict[str, Any], frame: dict[str, Any], anchor: dict[str, float]) -> bool:
    center = _match_center_normalized(match, frame)
    if anchor["y"] < 0.2:
        return 0.015 <= center["y"] <= min(0.24, anchor["y"] + 0.13)
    if anchor["y"] > 0.82 and 0.04 <= center["y"] <= 0.28:
        return True
    return abs(center["x"] - anchor["x"]) <= 0.48 and abs(center["y"] - anchor["y"]) <= 0.1


def _compact_input_verification_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def _find_compact_text_input_candidates(tokens: list[OcrToken], expected: str) -> list[dict[str, Any]]:
    needle = _compact_input_verification_text(expected)
    if len(needle) < 8:
        return []
    normalized = [_compact_input_verification_text(token.text) for token in tokens]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for start in range(len(tokens)):
        acc = ""
        for end in range(start, len(tokens)):
            acc += normalized[end]
            if acc == needle:
                match = token_match(tokens[start : end + 1], start, end + 1)
                match["matchedKind"] = "compact"
                key = (match["matchedTokenRange"]["start"], match["matchedTokenRange"]["endExclusive"])
                if key not in seen:
                    candidates.append(match)
                    seen.add(key)
                break
            if len(acc) > len(needle) + 30:
                break
    return candidates


def _verify_text_input_near_anchor(observation: dict[str, Any], text: str, anchor: dict[str, float]) -> dict[str, Any]:
    tokens = _observation_tokens(observation)
    frame = observation["frame"]
    candidates = find_exact_text_candidates(tokens, text, case_sensitive=False, min_confidence=0.0)
    candidates.extend(_find_compact_text_input_candidates(tokens, text))
    near = [match for match in candidates if _text_match_is_near_anchor(match, frame, anchor)]
    if near:
        return {
            "schema": "coretap.text-input.verification.v1",
            "status": "verified",
            "expectedText": text,
            "match": near[0],
            "candidateCount": len(candidates),
        }
    return {
        "schema": "coretap.text-input.verification.v1",
        "status": "failed",
        "expectedText": text,
        "candidateCount": len(candidates),
        "offTargetMatches": candidates[:5],
        "reason": "expected text was not visible near the text entry anchor",
    }


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


def _step_blocked(action: dict[str, Any], code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "coretap.step.execution.v1",
        "status": "blocked",
        "actionType": action["type"],
        "code": code,
        "reason": message,
        "details": details or {},
    }


def _bundle_launch_confirmed(launch: dict[str, Any]) -> bool:
    if launch.get("dryRun"):
        return True
    strategy = str(launch.get("strategy") or "")
    backend = str(launch.get("backend") or "")
    if strategy != "coredevice-dvt-launch" and backend != "device":
        return True
    try:
        return int(launch.get("pid") or 0) > 0
    except (TypeError, ValueError):
        return False


def _observation_frame(observation: dict[str, Any], *, reference: str = "source") -> dict[str, Any]:
    if reference == "preview":
        return observation["frame"]
    return observation.get("sourceFrame") or observation["frame"]


def _step_point_to_hid(action_point: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    reference = str(action_point.get("reference") or "source")
    frame = _observation_frame(before, reference=reference)
    return point_to_hid(
        float(action_point["x"]),
        float(action_point["y"]),
        width=int(frame["widthPx"]),
        height=int(frame["heightPx"]),
        space=str(action_point.get("space") or "normalized"),
    )


def _write_grounding_artifacts(run_dir: Path, grounded: dict[str, Any], *, stem: str) -> None:
    raw_tsv = grounded.pop("rawTsv", None)
    raw_output = grounded.pop("rawOutput", None)
    if raw_tsv is not None:
        (run_dir / f"{stem}.tsv").write_text(raw_tsv, encoding="utf-8")
    if raw_output is not None:
        (run_dir / f"{stem}.raw.txt").write_text(raw_output, encoding="utf-8")
    write_json(run_dir / f"{stem}.json", grounded)


def _step_before_observation_needs_ocr(action: dict[str, Any]) -> bool:
    if action.get("type") == "typeText" and _is_non_ascii_text(str(action.get("text") or "")):
        return True
    if action.get("type") == "typeText" and bool(action.get("replace")):
        return True
    return False


def _step_action_requires_before_observation(action: dict[str, Any]) -> bool:
    return action.get("type") not in {"terminateApp", "uninstallApp", "openUrl", "wait"}


def _text_entry_anchor_point_for_tap(action: dict[str, Any], point: dict[str, Any], source_frame: dict[str, Any]) -> dict[str, Any]:
    target = str(action.get("target") or "")
    if _target_suggests_relocated_search_entry(target, point):
        return point_to_hid(
            0.5,
            0.54,
            width=int(source_frame["widthPx"]),
            height=int(source_frame["heightPx"]),
            space="normalized",
        )
    if _target_suggests_top_text_entry(target, point):
        return point_to_hid(
            0.5,
            0.09,
            width=int(source_frame["widthPx"]),
            height=int(source_frame["heightPx"]),
            space="normalized",
        )
    return point


def _execute_step_tap_point(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    point = _step_point_to_hid(action["point"], before)
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    tap = backend.tap_normalized(
        args.device,
        point["normalized"]["x"],
        point["normalized"]["y"],
        dry_run=args.dry_run,
        hid_u16=point["hidU16"],
    )
    anchor = _remember_text_entry_anchor(args, point, source="last-tap", action_type="tapPoint")
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "tapPoint",
        "strategy": "explicit_point",
        "point": point,
        "textEntryAnchor": anchor,
        "tap": tap,
    }


def _execute_step_long_press(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    point = _step_point_to_hid(action["point"], before)
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    hold = backend.drag_normalized(
        args.device,
        point["normalized"]["x"],
        point["normalized"]["y"],
        point["normalized"]["x"],
        point["normalized"]["y"],
        dry_run=args.dry_run,
        start_hid_u16=point["hidU16"],
        end_hid_u16=point["hidU16"],
        steps=action["steps"],
        duration_ms=action["durationMs"],
    )
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "longPress",
        "strategy": "explicit_point_hold",
        "durationMs": action["durationMs"],
        "steps": action["steps"],
        "point": point,
        "hold": hold,
    }


def _execute_step_app_switcher(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    start = _step_point_to_hid(action["start"], before)
    end = _step_point_to_hid(action["end"], before)
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    gesture = backend.drag_normalized(
        args.device,
        start["normalized"]["x"],
        start["normalized"]["y"],
        end["normalized"]["x"],
        end["normalized"]["y"],
        dry_run=args.dry_run,
        start_hid_u16=start["hidU16"],
        end_hid_u16=end["hidU16"],
        steps=action["steps"],
        duration_ms=action["durationMs"],
    )
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "appSwitcher",
        "strategy": "home_indicator_up_and_hold",
        "start": start,
        "end": end,
        "steps": action["steps"],
        "durationMs": action["durationMs"],
        "gesture": gesture,
    }


def _execute_step_terminate_app(args: argparse.Namespace, action: dict[str, Any]) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    result = backend.terminate_app(
        args.device,
        action["bundleId"],
        signal=action["signal"],
        dry_run=args.dry_run,
    )
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "terminateApp",
        "strategy": "bundle_process_signal",
        "terminateResult": result,
    }


def _execute_step_uninstall_app(args: argparse.Namespace, action: dict[str, Any]) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    result = backend.uninstall_app(
        args.device,
        action["bundleId"],
        ignore_missing=action["ignoreMissing"],
        dry_run=args.dry_run,
    )
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "uninstallApp",
        "strategy": "bundle_uninstall",
        **({"name": action["name"]} if "name" in action else {}),
        "bundleId": action["bundleId"],
        "uninstallResult": result,
    }


def _execute_step_open_url(args: argparse.Namespace, action: dict[str, Any]) -> dict[str, Any]:
    backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
    result = backend.open_url(
        args.device,
        action["url"],
        timeout_sec=action["timeoutSec"],
        dry_run=args.dry_run,
    )
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "openUrl",
        "strategy": result.get("strategy") or "open-url",
        "url": action["url"],
        "openUrlResult": result,
    }


def _ground_target_with_recovery(image: Path, target: str, *, profile: str) -> dict[str, Any]:
    try:
        return ground_target(image, target, profile=profile)
    except CoretapError as exc:
        if exc.code != "MODEL_WORKER_CRASHED" or not exc.retryable:
            raise
        warm_model(profile)
        result = ground_target(image, target, profile=profile)
        result["modelRecovery"] = {
            "schema": "coretap.model-recovery.v1",
            "recoveredFrom": exc.code,
            "attempts": 2,
        }
        return result


def _model_input_summary(model_input: dict[str, Any]) -> dict[str, Any]:
    return {
        key: model_input.get(key)
        for key in ("path", "widthPx", "heightPx", "resized", "maxLongSidePx", "scale")
        if key in model_input
    }


def _grounding_response_summary(grounded: dict[str, Any], *, source: str) -> dict[str, Any]:
    point = grounded.get("point") if isinstance(grounded.get("point"), dict) else {}
    summary: dict[str, Any] = {
        "schema": grounded.get("schema") or "coretap.ground.result.v1",
        "status": grounded.get("status"),
        "source": source,
        "point": {
            key: point.get(key)
            for key in ("normalized", "framePx", "modelInputFramePx", "cropFramePx")
            if key in point
        },
        "frame": grounded.get("frame"),
    }
    for key in ("target", "model", "reason", "modelRecovery"):
        if key in grounded:
            summary[key] = grounded.get(key)
    return {key: value for key, value in summary.items() if value is not None}


def _refinement_edge_warning(grounded: dict[str, Any], crop: dict[str, Any], *, margin: float = 0.03) -> dict[str, Any] | None:
    point = grounded.get("point") if isinstance(grounded.get("point"), dict) else {}
    crop_frame_px = point.get("cropFramePx") if isinstance(point.get("cropFramePx"), dict) else None
    if not crop_frame_px:
        return None
    try:
        x = float(crop_frame_px["x"])
        y = float(crop_frame_px["y"])
        width = float(crop["width"])
        height = float(crop["height"])
    except (KeyError, TypeError, ValueError):
        return None
    margin_x = width * margin
    margin_y = height * margin
    near = x <= margin_x or x >= width - margin_x or y <= margin_y or y >= height - margin_y
    if not near:
        return None
    return {
        "code": "REFINED_POINT_NEAR_CROP_EDGE",
        "message": "refined grounding point is close to the crop edge",
        "margin": margin,
        "cropFramePx": {"x": x, "y": y},
        "cropSizePx": {"width": width, "height": height},
    }


def _refinement_error_summary(exc: CoretapError) -> dict[str, Any]:
    return {
        "code": exc.code,
        "category": exc.category,
        "stage": exc.stage,
        "message": str(exc),
        "retryable": exc.retryable,
        "details": exc.details,
    }


def _write_step_grounding_final(
    run_dir: Path,
    *,
    target: str,
    strategy: str,
    final_grounding: dict[str, Any],
    coarse: dict[str, Any],
    crop: dict[str, Any] | None = None,
    refined: dict[str, Any] | None = None,
    refine_error: dict[str, Any] | None = None,
    warning: dict[str, Any] | None = None,
) -> None:
    write_json(
        run_dir / "step-grounding-final.json",
        {
            "schema": "coretap.step.tap-refinement.v1",
            "target": target,
            "strategy": strategy,
            "final": final_grounding,
            "coarse": _grounding_response_summary(coarse, source="coarse"),
            "crop": crop,
            "refined": _grounding_response_summary(refined, source="refined") if isinstance(refined, dict) else None,
            "refineError": refine_error,
            "warning": warning,
        },
    )


def _execute_step_tap(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    target = action["target"]
    refine_crop_ratio = float(getattr(args, "refine_crop_ratio", DEFAULT_REFINEMENT_CROP_RATIO) or DEFAULT_REFINEMENT_CROP_RATIO)
    if refine_crop_ratio <= 0:
        raise CoretapError("INVALID_ARGUMENT", "step --refine-crop-ratio must be > 0", category="usage", stage="step-action")
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
    grounded = _ground_target_with_recovery(Path(model_input["path"]), target, profile=args.profile)
    grounded["modelInput"] = _model_input_summary(model_input)
    grounded = remap_grounding_to_source_frame(
        grounded,
        source_width=int(source_frame["widthPx"]),
        source_height=int(source_frame["heightPx"]),
    )
    if grounded.get("status") != "found":
        _write_grounding_artifacts(run_dir, copy.deepcopy(grounded), stem="step-grounding-coarse")
        final_grounding = _grounding_response_summary(grounded, source="coarse")
        _write_step_grounding_final(run_dir, target=target, strategy="vlm_grounding", final_grounding=final_grounding, coarse=grounded)
        return _step_blocked(
            action,
            _grounding_error_code(str(grounded.get("status") or "invalid")),
            f"Target was not found: {target}",
            details={"grounding": grounded, "modelInput": model_input},
        )
    _write_grounding_artifacts(run_dir, copy.deepcopy(grounded), stem="step-grounding-coarse")

    final_grounded = grounded
    final_grounding_source = "coarse"
    strategy = "vlm_grounding"
    refinement_crop: dict[str, Any] | None = None
    refined_grounded: dict[str, Any] | None = None
    refine_error: dict[str, Any] | None = None
    refine_warning: dict[str, Any] | None = None
    if not bool(getattr(args, "no_refine", False)):
        try:
            refinement_crop = prepare_refinement_crop(
                source_image,
                center=grounded["point"]["framePx"],
                output_dir=run_dir,
                crop_ratio=refine_crop_ratio,
            )
            refined_model_input = prepare_image_long_side(
                Path(refinement_crop["path"]),
                output_path=run_dir / "step-grounding-refine-model-input.png",
                max_long_side=args.max_long_side,
                stage="grounding-refine-preprocess",
            )
            refined_raw = _ground_target_with_recovery(Path(refined_model_input["path"]), target, profile=args.profile)
            refined_raw["modelInput"] = _model_input_summary(refined_model_input)
            refined_in_crop = remap_grounding_to_source_frame(
                refined_raw,
                source_width=int(refinement_crop["width"]),
                source_height=int(refinement_crop["height"]),
            )
            refined_grounded = remap_crop_grounding_to_source_frame(refined_in_crop, crop=refinement_crop)
            _write_grounding_artifacts(run_dir, copy.deepcopy(refined_grounded), stem="step-grounding-refined")
            if refined_grounded.get("status") == "found":
                final_grounded = refined_grounded
                final_grounding_source = "refined"
                strategy = "vlm_grounding_refined"
                refine_warning = _refinement_edge_warning(refined_grounded, refinement_crop)
            else:
                final_grounding_source = "coarse_fallback"
                strategy = "vlm_grounding_coarse_fallback"
        except CoretapError as exc:
            refine_error = _refinement_error_summary(exc)
            final_grounding_source = "coarse_fallback"
            strategy = "vlm_grounding_coarse_fallback"

    final_grounding = _grounding_response_summary(final_grounded, source=final_grounding_source)
    _write_step_grounding_final(
        run_dir,
        target=target,
        strategy=strategy,
        final_grounding=final_grounding,
        coarse=grounded,
        crop=refinement_crop,
        refined=refined_grounded,
        refine_error=refine_error,
        warning=refine_warning,
    )
    point_px = final_grounded["point"]["framePx"]
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
    anchor_point = _text_entry_anchor_point_for_tap(action, point, source_frame)
    anchor = _remember_text_entry_anchor(args, anchor_point, source="last-tap", action_type="tap", target=target)
    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "tap",
        "strategy": strategy,
        "target": target,
        "profile": args.profile,
        "grounding": final_grounding,
        "point": point,
        "textEntryAnchor": anchor,
        "tap": tap,
    }


def _execute_step_open_app(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    app_name = action["name"]
    bundle_id = action.get("bundleId")
    strategy = str(action.get("strategy") or "auto")
    search_target = str(action.get("searchTarget") or "the Search button at the bottom center of the iOS home screen")
    result_target = str(action.get("resultTarget") or f"the {app_name} app icon in Spotlight search results")
    substeps: list[dict[str, Any]] = []
    if args.dry_run:
        resolved_strategy = "bundle-launch" if bundle_id and strategy != "spotlight" else "spotlight-search"
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "openApp",
            "strategy": resolved_strategy,
            "app": app_name,
            "bundleId": bundle_id,
            "attempted": False,
            "dryRun": True,
        }

    if bundle_id and strategy in {"auto", "bundle"}:
        backend = backend_for(args.backend, developer_dir=args.developer_dir, coredevice_tunnel_mode=args.coredevice_tunnel_mode)
        try:
            launch = backend.launch_app(
                args.device,
                str(bundle_id),
                kill_existing=bool(action.get("killExisting", True)),
                dry_run=args.dry_run,
            )
            substeps.append({"name": "launch-bundle", "status": "executed", "result": launch})
            if not _bundle_launch_confirmed(launch):
                substeps.append(
                    {
                        "name": "confirm-bundle-launch",
                        "status": "blocked",
                        "reason": "bundle launch did not return a running process id",
                    }
                )
                if strategy == "bundle":
                    return _step_blocked(
                        action,
                        "APP_LAUNCH_NOT_CONFIRMED",
                        f"Could not confirm app launch by bundle id: {app_name}",
                        details={"bundleId": bundle_id, "launch": launch, "substeps": substeps},
                    )
            else:
                command_wait(argparse.Namespace(ms=800, wait_command=None))
                launched = _observe_into(args, run_dir=run_dir, label="open-app-after-launch", no_ocr=True)
                substeps.append({"name": "observe-after-launch", "status": "observed", "result": launched})
                return {
                    "schema": "coretap.step.execution.v1",
                    "status": "executed",
                    "actionType": "openApp",
                    "strategy": "bundle-launch",
                    "app": app_name,
                    "bundleId": bundle_id,
                    "substeps": substeps,
                }
        except CoretapError as exc:
            substeps.append(
                {
                    "name": "launch-bundle",
                    "status": "blocked",
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "category": exc.category,
                        "stage": exc.stage,
                        "details": exc.details,
                    },
                }
            )
            if strategy == "bundle":
                return _step_blocked(
                    action,
                    exc.code,
                    f"Could not launch app by bundle id: {app_name}",
                    details={"bundleId": bundle_id, "substeps": substeps},
                )

    press_args = argparse.Namespace(**vars(args))
    press_args.button = "home"
    press_args.state = "press"
    press_args.hold_ms = None
    press_result = command_press(press_args)
    substeps.append({"name": "press-home", "status": "executed", "result": press_result})
    command_wait(argparse.Namespace(ms=700, wait_command=None))

    home = _observe_into(args, run_dir=run_dir, label="open-app-home", no_ocr=True)
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

    search = _observe_into(args, run_dir=run_dir, label="open-app-search", no_ocr=True)
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

    results = _observe_into(args, run_dir=run_dir, label="open-app-results", no_ocr=True)
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

    launched = _observe_into(args, run_dir=run_dir, label="open-app-after-launch", no_ocr=True)
    substeps.append({"name": "observe-after-launch", "status": "observed", "result": launched})

    return {
        "schema": "coretap.step.execution.v1",
        "status": "executed",
        "actionType": "openApp",
        "strategy": "spotlight-search",
        "app": app_name,
        "substeps": substeps,
    }


def _type_text_step_strategy(type_result: dict[str, Any]) -> str:
    input_method = str(type_result.get("inputMethod") or "").strip()
    if input_method == "coredevice-pinyin-keyboard":
        return "coredevice_pinyin_keyboard"
    if input_method == "coredevice-pasteboard-keyboard-shortcut":
        return "pasteboard_keyboard_shortcut"
    if input_method == "coredevice-pasteboard-visual-menu":
        return "visual_paste_verified"
    if input_method in {"coredevice-hid-keyboard", "coredevice-virtual-keyboard"}:
        return "coredevice_hid_keyboard"
    if input_method == "coredevice-pasteboard-edit-menu":
        return "pasteboard_edit_menu"
    if input_method:
        return input_method.replace("-", "_")
    return "coredevice_text_input"


def _execute_step_type_text(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    context = {
        "schema": "coretap.text-entry-context.v1",
        "ready": True,
        "source": "vlm-first",
        "reason": "typeText no longer uses OCR to infer focus; focus text fields with an explicit VLM tap step before typing",
        "anchor": None,
    }
    ns = argparse.Namespace(**vars(args))
    ns.text = action["text"]
    ns.text_query = None
    ns.char_delay_ms = action["charDelayMs"]
    ns.inter_delay_ms = action["interDelayMs"]
    ns.paste_at = _resolve_type_text_paste_at(args, action, context)
    ns.paste_hold_ms = action["pasteHoldMs"]
    ns.verify_timeout_ms = action["verifyTimeoutMs"]
    ns.no_verify = action["noVerify"]
    ns.replace = action["replace"]
    if _is_unshifted_virtual_keyboard_text(action["text"]):
        focus_anchor = _resolve_keyboard_text_focus_anchor(args, ns.paste_at, context)
        focus_result = None
        if focus_anchor is not None:
            if ns.replace:
                replace_decision = _replace_decision_from_before_observation(
                    before,
                    {"x": float(focus_anchor["x"]), "y": float(focus_anchor["y"])},
                )
                context["replaceDecision"] = replace_decision
                if not replace_decision.get("shouldClear", True):
                    ns.replace = False
            focus_result = _tap_normalized_for_step(
                args,
                {"x": float(focus_anchor["x"]), "y": float(focus_anchor["y"])},
                reason="focus-text-field-before-keyboard-input",
            )
            command_wait(argparse.Namespace(ms=150, wait_command=None))
        type_result = command_type(ns)
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "typeText",
            "strategy": _type_text_step_strategy(type_result),
            "textEntryContext": context,
            "focusResult": focus_result,
            "resolvedPasteAt": ns.paste_at,
            "typeResult": type_result,
        }
    focus_anchor = _anchor_point_from_value(ns.paste_at, source="last-tap")
    if ns.replace:
        replace_decision = _replace_decision_from_before_observation(
            before,
            {"x": float(focus_anchor["x"]), "y": float(focus_anchor["y"])},
        )
        context["replaceDecision"] = replace_decision
        if not replace_decision.get("shouldClear", True):
            ns.replace = False
    visual_action = {**action, "replace": bool(ns.replace)}
    return _execute_step_type_text_visual(
        args,
        visual_action,
        before,
        run_dir,
        context=context,
        paste_at={
            "x": float(focus_anchor["x"]),
            "y": float(focus_anchor["y"]),
            "source": str(focus_anchor.get("source") or "last-tap"),
        },
    )


def _execute_step_type_text_visual(
    args: argparse.Namespace,
    action: dict[str, Any],
    before: dict[str, Any],
    run_dir: Path,
    *,
    context: dict[str, Any],
    paste_at: str | dict[str, Any] | None,
) -> dict[str, Any]:
    if getattr(args, "no_ocr", False):
        raise CoretapError(
            "ACTION_SCHEMA_INVALID",
            "paste-backed typeText requires OCR/VLM verification; remove --no-ocr",
            category="usage",
            stage="type",
        )
    if isinstance(paste_at, str):
        x, y = _parse_xy_pair(paste_at, option="pasteAt")
        paste_at = {"x": x, "y": y, "source": "explicit"}
    if not isinstance(paste_at, dict):
        raise CoretapError(
            "TEXT_INPUT_TARGET_UNKNOWN",
            "Paste-backed typeText requires a normalized paste anchor",
            category="usage",
            stage="type",
            details={"pasteAt": paste_at},
        )
    anchor = {"x": float(paste_at["x"]), "y": float(paste_at["y"])}
    attempts: list[dict[str, Any]] = []
    if getattr(args, "dry_run", False):
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": "typeText",
            "strategy": "visual_paste_verified",
            "textEntryContext": context,
            "resolvedPasteAt": paste_at,
            "attempts": [],
            "typeResult": {
                "attempted": False,
                "dryRun": True,
                "inputMethod": "coredevice-pasteboard-visual-menu",
                "confirmationStatus": "not_requested",
                "reason": "dry-run requested",
            },
        }

    for attempt_index in range(1, _TEXT_INPUT_VISUAL_ATTEMPTS + 1):
        attempt: dict[str, Any] = {"index": attempt_index, "anchor": anchor}
        attempts.append(attempt)
        pasteboard = _set_device_pasteboard_text(args, action["text"], verify=True)
        attempt["pasteboard"] = pasteboard

        before_ocr = before.get("ocr") if isinstance(before, dict) else None
        visible_observation = (
            before
            if attempt_index == 1 and isinstance(before_ocr, dict) and before_ocr.get("enabled")
            else _observe_into(args, run_dir=run_dir, label=f"type-visible-menu-{attempt_index:03d}", no_ocr=False)
        )
        attempt["visibleMenuObservation"] = {
            "frame": visible_observation.get("frame"),
            "ocr": _ocr_summary_for_observation(visible_observation),
        }
        visible_paste_candidate = _locate_paste_menu(
            args,
            visible_observation,
            anchor=anchor,
            run_dir=run_dir,
            label=f"type-visible-menu-{attempt_index:03d}",
        )
        if visible_paste_candidate is not None:
            attempt["pasteCandidate"] = {**visible_paste_candidate, "stage": "already-visible"}
            attempt["pasteTap"] = _tap_normalized_for_step(args, visible_paste_candidate["point"], reason="tap-visible-paste-menu")
            command_wait(argparse.Namespace(ms=700, wait_command=None))

            verify_observation = _observe_into(args, run_dir=run_dir, label=f"type-verify-{attempt_index:03d}", no_ocr=False)
            verification = _verify_text_input_near_anchor(verify_observation, action["text"], anchor)
            attempt["verification"] = verification
            attempt["verifyObservation"] = {
                "frame": verify_observation.get("frame"),
                "ocr": _ocr_summary_for_observation(verify_observation),
            }
            if verification["status"] == "verified":
                return {
                    "schema": "coretap.step.execution.v1",
                    "status": "executed",
                    "actionType": "typeText",
                    "strategy": "visual_paste_verified",
                    "textEntryContext": context,
                    "focusResult": None,
                    "resolvedPasteAt": paste_at,
                    "attempts": attempts,
                    "typeResult": {
                        "attempted": True,
                        "dryRun": False,
                        "inputMethod": "coredevice-pasteboard-visible-menu",
                        "confirmationStatus": "verified_text",
                        "pasteboardSet": True,
                        "pasteboardVerified": bool(pasteboard.get("pasteboardVerified")),
                        "pasteAnchor": {"source": paste_at.get("source", "explicit"), **anchor},
                        "pasteMenuTap": attempt["pasteTap"],
                        "verification": verification,
                        "attemptCount": attempt_index,
                        "clearExisting": action["replace"],
                        "typedCharacters": len(action["text"]),
                    },
                }
            attempt["status"] = "visible-menu-verification-failed"

        focus = _tap_normalized_for_step(args, anchor, reason="focus-text-field-before-visual-paste")
        attempt["focus"] = focus
        command_wait(argparse.Namespace(ms=150, wait_command=None))

        if action["replace"]:
            clear_args = argparse.Namespace(**vars(args))
            clear_args.count = 80
            clear_args.inter_delay_ms = 1
            attempt["clear"] = command_clear(clear_args)
            command_wait(argparse.Namespace(ms=120, wait_command=None))

        long_press_action = {
            "schema": "coretap.action.v2",
            "type": "longPress",
            "point": {"x": anchor["x"], "y": anchor["y"], "space": "normalized", "reference": "source"},
            "durationMs": action["pasteHoldMs"],
            "steps": 12,
        }
        attempt["openMenu"] = _execute_step_long_press(args, long_press_action, before)
        command_wait(argparse.Namespace(ms=250, wait_command=None))

        menu_observation = _observe_into(args, run_dir=run_dir, label=f"type-paste-menu-{attempt_index:03d}", no_ocr=False)
        attempt["menuObservation"] = {
            "frame": menu_observation.get("frame"),
            "ocr": _ocr_summary_for_observation(menu_observation),
        }
        paste_candidate = _locate_paste_menu(
            args,
            menu_observation,
            anchor=anchor,
            run_dir=run_dir,
            label=f"type-paste-menu-{attempt_index:03d}",
            allow_vlm=True,
        )
        attempt["pasteCandidate"] = paste_candidate
        if paste_candidate is None:
            attempt["status"] = "paste-menu-not-found"
            _tap_normalized_for_step(args, anchor, reason="dismiss-missing-paste-menu")
            command_wait(argparse.Namespace(ms=250, wait_command=None))
            continue

        attempt["pasteTap"] = _tap_normalized_for_step(args, paste_candidate["point"], reason="tap-visual-paste-menu")
        command_wait(argparse.Namespace(ms=700, wait_command=None))

        verify_observation = _observe_into(args, run_dir=run_dir, label=f"type-verify-{attempt_index:03d}", no_ocr=False)
        verification = _verify_text_input_near_anchor(verify_observation, action["text"], anchor)
        attempt["verification"] = verification
        attempt["verifyObservation"] = {
            "frame": verify_observation.get("frame"),
            "ocr": _ocr_summary_for_observation(verify_observation),
        }
        if verification["status"] == "verified":
            return {
                "schema": "coretap.step.execution.v1",
                "status": "executed",
                "actionType": "typeText",
                "strategy": "visual_paste_verified",
                "textEntryContext": context,
                "focusResult": focus,
                "resolvedPasteAt": paste_at,
                "attempts": attempts,
                "typeResult": {
                    "attempted": True,
                    "dryRun": False,
                    "inputMethod": "coredevice-pasteboard-visual-menu",
                    "confirmationStatus": "verified_text",
                    "pasteboardSet": True,
                    "pasteboardVerified": bool(pasteboard.get("pasteboardVerified")),
                    "pasteAnchor": {"source": paste_at.get("source", "last-tap"), **anchor},
                    "pasteMenuTap": attempt["pasteTap"],
                    "verification": verification,
                    "attemptCount": attempt_index,
                    "clearExisting": action["replace"],
                    "typedCharacters": len(action["text"]),
                },
            }
        attempt["status"] = "verification-failed"

    failure_code = "PASTE_MENU_NOT_FOUND" if all(item.get("pasteCandidate") is None for item in attempts) else "TEXT_INPUT_VERIFICATION_FAILED"
    raise CoretapError(
        failure_code,
        "Paste-backed text input could not be verified",
        category="assertion",
        stage="type",
        details={
            "text": action["text"],
            "anchor": anchor,
            "attempts": attempts,
            "textEntryContext": context,
        },
    )


def _ocr_summary_for_observation(observation: dict[str, Any]) -> dict[str, Any]:
    ocr = observation.get("ocr") if isinstance(observation, dict) else None
    if not isinstance(ocr, dict):
        return {"enabled": False}
    return {
        "enabled": bool(ocr.get("enabled")),
        "selectedEngine": ocr.get("selectedEngine"),
        "tokenCount": ocr.get("tokenCount"),
        "plainText": ocr.get("plainText"),
    }


def _execute_step_action(args: argparse.Namespace, action: dict[str, Any], before: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    action_type = action["type"]
    if action_type == "tap":
        return _execute_step_tap(args, action, before, run_dir)
    if action_type == "tapPoint":
        return _execute_step_tap_point(args, action, before)
    if action_type == "longPress":
        return _execute_step_long_press(args, action, before)
    if action_type == "openApp":
        return _execute_step_open_app(args, action, before, run_dir)
    if action_type == "openUrl":
        return _execute_step_open_url(args, action)
    if action_type == "typeText":
        return _execute_step_type_text(args, action, before, run_dir)
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
    if action_type == "appSwitcher":
        return _execute_step_app_switcher(args, action, before)
    if action_type == "terminateApp":
        return _execute_step_terminate_app(args, action)
    if action_type == "uninstallApp":
        return _execute_step_uninstall_app(args, action)
    if action_type == "wait":
        return {
            "schema": "coretap.step.execution.v1",
            "status": "executed",
            "actionType": action_type,
            "waitResult": command_wait(argparse.Namespace(ms=action["ms"], wait_command=None)),
        }
    raise CoretapError("ACTION_UNSUPPORTED", f"Unsupported step action type: {action_type}", category="usage", stage="step-action")


def _step_action_should_observe_page(action: dict[str, Any]) -> bool:
    return action.get("type") in {
        "tap",
        "tapPoint",
        "longPress",
        "openApp",
        "openUrl",
        "typeText",
        "key",
        "clear",
        "press",
        "scroll",
        "appSwitcher",
    }


def _attach_step_page_observation(args: argparse.Namespace, result: dict[str, Any], run_dir: Path) -> None:
    action = result.get("action")
    if not isinstance(action, dict) or not _step_action_should_observe_page(action):
        return
    if getattr(args, "dry_run", False) or getattr(args, "no_page", False):
        result["observation"] = {"status": "skipped", "reason": "disabled"}
        return
    page_wait_ms = max(0, int(getattr(args, "page_wait_ms", DEFAULT_STEP_PAGE_WAIT_MS) or 0))
    if page_wait_ms:
        command_wait(argparse.Namespace(ms=page_wait_ms, wait_command=None))
    page = _observe_into(
        args,
        run_dir=run_dir,
        label="step-page",
        no_ocr=bool(getattr(args, "no_ocr", False)),
        no_vlm=bool(getattr(args, "no_vlm", False)),
    )
    result["observation"] = page_observation_summary(page)


def command_step(args: argparse.Namespace) -> dict[str, Any]:
    with _command_artifacts(args) as run_dir:
        return _command_step_into(args, run_dir)


def _command_step_into(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    raw_action = _load_step_action(args)
    action = _normalize_step_action(raw_action)
    before_needs_ocr = _step_before_observation_needs_ocr(action)
    if _step_action_requires_before_observation(action):
        before = _observe_into(args, run_dir=run_dir, label="step-before", no_ocr=not before_needs_ocr, no_vlm=True)
    else:
        before = {
            "schema": "coretap.observe.result.v1",
            "skipped": True,
            "reason": "action does not require screen observation",
        }
    result: dict[str, Any] = {
        "schema": "coretap.step.result.v1",
        "action": action,
        "before": before,
        **_artifact_result(args, run_dir),
    }
    execution = _execute_step_action(args, action, before, run_dir)
    result["execution"] = execution

    if execution.get("status") == "blocked":
        result["status"] = "blocked"
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
        write_json(run_dir / "step.result.json", result)
        return result

    result["status"] = "executed"
    _attach_step_page_observation(args, result, run_dir)
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
    parser.add_argument("--backend", choices=["simulator", "device"], default="simulator")
    parser.add_argument("--device", default="booted")
    parser.add_argument("--developer-dir", default=None)
    parser.add_argument("--coredevice-tunnel-mode", choices=["userspace", "tunneld"], default=None)
    parser.add_argument("--artifact-root", default=os.environ.get("CORETAP_ARTIFACT_ROOT"))
    parser.add_argument("--keep-artifacts", action="store_true", default=_truthy_env("CORETAP_KEEP_ARTIFACTS"))
    parser.add_argument("--no-artifacts", action="store_true", default=_truthy_env("CORETAP_NO_ARTIFACTS"))
    parser.add_argument("--profile", default=PUBLIC_MODEL_PROFILE)
    parser.add_argument("--daemon", choices=["off", "auto", "on"], default="auto")
    parser.add_argument("--trace-id", default=os.environ.get("CORETAP_TRACE_ID"))
    parser.add_argument("--trace-title", default=os.environ.get("CORETAP_TRACE_TITLE"))

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
    observe.add_argument("--min-confidence", type=float, default=0.0)
    observe.add_argument("--no-ocr", action="store_true")
    observe.add_argument("--no-vlm", action="store_true")

    screenshot = sub.add_parser("screenshot")
    screenshot.add_argument("--label", default="screenshot")
    screenshot.add_argument("--out", default=None)

    step = sub.add_parser("step")
    step.add_argument("--action", default=None, help="Single Coretap step action JSON object")
    step.add_argument("--action-file", default=None, help="Path to a single Coretap step action JSON object")
    step.add_argument("--dry-run", action="store_true")
    step.add_argument("--page-wait-ms", type=int, default=DEFAULT_STEP_PAGE_WAIT_MS)
    step.add_argument("--no-page", action="store_true")
    step.add_argument("--min-confidence", type=float, default=0.0)
    step.add_argument("--max-long-side", type=int, default=DEFAULT_STEP_MODEL_INPUT_LONG_SIDE)
    step.add_argument("--no-refine", action="store_true")
    step.add_argument("--refine-crop-ratio", type=float, default=DEFAULT_REFINEMENT_CROP_RATIO)
    step.add_argument("--full-size", action="store_true")
    step.add_argument("--no-ocr", action="store_true")
    step.add_argument("--no-vlm", action="store_true")

    assert_text = sub.add_parser("assert")
    assert_sub = assert_text.add_subparsers(dest="assert_command", required=True)
    text = assert_sub.add_parser("text")
    text.add_argument("--text", required=True)
    text.add_argument("--image", default=None)
    text.add_argument("--timeout-ms", type=int, default=3000)
    text.add_argument("--poll-interval-ms", type=int, default=300)
    text.add_argument("--min-confidence", type=float, default=0.0)
    text.add_argument("--case-sensitive", action="store_true")

    wait = sub.add_parser("wait")
    wait.add_argument("wait_command", choices=["text"])
    wait.add_argument("--text", required=True)
    wait.add_argument("--image", default=None)
    wait.add_argument("--timeout-ms", type=int, default=3000)
    wait.add_argument("--poll-interval-ms", type=int, default=300)
    wait.add_argument("--min-confidence", type=float, default=0.0)
    wait.add_argument("--case-sensitive", action="store_true")
    return parser


COMMON_OPTIONS_WITH_VALUES = {
    "--backend",
    "--device",
    "--developer-dir",
    "--coredevice-tunnel-mode",
    "--artifact-root",
    "--profile",
    "--daemon",
    "--trace-id",
    "--trace-title",
}


COMMON_FLAG_OPTIONS = {
    "--keep-artifacts",
    "--no-artifacts",
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
        elif token in COMMON_FLAG_OPTIONS:
            moved.append(token)
            i += 1
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
    if args.command == "screenshot":
        return command_screenshot(args)
    if args.command == "observe":
        return command_observe(args)
    if args.command == "step":
        return command_step(args)
    if args.command == "assert" and args.assert_command == "text":
        return command_assert_text(args)
    if args.command == "wait":
        return command_wait(args)
    raise CoretapError("UNKNOWN_COMMAND", args.command, category="usage", stage="cli")


def _daemon_status_is_stale(status: dict[str, Any] | None, client_code: dict[str, Any]) -> bool:
    if not isinstance(status, dict):
        return True
    daemon_code = status.get("code")
    if not isinstance(daemon_code, dict):
        return True
    return daemon_code.get("fingerprint") != client_code.get("fingerprint")


def _ensure_current_daemon(daemon_mode: str) -> None:
    from coretap.daemon import ping_daemon, source_fingerprint, start_daemon, stop_daemon

    client_code = source_fingerprint()
    try:
        ping = ping_daemon(timeout=0.5)
    except CoretapError as exc:
        if daemon_mode == "auto" and exc.code == "DAEMON_UNAVAILABLE":
            start_daemon()
            return
        raise

    status = ping.get("result") if isinstance(ping, dict) else None
    if not _daemon_status_is_stale(status, client_code):
        return

    details = {
        "daemon": {
            "pid": status.get("pid") if isinstance(status, dict) else None,
            "code": status.get("code") if isinstance(status, dict) else None,
        },
        "client": {"code": client_code},
    }
    if daemon_mode != "auto":
        raise CoretapError(
            "DAEMON_STALE",
            "Coretap daemon is running older code. Restart it with `coretap daemon stop && coretap daemon start`.",
            category="infrastructure",
            stage="daemon",
            details=details,
        )

    try:
        stop_daemon(timeout=2.0)
    except CoretapError as exc:
        if exc.code != "DAEMON_UNAVAILABLE":
            raise
    start_daemon()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    normalized = normalize_global_args(list(argv if argv is not None else sys.argv[1:]))
    args = parser.parse_args(normalized)
    if args.command != "daemon" and args.daemon != "off":
        from coretap.daemon import request_daemon, start_daemon

        try:
            _ensure_current_daemon(args.daemon)
            data = request_daemon(normalized, cwd=str(Path.cwd()))
            emit(data)
            raise SystemExit(int(data.get("exitCode", 0 if data.get("ok") else 70)))
        except CoretapError as exc:
            if args.daemon == "auto" and exc.code == "DAEMON_UNAVAILABLE":
                start_daemon()
                data = request_daemon(normalized, cwd=str(Path.cwd()))
                emit(data)
                raise SystemExit(int(data.get("exitCode", 0 if data.get("ok") else 70)))
            if args.daemon == "on":
                data = response_error(args.command, exc)
                emit(data)
                raise SystemExit(EXIT_CODES.get(exc.code, 70))
            data = response_error(args.command, exc)
            emit(data)
            raise SystemExit(EXIT_CODES.get(exc.code, 70))
    started = time.monotonic()
    try:
        result = dispatch(args)
        data = response_ok(args.command, result)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        attach_trace(args, data, argv=normalized, cwd=str(Path.cwd()))
        emit(data)
        raise SystemExit(0)
    except CoretapError as exc:
        data = response_error(args.command, exc)
        data["durationMs"] = round((time.monotonic() - started) * 1000)
        attach_trace(args, data, argv=normalized, cwd=str(Path.cwd()))
        emit(data)
        raise SystemExit(EXIT_CODES.get(exc.code, 70))


if __name__ == "__main__":
    main()
