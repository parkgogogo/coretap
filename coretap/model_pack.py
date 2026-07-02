from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coretap.runtime import CoretapError, ensure_state, png_size, read_json, sha256_file, write_json


PUBLIC_MODEL_PROFILE = "builtin:mai-ui-2b-mlx-6bit@1"
PUBLIC_MODEL_SLUG = "builtin-mai-ui-2b-mlx-6bit"
PUBLIC_MODEL_PACK_VERSION = "0.1.0"
PUBLIC_MODEL_REPO = "mlx-community/MAI-UI-2B-6bit-v2"
PUBLIC_MODEL_REVISION = "cb57cf2fc99f28cb7691459f712d2a276342f804"
PUBLIC_PROMPT_VERSION = "grounding-v1"
PUBLIC_VISUAL_OBSERVE_PROMPT_VERSION = "visual-observe-v1"
PUBLIC_RUNTIME_ID = "mlx-vlm-process-worker-v1"
VISUAL_OBSERVE_ROLES = {"appIcon", "button", "tab", "input", "toggle", "image", "navigation", "unknown"}

INTERNAL_FIXTURE_PROFILE = "internal:test-fixture-grounder"

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _is_model_worker_process() -> bool:
    return os.environ.get("CORETAP_MODEL_WORKER") == "1"


@dataclass(frozen=True)
class ModelPaths:
    root: Path
    model: Path
    manifest: Path
    active: Path


def model_paths() -> ModelPaths:
    roots = ensure_state()
    root = roots["models"] / PUBLIC_MODEL_SLUG / PUBLIC_MODEL_PACK_VERSION
    return ModelPaths(
        root=root,
        model=root / "model",
        manifest=root / "manifest.json",
        active=roots["models"] / PUBLIC_MODEL_SLUG / "active.json",
    )


def public_profiles() -> dict[str, dict[str, Any]]:
    return {
        PUBLIC_MODEL_PROFILE: {
            "id": PUBLIC_MODEL_PROFILE,
            "kind": "production",
            "description": "Built-in MAI-UI 2B MLX 6-bit GUI grounding model pack.",
            "repo": PUBLIC_MODEL_REPO,
            "revision": PUBLIC_MODEL_REVISION,
            "packVersion": PUBLIC_MODEL_PACK_VERSION,
        }
    }


def internal_profiles() -> dict[str, dict[str, Any]]:
    return {
        INTERNAL_FIXTURE_PROFILE: {
            "id": INTERNAL_FIXTURE_PROFILE,
            "kind": "internal-test-fixture",
            "description": "OCR-backed fixture grounder for local simulator regression only.",
        }
    }


def _require_public_profile(profile: str) -> None:
    if profile != PUBLIC_MODEL_PROFILE:
        raise CoretapError(
            "UNKNOWN_MODEL_PROFILE",
            f"Unknown public model profile: {profile}",
            category="usage",
            stage="model",
            details={"profile": profile, "publicProfiles": list(public_profiles())},
        )


def model_installed() -> bool:
    paths = model_paths()
    return paths.manifest.exists() and paths.model.exists()


def build_manifest(paths: ModelPaths) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if paths.model.exists():
        for file in sorted(p for p in paths.model.rglob("*") if p.is_file()):
            rel = file.relative_to(paths.model).as_posix()
            if rel.startswith(".cache/"):
                continue
            files.append({"path": rel, "size": file.stat().st_size, "sha256": sha256_file(file)})
    return {
        "schema": "coretap.model-pack.v1",
        "profileId": PUBLIC_MODEL_PROFILE,
        "packVersion": PUBLIC_MODEL_PACK_VERSION,
        "installedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "upstream": {
            "repo": PUBLIC_MODEL_REPO,
            "revision": PUBLIC_MODEL_REVISION,
            "license": "Apache-2.0",
        },
        "artifact": {
            "format": "mlx",
            "quantization": {"bits": 6, "mode": "affine", "groupSize": 64},
            "files": files,
        },
        "runtime": {"id": PUBLIC_RUNTIME_ID, "loader": "mlx-vlm"},
        "grounding": {
            "promptVersion": PUBLIC_PROMPT_VERSION,
            "outputSchema": "coretap.ground.result.v1",
            "nativeCoordinateSpace": "model-1000",
        },
    }


def install_model(profile: str = PUBLIC_MODEL_PROFILE, *, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    _require_public_profile(profile)
    model_root_from_env()
    paths = model_paths()
    if paths.manifest.exists() and paths.model.exists() and not force:
        manifest = read_json(paths.manifest)
        return {
            "dryRun": dry_run,
            "wouldInstall": False,
            "installed": True,
            "changed": False,
            "profile": profile,
            "packVersion": manifest.get("packVersion"),
            "modelDir": str(paths.model),
            "manifest": str(paths.manifest),
        }

    if dry_run:
        return {
            "dryRun": True,
            "wouldInstall": True,
            "installed": paths.manifest.exists() and paths.model.exists(),
            "changed": False,
            "profile": profile,
            "packVersion": PUBLIC_MODEL_PACK_VERSION,
            "source": {"repo": PUBLIC_MODEL_REPO, "revision": PUBLIC_MODEL_REVISION},
            "modelDir": str(paths.model),
            "manifest": str(paths.manifest),
            "force": force,
        }

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CoretapError(
            "DEPENDENCY_MISSING",
            "huggingface_hub is required for model install",
            stage="model-install",
            details={"package": "huggingface-hub"},
        ) from exc

    paths.model.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=PUBLIC_MODEL_REPO,
        revision=PUBLIC_MODEL_REVISION,
        local_dir=paths.model,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    manifest = build_manifest(paths)
    write_json(paths.manifest, manifest)
    write_json(
        paths.active,
        {
            "schema": "coretap.active-model.v1",
            "profileId": PUBLIC_MODEL_PROFILE,
            "packVersion": PUBLIC_MODEL_PACK_VERSION,
            "path": str(paths.root),
            "modelPath": str(paths.model),
            "manifest": str(paths.manifest),
        },
    )
    return {
        "installed": True,
        "changed": True,
        "profile": profile,
        "packVersion": PUBLIC_MODEL_PACK_VERSION,
        "source": {"repo": PUBLIC_MODEL_REPO, "revision": PUBLIC_MODEL_REVISION},
        "downloadedPath": downloaded,
        "modelDir": str(paths.model),
        "manifest": str(paths.manifest),
        "fileCount": len(manifest["artifact"]["files"]),
    }


def check_model(profile: str = PUBLIC_MODEL_PROFILE, *, deep: bool = False) -> dict[str, Any]:
    _require_public_profile(profile)
    paths = model_paths()
    details: dict[str, Any] = {
        "profile": profile,
        "packVersion": PUBLIC_MODEL_PACK_VERSION,
        "modelDir": str(paths.model),
        "manifest": str(paths.manifest),
        "installed": model_installed(),
    }
    if not details["installed"]:
        return {**details, "ready": False, "state": "not-installed"}

    manifest = read_json(paths.manifest)
    problems: list[str] = []
    if manifest.get("profileId") != PUBLIC_MODEL_PROFILE:
        problems.append("manifest profileId mismatch")
    if manifest.get("upstream", {}).get("revision") != PUBLIC_MODEL_REVISION:
        problems.append("manifest revision mismatch")
    for required in ("config.json", "tokenizer_config.json"):
        if not (paths.model / required).exists():
            problems.append(f"missing {required}")
    if not any(paths.model.glob("*.safetensors")):
        problems.append("missing root safetensors file")

    if deep and not problems:
        try:
            from mlx_vlm.utils import load_config

            load_config(str(paths.model))
        except Exception as exc:  # pragma: no cover - depends on model runtime details
            problems.append(f"mlx-vlm config load failed: {exc}")

    return {
        **details,
        "ready": not problems,
        "state": "ready" if not problems else "invalid",
        "problems": problems,
        "source": {"repo": PUBLIC_MODEL_REPO, "revision": PUBLIC_MODEL_REVISION},
        "runtime": PUBLIC_RUNTIME_ID,
    }


def _error_payload(exc: CoretapError) -> dict[str, Any]:
    return {
        "code": exc.code,
        "message": str(exc),
        "category": exc.category,
        "stage": exc.stage,
        "details": exc.details,
        "retryable": exc.retryable,
    }


class _ModelProcessClient:
    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self.loaded_profiles: set[str] = set()

    def status(self) -> dict[str, Any]:
        proc = self._proc
        running = proc is not None and proc.poll() is None
        if not running:
            self.loaded_profiles.clear()
        return {
            "kind": "mlx-vlm-process-resident",
            "runtime": PUBLIC_RUNTIME_ID,
            "running": running,
            "pid": proc.pid if running and proc is not None else None,
            "loaded": bool(self.loaded_profiles),
            "loadedProfiles": sorted(self.loaded_profiles),
        }

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self.loaded_profiles.clear()
        if proc is None or proc.poll() is not None:
            return
        with suppress(Exception):
            proc.terminate()
            proc.wait(timeout=2)
        if proc.poll() is None:
            with suppress(Exception):
                proc.kill()
                proc.wait(timeout=2)

    def request(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        with self._lock:
            proc = self._ensure_started()
            assert proc.stdin is not None
            assert proc.stdout is not None
            try:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()
                line = self._readline(proc, timeout=timeout)
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._mark_crashed(proc)
                raise CoretapError(
                    "MODEL_WORKER_CRASHED",
                    f"MAI-UI model worker crashed while handling request: {exc}",
                    stage="model-worker",
                    category="model",
                    retryable=True,
                    details={"action": payload.get("action")},
                ) from exc
            if not line:
                returncode = proc.poll()
                self._mark_crashed(proc)
                raise CoretapError(
                    "MODEL_WORKER_CRASHED",
                    "MAI-UI model worker exited without returning a response",
                    stage="model-worker",
                    category="model",
                    retryable=True,
                    details={"action": payload.get("action"), "returncode": returncode},
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self._mark_crashed(proc)
                raise CoretapError(
                    "MODEL_WORKER_PROTOCOL_ERROR",
                    "MAI-UI model worker returned invalid JSON",
                    stage="model-worker",
                    category="model",
                    retryable=True,
                    details={"raw": line[:1000]},
                ) from exc
            if response.get("ok"):
                profile = str(payload.get("profile") or PUBLIC_MODEL_PROFILE)
                self.loaded_profiles.add(profile)
                return response["result"]
            error = response.get("error") if isinstance(response, dict) else None
            if isinstance(error, dict):
                raise CoretapError(
                    str(error.get("code") or "MODEL_RUN_FAILED"),
                    str(error.get("message") or "MAI-UI model request failed"),
                    category=str(error.get("category") or "model"),
                    stage=str(error.get("stage") or "model-worker"),
                    retryable=bool(error.get("retryable")),
                    details=error.get("details") if isinstance(error.get("details"), dict) else {},
                )
            raise CoretapError(
                "MODEL_WORKER_PROTOCOL_ERROR",
                "MAI-UI model worker returned an invalid error response",
                stage="model-worker",
                category="model",
                details={"response": response},
            )

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self.loaded_profiles.clear()
        env = os.environ.copy()
        env["CORETAP_MODEL_WORKER"] = "1"
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "coretap.model_pack", "model-worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        return self._proc

    def _readline(self, proc: subprocess.Popen[str], *, timeout: float) -> str:
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout)
            if not events:
                self._mark_crashed(proc, kill=True)
                raise CoretapError(
                    "MODEL_WORKER_TIMEOUT",
                    f"MAI-UI model worker did not respond within {timeout:.0f}s",
                    stage="model-worker",
                    category="model",
                    retryable=True,
                )
            return proc.stdout.readline()
        finally:
            selector.close()

    def _mark_crashed(self, proc: subprocess.Popen[str], *, kill: bool = False) -> None:
        if self._proc is proc:
            self._proc = None
        self.loaded_profiles.clear()
        if kill and proc.poll() is None:
            with suppress(Exception):
                proc.kill()
                proc.wait(timeout=2)


_MODEL_PROCESS_CLIENT = _ModelProcessClient()


def close_model_worker() -> None:
    _MODEL_PROCESS_CLIENT.close()


def model_worker_status() -> dict[str, Any]:
    if _is_model_worker_process():
        return {
            "kind": "mlx-vlm-process-local",
            "runtime": PUBLIC_RUNTIME_ID,
            "loaded": bool(_MODEL_CACHE),
            "loadedProfiles": sorted(_MODEL_CACHE),
        }
    return _MODEL_PROCESS_CLIENT.status()


def warm_model(profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    if not _is_model_worker_process():
        result = _MODEL_PROCESS_CLIENT.request({"action": "warm", "profile": profile}, timeout=90.0)
        return {
            **result,
            "worker": {
                "kind": "process-resident-child",
                "runtime": PUBLIC_RUNTIME_ID,
                "pid": _MODEL_PROCESS_CLIENT.status().get("pid"),
            },
        }
    return _warm_model_inprocess(profile)


def _warm_model_inprocess(profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    _require_public_profile(profile)
    status = check_model(profile)
    if not status["ready"]:
        raise CoretapError(
            "MODEL_NOT_INSTALLED" if not status["installed"] else "MODEL_INCOMPATIBLE",
            f"Model profile is not ready: {profile}",
            stage="model",
            category="model",
            details=status,
        )
    started = time.monotonic()
    _load_model()
    return {
        **status,
        "warm": True,
        "worker": {"kind": "process-local", "runtime": PUBLIC_RUNTIME_ID},
        "durationMs": round((time.monotonic() - started) * 1000),
    }


def _load_model() -> tuple[Any, Any]:
    cached = _MODEL_CACHE.get(PUBLIC_MODEL_PROFILE)
    if cached:
        return cached
    model_root_from_env()
    try:
        from mlx_vlm import load
    except ImportError as exc:
        raise CoretapError(
            "DEPENDENCY_MISSING",
            "mlx-vlm is required for MAI-UI grounding",
            stage="model-load",
            category="environment",
            details={"package": "mlx-vlm"},
        ) from exc
    paths = model_paths()
    try:
        loaded = load(str(paths.model), revision=PUBLIC_MODEL_REVISION)
    except Exception as exc:
        raise CoretapError(
            "MODEL_LOAD_FAILED",
            f"Failed to load MAI-UI model pack: {exc}",
            stage="model-load",
            category="model",
            details={"modelDir": str(paths.model), "profile": PUBLIC_MODEL_PROFILE},
        ) from exc
    _MODEL_CACHE[PUBLIC_MODEL_PROFILE] = loaded
    return loaded


def grounding_prompt(target: str, width: int, height: int) -> str:
    return f'Locate {target}. Output only {{"coordinate":[x,y]}} in 0-1000 coordinates.'


def visual_observe_prompt(width: int, height: int, *, max_elements: int = 40) -> str:
    return (
        "Inspect this mobile UI screenshot and list the important visible interactive visual elements, "
        "especially icons or controls that may not have OCR text. "
        "Return only one JSON object with keys summary and elements. "
        f"elements must contain at most {max_elements} items. "
        "Each item must use: label, role, center, bbox, confidence. "
        "role must be one of appIcon, button, tab, input, toggle, image, navigation, unknown. "
        "center must be [x,y] in 0-1000 coordinates. "
        "bbox, when known, must be [x1,y1,x2,y2] in 0-1000 coordinates. "
        "confidence is a number from 0 to 1. "
        "Focus on actionable icons/buttons/tabs/inputs and major visual cards; skip decorative backgrounds. "
        f"The screenshot size is {width}x{height}px."
    )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0:
        segment = raw[start:]
        try:
            data, _ = json.JSONDecoder().raw_decode(segment)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            repaired = _repair_json_like_object(segment)
            try:
                data, _ = json.JSONDecoder().raw_decode(repaired)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            repaired = _repair_json_like_object(raw[start : end + 1])
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                return None
    return None


def _repair_json_like_object(raw: str) -> str:
    repaired = raw
    repaired = re.sub(r'"([xy])\s*(-?\d+(?:\.\d+)?)', r'"\1":\2', repaired)
    repaired = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*=", r'\1"\2":', repaired)
    repaired = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:", r'\1"\2":', repaired)
    return repaired


def parse_grounding_output(raw: str, *, width: int, height: int) -> dict[str, Any]:
    data = _extract_json_object(raw)
    coordinate: Any = None
    if data:
        status = data.get("status")
        if status in {"not_found", "ambiguous"}:
            return {"status": status, "rawOutput": raw}
        coordinate = data.get("coordinate")
        if coordinate is None and isinstance(data.get("point"), dict):
            point = data["point"]
            coordinate = [point.get("x"), point.get("y")]
        if coordinate is None and isinstance(data.get("point_2d"), list) and len(data["point_2d"]) == 2:
            coordinate = data["point_2d"]
        if coordinate is None and isinstance(data.get("bbox_2d"), list) and len(data["bbox_2d"]) == 4:
            x1, y1, x2, y2 = data["bbox_2d"]
            coordinate = [(float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2]
    if coordinate is None:
        match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", raw)
        if match:
            coordinate = [float(match.group(1)), float(match.group(2))]
    if not isinstance(coordinate, list | tuple) or len(coordinate) != 2:
        return {"status": "invalid", "rawOutput": raw, "reason": "no coordinate"}
    try:
        x = float(coordinate[0])
        y = float(coordinate[1])
    except (TypeError, ValueError):
        return {"status": "invalid", "rawOutput": raw, "reason": "non-numeric coordinate"}
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return {
            "status": "invalid",
            "rawOutput": raw,
            "reason": "coordinate outside model-1000 space",
            "pointModel1000": {"x": x, "y": y},
        }
    frame_x = (x / 1000) * width
    frame_y = (y / 1000) * height
    return {
        "status": "found",
        "rawOutput": raw,
        "point": {
            "model1000": {"x": x, "y": y},
            "framePx": {"x": frame_x, "y": frame_y},
            "normalized": {"x": x / 1000, "y": y / 1000},
        },
        "frame": {"widthPx": width, "heightPx": height},
        "model": {
            "profile": PUBLIC_MODEL_PROFILE,
            "packVersion": PUBLIC_MODEL_PACK_VERSION,
            "promptVersion": PUBLIC_PROMPT_VERSION,
            "runtimeVersion": PUBLIC_RUNTIME_ID,
        },
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coordinate_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        x = _as_float(value.get("x"))
        y = _as_float(value.get("y"))
    elif isinstance(value, list | tuple) and len(value) == 2:
        x = _as_float(value[0])
        y = _as_float(value[1])
    else:
        return None
    if x is None or y is None or not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def _bbox_quad(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        x = _as_float(value.get("x"))
        y = _as_float(value.get("y"))
        width = _as_float(value.get("width"))
        height = _as_float(value.get("height"))
        if x is None or y is None or width is None or height is None:
            return None
        x1, y1, x2, y2 = x, y, x + width, y + height
    elif isinstance(value, list | tuple) and len(value) == 4:
        x1 = _as_float(value[0])
        y1 = _as_float(value[1])
        x2 = _as_float(value[2])
        y2 = _as_float(value[3])
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
    else:
        return None
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    if left < 0 or top < 0 or right > 1000 or bottom > 1000 or right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _confidence(value: Any) -> float | None:
    confidence = _as_float(value)
    if confidence is None:
        return None
    return max(0.0, min(1.0, confidence))


def parse_visual_observe_output(raw: str, *, width: int, height: int, max_elements: int = 40) -> dict[str, Any]:
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return {
            "schema": "coretap.visual.observe.v1",
            "enabled": True,
            "status": "invalid",
            "summary": "",
            "elements": [],
            "rawOutput": raw,
            "reason": "no json object",
            "model": {
                "profile": PUBLIC_MODEL_PROFILE,
                "packVersion": PUBLIC_MODEL_PACK_VERSION,
                "promptVersion": PUBLIC_VISUAL_OBSERVE_PROMPT_VERSION,
                "runtimeVersion": PUBLIC_RUNTIME_ID,
            },
        }
    raw_elements = data.get("elements")
    if not isinstance(raw_elements, list):
        raw_elements = []
    elements: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for item in raw_elements:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        role = str(item.get("role") or "unknown").strip() or "unknown"
        if role not in VISUAL_OBSERVE_ROLES:
            role = "unknown"
        bbox_model = _bbox_quad(item.get("bbox") if "bbox" in item else item.get("bbox_2d"))
        center_model = _coordinate_pair(item.get("center") if "center" in item else item.get("coordinate"))
        if center_model is None and bbox_model is not None:
            center_model = ((bbox_model[0] + bbox_model[2]) / 2, (bbox_model[1] + bbox_model[3]) / 2)
        if center_model is None:
            continue
        cx, cy = center_model
        bbox_normalized = None
        bbox_px = None
        if bbox_model is not None:
            x1, y1, x2, y2 = bbox_model
            bbox_normalized = {
                "x": x1 / 1000,
                "y": y1 / 1000,
                "width": (x2 - x1) / 1000,
                "height": (y2 - y1) / 1000,
            }
            bbox_px = {
                "x": (x1 / 1000) * width,
                "y": (y1 / 1000) * height,
                "width": ((x2 - x1) / 1000) * width,
                "height": ((y2 - y1) / 1000) * height,
            }
        normalized = {"x": cx / 1000, "y": cy / 1000}
        dedupe_key = (label.casefold(), role, round(normalized["x"] * 1000), round(normalized["y"] * 1000))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        elements.append(
            {
                "type": "visual",
                "source": "vlm",
                "label": label,
                "role": role,
                "confidence": _confidence(item.get("confidence")),
                "center": normalized,
                "centerPx": {"x": normalized["x"] * width, "y": normalized["y"] * height},
                "bbox": bbox_normalized,
                "bboxPx": bbox_px,
            }
        )
        if len(elements) >= max_elements:
            break
    return {
        "schema": "coretap.visual.observe.v1",
        "enabled": True,
        "status": "ready",
        "profile": PUBLIC_MODEL_PROFILE,
        "promptVersion": PUBLIC_VISUAL_OBSERVE_PROMPT_VERSION,
        "summary": str(data.get("summary") or "").strip(),
        "elements": elements,
        "rawElementCount": len(raw_elements),
        "rawOutput": raw,
        "frame": {"widthPx": width, "heightPx": height},
        "model": {
            "profile": PUBLIC_MODEL_PROFILE,
            "packVersion": PUBLIC_MODEL_PACK_VERSION,
            "promptVersion": PUBLIC_VISUAL_OBSERVE_PROMPT_VERSION,
            "runtimeVersion": PUBLIC_RUNTIME_ID,
        },
    }


def _run_model_prompt(image: Path, prompt: str, *, profile: str, max_tokens: int) -> str:
    _require_public_profile(profile)
    _load_model()
    model, processor = _MODEL_CACHE[PUBLIC_MODEL_PROFILE]
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        config = load_config(str(model_paths().model))
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        output = generate(model, processor, prompt=formatted_prompt, image=str(image), max_tokens=max_tokens, temp=0.0, verbose=False)
    except Exception as exc:
        raise CoretapError(
            "MODEL_RUN_FAILED",
            f"MAI-UI model request failed: {exc}",
            stage="model-run",
            category="model",
            details={"image": str(image)},
        ) from exc
    return getattr(output, "text", str(output)).strip()


def run_grounding_model(image: Path, target: str, *, profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    if not _is_model_worker_process():
        return _MODEL_PROCESS_CLIENT.request(
            {"action": "ground", "profile": profile, "image": str(image), "target": target},
            timeout=45.0,
        )
    return _run_grounding_model_inprocess(image, target, profile=profile)


def run_visual_observe_model(image: Path, *, profile: str = PUBLIC_MODEL_PROFILE, max_elements: int = 40) -> dict[str, Any]:
    if not _is_model_worker_process():
        return _MODEL_PROCESS_CLIENT.request(
            {"action": "observe", "profile": profile, "image": str(image), "maxElements": max_elements},
            timeout=60.0,
        )
    return _run_visual_observe_model_inprocess(image, profile=profile, max_elements=max_elements)


def _run_grounding_model_inprocess(image: Path, target: str, *, profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    _require_public_profile(profile)
    width, height = png_size(image)
    prompt = grounding_prompt(target, width, height)
    started = time.monotonic()
    raw = _run_model_prompt(image, prompt, profile=profile, max_tokens=16)
    parsed = parse_grounding_output(raw, width=width, height=height)
    parsed.update(
        {
            "schema": "coretap.ground.result.v1",
            "target": {"description": target},
            "durationMs": round((time.monotonic() - started) * 1000),
            "rawOutput": raw,
        }
    )
    return parsed


def _run_visual_observe_model_inprocess(image: Path, *, profile: str = PUBLIC_MODEL_PROFILE, max_elements: int = 40) -> dict[str, Any]:
    _require_public_profile(profile)
    width, height = png_size(image)
    prompt = visual_observe_prompt(width, height, max_elements=max_elements)
    started = time.monotonic()
    raw = _run_model_prompt(image, prompt, profile=profile, max_tokens=768)
    parsed = parse_visual_observe_output(raw, width=width, height=height, max_elements=max_elements)
    parsed.update(
        {
            "durationMs": round((time.monotonic() - started) * 1000),
            "rawOutput": raw,
        }
    )
    return parsed


def model_root_from_env() -> None:
    # Keep Hugging Face cache under Coretap unless the user deliberately overrides it.
    roots = ensure_state()
    os.environ.setdefault("HF_HOME", str(roots["cache"] / "huggingface"))


def _model_worker_loop() -> int:
    os.environ["CORETAP_MODEL_WORKER"] = "1"
    for line in sys.stdin:
        try:
            request = json.loads(line)
            action = request.get("action")
            profile = str(request.get("profile") or PUBLIC_MODEL_PROFILE)
            if action == "warm":
                result = _warm_model_inprocess(profile)
            elif action == "ground":
                image = Path(str(request.get("image") or ""))
                target = str(request.get("target") or "")
                result = _run_grounding_model_inprocess(image, target, profile=profile)
            elif action == "observe":
                image = Path(str(request.get("image") or ""))
                max_elements = int(request.get("maxElements") or 40)
                result = _run_visual_observe_model_inprocess(image, profile=profile, max_elements=max_elements)
            else:
                raise CoretapError(
                    "MODEL_WORKER_INVALID_REQUEST",
                    f"Unsupported model worker action: {action}",
                    stage="model-worker",
                    category="usage",
                    details={"action": action},
                )
            response = {"ok": True, "result": result}
        except CoretapError as exc:
            response = {"ok": False, "error": _error_payload(exc)}
        except Exception as exc:
            error = CoretapError(
                "MODEL_WORKER_FAILED",
                f"MAI-UI model worker request failed: {exc}",
                stage="model-worker",
                category="model",
                details={"errorType": type(exc).__name__},
            )
            response = {"ok": False, "error": _error_payload(error)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args == ["model-worker"]:
        return _model_worker_loop()
    print("usage: python -m coretap.model_pack model-worker", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
