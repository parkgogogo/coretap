from __future__ import annotations

import json
import os
import re
import time
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
PUBLIC_RUNTIME_ID = "mlx-vlm-process-worker-v1"

INTERNAL_FIXTURE_PROFILE = "internal:test-fixture-grounder"

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


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


def cache_status() -> dict[str, Any]:
    roots = ensure_state()
    paths = model_paths()
    return {
        "profile": PUBLIC_MODEL_PROFILE,
        "modelRoot": str(roots["models"]),
        "cacheRoot": str(roots["cache"]),
        "active": str(paths.active) if paths.active.exists() else None,
        "installed": model_installed(),
        "packDir": str(paths.root),
        "modelDir": str(paths.model),
    }


def gc_model(*, dry_run: bool = False) -> dict[str, Any]:
    roots = ensure_state()
    downloads = roots["downloads"]
    candidates = [p for p in downloads.rglob("*.part") if p.is_file()] if downloads.exists() else []
    total = sum(p.stat().st_size for p in candidates)
    if not dry_run:
        for path in candidates:
            path.unlink(missing_ok=True)
    return {"dryRun": dry_run, "deletedCount": 0 if dry_run else len(candidates), "candidateCount": len(candidates), "bytes": total}


def stop_model() -> dict[str, Any]:
    stopped = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    return {"stopped": stopped, "profile": PUBLIC_MODEL_PROFILE, "worker": {"kind": "process-local"}}


def model_worker_status() -> dict[str, Any]:
    return {
        "kind": "mlx-vlm-process-resident",
        "runtime": PUBLIC_RUNTIME_ID,
        "loaded": bool(_MODEL_CACHE),
        "loadedProfiles": sorted(_MODEL_CACHE),
    }


def warm_model(profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
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
    return (
        "You are a GUI grounding agent.\n"
        "Given an iOS screenshot and the user's grounding instruction, locate exactly one UI element.\n"
        "Return only a JSON object with this exact shape: {\"coordinate\": [x, y]}.\n"
        "The coordinate MUST use the model coordinate space where x and y are numbers from 0 to 1000, "
        f"with origin at the top-left of the full screenshot. The original screenshot is {width}x{height} pixels.\n"
        "If the target is not visible or ambiguous, return only: {\"status\":\"not_found\"}.\n"
        f"Instruction: {target}"
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
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None


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


def run_grounding_model(image: Path, target: str, *, profile: str = PUBLIC_MODEL_PROFILE) -> dict[str, Any]:
    _require_public_profile(profile)
    width, height = png_size(image)
    _load_model()
    model, processor = _MODEL_CACHE[PUBLIC_MODEL_PROFILE]
    prompt = grounding_prompt(target, width, height)
    started = time.monotonic()
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        config = load_config(str(model_paths().model))
        formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
        output = generate(model, processor, prompt=formatted_prompt, image=str(image), max_tokens=96, temp=0.0, verbose=False)
    except Exception as exc:
        raise CoretapError(
            "MODEL_RUN_FAILED",
            f"MAI-UI grounding request failed: {exc}",
            stage="model-run",
            category="model",
            details={"image": str(image), "target": target},
        ) from exc
    raw = getattr(output, "text", str(output)).strip()
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


def model_root_from_env() -> None:
    # Keep Hugging Face cache under Coretap unless the user deliberately overrides it.
    roots = ensure_state()
    os.environ.setdefault("HF_HOME", str(roots["cache"] / "huggingface"))
