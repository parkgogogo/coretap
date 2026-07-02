from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coretap.runtime import CoretapError, cache_root, command_env, require_success, run_command, which


DEFAULT_OCR_LANG = "zh-Hans+en-US"
DEFAULT_OCR_ENGINE = "vision"

_OCR_EQUIVALENCE_TRANSLATION = str.maketrans(
    {
        "紅": "红",
        "書": "书",
        "門": "门",
        "開": "开",
        "應": "应",
        "設": "设",
        "訊": "讯",
        "號": "号",
        "聯": "联",
        "絡": "络",
        "雲": "云",
        "國": "国",
        "臺": "台",
        "灣": "湾",
        "體": "体",
        "蘋": "苹",
        "獲": "获",
    }
)


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    engine: str = DEFAULT_OCR_ENGINE

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height / 2)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().translate(_OCR_EQUIVALENCE_TRANSLATION)
    return " ".join(normalized.split())


def vision_ocr_status() -> dict[str, Any]:
    xcrun = which("xcrun")
    return {
        "ready": bool(xcrun),
        "engine": DEFAULT_OCR_ENGINE,
        "executable": xcrun,
        "languages": ["zh-Hans", "en-US"],
        "defaultLang": DEFAULT_OCR_LANG,
        "defaultLangAvailable": bool(xcrun),
    }


def run_ocr(image: Path) -> tuple[list[OcrToken], dict[str, Any]]:
    raw: dict[str, Any] = {"engines": [], "errors": []}
    try:
        vision_tokens, vision_stdout = run_vision_ocr(image)
        raw["engines"].append("vision")
        raw["visionJson"] = vision_stdout
        return vision_tokens, raw
    except CoretapError as exc:
        raw["errors"].append({"engine": "vision", "code": exc.code, "message": str(exc), "details": exc.details})

    first = raw["errors"][0] if raw["errors"] else {}
    raise CoretapError(
        first.get("code") or "OCR_UNAVAILABLE",
        first.get("message") or "Vision OCR is unavailable",
        stage="ocr",
        category="environment",
        details={"image": str(image), "errors": raw["errors"]},
    )


def run_vision_ocr(image: Path) -> tuple[list[OcrToken], str]:
    helper = _vision_helper_binary()
    done = require_success(
        run_command([str(helper), str(image)], env=command_env(), timeout=30, max_output=10_000_000),
        code="VISION_OCR_FAILED",
        stage="ocr",
    )
    return parse_vision_json(done.stdout), done.stdout


def parse_vision_json(raw: str) -> list[OcrToken]:
    try:
        data, _ = json.JSONDecoder().raw_decode(raw.strip())
    except json.JSONDecodeError as exc:
        raise CoretapError(
            "VISION_OCR_FAILED",
            "Could not parse Vision OCR output",
            stage="ocr",
            details={"stdout": raw[:1000]},
        ) from exc
    if not isinstance(data, list):
        raise CoretapError(
            "VISION_OCR_FAILED",
            "Vision OCR output was not a JSON array",
            stage="ocr",
            details={"stdout": raw[:1000]},
        )
    tokens: list[OcrToken] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            tokens.append(
                OcrToken(
                    text=text,
                    confidence=float(item.get("confidence") or 0),
                    left=int(float(item.get("left") or 0)),
                    top=int(float(item.get("top") or 0)),
                    width=int(float(item.get("width") or 0)),
                    height=int(float(item.get("height") or 0)),
                    engine="vision",
                )
            )
        except (TypeError, ValueError):
            continue
    return tokens


def find_text(tokens: list[OcrToken], expected: str, *, case_sensitive: bool = False) -> dict[str, Any] | None:
    if not expected:
        return None
    needle = expected if case_sensitive else normalize_text(expected)
    for idx, token in enumerate(tokens):
        hay = token.text if case_sensitive else normalize_text(token.text)
        if needle in hay:
            return token_match([token], idx, idx + 1)
    # Contiguous phrase search.
    normalized = [t.text if case_sensitive else normalize_text(t.text) for t in tokens]
    for start in range(len(tokens)):
        acc = ""
        for end in range(start, len(tokens)):
            acc = (acc + " " + normalized[end]).strip()
            if needle in acc:
                return token_match(tokens[start : end + 1], start, end + 1)
            if len(acc) > len(needle) + 80:
                break
    return None


def find_exact_text_candidates(
    tokens: list[OcrToken],
    expected: str,
    *,
    case_sensitive: bool = False,
    min_confidence: float = 50.0,
) -> list[dict[str, Any]]:
    if not expected:
        return []
    needle = expected if case_sensitive else normalize_text(expected)
    normalized = [t.text if case_sensitive else normalize_text(t.text) for t in tokens]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for start in range(len(tokens)):
        acc = ""
        for end in range(start, len(tokens)):
            acc = (acc + " " + normalized[end]).strip()
            if acc == needle:
                match = token_match(tokens[start : end + 1], start, end + 1)
                match["matchedKind"] = "exact"
                if _passes_min_confidence(match, min_confidence):
                    key = (match["matchedTokenRange"]["start"], match["matchedTokenRange"]["endExclusive"])
                    if key not in seen:
                        candidates.append(match)
                        seen.add(key)
                break
            if len(acc) > len(needle) + 40:
                break
    for idx, hay in enumerate(normalized):
        if _token_is_exact_with_ui_prefix(hay, needle, engine=tokens[idx].engine):
            match = token_match([tokens[idx]], idx, idx + 1)
            match["matchedKind"] = "exact"
            match["exactMatchStrategy"] = "ui-prefix-stripped"
            if _passes_min_confidence(match, min_confidence):
                key = (match["matchedTokenRange"]["start"], match["matchedTokenRange"]["endExclusive"])
                if key not in seen:
                    candidates.append(match)
                    seen.add(key)
    if candidates:
        return candidates

    for idx, token in enumerate(tokens):
        if needle and needle in normalized[idx]:
            match = token_match([token], idx, idx + 1)
            match["matchedKind"] = "token_contains"
            if _passes_min_confidence(match, min_confidence):
                key = (match["matchedTokenRange"]["start"], match["matchedTokenRange"]["endExclusive"])
                if key not in seen:
                    candidates.append(match)
                    seen.add(key)
    return candidates


def _token_is_exact_with_ui_prefix(hay: str, needle: str, *, engine: str | None = None) -> bool:
    if not hay or not needle or hay == needle:
        return False
    if not hay.endswith(needle):
        return False
    prefix = hay[: -len(needle)].strip()
    if not prefix:
        return False
    return _is_ocr_ui_prefix(prefix, engine=engine)


def _is_ocr_ui_prefix(prefix: str, *, engine: str | None = None) -> bool:
    compact = re.sub(r"\s+", "", prefix.casefold())
    if compact in {"q", "搜索", "search"}:
        return True
    if re.fullmatch(r"[0-9]+", compact):
        return True
    if engine == "vision" and re.fullmatch(r"[a-z]", compact):
        # Apple Vision sometimes folds small adjacent app/developer icons into
        # a single-letter prefix in the same token, for example "g Xingin".
        return True
    # OCR often folds search icons, bullets, or counters into the same token.
    return bool(re.fullmatch(r"[q0-9#•·.,:;!?~_+*/|()（）\\[\\]【】<>《》-]+", compact))


def token_match(tokens: list[OcrToken], start: int, end: int) -> dict[str, Any]:
    left = min(t.left for t in tokens)
    top = min(t.top for t in tokens)
    right = max(t.left + t.width for t in tokens)
    bottom = max(t.top + t.height for t in tokens)
    text = " ".join(t.text for t in tokens)
    confidences = [t.confidence for t in tokens]
    engines = sorted({t.engine for t in tokens})
    return {
        "matchedText": text,
        "matchedEngines": engines,
        "matchedTokenRange": {"start": start, "endExclusive": end},
        "matchedTokenMeanConfidence": sum(confidences) / len(confidences),
        "matchedTokenMinimumConfidence": min(confidences),
        "matchedBoxPx": {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
        },
    }


def _passes_min_confidence(match: dict[str, Any], min_confidence: float) -> bool:
    threshold = min_confidence
    if "vision" in match.get("matchedEngines", []):
        threshold = min(threshold, 25.0)
    return float(match["matchedTokenMinimumConfidence"]) >= threshold


def _vision_helper_binary() -> Path:
    source = _VISION_OCR_SWIFT_SOURCE
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    root = cache_root() / "vision-ocr" / digest
    binary = root / "coretap-vision-ocr"
    if binary.exists():
        return binary
    swiftc = which("xcrun")
    if not swiftc:
        raise CoretapError(
            "VISION_OCR_UNAVAILABLE",
            "xcrun is required for macOS Vision OCR",
            stage="ocr",
            category="environment",
        )
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "main.swift"
    source_path.write_text(source, encoding="utf-8")
    done = require_success(
        run_command(["xcrun", "swiftc", str(source_path), "-o", str(binary)], env=command_env(), timeout=120, max_output=2_000_000),
        code="VISION_OCR_UNAVAILABLE",
        stage="ocr",
    )
    if not binary.exists():
        raise CoretapError(
            "VISION_OCR_UNAVAILABLE",
            "Vision OCR helper did not produce a binary",
            stage="ocr",
            category="environment",
            details={"stdout": done.stdout[-1000:], "stderr": done.stderr[-1000:]},
        )
    return binary


_VISION_OCR_SWIFT_SOURCE = r'''
import Foundation
import Vision
import CoreGraphics
import ImageIO

struct Token: Codable {
    let text: String
    let confidence: Double
    let left: Int
    let top: Int
    let width: Int
    let height: Int
}

let args = CommandLine.arguments
if args.count < 2 {
    fputs("usage: coretap-vision-ocr image\n", stderr)
    exit(2)
}

let url = URL(fileURLWithPath: args[1])
guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fputs("could not load image\n", stderr)
    exit(3)
}

let imageWidth = image.width
let imageHeight = image.height
var tokens: [Token] = []
let request = VNRecognizeTextRequest { request, error in
    if let error = error {
        fputs("vision error: \(error)\n", stderr)
        exit(4)
    }
    let observations = (request.results as? [VNRecognizedTextObservation]) ?? []
    for observation in observations {
        guard let candidate = observation.topCandidates(1).first else { continue }
        let box = observation.boundingBox
        let left = Int((box.minX * CGFloat(imageWidth)).rounded())
        let top = Int(((1.0 - box.maxY) * CGFloat(imageHeight)).rounded())
        let width = Int((box.width * CGFloat(imageWidth)).rounded())
        let height = Int((box.height * CGFloat(imageHeight)).rounded())
        tokens.append(Token(
            text: candidate.string,
            confidence: Double(candidate.confidence * 100.0),
            left: left,
            top: top,
            width: width,
            height: height
        ))
    }
}

request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["zh-Hans", "en-US"]

let handler = VNImageRequestHandler(cgImage: image, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("perform error: \(error)\n", stderr)
    exit(5)
}

let data = try JSONEncoder().encode(tokens)
FileHandle.standardOutput.write(data)
'''
