from __future__ import annotations

import csv
import io
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coretap.runtime import CoretapError, require_success, run_command, which


DEFAULT_OCR_LANG = "chi_sim+eng"


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height / 2)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_tesseract_languages(output: str) -> list[str]:
    languages: list[str] = []
    for line in output.splitlines():
        item = line.strip()
        if not item or item.startswith("List of available languages"):
            continue
        languages.append(item)
    return languages


def required_tesseract_languages(lang: str = DEFAULT_OCR_LANG) -> list[str]:
    return [part for part in lang.split("+") if part]


def missing_tesseract_languages(languages: list[str], lang: str = DEFAULT_OCR_LANG) -> list[str]:
    available = set(languages)
    return [part for part in required_tesseract_languages(lang) if part not in available]


def tesseract_status() -> dict[str, Any]:
    exe = which("tesseract")
    if not exe:
        return {
            "ready": False,
            "executable": None,
            "version": None,
            "defaultLang": DEFAULT_OCR_LANG,
            "languages": [],
            "defaultLangAvailable": False,
            "missingLanguages": required_tesseract_languages(),
        }
    done = run_command([exe, "--version"], timeout=5)
    first = done.stdout.splitlines()[0] if done.stdout else ""
    languages_done = run_command([exe, "--list-langs"], timeout=5)
    languages = parse_tesseract_languages(languages_done.stdout) if languages_done.returncode == 0 else []
    missing = missing_tesseract_languages(languages)
    return {
        "ready": done.returncode == 0,
        "executable": exe,
        "version": first,
        "defaultLang": DEFAULT_OCR_LANG,
        "languages": languages,
        "defaultLangAvailable": done.returncode == 0 and languages_done.returncode == 0 and not missing,
        "missingLanguages": missing,
    }


def run_tesseract(image: Path, *, lang: str = DEFAULT_OCR_LANG, psm: int = 11) -> tuple[list[OcrToken], str]:
    exe = which("tesseract")
    if not exe:
        raise CoretapError("OCR_UNAVAILABLE", "tesseract not found in PATH", stage="ocr")
    done = require_success(
        run_command([exe, str(image), "stdout", "-l", lang, "--oem", "1", "--psm", str(psm), "tsv"], timeout=10),
        code="OCR_PROCESS_FAILED",
        stage="ocr",
    )
    return parse_tsv(done.stdout), done.stdout


def parse_tsv(tsv: str) -> list[OcrToken]:
    tokens: list[OcrToken] = []
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row.get("conf") or -1)
            left = int(float(row.get("left") or 0))
            top = int(float(row.get("top") or 0))
            width = int(float(row.get("width") or 0))
            height = int(float(row.get("height") or 0))
        except ValueError:
            continue
        if conf < 0:
            continue
        tokens.append(OcrToken(text, conf, left, top, width, height))
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
                if match["matchedTokenMinimumConfidence"] >= min_confidence:
                    key = (match["matchedTokenRange"]["start"], match["matchedTokenRange"]["endExclusive"])
                    if key not in seen:
                        candidates.append(match)
                        seen.add(key)
                break
            if len(acc) > len(needle) + 40:
                break
    return candidates


def token_match(tokens: list[OcrToken], start: int, end: int) -> dict[str, Any]:
    left = min(t.left for t in tokens)
    top = min(t.top for t in tokens)
    right = max(t.left + t.width for t in tokens)
    bottom = max(t.top + t.height for t in tokens)
    text = " ".join(t.text for t in tokens)
    confidences = [t.confidence for t in tokens]
    return {
        "matchedText": text,
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
