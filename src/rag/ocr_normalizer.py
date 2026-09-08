"""
OCR NORMALIZATION + QUALITY VALIDATION

Purpose:
1. Clean OCR text without destroying meaningful information.
2. Normalize common OCR formatting problems.
3. Remove obvious page noise.
4. Detect suspicious/low-quality OCR.
5. Preserve numbers, units, scheme names and important content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TEXT_LENGTH = 80
MIN_WORD_COUNT = 15
LOW_CONFIDENCE_THRESHOLD = 0.65

# Common noise found in scanned government PDFs.
NOISE_PATTERNS = [
    r"^\s*vikaspedia\.in\s*$",
    r"^\s*www\.vikaspedia\.in\s*$",
]

# Common OCR substitutions that are relatively safe.
CHARACTER_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00a0": " ",
    "\u200b": "",
    "\u200c": "",
    "\u200d": "",
}


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class OCRQuality:
    confidence: float
    character_count: int
    word_count: int
    line_count: int
    has_meaningful_text: bool
    warning: bool
    issues: List[str]


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters while preserving useful text.
    """

    for old, new in CHARACTER_REPLACEMENTS.items():
        text = text.replace(old, new)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces without aggressively joining words.
    """

    # Replace tabs with spaces.
    text = text.replace("\t", " ")

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Normalize repeated spaces.
    text = re.sub(r"[ ]{2,}", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def repair_common_word_spacing(text: str) -> str:
    """
    Repair a limited set of common OCR word-spacing problems.

    This intentionally does NOT attempt aggressive dictionary-based
    word reconstruction because that can damage legitimate terms.
    """

    replacements = {
        "Infrastructurefor": "Infrastructure for",
        "financingfacility": "financing facility",
        "post-harvestmanagement": "post-harvest management",
        "farm-gateand": "farm-gate and",
        "Eligibleprojects": "Eligible projects",
        "beneficiariesA": "beneficiaries A",
        "smartand": "smart and",
        "precisionagriculture": "precision agriculture",
        "Howto": "How to",
        "Relatedresources": "Related resources",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_bullets(text: str) -> str:
    """
    Normalize common OCR bullet characters.
    """

    text = text.replace("•", "*")
    text = text.replace("▪", "*")
    text = text.replace("●", "*")
    text = text.replace("·", "*")

    return text


def remove_noise_lines(text: str) -> str:
    """
    Remove obvious standalone noise lines.
    """

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        remove_line = False

        for pattern in NOISE_PATTERNS:
            if re.match(pattern, stripped, flags=re.IGNORECASE):
                remove_line = True
                break

        if not remove_line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_ocr_text(text: str) -> str:
    """
    Main OCR normalization pipeline.
    """

    if not text:
        return ""

    text = normalize_unicode(text)
    text = remove_noise_lines(text)
    text = normalize_bullets(text)
    text = repair_common_word_spacing(text)
    text = normalize_whitespace(text)

    # Remove leading/trailing whitespace.
    text = text.strip()

    return text


# ============================================================
# OCR ARTIFACT DETECTION
# ============================================================

def detect_ocr_artifacts(text: str) -> List[str]:
    """
    Detect suspicious OCR patterns.

    These are warnings only.
    We do not automatically modify the text based on them.
    """

    issues = []

    if not text:
        issues.append("empty_text")
        return issues

    words = text.split()

    if len(text) < MIN_TEXT_LENGTH:
        issues.append("very_short_text")

    if len(words) < MIN_WORD_COUNT:
        issues.append("low_word_count")

    # Excessive strange symbols.
    strange_chars = re.findall(
        r"[^A-Za-z0-9\s.,;:!?%()\-/'\"&+*=₹$€£]",
        text
    )

    if len(strange_chars) > max(10, len(text) * 0.03):
        issues.append("many_unusual_characters")

    # Very long sequences without spaces can indicate OCR damage.
    long_tokens = [
        word for word in words
        if len(word) >= 35
    ]

    if long_tokens:
        issues.append("possible_word_joining")

    # Repeated punctuation/symbol noise.
    if re.search(r"[^\w\s]{6,}", text):
        issues.append("symbol_sequence")

    return issues


# ============================================================
# QUALITY ANALYSIS
# ============================================================

def analyze_ocr_quality(
    text: str,
    confidence: float,
) -> OCRQuality:
    """
    Calculate quality information for one OCR page.
    """

    issues = detect_ocr_artifacts(text)

    character_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())

    meaningful_text = (
        character_count >= MIN_TEXT_LENGTH
        and word_count >= MIN_WORD_COUNT
    )

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        issues.append("low_ocr_confidence")

    warning = (
        confidence < LOW_CONFIDENCE_THRESHOLD
        or not meaningful_text
        or len(issues) >= 3
    )

    return OCRQuality(
        confidence=confidence,
        character_count=character_count,
        word_count=word_count,
        line_count=line_count,
        has_meaningful_text=meaningful_text,
        warning=warning,
        issues=issues,
    )


# ============================================================
# PAGE VALIDATION
# ============================================================

def validate_ocr_page(
    text: str,
    confidence: float,
) -> tuple[str, OCRQuality]:
    """
    Normalize and validate a single OCR page.
    """

    cleaned_text = normalize_ocr_text(text)

    quality = analyze_ocr_quality(
        cleaned_text,
        confidence,
    )

    return cleaned_text, quality


# ============================================================
# PRINTING
# ============================================================

def print_ocr_quality(
    quality: OCRQuality,
) -> None:

    print(f"Confidence      : {quality.confidence:.2f}")
    print(f"Characters      : {quality.character_count}")
    print(f"Words           : {quality.word_count}")
    print(f"Lines           : {quality.line_count}")
    print(
        f"Meaningful text : "
        f"{'YES' if quality.has_meaningful_text else 'NO'}"
    )
    print(
        f"Warning         : "
        f"{'YES' if quality.warning else 'NO'}"
    )

    if quality.issues:
        print("Issues          :", ", ".join(quality.issues))
    else:
        print("Issues          : None")