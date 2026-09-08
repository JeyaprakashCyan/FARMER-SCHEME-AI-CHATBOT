from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ============================================================
# SECTION DATA MODEL
# ============================================================

@dataclass
class Section:
    page_number: int
    section: str
    subsection: str
    text: str
    heading: str | None = None
    heading_level: int = 1


# ============================================================
# HEADING PATTERNS
# ============================================================

NUMBERED_HEADING = re.compile(
    r"^\s*\d+[\.\)]\s+[A-Za-z]"
)

SUBNUMBERED_HEADING = re.compile(
    r"^\s*\d+(?:\.\d+)+[\.\)]?\s+[A-Za-z]"
)

LETTER_HEADING = re.compile(
    r"^\s*[A-Z][\.\)]\s+[A-Za-z]"
)

LOWER_LETTER_HEADING = re.compile(
    r"^\s*[a-z][\.\)]\s+[A-Za-z]"
)


# ============================================================
# COMMON GOVERNMENT HEADINGS
# ============================================================

COMMON_HEADINGS = {
    "introduction",
    "background",
    "overview",
    "objective",
    "objectives",
    "scheme objective",
    "scheme duration",
    "scheme period",
    "duration",
    "intended beneficiaries",
    "beneficiaries",
    "eligible beneficiaries",
    "eligibility",
    "eligibility criteria",
    "eligible projects",
    "eligible projects for all beneficiaries",
    "eligible projects for building community farming assets",
    "eligible activities",
    "financial assistance",
    "financial support",
    "benefits",
    "interest subvention",
    "interest subsidy",
    "loan assistance",
    "credit facility",
    "application process",
    "how to apply",
    "application procedure",
    "documents required",
    "documents",
    "checklist of documents",
    "implementation",
    "implementation mechanism",
    "monitoring",
    "monitoring mechanism",
    "funding pattern",
    "funding",
    "project components",
    "project cost",
    "project activities",
    "guidelines",
    "terms and conditions",
    "conditions",
    "coverage",
    "scope",
    "target beneficiaries",
    "assistance",
    "support",
    "conclusion",
    "related resources",
}


# ============================================================
# OCR / DEBUG MARKER DETECTION
# ============================================================

def is_debug_or_page_marker(text: str) -> bool:

    if not text:
        return False

    normalized = text.strip().lower()

    # PAGE 1 / PAGE 2 / PAGE 5
    if re.fullmatch(r"page\s+\d+", normalized):
        return True

    # Debug metadata
    if normalized.startswith("extraction method:"):
        return True

    if normalized.startswith("image count:"):
        return True

    # Separator lines
    if re.fullmatch(r"[#=_\-]{5,}", text.strip()):
        return True

    return False


# ============================================================
# BULLET DETECTION
# ============================================================

def is_bullet_point(text: str) -> bool:

    if not text:
        return False

    stripped = text.strip()

    # Normal bullets + OCR-converted bullets
    bullet_pattern = r"^[*•°◦▪▫‣⁃·\-]\s*"

    return bool(re.match(bullet_pattern, stripped))


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_heading_text(text: str) -> str:

    text = text.strip()

    # Remove numbering
    text = re.sub(
        r"^\s*\d+(?:\.\d+)*[\.\)]?\s*",
        "",
        text
    )

    # Remove letter numbering
    text = re.sub(
        r"^\s*[A-Za-z][\.\)]\s*",
        "",
        text
    )

    # Remove OCR bullet
    text = re.sub(
        r"^[*•°◦▪▫‣⁃·\-]\s*",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_count(text: str) -> int:

    return len(
        re.findall(r"\b\w+\b", text)
    )


def alphabetic_ratio(text: str) -> float:

    if not text:
        return 0.0

    alphabetic = sum(
        char.isalpha()
        for char in text
    )

    return alphabetic / max(len(text), 1)


def is_mostly_uppercase(text: str) -> bool:

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if len(letters) < 4:
        return False

    uppercase = sum(
        char.isupper()
        for char in letters
    )

    return (
        uppercase / len(letters)
        >= 0.80
    )


def looks_like_sentence(text: str) -> bool:

    words = word_count(text)

    # Long lines are almost always paragraphs
    if words > 15:
        return True

    # Sentence punctuation
    if text.endswith(
        (".", ",", ";", ":")
    ):
        return True

    return False


def has_heading_keywords(text: str) -> bool:

    normalized = normalize_heading_text(
        text
    ).lower()

    if normalized in COMMON_HEADINGS:
        return True

    for heading in COMMON_HEADINGS:

        if normalized.startswith(
            heading + ":"
        ):
            return True

    return False


# ============================================================
# EXACT COMMON HEADING MATCH
# ============================================================

def get_common_heading(text: str) -> str | None:

    normalized = normalize_heading_text(
        text
    ).lower()

    # Exact match first
    if normalized in COMMON_HEADINGS:
        return normalize_heading_text(text)

    # Detect heading followed by paragraph
    for heading in sorted(
        COMMON_HEADINGS,
        key=len,
        reverse=True
    ):

        if normalized.startswith(
            heading + " "
        ):

            return text[
                :len(heading)
            ].strip()

    return None


# ============================================================
# HEADING + CONTENT SPLITTING
# ============================================================

def split_heading_and_content(
    text: str
):
    """
    Detect cases where OCR puts the heading and
    paragraph on the same line.

    Example:

    Eligible projects for building community farming assets
    In addition to above activities farmer groups...

    OCR may produce:

    Eligible projects for building community farming assets In addition to above activities...

    Returns:

        (heading, remaining_content)

    or

        (None, text)
    """

    if not text:
        return None, text

    normalized = normalize_heading_text(
        text
    )

    normalized_lower = normalized.lower()

    # --------------------------------------------------------
    # 1. Exact heading
    # --------------------------------------------------------

    if normalized_lower in COMMON_HEADINGS:

        return normalize_heading_text(text), ""

    # --------------------------------------------------------
    # 2. Heading followed by content
    # --------------------------------------------------------

    for heading in sorted(
        COMMON_HEADINGS,
        key=len,
        reverse=True
    ):

        heading_lower = heading.lower()

        if normalized_lower.startswith(
            heading_lower + " "
        ):

            heading_length = len(heading)

            remaining = normalized[
                heading_length:
            ].strip()

            if remaining:

                return heading, remaining

    return None, text


# ============================================================
# HEADING DETECTION
# ============================================================

def looks_like_heading(text: str) -> bool:

    if not text:
        return False

    text = text.strip()

    # --------------------------------------------------------
    # Debug/page markers
    # --------------------------------------------------------

    if is_debug_or_page_marker(text):
        return False

    # --------------------------------------------------------
    # Bullet points
    # --------------------------------------------------------

    if is_bullet_point(text):
        return False

    # --------------------------------------------------------
    # Very short text
    # --------------------------------------------------------

    if len(text) < 3:
        return False

    # --------------------------------------------------------
    # Numbered headings
    # --------------------------------------------------------

    if SUBNUMBERED_HEADING.match(text):
        return True

    if NUMBERED_HEADING.match(text):
        return True

    # --------------------------------------------------------
    # Letter headings
    # --------------------------------------------------------

    if LETTER_HEADING.match(text):
        return True

    if LOWER_LETTER_HEADING.match(text):
        return True

    # --------------------------------------------------------
    # Exact known government heading
    # --------------------------------------------------------

    normalized = normalize_heading_text(
        text
    ).lower()

    if normalized in COMMON_HEADINGS:
        return True

    # --------------------------------------------------------
    # Do NOT classify heading + paragraph as heading
    # --------------------------------------------------------

    heading, remaining = split_heading_and_content(
        text
    )

    if heading and remaining:

        # The whole line is content containing a heading,
        # not a pure heading.
        return False

    # --------------------------------------------------------
    # Mostly uppercase
    # --------------------------------------------------------

    if is_mostly_uppercase(text):

        if word_count(text) <= 12:
            return True

    # --------------------------------------------------------
    # Long sentence
    # --------------------------------------------------------

    if looks_like_sentence(text):
        return False

    # --------------------------------------------------------
    # Title-like heading
    # --------------------------------------------------------

    words = text.split()

    if 1 <= len(words) <= 10:

        title_like = 0

        for word in words:

            cleaned = re.sub(
                r"[^A-Za-z]",
                "",
                word
            )

            if not cleaned:
                continue

            if cleaned[0].isupper():
                title_like += 1

        if title_like >= max(
            1,
            len(words) * 0.5
        ):
            return True

    return False


# ============================================================
# HEADING LEVEL
# ============================================================

def heading_level(text: str) -> int:

    if SUBNUMBERED_HEADING.match(text):

        match = re.match(
            r"^\s*(\d+(?:\.\d+)+)",
            text
        )

        if match:

            depth = match.group(1).count(".")

            if depth >= 2:
                return 3

            return 2

    if NUMBERED_HEADING.match(text):
        return 1

    if LETTER_HEADING.match(text):
        return 2

    if LOWER_LETTER_HEADING.match(text):
        return 3

    return 1


# ============================================================
# CLEAN HEADING
# ============================================================

def clean_heading(text: str) -> str:

    text = text.strip()

    # Numbered headings
    text = re.sub(
        r"^\s*\d+(?:\.\d+)*[\.\)]?\s*",
        "",
        text
    )

    # Letter headings
    text = re.sub(
        r"^\s*[A-Za-z][\.\)]\s*",
        "",
        text
    )

    # Bullet characters
    text = re.sub(
        r"^[*•°◦▪▫‣⁃·\-]\s*",
        "",
        text
    )

    # Spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_sections(
    page_number: int,
    text: str
) -> List[Section]:

    if not text:
        return []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    sections: List[Section] = []

    current_section = None
    current_subsection = None
    current_heading = None
    current_level = 1

    current_content: List[str] = []

    def flush_current():

        nonlocal current_content

        if not current_content:
            return

        content = "\n".join(
            current_content
        ).strip()

        if not content:
            current_content = []
            return

        section_name = (
            current_section
            or "General"
        )

        subsection_name = (
            current_subsection
            or "-"
        )

        sections.append(
            Section(
                page_number=page_number,
                section=section_name,
                subsection=subsection_name,
                text=content,
                heading=current_heading,
                heading_level=current_level,
            )
        )

        current_content = []

    # ========================================================
    # PROCESS LINES
    # ========================================================

    for line in lines:

        # ----------------------------------------------------
        # First check if line contains:
        #
        # heading + paragraph
        # ----------------------------------------------------

        embedded_heading, remaining_content = (
            split_heading_and_content(line)
        )

        if (
            embedded_heading
            and remaining_content
        ):

            # Create heading
            heading = clean_heading(
                embedded_heading
            )

            level = heading_level(
                embedded_heading
            )

            # Save previous section
            flush_current()

            if level == 1:

                current_section = heading
                current_subsection = None

            elif level == 2:

                if current_section is None:
                    current_section = heading
                else:
                    current_subsection = heading

            else:

                if current_section is None:
                    current_section = heading
                else:
                    current_subsection = heading

            current_heading = heading
            current_level = level

            # Add remaining paragraph to content
            current_content.append(
                remaining_content
            )

            continue

        # ----------------------------------------------------
        # Normal heading
        # ----------------------------------------------------

        if looks_like_heading(line):

            flush_current()

            level = heading_level(line)

            heading = clean_heading(line)

            if level == 1:

                current_section = heading
                current_subsection = None

            elif level == 2:

                if current_section is None:

                    current_section = heading

                else:

                    current_subsection = heading

            else:

                if current_section is None:

                    current_section = heading

                else:

                    current_subsection = heading

            current_heading = heading
            current_level = level

            continue

        # ----------------------------------------------------
        # Normal content / bullets
        # ----------------------------------------------------

        current_content.append(line)

    # ========================================================
    # FINAL FLUSH
    # ========================================================

    flush_current()

    return sections


# ============================================================
# PRINT SECTIONS
# ============================================================

def print_sections(
    sections: List[Section]
):

    print()

    print("=" * 100)
    print("DETECTED SECTIONS")
    print("=" * 100)

    for index, section in enumerate(
        sections,
        start=1
    ):

        print()

        print(
            f"[{index}] Page       : "
            f"{section.page_number}"
        )

        print(
            f"    Section    : "
            f"{section.section}"
        )

        print(
            f"    Subsection : "
            f"{section.subsection}"
        )

        print(
            f"    Level      : "
            f"{section.heading_level}"
        )

        print(
            f"    Heading    : "
            f"{section.heading}"
        )

        print(
            f"    Characters : "
            f"{len(section.text)}"
        )

        preview = " ".join(
            section.text.split()
        )

        if len(preview) > 300:
            preview = (
                preview[:300]
                + "..."
            )

        print(
            f"    Preview    : "
            f"{preview}"
        )

    print()

    print(
        f"Total sections detected: "
        f"{len(sections)}"
    )


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Detect sections from "
            "OCR/PDF text"
        )
    )

    parser.add_argument(
        "text_file",
        nargs="?",
        type=Path,
        help=(
            "Path to text file (defaults to the repository debug OCR file)"
        )
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    text_path = args.text_file or (
        project_root
        / "storage"
        / "debug"
        / "Agriculture Infrastructure Fund_ocr.txt"
    )

    if not text_path.is_absolute():
        text_path = project_root / text_path

    if not text_path.exists():
        raise FileNotFoundError(
            f"Text file not found: {text_path}"
        )

    text = text_path.read_text(encoding="utf-8")

    sections = detect_sections(
        page_number=1,
        text=text
    )

    print_sections(sections)
