from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import argparse
import re


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class NormalizedTable:
    """
    Represents one detected and normalized table.

    Attributes:
        page_number: PDF page number.
        table_number: Table number on that page.
        table_id: Unique table identifier.
        raw_lines: Original OCR lines detected as table content.
        rows: Reconstructed semantic rows.
        rag_text: Search/RAG-friendly normalized representation.
        confidence: Estimated normalization confidence from 0 to 1.
        notes: Diagnostic notes about the normalization.
    """

    page_number: int
    table_number: int
    table_id: str

    raw_lines: List[str] = field(default_factory=list)
    rows: List[str] = field(default_factory=list)

    rag_text: str = ""

    confidence: float = 0.0

    notes: List[str] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Backward-compatible properties
    # -------------------------------------------------------------------------

    @property
    def raw_text(self) -> str:
        """
        Return the original detected table text.
        """
        return "\n".join(self.raw_lines)

    @property
    def normalized_text(self) -> str:
        """
        Backward-compatible alias for rag_text.

        This prevents older test scripts from failing if they use:
            table.normalized_text
        """
        return self.rag_text


# =============================================================================
# REGEX PATTERNS
# =============================================================================

ROMAN_MARKER_PATTERN = re.compile(
    r"(?<!\w)"
    r"\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi)\)"
    r"\s*",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"""
    (?:
        ₹?\s*
        [0-9]+(?:\.[0-9]+)?
        \s*
        (?:crore|crores|lakh|lakhs|lac|lacs|rs\.?|rupees)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

PERCENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)

CAPACITY_PATTERN = re.compile(
    r"""
    \b
    \d+(?:\.\d+)?
    \s*
    (?:MT|MTPD|KG|KGS|TON|TONS|LITRE|LITRES|L|KL|MW|KW)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

TABLE_KEYWORDS = [
    "item",
    "items",
    "activity",
    "activities",
    "cost norms",
    "cost norm",
    "financial assistance",
    "assistance",
    "amount",
    "subsidy",
    "percentage",
    "project",
    "per unit",
    "per project",
    "warehouse",
    "warehouses",
    "silos",
    "pack house",
    "pack houses",
    "cold storage",
    "cold chain",
    "logistics",
    "processing",
    "ripening",
    "assaying",
    "sorting",
    "grading",
]


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_line(text: str) -> str:
    """
    Clean OCR text while preserving useful information.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize common OCR whitespace.
    text = text.replace("\t", " ")

    # Normalize repeated whitespace.
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces immediately before punctuation.
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    return text.strip()


def normalize_spaces(text: str) -> str:
    """
    Normalize whitespace without aggressively changing OCR content.
    """

    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


# =============================================================================
# TABLE DETECTION
# =============================================================================

def is_table_like_line(line: str) -> bool:
    """
    Determine whether a line looks like part of a table.

    This is intentionally conservative.
    """

    if not line:
        return False

    normalized = normalize_spaces(line).lower()

    # Strong table indicators.
    if "|" in line:
        return True

    if "\t" in line:
        return True

    # Roman numbered table rows.
    if ROMAN_MARKER_PATTERN.search(line):
        return True

    # Financial/table keywords.
    keyword_count = 0

    for keyword in TABLE_KEYWORDS:
        if keyword in normalized:
            keyword_count += 1

    # Lines containing financial values are likely table content.
    has_money = bool(MONEY_PATTERN.search(line))

    has_percent = bool(PERCENT_PATTERN.search(line))

    has_capacity = bool(CAPACITY_PATTERN.search(line))

    if keyword_count >= 2:
        return True

    if keyword_count >= 1 and (has_money or has_percent or has_capacity):
        return True

    if has_money and has_capacity:
        return True

    return False


def detect_table_blocks(text: str) -> List[List[str]]:
    """
    Detect consecutive table-like lines.

    Returns:
        List of table blocks.
    """

    if not text:
        return []

    lines = text.splitlines()

    blocks: List[List[str]] = []

    current_block: List[str] = []

    for raw_line in lines:

        line = clean_line(raw_line)

        if not line:
            if len(current_block) >= 2:
                blocks.append(current_block)

            current_block = []
            continue

        if is_table_like_line(line):

            current_block.append(line)

        else:

            if len(current_block) >= 2:
                blocks.append(current_block)

            current_block = []

    if len(current_block) >= 2:
        blocks.append(current_block)

    return blocks


# =============================================================================
# ROMAN NUMBER HANDLING
# =============================================================================

ROMAN_VALUES = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
    "xiii": 13,
    "xiv": 14,
    "xv": 15,
    "xvi": 16,
}


def roman_to_int(value: str) -> Optional[int]:
    """
    Convert roman numeral to integer.
    """

    if not value:
        return None

    value = value.lower().strip()

    return ROMAN_VALUES.get(value)


def split_roman_rows(text: str) -> List[str]:
    """
    Split OCR text using markers such as:

        (i)
        (ii)
        (iii)

    This is useful when OCR has merged multiple visual rows.
    """

    if not text:
        return []

    matches = list(ROMAN_MARKER_PATTERN.finditer(text))

    if not matches:
        return [text.strip()]

    rows = []

    # Preserve text before first marker if meaningful.
    first_start = matches[0].start()

    if first_start > 0:
        prefix = text[:first_start].strip()

        if prefix:
            rows.append(prefix)

    for index, match in enumerate(matches):

        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        row = text[start:end].strip()

        if row:
            rows.append(row)

    return rows


# =============================================================================
# ACTIVITY EXTRACTION
# =============================================================================

def extract_activity_name(text: str) -> str:
    """
    Attempt to extract the activity/item name from an OCR row.

    IMPORTANT:
    This function is conservative.

    It does not invent information when OCR is ambiguous.
    """

    if not text:
        return ""

    cleaned = normalize_spaces(text)

    # Remove roman marker.
    cleaned = ROMAN_MARKER_PATTERN.sub("", cleaned, count=1).strip()

    # Stop before obvious financial information.
    amount_match = MONEY_PATTERN.search(cleaned)

    if amount_match:
        activity = cleaned[:amount_match.start()].strip()
    else:
        activity = cleaned

    activity = activity.strip(" :-|")

    # Avoid returning extremely long corrupted OCR text as activity.
    words = activity.split()

    if len(words) > 25:
        activity = " ".join(words[:25]) + "..."

    return activity


# =============================================================================
# FINANCIAL VALUE EXTRACTION
# =============================================================================

def extract_money_values(text: str) -> List[str]:
    """
    Extract monetary values from a row.
    """

    if not text:
        return []

    values = []

    for match in MONEY_PATTERN.finditer(text):

        value = normalize_spaces(match.group(0))

        if value:
            values.append(value)

    return values


def extract_percentages(text: str) -> List[str]:
    """
    Extract percentages.
    """

    if not text:
        return []

    return [
        normalize_spaces(match.group(0))
        for match in PERCENT_PATTERN.finditer(text)
    ]


def extract_capacities(text: str) -> List[str]:
    """
    Extract capacity values such as 6MT.
    """

    if not text:
        return []

    return [
        normalize_spaces(match.group(0))
        for match in CAPACITY_PATTERN.finditer(text)
    ]


# =============================================================================
# SEMANTIC ROW RECONSTRUCTION
# =============================================================================

def reconstruct_row(text: str) -> str:
    """
    Convert one OCR fragment into a RAG-friendly semantic row.

    This does NOT attempt aggressive column guessing.
    """

    text = normalize_spaces(text)

    if not text:
        return ""

    activity = extract_activity_name(text)
    money_values = extract_money_values(text)
    percentages = extract_percentages(text)
    capacities = extract_capacities(text)

    parts = []

    if activity:
        parts.append(f"Activity: {activity}")

    if money_values:
        parts.append(
            "Cost/Financial values: " + ", ".join(money_values)
        )

    if percentages:
        parts.append(
            "Percentages: " + ", ".join(percentages)
        )

    if capacities:
        parts.append(
            "Capacity: " + ", ".join(capacities)
        )

    # If extraction was unsuccessful, preserve original OCR.
    if not parts:
        return f"Original OCR: {text}"

    # Preserve original row when OCR contains significant additional text.
    extracted_text = " ".join(
        [
            activity,
            *money_values,
            *percentages,
            *capacities,
        ]
    )

    if len(text) > len(extracted_text) + 40:
        parts.append(f"OCR text: {text}")

    return " | ".join(parts)


def reconstruct_rows(raw_lines: List[str]) -> List[str]:
    """
    Reconstruct semantic rows from raw OCR table lines.
    """

    rows: List[str] = []

    for line in raw_lines:

        line = normalize_spaces(line)

        if not line:
            continue

        # Split multiple numbered rows inside one OCR line.
        split_rows = split_roman_rows(line)

        for row in split_rows:

            row = normalize_spaces(row)

            if not row:
                continue

            normalized_row = reconstruct_row(row)

            if normalized_row:
                rows.append(normalized_row)

    return rows


# =============================================================================
# CONFIDENCE
# =============================================================================

def calculate_confidence(
    raw_lines: List[str],
    rows: List[str],
) -> float:
    """
    Estimate normalization confidence.

    This is NOT a machine-learning confidence score.
    It is a heuristic used to decide whether the table
    should be trusted for RAG.
    """

    if not raw_lines:
        return 0.0

    score = 0.0

    # Basic table structure.
    if len(raw_lines) >= 2:
        score += 0.20

    # Rows successfully reconstructed.
    if rows:
        score += 0.20

    # Numbered rows are a strong signal.
    roman_count = sum(
        len(ROMAN_MARKER_PATTERN.findall(line))
        for line in raw_lines
    )

    if roman_count >= 2:
        score += 0.20

    # Financial values detected.
    money_count = sum(
        len(extract_money_values(line))
        for line in raw_lines
    )

    if money_count >= 1:
        score += 0.15

    # Percentages.
    percent_count = sum(
        len(extract_percentages(line))
        for line in raw_lines
    )

    if percent_count >= 1:
        score += 0.10

    # Capacity.
    capacity_count = sum(
        len(extract_capacities(line))
        for line in raw_lines
    )

    if capacity_count >= 1:
        score += 0.10

    # Pipe/tab structure is stronger than pure OCR spacing.
    if any("|" in line or "\t" in line for line in raw_lines):
        score += 0.05

    return min(score, 1.0)


# =============================================================================
# NOTES / QUALITY CHECKS
# =============================================================================

def generate_notes(
    raw_lines: List[str],
    rows: List[str],
    confidence: float,
) -> List[str]:

    notes: List[str] = []

    if not raw_lines:
        notes.append("No raw table lines detected.")

    if not rows:
        notes.append("No semantic rows could be reconstructed.")

    if len(raw_lines) == 1:
        notes.append(
            "Only one table-like OCR line detected; table structure may be incomplete."
        )

    roman_count = sum(
        len(ROMAN_MARKER_PATTERN.findall(line))
        for line in raw_lines
    )

    if roman_count >= 2:
        notes.append(
            f"Detected {roman_count} numbered table-row markers."
        )

    money_count = sum(
        len(extract_money_values(line))
        for line in raw_lines
    )

    if money_count:
        notes.append(
            f"Detected {money_count} financial value(s)."
        )

    percent_count = sum(
        len(extract_percentages(line))
        for line in raw_lines
    )

    if percent_count:
        notes.append(
            f"Detected {percent_count} percentage value(s)."
        )

    capacity_count = sum(
        len(extract_capacities(line))
        for line in raw_lines
    )

    if capacity_count:
        notes.append(
            f"Detected {capacity_count} capacity value(s)."
        )

    # Important warning for the current OCR problem.
    very_long_lines = [
        line for line in raw_lines
        if len(line) > 300
    ]

    if very_long_lines:
        notes.append(
            "One or more OCR lines are very long. "
            "Possible multi-column OCR interleaving detected."
        )

    if confidence < 0.60:
        notes.append(
            "Low normalization confidence. "
            "Do not rely on reconstructed column relationships without validation."
        )

    elif confidence < 0.80:
        notes.append(
            "Medium normalization confidence. "
            "Review reconstructed rows before production ingestion."
        )

    else:
        notes.append(
            "High heuristic confidence."
        )

    return notes


# =============================================================================
# TABLE PROCESSING
# =============================================================================

def process_table_block(
    block: List[str],
    page_number: int,
    table_number: int,
) -> NormalizedTable:
    """
    Process one detected table block.
    """

    cleaned_lines = [
        clean_line(line)
        for line in block
        if clean_line(line)
    ]

    table_id = f"page_{page_number}_table_{table_number}"

    rows = reconstruct_rows(cleaned_lines)

    confidence = calculate_confidence(
        raw_lines=cleaned_lines,
        rows=rows,
    )

    notes = generate_notes(
        raw_lines=cleaned_lines,
        rows=rows,
        confidence=confidence,
    )

    rag_parts = []

    rag_parts.append(
        f"Table: {table_id}"
    )

    rag_parts.append(
        f"Page: {page_number}"
    )

    if rows:

        rag_parts.append(
            "Rows:"
        )

        for index, row in enumerate(rows, start=1):

            rag_parts.append(
                f"Row {index}: {row}"
            )

    else:

        rag_parts.append(
            "Original OCR:"
        )

        rag_parts.extend(cleaned_lines)

    rag_text = "\n".join(rag_parts)

    return NormalizedTable(
        page_number=page_number,
        table_number=table_number,
        table_id=table_id,
        raw_lines=cleaned_lines,
        rows=rows,
        rag_text=rag_text,
        confidence=confidence,
        notes=notes,
    )


def process_page(
    text: str,
    page_number: int,
) -> List[NormalizedTable]:
    """
    Detect and normalize all table-like blocks on a page.
    """

    blocks = detect_table_blocks(text)

    tables: List[NormalizedTable] = []

    for table_number, block in enumerate(blocks, start=1):

        table = process_table_block(
            block=block,
            page_number=page_number,
            table_number=table_number,
        )

        tables.append(table)

    return tables


# =============================================================================
# PRINT / DEBUG
# =============================================================================

def print_normalized_table(table: NormalizedTable) -> None:
    """
    Pretty-print one normalized table.
    """

    print()
    print("=" * 100)

    print(
        f"PAGE {table.page_number}"
    )

    print(
        f"TABLE #{table.table_number}"
    )

    print(
        f"TABLE ID: {table.table_id}"
    )

    print("=" * 100)

    print()
    print("RAW TABLE:")
    print("-" * 100)

    print(table.raw_text)

    print()
    print("RECONSTRUCTED ROWS:")
    print("-" * 100)

    if table.rows:

        for index, row in enumerate(
            table.rows,
            start=1,
        ):
            print(
                f"Row {index}: {row}"
            )

    else:

        print("No rows reconstructed.")

    print()
    print("NORMALIZED RAG TEXT:")
    print("-" * 100)

    print(table.rag_text)

    print()
    print("CONFIDENCE:")
    print("-" * 100)

    print(
        f"{table.confidence:.2f}"
    )

    print()
    print("NOTES:")
    print("-" * 100)

    if table.notes:

        for note in table.notes:
            print(
                f"- {note}"
            )

    else:

        print("No notes.")

    print()


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Detect and normalize OCR table text."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Path to a text file containing OCR output.",
    )

    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number for the supplied text.",
    )

    args = parser.parse_args()

    if not args.text:

        print(
            "No --text file supplied."
        )

        print()

        print(
            "Example:"
        )

        print(
            r"python src/parser/table_parser.py --text storage/debug/aif_page2.txt --page 2"
        )

        return

    text_path = Path(args.text)

    if not text_path.exists():

        raise FileNotFoundError(
            f"Text file not found: {text_path}"
        )

    text = text_path.read_text(
        encoding="utf-8"
    )

    tables = process_page(
        text=text,
        page_number=args.page,
    )

    if not tables:

        print(
            "No table-like blocks detected."
        )

        return

    for table in tables:

        print_normalized_table(table)


if __name__ == "__main__":
    main()