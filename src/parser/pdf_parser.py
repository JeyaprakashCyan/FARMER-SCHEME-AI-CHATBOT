from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum amount of useful text expected from a page.
MIN_TEXT_CHARS_PER_PAGE = 100

# OCR rendering quality.
# 250 DPI gives a good balance between OCR accuracy and processing time.
OCR_DPI = 250

# Tesseract page segmentation mode.
# PSM 6 = Assume a single uniform block of text.
OCR_CONFIG = "--oem 3 --psm 6"

# Project root:
# farmer-scheme-chatbot/
# └── src/
#     └── parser/
#         └── pdf_parser.py
#
# parents[0] = parser
# parents[1] = src
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Debug output directory.
DEBUG_DIR = PROJECT_ROOT / "storage" / "debug"


# =============================================================================
# TESSERACT DETECTION
# =============================================================================

def find_tesseract() -> Optional[str]:
    """
    Try to find the Tesseract executable.

    Priority:
    1. Existing Tesseract available in PATH
    2. Common Windows installation locations
    """

    # -------------------------------------------------------------------------
    # 1. Check PATH
    # -------------------------------------------------------------------------

    tesseract_path = shutil.which("tesseract")

    if tesseract_path:
        return tesseract_path

    # -------------------------------------------------------------------------
    # 2. Common Windows locations
    # -------------------------------------------------------------------------

    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Tesseract-OCR\tesseract.exe"),
    ]

    for path in common_paths:
        if path.exists():
            return str(path)

    return None


# Detect Tesseract once when module is loaded.
TESSERACT_PATH = find_tesseract()

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ParsedPage:
    """
    Represents one parsed PDF page.
    """

    page_number: int
    text: str
    extraction_method: str

    image_count: int = 0

    # Optional block-level information.
    blocks: List[dict] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """
    Represents a complete parsed PDF document.
    """

    pdf_path: str
    pages: List[ParsedPage]

    @property
    def full_text(self) -> str:
        """
        Combine all page text into one document.
        """

        return "\n\n".join(
            page.text
            for page in self.pages
            if page.text.strip()
        )


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_unicode(text: str) -> str:
    """
    Normalize common Unicode characters produced by OCR.

    This avoids aggressive ASCII-only cleaning because scheme documents
    may contain Unicode characters.
    """

    if not text:
        return ""

    # Normalize Unicode representation.
    text = unicodedata.normalize("NFKC", text)

    # Smart quotes.
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",

        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',

        # Dashes.
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",

        # Non-breaking space.
        "\u00a0": " ",

        # Ellipsis.
        "\u2026": "...",

        # Bullet.
        "\u2022": "-",

        # Other common OCR characters.
        "\u00ad": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def remove_control_characters(text: str) -> str:
    """
    Remove invisible/control characters while preserving:
    - newline
    - carriage return
    - tab
    """

    if not text:
        return ""

    cleaned = []

    for char in text:
        if char in ("\n", "\r", "\t"):
            cleaned.append(char)
            continue

        category = unicodedata.category(char)

        if category.startswith("C"):
            cleaned.append(" ")
        else:
            cleaned.append(char)

    return "".join(cleaned)


def normalize_spaces(text: str) -> str:
    """
    Normalize spaces and tabs while preserving paragraphs.
    """

    if not text:
        return ""

    # Replace tabs with spaces.
    text = text.replace("\t", " ")

    # Collapse multiple spaces.
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces at beginning/end of lines.
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n[ ]+", "\n", text)

    return text


def fix_hyphenated_line_breaks(text: str) -> str:
    """
    Fix words split across lines.

    Example:

        infrastruc-
        ture

    becomes:

        infrastructure

    Only alphabetic word splits are handled.
    """

    if not text:
        return ""

    pattern = r"([A-Za-z])-\n([A-Za-z])"

    return re.sub(
        pattern,
        lambda match: match.group(1) + match.group(2),
        text,
    )


def fix_wrapped_lines(text: str) -> str:
    """
    Join lines that are clearly part of the same sentence.

    Example:

        The Agriculture Infrastructure
        Fund provides financial assistance

    becomes:

        The Agriculture Infrastructure Fund provides financial assistance

    A line ending in punctuation remains a separate line.
    """

    if not text:
        return ""

    lines = text.splitlines()

    output = []
    current = ""

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            if current:
                output.append(current.strip())
                current = ""

            output.append("")
            continue

        # First line.
        if not current:
            current = line
            continue

        previous = current.rstrip()

        # ---------------------------------------------------------------------
        # Do not join obvious headings / numbered items.
        # ---------------------------------------------------------------------

        looks_like_new_heading = bool(
            re.match(
                r"^(?:"
                r"\d+(?:\.\d+)*[\.\)]?\s+"
                r"|[A-Z][A-Z\s&/\-]{3,}"
                r")",
                line,
            )
        )

        # ---------------------------------------------------------------------
        # Previous line ending with punctuation normally indicates
        # sentence completion.
        # ---------------------------------------------------------------------

        previous_ends_sentence = bool(
            re.search(r"[.!?:;]$", previous)
        )

        # ---------------------------------------------------------------------
        # Current line looks like a bullet.
        # ---------------------------------------------------------------------

        current_is_bullet = bool(
            re.match(
                r"^(?:[-*•]|\(?\d+[\.\)]|\(?[a-zA-Z][\.\)])\s+",
                line,
            )
        )

        # ---------------------------------------------------------------------
        # Join lower-case continuation lines.
        # ---------------------------------------------------------------------

        starts_lowercase = bool(
            re.match(r"^[a-z]", line)
        )

        if (
            not looks_like_new_heading
            and not current_is_bullet
            and not previous_ends_sentence
        ):
            current = previous + " " + line

        elif starts_lowercase and not looks_like_new_heading:
            current = previous + " " + line

        else:
            output.append(previous)
            current = line

    if current:
        output.append(current.strip())

    return "\n".join(output)


def normalize_punctuation_spacing(text: str) -> str:
    """
    Normalize spaces around common punctuation.

    Avoid aggressive transformations because this is scheme/legal content.
    """

    if not text:
        return ""

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Ensure a reasonable space after punctuation where appropriate.
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)

    return text


def collapse_repeated_punctuation(text: str) -> str:
    """
    Remove obvious OCR punctuation duplication.

    Example:
        "...." -> "..."
        "!!!!" -> "!"
    """

    if not text:
        return ""

    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    return text


def clean_text(text: str) -> str:
    """
    Main text cleaning pipeline.

    Important:
    This function intentionally does NOT perform aggressive dictionary-based
    OCR corrections such as:

        InfrastructUrefor -> Infrastructure for

    because such automatic replacements can accidentally change legal,
    financial, scheme, organization, or technical terminology.
    """

    if not text:
        return ""

    # Step 1 - Unicode normalization.
    text = normalize_unicode(text)

    # Step 2 - Remove invisible/control characters.
    text = remove_control_characters(text)

    # Step 3 - Normalize spaces.
    text = normalize_spaces(text)

    # Step 4 - Fix words split across lines.
    text = fix_hyphenated_line_breaks(text)

    # Step 5 - Normalize spaces again.
    text = normalize_spaces(text)

    # Step 6 - Join obvious sentence continuation lines.
    text = fix_wrapped_lines(text)

    # Step 7 - Normalize punctuation spacing.
    text = normalize_punctuation_spacing(text)

    # Step 8 - Remove obvious repeated punctuation.
    text = collapse_repeated_punctuation(text)

    # Step 9 - Normalize excessive blank lines.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Step 10 - Remove leading/trailing whitespace.
    text = text.strip()

    return text


# =============================================================================
# TEXT QUALITY
# =============================================================================

def text_quality_is_sufficient(text: str) -> bool:
    """
    Determine whether extracted text contains enough useful information.

    This is intentionally conservative.

    A page is considered useful when it contains:
    - At least MIN_TEXT_CHARS_PER_PAGE characters
    - At least 20 words
    - At least 5 unique words
    - At least 40 alphabetic characters
    - Not just a URL/domain
    """

    if not text:
        return False

    cleaned = clean_text(text)

    if len(cleaned) < MIN_TEXT_CHARS_PER_PAGE:
        return False

    # Extract words.
    words = re.findall(r"\b\w+\b", cleaned, flags=re.UNICODE)

    if len(words) < 20:
        return False

    # Unique words.
    unique_words = {
        word.lower()
        for word in words
        if len(word) > 1
    }

    if len(unique_words) < 5:
        return False

    # Require a reasonable amount of alphabetic content.
    alphabetic_chars = sum(
        1
        for char in cleaned
        if char.isalpha()
    )

    if alphabetic_chars < 40:
        return False

    # Reject pages containing essentially only a URL/domain.
    url_pattern = re.compile(
        r"^(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/?$"
    )

    if url_pattern.fullmatch(cleaned):
        return False

    return True


# =============================================================================
# IMAGE / OCR
# =============================================================================

def render_page_to_image(
    page,
    dpi: int = OCR_DPI,
) -> Image.Image:
    """
    Render a PDF page as a PIL image.
    """

    zoom = dpi / 72.0

    matrix = pymupdf.Matrix(
        zoom,
        zoom,
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    return image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess an OCR image.

    Processing:
    1. Convert to grayscale
    2. Auto contrast
    3. Mild sharpening

    We intentionally avoid aggressive thresholding because it can damage:
    - tables
    - thin characters
    - numbers
    - financial values
    """

    if image.mode != "L":
        image = ImageOps.grayscale(image)

    image = ImageOps.autocontrast(image)

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


def ocr_page(page) -> str:
    """
    OCR one PDF page using Tesseract.
    """

    if not TESSERACT_PATH:
        raise RuntimeError(
            "Tesseract OCR was not found.\n\n"
            "Install Tesseract OCR and make sure it is available in PATH.\n"
            "On Windows, a common installation path is:\n"
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    # Render PDF page.
    image = render_page_to_image(
        page,
        dpi=OCR_DPI,
    )

    # Preprocess.
    image = preprocess_image(image)

    # OCR.
    text = pytesseract.image_to_string(
        image,
        config=OCR_CONFIG,
    )

    return clean_text(text)


# =============================================================================
# PDF PARSING
# =============================================================================

def parse_pdf(
    pdf_path: str | Path,
    enable_ocr: bool = True,
) -> ParsedDocument:
    """
    Parse a PDF using:

        PyMuPDF
            ↓
        text quality check
            ↓
        OCR fallback
            ↓
        cleaned text

    OCR is only attempted when native PDF extraction does not contain
    sufficient useful text.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {pdf_path}"
        )

    print()
    print("=" * 80)
    print("PDF PARSER")
    print("=" * 80)
    print(f"PDF file       : {pdf_path}")
    print(f"OCR enabled    : {enable_ocr}")
    print(f"OCR DPI        : {OCR_DPI}")
    print(f"OCR config     : {OCR_CONFIG}")
    print(f"Tesseract      : {TESSERACT_PATH or 'NOT FOUND'}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Open PDF.
    # -------------------------------------------------------------------------

    try:
        pdf_document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to open PDF: {pdf_path}\n"
            f"Error: {exc}"
        ) from exc

    pages: List[ParsedPage] = []

    try:

        total_pages = len(pdf_document)

        print(f"Total pages    : {total_pages}")
        print()

        for page_index in range(total_pages):

            page_number = page_index + 1

            page = pdf_document[page_index]

            # -----------------------------------------------------------------
            # Native PyMuPDF extraction.
            # -----------------------------------------------------------------

            try:
                normal_text = page.get_text("text")
            except Exception as exc:
                print(
                    f"Page {page_number}: "
                    f"PyMuPDF extraction error: {exc}"
                )
                normal_text = ""

            normal_text = clean_text(normal_text)

            # -----------------------------------------------------------------
            # Images.
            # -----------------------------------------------------------------

            try:
                images = page.get_images(full=True)
                image_count = len(images)
            except Exception:
                image_count = 0

            # -----------------------------------------------------------------
            # Determine whether native text is useful.
            # -----------------------------------------------------------------

            normal_quality = text_quality_is_sufficient(
                normal_text
            )

            extraction_method = "pymupdf"
            final_text = normal_text

            print(
                f"Page {page_number:>3}/{total_pages:<3} | "
                f"PyMuPDF chars: {len(normal_text):>5} | "
                f"images: {image_count:>2} | "
                f"quality: {'OK' if normal_quality else 'LOW'}"
            )

            # -----------------------------------------------------------------
            # OCR fallback.
            # -----------------------------------------------------------------

            if enable_ocr and not normal_quality:

                if not TESSERACT_PATH:
                    print(
                        f"  -> OCR skipped: "
                        f"Tesseract not found."
                    )

                else:

                    print(
                        f"  -> Running OCR for page {page_number}..."
                    )

                    try:

                        ocr_text = ocr_page(page)

                        ocr_quality = text_quality_is_sufficient(
                            ocr_text
                        )

                        print(
                            f"  -> OCR chars: {len(ocr_text):>5} | "
                            f"quality: "
                            f"{'OK' if ocr_quality else 'LOW'}"
                        )

                        # -----------------------------------------------------
                        # Prefer OCR when it passes quality validation.
                        # -----------------------------------------------------

                        if ocr_quality:

                            final_text = ocr_text
                            extraction_method = "ocr"

                        # -----------------------------------------------------
                        # If OCR quality is low but has more useful text,
                        # retain it rather than losing the page entirely.
                        # -----------------------------------------------------

                        elif len(ocr_text) > len(normal_text):

                            final_text = ocr_text
                            extraction_method = "ocr_low_quality"

                        # -----------------------------------------------------
                        # Otherwise retain PyMuPDF text.
                        # -----------------------------------------------------

                        else:

                            final_text = normal_text
                            extraction_method = "pymupdf_low_quality"

                    except Exception as exc:

                        print(
                            f"  -> OCR ERROR on page "
                            f"{page_number}: {exc}"
                        )

                        final_text = normal_text

                        if normal_text:
                            extraction_method = "pymupdf"
                        else:
                            extraction_method = "failed"

            # -----------------------------------------------------------------
            # No OCR requested.
            # -----------------------------------------------------------------

            elif not enable_ocr:

                if normal_quality:
                    extraction_method = "pymupdf"
                else:
                    extraction_method = "pymupdf_low_quality"

            # -----------------------------------------------------------------
            # Completely failed extraction.
            # -----------------------------------------------------------------

            if not final_text.strip():

                extraction_method = "failed"

            # -----------------------------------------------------------------
            # Store page.
            # -----------------------------------------------------------------

            parsed_page = ParsedPage(
                page_number=page_number,
                text=final_text,
                extraction_method=extraction_method,
                image_count=image_count,
            )

            pages.append(parsed_page)

            print(
                f"  -> Final: {extraction_method}, "
                f"chars: {len(final_text)}"
            )

            print()

    finally:

        pdf_document.close()

    # =============================================================================
    # DOCUMENT RESULT
    # =============================================================================

    parsed_document = ParsedDocument(
        pdf_path=str(pdf_path),
        pages=pages,
    )

    # =============================================================================
    # SUMMARY
    # =============================================================================

    pymupdf_pages = sum(
        1
        for page in pages
        if page.extraction_method == "pymupdf"
    )

    ocr_pages = sum(
        1
        for page in pages
        if page.extraction_method == "ocr"
    )

    low_quality_pages = sum(
        1
        for page in pages
        if page.extraction_method
        in {
            "ocr_low_quality",
            "pymupdf_low_quality",
        }
    )

    failed_pages = sum(
        1
        for page in pages
        if page.extraction_method == "failed"
    )

    print("=" * 80)
    print("PARSING COMPLETE")
    print("=" * 80)
    print(f"PDF                  : {pdf_path.name}")
    print(f"Pages                : {len(pages)}")
    print(f"PyMuPDF pages        : {pymupdf_pages}")
    print(f"OCR pages            : {ocr_pages}")
    print(f"Low-quality pages    : {low_quality_pages}")
    print(f"Failed pages         : {failed_pages}")
    print(
        f"Total extracted chars: "
        f"{len(parsed_document.full_text):,}"
    )
    print("=" * 80)

    return parsed_document


# =============================================================================
# DEBUG OUTPUT
# =============================================================================

def save_debug_text(
    parsed_document: ParsedDocument,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Save parsed text for manual inspection.

    The debug file includes:
    - PDF name
    - page number
    - extraction method
    - image count
    - extracted text
    """

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path is None:

        pdf_name = Path(
            parsed_document.pdf_path
        ).stem

        output_path = (
            DEBUG_DIR
            / f"{pdf_name}_ocr.txt"
        )

    output_path = Path(output_path)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 100
            + "\n"
        )

        file.write(
            f"PDF: {parsed_document.pdf_path}\n"
        )

        file.write(
            "=" * 100
            + "\n\n"
        )

        for page in parsed_document.pages:

            file.write(
                "\n"
                + "#" * 100
                + "\n"
            )

            file.write(
                f"PAGE {page.page_number}\n"
            )

            file.write(
                f"Extraction method: "
                f"{page.extraction_method}\n"
            )

            file.write(
                f"Image count: "
                f"{page.image_count}\n"
            )

            file.write(
                "#" * 100
                + "\n\n"
            )

            file.write(
                page.text
                + "\n"
            )

            file.write(
                "\n\n"
            )

    print()
    print(
        f"Debug text saved to:\n"
        f"{output_path}"
    )

    return output_path


# =============================================================================
# PAGE SUMMARY
# =============================================================================

def print_page_summary(
    parsed_document: ParsedDocument,
) -> None:
    """
    Print a compact summary of each page.
    """

    print()
    print("=" * 80)
    print("PAGE SUMMARY")
    print("=" * 80)

    for page in parsed_document.pages:

        preview = (
            page.text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 150:
            preview = preview[:150] + "..."

        print(
            f"Page {page.page_number:>3} | "
            f"{page.extraction_method:<22} | "
            f"{len(page.text):>5} chars | "
            f"{preview}"
        )

    print("=" * 80)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parse farmer scheme PDFs using PyMuPDF "
            "with OCR fallback."
        )
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file",
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback",
    )

    parser.add_argument(
        "--output",
        help=(
            "Optional path for debug text output"
        ),
        default=None,
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Parse.
    # -------------------------------------------------------------------------

    parsed_document = parse_pdf(
        pdf_path=args.pdf_path,
        enable_ocr=not args.no_ocr,
    )

    # -------------------------------------------------------------------------
    # Print summary.
    # -------------------------------------------------------------------------

    print_page_summary(
        parsed_document
    )

    # -------------------------------------------------------------------------
    # Save debug output.
    # -------------------------------------------------------------------------

    save_debug_text(
        parsed_document,
        output_path=args.output,
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()