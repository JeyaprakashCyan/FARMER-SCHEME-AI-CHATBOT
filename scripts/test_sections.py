from __future__ import annotations

import sys
from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# IMPORTS
# =============================================================================

from src.parser.pdf_parser import parse_pdf
from src.parser.section_detector import detect_sections


# =============================================================================
# CONFIGURATION
# =============================================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "pdf"
    / "Agriculture Infrastructure Fund.pdf"
)


# =============================================================================
# MAIN TEST
# =============================================================================

def main():

    print()
    print("=" * 100)
    print("SECTION DETECTION TEST")
    print("=" * 100)

    print(f"PDF: {PDF_PATH}")

    if not PDF_PATH.exists():

        print()
        print("ERROR: PDF file not found.")
        print(f"Expected: {PDF_PATH}")
        print()

        return

    # =========================================================================
    # STEP 1 - Parse PDF
    # =========================================================================

    print()
    print("=" * 100)
    print("STEP 1 - PARSING PDF")
    print("=" * 100)

    parsed_document = parse_pdf(
        PDF_PATH,
        enable_ocr=True,
    )

    print()
    print(
        f"Parsed pages: "
        f"{len(parsed_document.pages)}"
    )

    # =========================================================================
    # STEP 2 - Detect sections page by page
    # =========================================================================

    print()
    print("=" * 100)
    print("STEP 2 - SECTION DETECTION")
    print("=" * 100)

    all_sections = []

    for page in parsed_document.pages:

        print()
        print("-" * 100)
        print(
            f"PAGE {page.page_number}"
        )
        print("-" * 100)

        if not page.text.strip():

            print(
                "WARNING: Page contains no extracted text."
            )

            continue

        sections = detect_sections(
            page_number=page.page_number,
            text=page.text,
        )

        if not sections:

            print(
                "WARNING: No sections detected."
            )

            continue

        for index, section in enumerate(
            sections,
            start=1,
        ):

            all_sections.append(section)

            print()
            print(
                f"Section #{index}"
            )

            print(
                f"  Page       : "
                f"{section.page_number}"
            )

            print(
                f"  Section    : "
                f"{section.section}"
            )

            print(
                f"  Subsection : "
                f"{section.subsection or '-'}"
            )

            print(
                f"  Level      : "
                f"{section.heading_level}"
            )

            print(
                f"  Heading    : "
                f"{section.heading or '-'}"
            )

            print(
                f"  Characters : "
                f"{len(section.text)}"
            )

            # -------------------------------------------------------------
            # Preview
            # -------------------------------------------------------------

            preview = (
                section.text
                .replace("\n", " ")
                .strip()
            )

            if len(preview) > 250:
                preview = preview[:250] + "..."

            print(
                f"  Preview    : "
                f"{preview}"
            )

    # =========================================================================
    # STEP 3 - Summary
    # =========================================================================

    print()
    print("=" * 100)
    print("STEP 3 - SECTION SUMMARY")
    print("=" * 100)

    print(
        f"Total pages       : "
        f"{len(parsed_document.pages)}"
    )

    print(
        f"Total sections    : "
        f"{len(all_sections)}"
    )

    # =========================================================================
    # Group by page
    # =========================================================================

    print()
    print("Sections by page:")

    for page_number in range(
        1,
        len(parsed_document.pages) + 1,
    ):

        page_sections = [
            section
            for section in all_sections
            if section.page_number == page_number
        ]

        print(
            f"  Page {page_number}: "
            f"{len(page_sections)} section(s)"
        )

    # =========================================================================
    # Print section hierarchy
    # =========================================================================

    print()
    print("=" * 100)
    print("DETECTED SECTION HIERARCHY")
    print("=" * 100)

    for section in all_sections:

        if section.subsection:

            print(
                f"Page {section.page_number} | "
                f"LEVEL {section.heading_level} | "
                f"{section.section} "
                f"> "
                f"{section.subsection}"
            )

        else:

            print(
                f"Page {section.page_number} | "
                f"LEVEL {section.heading_level} | "
                f"{section.section}"
            )

    # =========================================================================
    # Basic validation
    # =========================================================================

    print()
    print("=" * 100)
    print("VALIDATION")
    print("=" * 100)

    # These should NOT appear as real sections.
    invalid_sections = []

    for section in all_sections:

        section_name = (
            section.section or ""
        ).strip().lower()

        subsection_name = (
            section.subsection or ""
        ).strip().lower()

        combined = (
            f"{section_name} "
            f"{subsection_name}"
        )

        if section_name.startswith("page "):
            invalid_sections.append(
                f"PAGE marker: {section.section}"
            )

        if "extraction method:" in combined:
            invalid_sections.append(
                f"Extraction marker: {section.section}"
            )

        if "image count:" in combined:
            invalid_sections.append(
                f"Image marker: {section.section}"
            )

        if section.heading and section.heading.startswith("*"):
            invalid_sections.append(
                f"Bullet heading: {section.heading}"
            )

    if invalid_sections:

        print()
        print(
            "WARNING: Potential false headings found:"
        )

        for item in invalid_sections:
            print(
                f"  - {item}"
            )

    else:

        print(
            "SUCCESS: No obvious debug headers or "
            "bullet headings detected."
        )

    # =========================================================================
    # Final
    # =========================================================================

    print()
    print("=" * 100)
    print("SECTION DETECTION TEST COMPLETE")
    print("=" * 100)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()