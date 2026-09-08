from __future__ import annotations

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORTS
# ============================================================

from src.parser.pdf_parser import parse_pdf
from src.parser.table_parser import process_page


# ============================================================
# TEST PDF
# ============================================================

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "pdf"
    / "Agriculture Infrastructure Fund.pdf"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("TABLE NORMALIZATION TEST")
    print("=" * 100)

    print(
        f"PDF: {PDF_PATH}"
    )

    if not PDF_PATH.exists():

        print()
        print(
            "ERROR: PDF file not found."
        )

        print(
            f"Expected: {PDF_PATH}"
        )

        return

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 1 - PARSING PDF")
    print("=" * 100)

    parsed_document = parse_pdf(
        PDF_PATH,
        enable_ocr=True
    )

    print()
    print(
        f"Total pages: "
        f"{len(parsed_document.pages)}"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 2 - TABLE DETECTION")
    print("=" * 100)

    all_tables = []

    for page in parsed_document.pages:

        print()
        print(
            "-" * 100
        )

        print(
            f"PAGE {page.page_number}"
        )

        print(
            f"Extraction method: "
            f"{page.extraction_method}"
        )

        tables = process_page(
            page_number=page.page_number,
            text=page.text,
        )

        if not tables:

            print(
                "No table-like blocks detected."
            )

            continue

        for table in tables:

            all_tables.append(table)

            print()
            print(
                f"TABLE #{table.table_number}"
            )

            print(
                "Raw table:"
            )

            print(
                table.raw_text
            )

            print()
            print(
                "Normalized RAG text:"
            )

            print(
                table.normalized_text
            )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 3 - TABLE SUMMARY")
    print("=" * 100)

    print(
        f"Total pages processed : "
        f"{len(parsed_document.pages)}"
    )

    print(
        f"Total tables detected : "
        f"{len(all_tables)}"
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 4 - VALIDATION")
    print("=" * 100)

    problems = []

    for table in all_tables:

        if not table.normalized_text.strip():

            problems.append(
                f"Page {table.page_number}, "
                f"Table {table.table_number}: "
                f"empty normalized text"
            )

        if not table.rows:

            problems.append(
                f"Page {table.page_number}, "
                f"Table {table.table_number}: "
                f"no rows"
            )

    if problems:

        print(
            "WARNING: Problems detected."
        )

        for problem in problems:

            print(
                f"  - {problem}"
            )

    else:

        print(
            "SUCCESS: All detected tables "
            "have normalized RAG text."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 100)
    print(
        "TABLE NORMALIZATION TEST COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":

    main()

