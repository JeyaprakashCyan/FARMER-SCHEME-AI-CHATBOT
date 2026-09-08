import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import process_pdf

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect parent and child chunks created from a PDF."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "pdf" / "Agriculture Infrastructure Fund.pdf",
        help="PDF to inspect (defaults to Agriculture Infrastructure Fund.pdf)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback",
    )
    args = parser.parse_args()
    pdf_path = args.pdf_path
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"File not found: {pdf_path}"
        )

    parents, children = process_pdf(
        pdf_path,
        enable_ocr=not args.no_ocr,
    )

    print("\n")
    print("=" * 80)
    print("PARENT CHUNKS")
    print("=" * 80)

    for index, doc in enumerate(
        parents[:10],
        start=1
    ):

        print("\n")
        print(
            f"PARENT #{index}"
        )

        print(
            "Page:",
            doc.metadata.get(
                "page_number"
            )
        )

        print(
            "Section:",
            doc.metadata.get(
                "section"
            )
        )

        print(
            "Subsection:",
            doc.metadata.get(
                "subsection"
            )
        )

        print(
            "Parent ID:",
            doc.metadata.get(
                "parent_id"
            )
        )

        print(
            "\nContent:"
        )

        print(
            doc.page_content[:1000]
        )

    print("\n")
    print("=" * 80)
    print("CHILD CHUNKS")
    print("=" * 80)

    for index, doc in enumerate(
        children[:20],
        start=1
    ):

        print("\n")
        print(
            f"CHILD #{index}"
        )

        print(
            "Page:",
            doc.metadata.get(
                "page_number"
            )
        )

        print(
            "Section:",
            doc.metadata.get(
                "section"
            )
        )

        print(
            "Subsection:",
            doc.metadata.get(
                "subsection"
            )
        )

        print(
            "Parent ID:",
            doc.metadata.get(
                "parent_id"
            )
        )

        print(
            "Chunk ID:",
            doc.metadata.get(
                "chunk_id"
            )
        )

        print(
            "\nContent:"
        )

        print(
            doc.page_content[:1000]
        )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)

    print(
        "Parents:",
        len(parents)
    )

    print(
        "Children:",
        len(children)
    )

    if children:

        lengths = [
            len(doc.page_content)
            for doc in children
        ]

        print(
            "Smallest child:",
            min(lengths)
        )

        print(
            "Largest child:",
            max(lengths)
        )

        print(
            "Average child:",
            round(
                sum(lengths)
                / len(lengths)
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())