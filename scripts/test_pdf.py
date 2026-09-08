import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.pdf_parser import parse_pdf


PDF_DIR = PROJECT_ROOT / "data" / "pdf"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect text extraction and OCR for repository PDFs."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        type=Path,
        help="One PDF to inspect; all PDFs in data/pdf when omitted",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback",
    )
    args = parser.parse_args()

    if args.pdf_path:
        pdf_path = args.pdf_path
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_ROOT / pdf_path
        pdf_files = [pdf_path]
    else:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        parser.error(f"No PDF files found in {PDF_DIR}")

    print(
        f"Found {len(pdf_files)} PDFs"
    )

    for pdf_path in pdf_files:

        print("\n" + "=" * 80)

        print(
            pdf_path.name
        )

        print("=" * 80)

        parsed = parse_pdf(
            pdf_path,
            enable_ocr=not args.no_ocr
        )

        pymupdf_pages = 0
        ocr_pages = 0
        empty_pages = 0

        for page in parsed.pages:

            chars = len(
                page.text.strip()
            )

            print(
                f"Page {page.page_number}: "
                f"{chars} chars | "
                f"{page.extraction_method}"
            )

            if (
                page.extraction_method
                == "ocr"
            ):

                ocr_pages += 1

            else:

                pymupdf_pages += 1

            if chars == 0:

                empty_pages += 1

        print(
            "\nPyMuPDF pages:",
            pymupdf_pages
        )

        print(
            "OCR pages:",
            ocr_pages
        )

        print(
            "Empty pages:",
            empty_pages
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())