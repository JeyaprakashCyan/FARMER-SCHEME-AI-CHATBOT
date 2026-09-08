from pathlib import Path
import sys

import pymupdf
import pytesseract
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.ocr_table_reconstructor import (
    reconstruct_table_from_ocr_data,
    print_reconstructed_table,
)


PDF_PATH = Path(
    PROJECT_ROOT
    / "data"
    / "pdf"
    / "Agriculture Infrastructure Fund.pdf"
)

TEST_PAGE = 2

OCR_DPI = 300

TESSERACT_CONFIG = "--oem 3 --psm 6"


def render_page(
    pdf_path: Path,
    page_number: int,
):
    """
    Render PDF page into a PIL image.
    """

    document = pymupdf.open(
        str(pdf_path)
    )

    try:

        page = document[
            page_number - 1
        ]

        zoom = OCR_DPI / 72

        matrix = pymupdf.Matrix(
            zoom,
            zoom,
        )

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [
                pixmap.width,
                pixmap.height,
            ],
            pixmap.samples,
        )

        return image

    finally:

        document.close()


def main():

    print("=" * 100)
    print("STEP 12 - COORDINATE BASED OCR TABLE TEST")
    print("=" * 100)

    print(
        f"PDF       : {PDF_PATH}"
    )

    print(
        f"Page      : {TEST_PAGE}"
    )

    print(
        f"OCR DPI   : {OCR_DPI}"
    )

    print(
        f"Tesseract : {pytesseract.get_tesseract_version()}"
    )

    print()

    print(
        "Rendering PDF page..."
    )

    image = render_page(
        PDF_PATH,
        TEST_PAGE,
    )

    print(
        f"Image size: {image.size}"
    )

    print()

    print(
        "Running Tesseract image_to_data()..."
    )

    data = pytesseract.image_to_data(
        image,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT,
    )

    print(
        "OCR coordinate extraction complete."
    )

    print()

    print(
        "Reconstructing table..."
    )

    result = reconstruct_table_from_ocr_data(
        data=data,
        page_number=TEST_PAGE,
    )

    print_reconstructed_table(
        result
    )

    print()

    print("=" * 100)
    print("STEP 12 TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()