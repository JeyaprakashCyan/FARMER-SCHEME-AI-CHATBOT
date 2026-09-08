# ============================================================
# PRODUCTION PDF INGESTION
# STEP 12.16.3
# OCR RETRY + PAGE VALIDATION
# ============================================================

from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf
import pytesseract
from PIL import Image, ImageOps, ImageFilter

from langchain_core.documents import Document

from src.rag.metadata_schema import (
    DocumentMetadata,
    metadata_to_dict,
    validate_metadata,
)

from src.rag.chunking import (
    chunk_documents,
    print_chunk_summary,
)

from src.rag.embeddings import create_embedding_model

from src.rag.vector_store import (
    build_faiss_index,
    save_faiss_index,
)

from src.rag.ocr_normalizer import validate_ocr_page


# ============================================================
# CONFIGURATION
# ============================================================

RAW_PDF_DIR = PROJECT_ROOT / "data" / "pdf"

FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "vector_store" / "faiss_index"

OCR_DPI = 250

OCR_LANGUAGE = "eng"

# First-pass OCR quality threshold
LOW_CONFIDENCE_THRESHOLD = 0.65

# Minimum useful OCR text
MIN_TEXT_LENGTH = 80
MIN_WORD_COUNT = 15


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

COMMON_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def configure_tesseract():
    """
    Automatically locate Tesseract on Windows.
    """

    for path in COMMON_TESSERACT_PATHS:

        if path.exists():

            pytesseract.pytesseract.tesseract_cmd = str(path)

            print(
                f"Tesseract executable: {path}"
            )

            return path

    print(
        "WARNING: Tesseract executable was not "
        "found in the standard Windows locations."
    )

    return None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Remove common web footer noise
    noise_patterns = [
        r"vikaspedia\.in",
        r"www\.vikaspedia\.in",
        r"http[s]?://\S+",
    ]

    for pattern in noise_patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE,
        )

    # Normalize unicode
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Preserve line structure
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# OCR QUALITY
# ============================================================

def calculate_text_quality(
    text: str,
    confidence: float,
):
    """
    Calculate whether OCR result is usable.
    """

    cleaned = clean_text(text)

    words = re.findall(
        r"\b[A-Za-z0-9]+\b",
        cleaned,
    )

    word_count = len(words)

    text_length = len(cleaned)

    issues = []

    if text_length == 0:
        issues.append("empty_text")

    if text_length < MIN_TEXT_LENGTH:
        issues.append("very_short_text")

    if word_count < MIN_WORD_COUNT:
        issues.append("low_word_count")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        issues.append("low_ocr_confidence")

    return {
        "text": cleaned,
        "confidence": confidence,
        "word_count": word_count,
        "text_length": text_length,
        "issues": issues,
    }


# ============================================================
# OCR CONFIDENCE
# ============================================================

def calculate_ocr_confidence(
    image: Image.Image,
    config: str,
):
    """
    Run Tesseract and calculate average word confidence.
    """

    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANGUAGE,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confidences = []

    for i, raw_conf in enumerate(
        data["conf"]
    ):

        try:
            confidence = float(raw_conf)
        except (ValueError, TypeError):
            continue

        word = data["text"][i].strip()

        if word and confidence >= 0:

            words.append(word)
            confidences.append(
                confidence / 100.0
            )

    if not confidences:

        return "", 0.0

    average_confidence = (
        sum(confidences)
        / len(confidences)
    )

    text = " ".join(words)

    return (
        text,
        average_confidence,
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    image: Image.Image,
):
    """
    Create several image versions for OCR retry.
    """

    original = image.convert("RGB")

    grayscale = ImageOps.grayscale(
        original
    )

    # Improve contrast
    contrast = ImageOps.autocontrast(
        grayscale
    )

    # Light sharpening
    sharpened = contrast.filter(
        ImageFilter.SHARPEN
    )

    # Binary threshold
    threshold = sharpened.point(
        lambda pixel: 255
        if pixel > 170
        else 0
    )

    return [
        (
            "grayscale",
            grayscale,
        ),
        (
            "enhanced",
            sharpened,
        ),
        (
            "threshold",
            threshold,
        ),
    ]


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_page(
    page,
    dpi=OCR_DPI,
):

    scale = dpi / 72

    matrix = pymupdf.Matrix(
        scale,
        scale,
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


# ============================================================
# NORMAL OCR
# ============================================================

def run_primary_ocr(
    image: Image.Image,
):

    configs = [
        "--psm 6",
        "--psm 11",
    ]

    candidates = []

    for config in configs:

        try:

            text, confidence = (
                calculate_ocr_confidence(
                    image,
                    config,
                )
            )

            quality = calculate_text_quality(
                text,
                confidence,
            )

            quality["config"] = config
            quality["preprocessing"] = (
                "original"
            )

            candidates.append(
                quality
            )

        except Exception as exc:

            print(
                f"       OCR error "
                f"({config}): {exc}"
            )

    return candidates


# ============================================================
# OCR RETRY
# ============================================================

def run_retry_ocr(
    image: Image.Image,
):

    candidates = []

    processed_images = (
        preprocess_image(image)
    )

    configs = [
        "--psm 6",
        "--psm 11",
        "--psm 4",
    ]

    for preprocessing_name, processed_image in (
        processed_images
    ):

        for config in configs:

            try:

                text, confidence = (
                    calculate_ocr_confidence(
                        processed_image,
                        config,
                    )
                )

                quality = calculate_text_quality(
                    text,
                    confidence,
                )

                quality["config"] = config
                quality["preprocessing"] = (
                    preprocessing_name
                )

                candidates.append(
                    quality
                )

            except Exception as exc:

                print(
                    f"       Retry OCR error "
                    f"({preprocessing_name}, "
                    f"{config}): {exc}"
                )

    return candidates


# ============================================================
# CHOOSE BEST OCR RESULT
# ============================================================

def choose_best_ocr(
    candidates,
):

    if not candidates:
        return {
            "text": "",
            "confidence": 0.0,
            "word_count": 0,
            "text_length": 0,
            "issues": [
                "ocr_failed"
            ],
            "config": "",
            "preprocessing": "",
        }

    def score(candidate):

        text_score = min(
            candidate["text_length"]
            / 1000.0,
            1.0,
        )

        word_score = min(
            candidate["word_count"]
            / 150.0,
            1.0,
        )

        confidence_score = (
            candidate["confidence"]
        )

        empty_penalty = (
            0.0
            if candidate["text"]
            else -1.0
        )

        return (
            confidence_score * 0.55
            + text_score * 0.25
            + word_score * 0.20
            + empty_penalty
        )

    return max(
        candidates,
        key=score,
    )


# ============================================================
# EXTRACT PDF
# ============================================================

def extract_pdf_documents(
    pdf_path: Path,
):

    print(
        f"\nText extraction: "
        f"{pdf_path.name}"
    )

    pdf = pymupdf.open(
        str(pdf_path)
    )

    total_pages = len(pdf)

    print(
        f"  Total pages: "
        f"{total_pages}"
    )

    documents = []

    low_quality_pages = []

    empty_pages = []

    retry_pages = []

    successful_pages = []

    for page_index in range(
        total_pages
    ):

        page_number = page_index + 1

        page = pdf[page_index]

        # ----------------------------------------------------
        # First attempt: native PDF text
        # ----------------------------------------------------

        native_text = clean_text(
            page.get_text("text")
        )

        native_words = re.findall(
            r"\b[A-Za-z0-9]+\b",
            native_text,
        )

        native_good = (
            len(native_text)
            >= MIN_TEXT_LENGTH
            and len(native_words)
            >= MIN_WORD_COUNT
        )

        if native_good:

            final_text = native_text

            extraction_method = (
                "pymupdf_text"
            )

            confidence = 1.0

            warnings = []

            print(
                f"    → Page {page_number}: "
                f"native text"
            )

        else:

            # ------------------------------------------------
            # OCR required
            # ------------------------------------------------

            print(
                f"    → Page {page_number}: "
                f"OCR required"
            )

            image = render_page(
                page
            )

            primary_candidates = (
                run_primary_ocr(
                    image
                )
            )

            primary_best = (
                choose_best_ocr(
                    primary_candidates
                )
            )

            final_candidate = (
                primary_best
            )

            print(
                f"       Primary OCR confidence: "
                f"{primary_best['confidence']:.2f}"
            )

            # ------------------------------------------------
            # Retry bad OCR
            # ------------------------------------------------

            needs_retry = (
                primary_best[
                    "confidence"
                ]
                < LOW_CONFIDENCE_THRESHOLD
                or not primary_best[
                    "text"
                ]
                or len(
                    primary_best[
                        "text"
                    ]
                )
                < MIN_TEXT_LENGTH
            )

            if needs_retry:

                retry_pages.append(
                    page_number
                )

                print(
                    f"       ↻ OCR retry started"
                )

                retry_candidates = (
                    run_retry_ocr(
                        image
                    )
                )

                retry_best = (
                    choose_best_ocr(
                        retry_candidates
                    )
                )

                print(
                    f"       Retry OCR confidence: "
                    f"{retry_best['confidence']:.2f}"
                )

                # Choose retry only when
                # it is actually better
                if (
                    retry_best[
                        "text_length"
                    ]
                    > primary_best[
                        "text_length"
                    ]
                    and
                    retry_best[
                        "word_count"
                    ]
                    >= primary_best[
                        "word_count"
                    ]
                ):

                    final_candidate = (
                        retry_best
                    )

                    print(
                        "       ✓ Retry result selected"
                    )

                elif (
                    retry_best[
                        "confidence"
                    ]
                    > primary_best[
                        "confidence"
                    ]
                    + 0.05
                ):

                    final_candidate = (
                        retry_best
                    )

                    print(
                        "       ✓ Higher-confidence "
                        "retry selected"
                    )

                else:

                    print(
                        "       ✓ Primary OCR retained"
                    )

            final_text = final_candidate[
                "text"
            ]

            confidence = final_candidate[
                "confidence"
            ]

            warnings = final_candidate[
                "issues"
            ]

            extraction_method = (
                "tesseract_ocr"
            )

            if (
                final_candidate[
                    "preprocessing"
                ]
                != "original"
            ):

                extraction_method = (
                    "tesseract_ocr_retry"
                )

            # ------------------------------------------------
            # Final validation
            # ------------------------------------------------

            if not final_text:

                empty_pages.append(
                    (
                        pdf_path.name,
                        page_number,
                    )
                )

                print(
                    "       ✗ EMPTY PAGE AFTER OCR"
                )

            elif (
                confidence
                < LOW_CONFIDENCE_THRESHOLD
            ):

                low_quality_pages.append(
                    (
                        pdf_path.name,
                        page_number,
                        confidence,
                        warnings,
                    )
                )

                print(
                    f"       ⚠ Low-quality OCR: "
                    f"{confidence:.2f}"
                )

            else:

                successful_pages.append(
                    page_number
                )

        # ----------------------------------------------------
        # Normalize OCR output
        # ----------------------------------------------------

        if final_text:

            normalized_text, ocr_quality = (
                validate_ocr_page(
                    final_text,
                    confidence,
                )
            )

            final_text = clean_text(
                normalized_text
            )

            # Add warnings detected
            # after normalization
            if hasattr(
                ocr_quality,
                "warnings",
            ):

                warnings = list(
                    set(
                        warnings
                        + list(
                            ocr_quality.warnings
                        )
                    )
                )

        # ----------------------------------------------------
        # Build metadata
        # ----------------------------------------------------

        if not final_text:

            # Do NOT create an empty
            # LangChain Document.
            continue

        metadata_object = (
            DocumentMetadata(
                source=pdf_path.name,
                file_name=pdf_path.name,
                document_type="scheme",
                page=page_number,
                page_number=page_number,
                section="",
                content_type="paragraph",
                activity_number="",
                activity_marker="",
                item_number="",
                ocr_confidence=confidence,
                has_ocr_warning=bool(
                    warnings
                ),
                parser="pymupdf+tesseract",
                extraction_method=(
                    extraction_method
                ),
                language="en",
                version="1.0",
                raw_description="",
                raw_cost="",
            )
        )

        metadata = metadata_to_dict(
            metadata_object
        )

        # ----------------------------------------------------
        # Validate metadata
        # ----------------------------------------------------

        metadata_errors = validate_metadata(
            metadata
        )

        if metadata_errors:

            print(
                f"       ✗ Invalid metadata for page "
                f"{page_number}: {metadata_errors}"
            )

            continue

        # ----------------------------------------------------
        # Create LangChain Document
        # ----------------------------------------------------

        document = Document(
            page_content=final_text,
            metadata=metadata,
        )

        documents.append(
            document
        )

    pdf.close()

    print(
        f"  Documents produced: "
        f"{len(documents)}"
    )

    return {
        "documents": documents,
        "total_pages": total_pages,
        "low_quality_pages": (
            low_quality_pages
        ),
        "empty_pages": empty_pages,
        "retry_pages": retry_pages,
    }


# ============================================================
# DISCOVER PDFs
# ============================================================

def discover_pdfs():

    if not RAW_PDF_DIR.exists():

        raise FileNotFoundError(
            f"PDF directory not found: "
            f"{RAW_PDF_DIR.resolve()}"
        )

    pdf_files = sorted(
        RAW_PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found in: "
            f"{RAW_PDF_DIR.resolve()}"
        )

    return pdf_files


# ============================================================
# VALIDATE PAGE COVERAGE
# ============================================================

def validate_page_coverage(
    pdf_results,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PAGE COVERAGE VALIDATION"
    )

    print(
        "=" * 60
    )

    total_expected = 0
    total_documents = 0

    missing_pages = []

    for result in pdf_results:

        pdf_name = result["pdf_name"]

        expected = result[
            "total_pages"
        ]

        produced = len(
            result["documents"]
        )

        total_expected += expected
        total_documents += produced

        if produced != expected:

            produced_pages = {
                doc.metadata.get(
                    "page_number"
                )
                for doc in result[
                    "documents"
                ]
            }

            expected_pages = set(
                range(
                    1,
                    expected + 1,
                )
            )

            missing = sorted(
                expected_pages
                - produced_pages
            )

            for page in missing:

                missing_pages.append(
                    (
                        pdf_name,
                        page,
                    )
                )

    print(
        f"Expected pages  : "
        f"{total_expected}"
    )

    print(
        f"Documents       : "
        f"{total_documents}"
    )

    if missing_pages:

        print(
            "\n✗ MISSING PAGES:"
        )

        for pdf_name, page in (
            missing_pages
        ):

            print(
                f"  {pdf_name} "
                f"→ Page {page}"
            )

    else:

        print(
            "✓ All PDF pages have "
            "usable documents."
        )

    return missing_pages


# ============================================================
# PRINT QUALITY REPORT
# ============================================================

def print_quality_report(
    pdf_results,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "OCR RETRY / QUALITY REPORT"
    )

    print(
        "=" * 60
    )

    total_retry_pages = 0
    total_low_quality = 0
    total_empty = 0

    for result in pdf_results:

        retry_pages = result[
            "retry_pages"
        ]

        low_quality = result[
            "low_quality_pages"
        ]

        empty_pages = result[
            "empty_pages"
        ]

        total_retry_pages += len(
            retry_pages
        )

        total_low_quality += len(
            low_quality
        )

        total_empty += len(
            empty_pages
        )

        if retry_pages:

            print(
                f"\n{result['pdf_name']}"
            )

            print(
                f"  Retry pages: "
                f"{retry_pages}"
            )

            if low_quality:

                for (
                    pdf_name,
                    page,
                    confidence,
                    warnings,
                ) in low_quality:

                    print(
                        f"  Low quality → "
                        f"Page {page}, "
                        f"confidence "
                        f"{confidence:.2f}, "
                        f"warnings={warnings}"
                    )

            if empty_pages:

                for (
                    pdf_name,
                    page,
                ) in empty_pages:

                    print(
                        f"  EMPTY → "
                        f"Page {page}"
                    )

    print(
        "\n"
        + "-" * 60
    )

    print(
        f"Pages requiring retry : "
        f"{total_retry_pages}"
    )

    print(
        f"Low-quality after retry: "
        f"{total_low_quality}"
    )

    print(
        f"Empty after retry      : "
        f"{total_empty}"
    )

    return (
        total_retry_pages,
        total_low_quality,
        total_empty,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    start_time = time.time()

    print(
        "=" * 60
    )

    print(
        "PRODUCTION PDF INGESTION"
    )

    print(
        "STEP 12.16.3"
    )

    print(
        "OCR RETRY + PAGE VALIDATION"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Configure Tesseract
    # --------------------------------------------------------

    configure_tesseract()

    # --------------------------------------------------------
    # Discover PDFs
    # --------------------------------------------------------

    pdf_files = discover_pdfs()

    print(
        f"PDF directory : "
        f"{RAW_PDF_DIR.resolve()}"
    )

    print(
        f"PDF files     : "
        f"{len(pdf_files)}"
    )

    # --------------------------------------------------------
    # Process PDFs
    # --------------------------------------------------------

    all_documents = []

    pdf_results = []

    successful_pdfs = 0
    failed_pdfs = 0

    total_pages = 0
    total_retry_pages = 0

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"[{index}/{len(pdf_files)}] "
            f"{pdf_path.name}"
        )

        try:

            result = (
                extract_pdf_documents(
                    pdf_path
                )
            )

            documents = result[
                "documents"
            ]

            result["pdf_name"] = (
                pdf_path.name
            )

            pdf_results.append(
                result
            )

            all_documents.extend(
                documents
            )

            total_pages += result[
                "total_pages"
            ]

            total_retry_pages += len(
                result[
                    "retry_pages"
                ]
            )

            successful_pdfs += 1

        except Exception as exc:

            failed_pdfs += 1

            print(
                f"  ✗ FAILED: {exc}"
            )

    # --------------------------------------------------------
    # Processing summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PDF PROCESSING COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"PDFs discovered      : "
        f"{len(pdf_files)}"
    )

    print(
        f"PDFs successful      : "
        f"{successful_pdfs}"
    )

    print(
        f"PDFs failed          : "
        f"{failed_pdfs}"
    )

    print(
        f"Total pages          : "
        f"{total_pages}"
    )

    print(
        f"Documents produced   : "
        f"{len(all_documents)}"
    )

    print(
        f"OCR retry pages      : "
        f"{total_retry_pages}"
    )

    # --------------------------------------------------------
    # Page coverage
    # --------------------------------------------------------

    missing_pages = (
        validate_page_coverage(
            pdf_results
        )
    )

    # --------------------------------------------------------
    # Quality report
    # --------------------------------------------------------

    (
        retry_count,
        low_quality_count,
        empty_count,
    ) = print_quality_report(
        pdf_results
    )

    # --------------------------------------------------------
    # HARD VALIDATION
    # --------------------------------------------------------

    if failed_pdfs > 0:

        raise RuntimeError(
            "Ingestion stopped because "
            "one or more PDFs failed."
        )

    if missing_pages:

        print(
            "\n"
            "⚠ WARNING: Some pages are "
            "still missing after OCR retry."
        )

        print(
            "FAISS build will NOT continue."
        )

        raise RuntimeError(
            "Page coverage validation failed."
        )

    if empty_count > 0:

        raise RuntimeError(
            "Empty pages remain after "
            "OCR retry."
        )

    # --------------------------------------------------------
    # Document validation
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "DOCUMENT VALIDATION"
    )

    print(
        "=" * 60
    )

    invalid_documents = 0

    for document in all_documents:

        if not document.page_content.strip():

            invalid_documents += 1

    print(
        f"Documents checked   : "
        f"{len(all_documents)}"
    )

    print(
        f"Empty documents     : "
        f"{invalid_documents}"
    )

    if invalid_documents:

        raise RuntimeError(
            "Empty documents detected."
        )

    print(
        "✓ Document validation passed"
    )

    # --------------------------------------------------------
    # Smart chunking
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SMART CHUNKING"
    )

    print(
        "=" * 60
    )

    chunks = chunk_documents(
        all_documents
    )

    print_chunk_summary(
        all_documents,
        chunks
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EMBEDDINGS"
    )

    print(
        "=" * 60
    )

    embedding_model = (
        create_embedding_model()
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FAISS VECTOR STORE"
    )

    print(
        "=" * 60
    )

    faiss_index = build_faiss_index(
        chunks,
        embedding_model,
    )

    save_faiss_index(
        faiss_index,
        FAISS_INDEX_DIR,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL OCR QUALITY SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"PDFs processed       : "
        f"{successful_pdfs}/{len(pdf_files)}"
    )

    print(
        f"Pages processed      : "
        f"{total_pages}"
    )

    print(
        f"Documents produced   : "
        f"{len(all_documents)}"
    )

    print(
        f"Pages retried        : "
        f"{retry_count}"
    )

    print(
        f"Low-quality pages    : "
        f"{low_quality_count}"
    )

    print(
        f"Empty pages          : "
        f"{empty_count}"
    )

    print(
        f"Final chunks         : "
        f"{len(chunks)}"
    )

    print(
        f"Processing time      : "
        f"{elapsed:.2f} seconds"
    )

    print(
        "\n"
        "✓ OCR / DOCUMENT VALIDATION PASSED"
    )

    print(
        "\n"
        "✓ PRODUCTION INGESTION COMPLETE"
    )

    print(
        f"FAISS location       : "
        f"{FAISS_INDEX_DIR.resolve()}"
    )

    print(
        "Knowledge base is ready."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()