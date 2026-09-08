from __future__ import annotations

"""
STEP 12.9
Metadata Architecture

Purpose:
    Define a consistent metadata schema for all documents
    entering the RAG pipeline.

The metadata must survive:

    PDF
      ↓
    OCR / extraction
      ↓
    LangChain Document
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    FAISS
      ↓
    Retrieval
      ↓
    Reranking
      ↓
    Final answer / citations

IMPORTANT:
    This schema is intentionally generic.

    It should work for:
        - Agriculture Infrastructure Fund
        - future PDFs
        - OCR documents
        - normal text PDFs
        - tables
        - sections
        - paragraphs
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


# ============================================================================
# METADATA VERSION
# ============================================================================

METADATA_VERSION = "1.0"


# ============================================================================
# DOCUMENT METADATA
# ============================================================================

@dataclass
class DocumentMetadata:
    """
    Standard metadata object used by the RAG pipeline.
    """

    # ------------------------------------------------------------------------
    # DOCUMENT IDENTITY
    # ------------------------------------------------------------------------

    source: str

    file_name: str

    document_type: str

    # ------------------------------------------------------------------------
    # LOCATION
    # ------------------------------------------------------------------------

    page: int

    page_number: int

    section: str

    # ------------------------------------------------------------------------
    # CONTENT
    # ------------------------------------------------------------------------

    content_type: str

    # ------------------------------------------------------------------------
    # OPTIONAL STRUCTURAL INFORMATION
    # ------------------------------------------------------------------------

    activity_number: Optional[int] = None

    activity_marker: Optional[str] = None

    item_number: Optional[int] = None

    # ------------------------------------------------------------------------
    # QUALITY
    # ------------------------------------------------------------------------

    ocr_confidence: float = 1.0

    has_ocr_warning: bool = False

    # ------------------------------------------------------------------------
    # EXTRACTION / PROCESSING
    # ------------------------------------------------------------------------

    parser: str = ""

    extraction_method: str = ""

    language: str = "en"

    version: str = METADATA_VERSION

    # ------------------------------------------------------------------------
    # OPTIONAL RAW OCR INFORMATION
    # ------------------------------------------------------------------------

    raw_description: Optional[str] = None

    raw_cost: Optional[str] = None


# ============================================================================
# CONVERSION
# ============================================================================

def metadata_to_dict(
    metadata: DocumentMetadata,
) -> Dict[str, Any]:
    """
    Convert metadata dataclass into a normal dictionary.

    LangChain Document.metadata expects a dictionary.
    """

    data = asdict(
        metadata
    )

    # Remove optional fields that are not used.
    data = {
        key: value
        for key, value in data.items()
        if value is not None
    }

    return data


# ============================================================================
# METADATA VALIDATION
# ============================================================================

REQUIRED_METADATA_FIELDS = [
    "source",
    "file_name",
    "document_type",
    "page",
    "page_number",
    "section",
    "content_type",
    "ocr_confidence",
    "has_ocr_warning",
    "parser",
    "extraction_method",
    "language",
    "version",
]


def validate_metadata(
    metadata: Dict[str, Any],
) -> list[str]:
    """
    Validate a metadata dictionary.

    Returns:
        List of validation errors.

    Empty list means validation passed.
    """

    errors = []

    # ------------------------------------------------------------------------
    # Required fields
    # ------------------------------------------------------------------------

    for field in REQUIRED_METADATA_FIELDS:

        if field not in metadata:

            errors.append(
                f"Missing required metadata field: {field}"
            )

    # ------------------------------------------------------------------------
    # Basic value validation
    # ------------------------------------------------------------------------

    if "source" in metadata:

        if not str(
            metadata["source"]
        ).strip():

            errors.append(
                "Metadata 'source' cannot be empty."
            )

    if "file_name" in metadata:

        if not str(
            metadata["file_name"]
        ).strip():

            errors.append(
                "Metadata 'file_name' cannot be empty."
            )

    if "page" in metadata:

        if not isinstance(
            metadata["page"],
            int,
        ):

            errors.append(
                "Metadata 'page' must be an integer."
            )

        elif metadata["page"] < 1:

            errors.append(
                "Metadata 'page' must be >= 1."
            )

    if "page_number" in metadata:

        if not isinstance(
            metadata["page_number"],
            int,
        ):

            errors.append(
                "Metadata 'page_number' must be an integer."
            )

        elif metadata["page_number"] < 1:

            errors.append(
                "Metadata 'page_number' must be >= 1."
            )

    # ------------------------------------------------------------------------
    # OCR confidence
    # ------------------------------------------------------------------------

    if "ocr_confidence" in metadata:

        try:

            confidence = float(
                metadata["ocr_confidence"]
            )

            if not 0.0 <= confidence <= 1.0:

                errors.append(
                    "Metadata 'ocr_confidence' "
                    "must be between 0.0 and 1.0."
                )

        except (
            TypeError,
            ValueError,
        ):

            errors.append(
                "Metadata 'ocr_confidence' "
                "must be numeric."
            )

    # ------------------------------------------------------------------------
    # Boolean validation
    # ------------------------------------------------------------------------

    if "has_ocr_warning" in metadata:

        if not isinstance(
            metadata["has_ocr_warning"],
            bool,
        ):

            errors.append(
                "Metadata 'has_ocr_warning' "
                "must be boolean."
            )

    return errors


# ============================================================================
# CONTENT TYPE VALIDATION
# ============================================================================

VALID_CONTENT_TYPES = {
    "activity",
    "cost_norm",
    "paragraph",
    "table",
    "heading",
    "list",
    "other",
}


def validate_content_type(
    content_type: str,
) -> bool:
    """
    Check whether a content type is supported.
    """

    return content_type in VALID_CONTENT_TYPES


# ============================================================================
# DOCUMENT TYPE VALIDATION
# ============================================================================

def normalize_document_type(
    document_type: str,
) -> str:
    """
    Convert document type into a consistent format.

    Example:

        Agriculture Infrastructure Fund

    becomes:

        agriculture_infrastructure_fund
    """

    normalized = (
        document_type
        .strip()
        .lower()
    )

    normalized = normalized.replace(
        "-",
        "_",
    )

    normalized = normalized.replace(
        " ",
        "_",
    )

    return normalized


# ============================================================================
# METADATA SUMMARY
# ============================================================================

def print_metadata_summary(
    metadata: Dict[str, Any],
) -> None:

    print(
        "\n" + "=" * 80
    )

    print(
        "METADATA SUMMARY"
    )

    print(
        "=" * 80
    )

    for key, value in metadata.items():

        print(
            f"{key:22}: {value}"
        )