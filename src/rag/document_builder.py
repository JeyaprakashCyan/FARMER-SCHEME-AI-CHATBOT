from __future__ import annotations

"""
STEP 12.8
Standardize reconstructed OCR content into LangChain Documents.

Input:
    ReconstructionResult from Step 12.7

Output:
    List[langchain_core.documents.Document]

These Documents will later be used for:
    - chunking
    - embeddings
    - FAISS
    - metadata filtering
    - reranking
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document


# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_TYPE = "pdf"

DOCUMENT_TYPE = "agriculture_infrastructure_fund"

PARSER_NAME = "aif_page2_ocr_reconstructor"

EXTRACTION_METHOD = "ocr_reconstruction"


# ============================================================================
# ACTIVITY DOCUMENTS
# ============================================================================

def build_activity_documents(
    reconstruction_result,
) -> List[Document]:

    documents = []

    document_name = (
        reconstruction_result.document
    )

    page_number = (
        reconstruction_result.page_number
    )

    for index, activity in enumerate(
        reconstruction_result.activities,
        start=1,
    ):

        text = (
            f"Eligible Activity\n"
            f"{activity.marker} {activity.text}"
        )

        metadata = {
            "source": document_name,
            "file_name": Path(
                document_name
            ).name,

            "page": page_number,
            "page_number": page_number,

            "document_type": DOCUMENT_TYPE,

            "section": "Eligible Activities",

            "content_type": "activity",

            "activity_number": index,
            "activity_marker": activity.marker,

            "parser": PARSER_NAME,

            "extraction_method": EXTRACTION_METHOD,

            "ocr_confidence": float(
                activity.confidence
            ),

            "has_ocr_warning": bool(
                activity.warnings
            ),

            "language": "en",

            "version": "1.0",
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


# ============================================================================
# COST-NORM DOCUMENTS
# ============================================================================

def build_cost_documents(
    reconstruction_result,
) -> List[Document]:

    documents = []

    if not reconstruction_result.cost_table:
        return documents

    document_name = (
        reconstruction_result.document
    )

    page_number = (
        reconstruction_result.page_number
    )

    for row in reconstruction_result.cost_table.rows:

        text = (
            f"Cost Norm\n"
            f"Item {row.item_number}\n"
            f"Description: {row.description}\n"
            f"Cost Norm: {row.cost}"
        )

        metadata = {
            "source": document_name,
            "file_name": Path(
                document_name
            ).name,

            "page": page_number,
            "page_number": page_number,

            "document_type": DOCUMENT_TYPE,

            "section": "Cost Norms",

            "content_type": "cost_norm",

            "item_number": row.item_number,

            "parser": PARSER_NAME,

            "extraction_method": EXTRACTION_METHOD,

            "ocr_confidence": float(
                row.confidence
            ),

            "has_ocr_warning": bool(
                row.warnings
            ),

            "raw_description": row.raw_description,

            "raw_cost": row.raw_cost,

            "language": "en",

            "version": "1.0",
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


# ============================================================================
# FULL DOCUMENT BUILDER
# ============================================================================

def build_langchain_documents(
    reconstruction_result,
) -> List[Document]:

    documents = []

    documents.extend(
        build_activity_documents(
            reconstruction_result
        )
    )

    documents.extend(
        build_cost_documents(
            reconstruction_result
        )
    )

    return documents


# ============================================================================
# DEBUG DISPLAY
# ============================================================================

def print_document_summary(
    documents: List[Document],
) -> None:

    print("\n" + "=" * 80)
    print("STEP 12.8 - LANGCHAIN DOCUMENT SUMMARY")
    print("=" * 80)

    print(
        f"Total LangChain Documents: "
        f"{len(documents)}"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print("\n" + "-" * 80)

        print(
            f"Document #{index}"
        )

        print(
            f"Content Type : "
            f"{document.metadata.get('content_type')}"
        )

        print(
            f"Section      : "
            f"{document.metadata.get('section')}"
        )

        print(
            f"Page         : "
            f"{document.metadata.get('page')}"
        )

        print(
            f"Confidence   : "
            f"{document.metadata.get('ocr_confidence')}"
        )

        print(
            f"Content      : "
            f"{document.page_content}"
        )

        print(
            "Metadata:"
        )

        for key, value in document.metadata.items():
            print(
                f"  {key}: {value}"
            )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_langchain_documents(
    documents: List[Document],
) -> List[str]:

    errors = []

    if not documents:
        errors.append(
            "No LangChain Documents were generated."
        )

        return errors

    for index, document in enumerate(
        documents,
        start=1,
    ):

        if not document.page_content.strip():
            errors.append(
                f"Document {index} has empty page_content."
            )

        required_metadata = [
            "source",
            "file_name",
            "page",
            "document_type",
            "section",
            "content_type",
            "parser",
            "extraction_method",
            "ocr_confidence",
        ]

        for key in required_metadata:

            if key not in document.metadata:
                errors.append(
                    f"Document {index} missing "
                    f"metadata: {key}"
                )

    return errors