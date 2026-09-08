from __future__ import annotations

"""
STEP 12.10
Smart Chunking for the RAG Pipeline

Purpose:
    Convert standardized LangChain Documents into retrieval-friendly chunks.

Rules:
    1. Activity documents -> keep intact
    2. Cost norm documents -> keep intact
    3. Headings -> keep intact
    4. Short lists -> keep intact
    5. Long paragraphs -> recursive character splitting
    6. Tables -> preserve logical content where possible
    7. All metadata -> preserved on every chunk

Pipeline:

    LangChain Documents
            |
            v
       Smart Chunking
            |
            v
        Chunked Docs
            |
            v
         Embeddings
            |
            v
           FAISS
"""

from copy import deepcopy
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================

# Maximum size for normal textual chunks.
CHUNK_SIZE = 800

# Overlap between normal textual chunks.
CHUNK_OVERLAP = 120

# Small documents below this length will remain intact.
MIN_SPLIT_LENGTH = 900


# ============================================================================
# CONTENT TYPES
# ============================================================================

KEEP_INTACT_TYPES = {
    "activity",
    "cost_norm",
    "heading",
}

SPLITTABLE_TYPES = {
    "paragraph",
    "table",
    "list",
    "other",
}


# ============================================================================
# TEXT SPLITTER
# ============================================================================

def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the standard recursive text splitter.

    Separators are ordered from larger logical boundaries
    to smaller boundaries.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )


# ============================================================================
# METADATA COPY
# ============================================================================

def copy_metadata(
    metadata: dict,
) -> dict:
    """
    Make a safe copy of document metadata.

    Each generated chunk receives its own metadata dictionary.
    """

    return deepcopy(
        metadata
    )


# ============================================================================
# ADD CHUNK INFORMATION
# ============================================================================

def add_chunk_metadata(
    metadata: dict,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    """
    Add chunk-level metadata while preserving the original metadata.
    """

    new_metadata = copy_metadata(
        metadata
    )

    new_metadata[
        "chunk_index"
    ] = chunk_index

    new_metadata[
        "total_chunks"
    ] = total_chunks

    return new_metadata


# ============================================================================
# KEEP DOCUMENT INTACT
# ============================================================================

def keep_document_intact(
    document: Document,
) -> List[Document]:
    """
    Keep a document as one chunk.

    Used for:
        - activities
        - cost norms
        - headings
    """

    metadata = add_chunk_metadata(
        document.metadata,
        chunk_index=0,
        total_chunks=1,
    )

    return [
        Document(
            page_content=document.page_content.strip(),
            metadata=metadata,
        )
    ]


# ============================================================================
# SPLIT NORMAL DOCUMENT
# ============================================================================

def split_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Split a normal document using RecursiveCharacterTextSplitter.

    LangChain's splitter preserves metadata when creating
    Document chunks, but we explicitly rebuild the metadata
    here so our chunk metadata is deterministic.
    """

    text = document.page_content.strip()

    if not text:
        return []

    # ------------------------------------------------------------------------
    # Short document -> keep intact
    # ------------------------------------------------------------------------

    if len(text) <= MIN_SPLIT_LENGTH:

        return keep_document_intact(
            document
        )

    # ------------------------------------------------------------------------
    # Split long document
    # ------------------------------------------------------------------------

    raw_chunks = splitter.split_text(
        text
    )

    total_chunks = len(
        raw_chunks
    )

    chunks = []

    for index, chunk_text in enumerate(
        raw_chunks
    ):

        chunk_text = chunk_text.strip()

        if not chunk_text:
            continue

        metadata = add_chunk_metadata(
            document.metadata,
            chunk_index=index,
            total_chunks=total_chunks,
        )

        chunks.append(
            Document(
                page_content=chunk_text,
                metadata=metadata,
            )
        )

    # ------------------------------------------------------------------------
    # Safety fallback
    # ------------------------------------------------------------------------

    if not chunks:

        return keep_document_intact(
            document
        )

    return chunks


# ============================================================================
# TABLE HANDLING
# ============================================================================

def split_table_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Handle table-like documents.

    The first strategy is conservative:
        - short tables remain intact
        - long tables are split using newline boundaries

    This prevents blindly separating table rows whenever possible.
    """

    text = document.page_content.strip()

    if not text:
        return []

    # ------------------------------------------------------------------------
    # Short table -> keep intact
    # ------------------------------------------------------------------------

    if len(text) <= CHUNK_SIZE:

        return keep_document_intact(
            document
        )

    # ------------------------------------------------------------------------
    # Try line-aware splitting first
    # ------------------------------------------------------------------------

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:

        return keep_document_intact(
            document
        )

    groups = []

    current_group = []
    current_length = 0

    for line in lines:

        line_length = len(line)

        # ------------------------------------------------------------
        # If adding the next row exceeds chunk size,
        # finalize the current group.
        # ------------------------------------------------------------

        if (
            current_group
            and current_length + line_length + 1
            > CHUNK_SIZE
        ):

            groups.append(
                "\n".join(
                    current_group
                )
            )

            current_group = []
            current_length = 0

        current_group.append(
            line
        )

        current_length += (
            line_length + 1
        )

    if current_group:

        groups.append(
            "\n".join(
                current_group
            )
        )

    # ------------------------------------------------------------------------
    # If line-aware splitting didn't help,
    # fall back to recursive splitting.
    # ------------------------------------------------------------------------

    if len(groups) == 1:

        return split_document(
            document,
            splitter,
        )

    total_chunks = len(
        groups
    )

    chunks = []

    for index, group in enumerate(
        groups
    ):

        metadata = add_chunk_metadata(
            document.metadata,
            chunk_index=index,
            total_chunks=total_chunks,
        )

        chunks.append(
            Document(
                page_content=group,
                metadata=metadata,
            )
        )

    return chunks


# ============================================================================
# MAIN DOCUMENT CHUNKER
# ============================================================================

def chunk_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    """
    Decide the appropriate chunking strategy based on content_type.
    """

    content_type = document.metadata.get(
        "content_type",
        "other",
    )

    # ------------------------------------------------------------------------
    # Keep important atomic content intact
    # ------------------------------------------------------------------------

    if content_type in KEEP_INTACT_TYPES:

        return keep_document_intact(
            document
        )

    # ------------------------------------------------------------------------
    # Tables get special handling
    # ------------------------------------------------------------------------

    if content_type == "table":

        return split_table_document(
            document,
            splitter,
        )

    # ------------------------------------------------------------------------
    # Lists
    # ------------------------------------------------------------------------

    if content_type == "list":

        # Lists are often semantically connected.
        # Keep short lists intact.
        if len(
            document.page_content
        ) <= CHUNK_SIZE:

            return keep_document_intact(
                document
            )

        return split_document(
            document,
            splitter,
        )

    # ------------------------------------------------------------------------
    # Paragraph / other
    # ------------------------------------------------------------------------

    return split_document(
        document,
        splitter,
    )


# ============================================================================
# CHUNK ALL DOCUMENTS
# ============================================================================

def chunk_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Convert a list of LangChain Documents into chunks.
    """

    if not documents:
        return []

    splitter = create_text_splitter()

    all_chunks = []

    for document in documents:

        chunks = chunk_document(
            document,
            splitter,
        )

        all_chunks.extend(
            chunks
        )

    return all_chunks


# ============================================================================
# CHUNK SUMMARY
# ============================================================================

def print_chunk_summary(
    input_documents: List[Document],
    chunks: List[Document],
) -> None:

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 12.10 - SMART CHUNKING SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Input Documents : "
        f"{len(input_documents)}"
    )

    print(
        f"Output Chunks   : "
        f"{len(chunks)}"
    )

    # ------------------------------------------------------------------------
    # Input content types
    # ------------------------------------------------------------------------

    input_counts = {}

    for document in input_documents:

        content_type = document.metadata.get(
            "content_type",
            "unknown",
        )

        input_counts[
            content_type
        ] = (
            input_counts.get(
                content_type,
                0,
            )
            + 1
        )

    print(
        "\nInput Content Types:"
    )

    for content_type, count in sorted(
        input_counts.items()
    ):

        print(
            f"  {content_type:15}: "
            f"{count}"
        )

    # ------------------------------------------------------------------------
    # Output content types
    # ------------------------------------------------------------------------

    output_counts = {}

    for chunk in chunks:

        content_type = chunk.metadata.get(
            "content_type",
            "unknown",
        )

        output_counts[
            content_type
        ] = (
            output_counts.get(
                content_type,
                0,
            )
            + 1
        )

    print(
        "\nOutput Chunk Types:"
    )

    for content_type, count in sorted(
        output_counts.items()
    ):

        print(
            f"  {content_type:15}: "
            f"{count}"
        )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_chunks(
    input_documents: List[Document],
    chunks: List[Document],
) -> List[str]:
    """
    Validate chunking output.

    Checks:
        - chunks exist
        - page_content exists
        - metadata survives
        - source survives
        - page survives
        - content_type survives
        - atomic documents remain intact
    """

    errors = []

    # ------------------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------------------

    if not input_documents:

        errors.append(
            "No input documents were provided."
        )

        return errors

    if not chunks:

        errors.append(
            "No chunks were generated."
        )

        return errors

    # ------------------------------------------------------------------------
    # Required metadata
    # ------------------------------------------------------------------------

    required_metadata = [
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
        "chunk_index",
        "total_chunks",
    ]

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        # ------------------------------------------------------------
        # Content
        # ------------------------------------------------------------

        if not chunk.page_content.strip():

            errors.append(
                f"Chunk {index} has empty page_content."
            )

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        for field in required_metadata:

            if field not in chunk.metadata:

                errors.append(
                    f"Chunk {index} missing metadata: "
                    f"{field}"
                )

    # ------------------------------------------------------------------------
    # Atomic content validation
    # ------------------------------------------------------------------------

    for document in input_documents:

        content_type = document.metadata.get(
            "content_type"
        )

        if content_type not in KEEP_INTACT_TYPES:
            continue

        source = document.metadata.get(
            "source"
        )

        page = document.metadata.get(
            "page"
        )

        document_content = document.page_content.strip()

        matching_chunks = [

            chunk

            for chunk in chunks

            if (
                chunk.metadata.get(
                    "source"
                )
                == source
                and chunk.metadata.get(
                    "page"
                )
                == page
                and chunk.metadata.get(
                    "content_type"
                )
                == content_type
                and chunk.page_content.strip()
                == document_content
            )
        ]

        if len(
            matching_chunks
        ) != 1:

            errors.append(
                f"Atomic document was not kept intact: "
                f"{content_type}, "
                f"source={source}, "
                f"page={page}"
            )

    return errors


# ============================================================================
# DEBUG PRINT
# ============================================================================

def print_chunks(
    chunks: List[Document],
) -> None:

    print(
        "\n" + "=" * 80
    )

    print(
        "GENERATED CHUNKS"
    )

    print(
        "=" * 80
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            "\n" + "-" * 80
        )

        print(
            f"Chunk #{index}"
        )

        print(
            f"Content Type : "
            f"{chunk.metadata.get('content_type')}"
        )

        print(
            f"Chunk Index  : "
            f"{chunk.metadata.get('chunk_index')}"
        )

        print(
            f"Total Chunks : "
            f"{chunk.metadata.get('total_chunks')}"
        )

        print(
            f"Source       : "
            f"{chunk.metadata.get('source')}"
        )

        print(
            f"Page         : "
            f"{chunk.metadata.get('page')}"
        )

        print(
            f"Length       : "
            f"{len(chunk.page_content)}"
        )

        print(
            "Content:"
        )

        print(
            chunk.page_content
        )