"""
Step 12.13 - Retriever

Purpose:
    Retrieve a broad set of candidate documents from FAISS.

Pipeline:

User Query
    ↓
FAISS Similarity Search
    ↓
Top-K Candidates
    ↓
Metadata Filtering
    ↓
Candidates for Reranker
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_RETRIEVAL_K = 20


# ============================================================
# BASIC SIMILARITY RETRIEVAL
# ============================================================

def retrieve_documents(
    vector_store: FAISS,
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve top-k documents from FAISS.

    This performs broad candidate retrieval.
    Reranking will happen later.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results


# ============================================================
# SIMILARITY SEARCH WITH SCORES
# ============================================================

def retrieve_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
):
    """
    Retrieve documents together with FAISS similarity scores.

    Lower distance generally indicates greater similarity
    for the FAISS distance strategy being used.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    return results


# ============================================================
# METADATA FILTER
# ============================================================

def filter_documents(
    documents: List[Document],
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Filter retrieved documents using metadata.

    Example:

        filters = {
            "content_type": "cost_norm"
        }

    Multiple filters use AND logic.
    """

    if not filters:
        return documents

    filtered_documents = []

    for document in documents:

        metadata = document.metadata

        matches = True

        for key, expected_value in filters.items():

            actual_value = metadata.get(key)

            if actual_value != expected_value:
                matches = False
                break

        if matches:
            filtered_documents.append(document)

    return filtered_documents


# ============================================================
# RETRIEVE + FILTER
# ============================================================

def retrieve_with_filters(
    vector_store: FAISS,
    query: str,
    k: int = DEFAULT_RETRIEVAL_K,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Retrieve documents from FAISS and then apply
    metadata filters.
    """

    documents = retrieve_documents(
        vector_store=vector_store,
        query=query,
        k=k,
    )

    filtered_documents = filter_documents(
        documents=documents,
        filters=filters,
    )

    return filtered_documents


# ============================================================
# CONVENIENCE FILTER FUNCTIONS
# ============================================================

def retrieve_by_content_type(
    vector_store: FAISS,
    query: str,
    content_type: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve documents of a specific content type.
    """

    return retrieve_with_filters(
        vector_store=vector_store,
        query=query,
        k=k,
        filters={
            "content_type": content_type
        },
    )


def retrieve_by_source(
    vector_store: FAISS,
    query: str,
    source: str,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve documents from a specific source.
    """

    return retrieve_with_filters(
        vector_store=vector_store,
        query=query,
        k=k,
        filters={
            "source": source
        },
    )


def retrieve_by_page(
    vector_store: FAISS,
    query: str,
    page: int,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve documents from a specific page.
    """

    return retrieve_with_filters(
        vector_store=vector_store,
        query=query,
        k=k,
        filters={
            "page": page
        },
    )


def retrieve_by_activity(
    vector_store: FAISS,
    query: str,
    activity_number: int,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve a specific activity.
    """

    return retrieve_with_filters(
        vector_store=vector_store,
        query=query,
        k=k,
        filters={
            "activity_number": activity_number
        },
    )


def retrieve_by_item(
    vector_store: FAISS,
    query: str,
    item_number: int,
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[Document]:
    """
    Retrieve a specific cost-norm item.
    """

    return retrieve_with_filters(
        vector_store=vector_store,
        query=query,
        k=k,
        filters={
            "item_number": item_number
        },
    )


# ============================================================
# PRINT DOCUMENTS
# ============================================================

def print_documents(
    documents: List[Document],
    title: str = "RETRIEVAL RESULTS",
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if not documents:
        print("No documents found.")
        return

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print()
        print(f"Result #{index}")
        print("-" * 80)

        print(
            f"Source       : "
            f"{document.metadata.get('source')}"
        )

        print(
            f"File         : "
            f"{document.metadata.get('file_name')}"
        )

        print(
            f"Page         : "
            f"{document.metadata.get('page')}"
        )

        print(
            f"Content Type : "
            f"{document.metadata.get('content_type')}"
        )

        print(
            f"Activity     : "
            f"{document.metadata.get('activity_number')}"
        )

        print(
            f"Item         : "
            f"{document.metadata.get('item_number')}"
        )

        print(
            f"Chunk        : "
            f"{document.metadata.get('chunk_index')}"
        )

        print(
            f"Content      : "
            f"{document.page_content}"
        )