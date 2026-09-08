"""
Step 12.14 - Cross-Encoder Reranker

Pipeline:

User Query
    ↓
FAISS Top-K Candidates
    ↓
Cross-Encoder Reranker
    ↓
Relevance Scores
    ↓
Sorted Documents
    ↓
Top-N Documents
"""

from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

RERANKER_DEVICE = "cpu"

DEFAULT_TOP_N = 5


# ============================================================
# CREATE RERANKER
# ============================================================

def create_reranker() -> CrossEncoder:
    """
    Create and return the cross-encoder reranker.
    """

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME,
        device=RERANKER_DEVICE,
    )

    return reranker


# ============================================================
# PREPARE QUERY-DOCUMENT PAIRS
# ============================================================

def create_query_document_pairs(
    query: str,
    documents: List[Document],
) -> List[Tuple[str, str]]:
    """
    Create query-document pairs for the cross encoder.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not documents:
        raise ValueError(
            "No documents supplied for reranking."
        )

    pairs = []

    for document in documents:

        pairs.append(
            (
                query,
                document.page_content,
            )
        )

    return pairs


# ============================================================
# SCORE DOCUMENTS
# ============================================================

def score_documents(
    query: str,
    documents: List[Document],
    reranker: CrossEncoder,
) -> List[Tuple[Document, float]]:
    """
    Score every document against the query.

    Returns:

        [
            (Document, score),
            ...
        ]
    """

    pairs = create_query_document_pairs(
        query,
        documents,
    )

    scores = reranker.predict(
        pairs
    )

    scored_documents = []

    for document, score in zip(
        documents,
        scores,
    ):

        scored_documents.append(
            (
                document,
                float(score),
            )
        )

    return scored_documents


# ============================================================
# RERANK DOCUMENTS
# ============================================================

def rerank_documents(
    query: str,
    documents: List[Document],
    reranker: CrossEncoder,
    top_n: int = DEFAULT_TOP_N,
) -> List[Tuple[Document, float]]:
    """
    Rerank documents and return the top-N results.

    Higher score = more relevant.
    """

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    if not documents:
        return []

    scored_documents = score_documents(
        query=query,
        documents=documents,
        reranker=reranker,
    )

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_documents[:top_n]


# ============================================================
# ADD RERANK SCORE TO METADATA
# ============================================================

def attach_rerank_scores(
    scored_documents: List[Tuple[Document, float]],
) -> List[Document]:
    """
    Create copies of documents with the reranker
    score stored in metadata.
    """

    results = []

    for document, score in scored_documents:

        new_metadata = dict(
            document.metadata
        )

        new_metadata[
            "rerank_score"
        ] = score

        new_document = Document(
            page_content=document.page_content,
            metadata=new_metadata,
        )

        results.append(
            new_document
        )

    return results


# ============================================================
# COMPLETE RERANK PIPELINE
# ============================================================

def rerank(
    query: str,
    documents: List[Document],
    reranker: CrossEncoder,
    top_n: int = DEFAULT_TOP_N,
) -> List[Document]:
    """
    Complete reranking pipeline.

    Returns only the top-N documents.
    """

    scored_documents = rerank_documents(
        query=query,
        documents=documents,
        reranker=reranker,
        top_n=top_n,
    )

    return attach_rerank_scores(
        scored_documents
    )


# ============================================================
# PRINT RERANK RESULTS
# ============================================================

def print_rerank_results(
    documents: List[Document],
) -> None:

    print()
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        score = document.metadata.get(
            "rerank_score"
        )

        print()
        print(
            f"Rank #{index}"
        )

        print("-" * 80)

        print(
            f"Rerank Score : {score:.6f}"
        )

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
            f"Content      : "
            f"{document.page_content}"
        )