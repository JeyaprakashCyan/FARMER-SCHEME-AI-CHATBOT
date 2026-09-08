"""
Step 12.14 - Reranker Test

Pipeline:

FAISS
  ↓
Top 20
  ↓
Cross Encoder
  ↓
Top 5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import (
    create_embedding_model,
)

from src.rag.vector_store import (
    load_faiss_index,
)

from src.rag.retriever import (
    retrieve_documents,
)

from src.rag.reranker import (
    create_reranker,
    score_documents,
    rerank_documents,
    rerank,
    print_rerank_results,
)


def main():

    print("=" * 80)
    print("STEP 12.14 - RERANKER TEST")
    print("=" * 80)

    # ========================================================
    # STEP 1 - LOAD EMBEDDING MODEL
    # ========================================================

    print(
        "\n[1/6] Loading embedding model..."
    )

    embedding_model = create_embedding_model()

    print(
        "✓ Embedding model loaded."
    )

    # ========================================================
    # STEP 2 - LOAD FAISS
    # ========================================================

    print(
        "\n[2/6] Loading FAISS index..."
    )

    vector_store = load_faiss_index(
        embedding_model
    )

    print(
        "✓ FAISS index loaded."
    )

    # ========================================================
    # STEP 3 - RETRIEVE CANDIDATES
    # ========================================================

    print(
        "\n[3/6] Retrieving candidates..."
    )

    query = (
        "What is the cost norm for "
        "integrated pack house?"
    )

    candidates = retrieve_documents(
        vector_store=vector_store,
        query=query,
        k=20,
    )

    print(
        f"✓ Candidates retrieved: "
        f"{len(candidates)}"
    )

    if not candidates:
        raise ValueError(
            "FAISS returned no candidates."
        )

    # ========================================================
    # STEP 4 - LOAD RERANKER
    # ========================================================

    print(
        "\n[4/6] Loading reranker..."
    )

    reranker = create_reranker()

    print(
        "✓ Reranker loaded."
    )

    # ========================================================
    # STEP 5 - SCORE CANDIDATES
    # ========================================================

    print(
        "\n[5/6] Scoring candidates..."
    )

    scored = score_documents(
        query=query,
        documents=candidates,
        reranker=reranker,
    )

    print(
        f"✓ Scored documents: "
        f"{len(scored)}"
    )

    # ========================================================
    # SHOW TOP SCORES
    # ========================================================

    print()
    print("=" * 80)
    print("RERANK SCORES")
    print("=" * 80)

    sorted_scores = sorted(
        scored,
        key=lambda item: item[1],
        reverse=True,
    )

    for index, (
        document,
        score,
    ) in enumerate(
        sorted_scores,
        start=1,
    ):

        print(
            f"{index:02d}. "
            f"{score:.6f} | "
            f"{document.metadata.get('content_type')} | "
            f"{document.page_content[:100]}"
        )

    # ========================================================
    # STEP 6 - FINAL TOP 5
    # ========================================================

    print(
        "\n[6/6] Selecting top 5..."
    )

    final_results = rerank(
        query=query,
        documents=candidates,
        reranker=reranker,
        top_n=5,
    )

    if not final_results:
        raise ValueError(
            "Reranker returned no results."
        )

    if len(final_results) > 5:
        raise ValueError(
            "Reranker returned more than top-5."
        )

    print(
        f"✓ Final results: "
        f"{len(final_results)}"
    )

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print_rerank_results(
        final_results
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    # Check that rerank scores exist.

    for document in final_results:

        if "rerank_score" not in document.metadata:

            raise ValueError(
                "Rerank score missing from metadata."
            )

    # Check descending order.

    scores = [
        document.metadata["rerank_score"]
        for document in final_results
    ]

    if scores != sorted(
        scores,
        reverse=True,
    ):

        raise ValueError(
            "Reranked documents are not sorted "
            "by descending relevance."
        )

    print()
    print("=" * 80)
    print("✓ STEP 12.14 VALIDATION SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()