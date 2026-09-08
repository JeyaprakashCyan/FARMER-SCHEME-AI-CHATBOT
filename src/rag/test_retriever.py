"""
Step 12.13 - Retriever Test

Tests:

1. Load FAISS
2. Basic top-k retrieval
3. Retrieval with scores
4. Metadata filtering
5. Content-type filtering
6. Activity filtering
7. Cost-norm filtering
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
    retrieve_with_scores,
    retrieve_with_filters,
    retrieve_by_content_type,
    retrieve_by_activity,
    retrieve_by_item,
    print_documents,
)


def main():

    print("=" * 80)
    print("STEP 12.13 - RETRIEVER TEST")
    print("=" * 80)

    # ========================================================
    # STEP 1 - LOAD EMBEDDING MODEL
    # ========================================================

    print("\n[1/7] Loading embedding model...")

    embedding_model = create_embedding_model()

    print(
        "✓ Embedding model loaded."
    )

    # ========================================================
    # STEP 2 - LOAD FAISS
    # ========================================================

    print("\n[2/7] Loading FAISS index...")

    vector_store = load_faiss_index(
        embedding_model
    )

    print(
        "✓ FAISS index loaded."
    )

    # ========================================================
    # STEP 3 - BASIC RETRIEVAL
    # ========================================================

    print("\n[3/7] Testing basic retrieval...")

    query = "What are the cost norms?"

    results = retrieve_documents(
        vector_store=vector_store,
        query=query,
        k=5,
    )

    print(
        f"✓ Retrieved {len(results)} documents."
    )

    if not results:
        raise ValueError(
            "Basic retrieval returned no results."
        )

    print_documents(
        results,
        title="BASIC RETRIEVAL RESULTS",
    )

    # ========================================================
    # STEP 4 - RETRIEVAL WITH SCORES
    # ========================================================

    print(
        "\n[4/7] Testing retrieval with scores..."
    )

    scored_results = retrieve_with_scores(
        vector_store=vector_store,
        query=query,
        k=5,
    )

    if not scored_results:
        raise ValueError(
            "Scored retrieval returned no results."
        )

    print(
        f"✓ Retrieved {len(scored_results)} scored results."
    )

    print()

    for index, (
        document,
        score,
    ) in enumerate(
        scored_results,
        start=1,
    ):

        print(
            f"Result #{index} | "
            f"Score: {score:.6f} | "
            f"Type: {document.metadata.get('content_type')}"
        )

    # ========================================================
    # STEP 5 - CONTENT TYPE FILTER
    # ========================================================

    print(
        "\n[5/7] Testing content-type filtering..."
    )

    cost_results = retrieve_by_content_type(
        vector_store=vector_store,
        query="What are the cost norms?",
        content_type="cost_norm",
        k=20,
    )

    if not cost_results:
        raise ValueError(
            "Cost norm filtering returned no results."
        )

    for document in cost_results:

        if document.metadata.get(
            "content_type"
        ) != "cost_norm":

            raise ValueError(
                "Content-type filter returned "
                "an incorrect document."
            )

    print(
        f"✓ Cost norm results: {len(cost_results)}"
    )

    # ========================================================
    # STEP 6 - ACTIVITY FILTER
    # ========================================================

    print(
        "\n[6/7] Testing activity filtering..."
    )

    activity_results = retrieve_by_activity(
        vector_store=vector_store,
        query="activity",
        activity_number=8,
        k=20,
    )

    if not activity_results:
        raise ValueError(
            "Activity filter returned no results."
        )

    for document in activity_results:

        if document.metadata.get(
            "activity_number"
        ) != 8:

            raise ValueError(
                "Activity filter returned "
                "an incorrect activity."
            )

    print(
        f"✓ Activity #8 results: "
        f"{len(activity_results)}"
    )

    # ========================================================
    # STEP 7 - COST ITEM FILTER
    # ========================================================

    print(
        "\n[7/7] Testing cost item filtering..."
    )

    item_results = retrieve_by_item(
        vector_store=vector_store,
        query="integrated pack house",
        item_number=2,
        k=20,
    )

    if not item_results:
        raise ValueError(
            "Cost item filter returned no results."
        )

    for document in item_results:

        if document.metadata.get(
            "item_number"
        ) != 2:

            raise ValueError(
                "Cost item filter returned "
                "an incorrect item."
            )

    print(
        f"✓ Cost item #2 results: "
        f"{len(item_results)}"
    )

    # ========================================================
    # DISPLAY IMPORTANT RESULTS
    # ========================================================

    print_documents(
        cost_results,
        title="COST-NORM FILTER RESULTS",
    )

    print_documents(
        activity_results,
        title="ACTIVITY #8 RESULTS",
    )

    print_documents(
        item_results,
        title="COST ITEM #2 RESULTS",
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("✓ STEP 12.13 VALIDATION SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()