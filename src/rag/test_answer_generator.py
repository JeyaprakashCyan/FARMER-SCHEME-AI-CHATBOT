"""
Step 12.15 - RAG Answer Generator Test
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
    rerank,
)

from src.rag.answer_generator import (
    create_llm,
    build_context,
    extract_sources,
    generate_answer,
    print_answer,
)


def main():

    print("=" * 80)
    print("STEP 12.15 - RAG ANSWER GENERATION TEST")
    print("=" * 80)

    # ========================================================
    # STEP 1 - EMBEDDING MODEL
    # ========================================================

    print(
        "\n[1/8] Loading embedding model..."
    )

    embedding_model = create_embedding_model()

    print(
        "✓ Embedding model loaded."
    )

    # ========================================================
    # STEP 2 - FAISS
    # ========================================================

    print(
        "\n[2/8] Loading FAISS index..."
    )

    vector_store = load_faiss_index(
        embedding_model
    )

    print(
        "✓ FAISS index loaded."
    )

    # ========================================================
    # STEP 3 - RERANKER
    # ========================================================

    print(
        "\n[3/8] Loading reranker..."
    )

    reranker = create_reranker()

    print(
        "✓ Reranker loaded."
    )

    # ========================================================
    # STEP 4 - LLM
    # ========================================================

    print(
        "\n[4/8] Loading LLM..."
    )

    llm = create_llm()

    print(
        "✓ LLM configured."
    )

    # ========================================================
    # STEP 5 - USER QUESTION
    # ========================================================

    print(
        "\n[5/8] Preparing question..."
    )

    question = (
        "What is the cost norm for "
        "integrated pack house?"
    )

    print(
        f"Question: {question}"
    )

    # ========================================================
    # STEP 6 - RETRIEVE
    # ========================================================

    print(
        "\n[6/8] Retrieving candidates..."
    )

    candidates = retrieve_documents(
        vector_store=vector_store,
        query=question,
        k=20,
    )

    if not candidates:
        raise ValueError(
            "No retrieval candidates found."
        )

    print(
        f"✓ Candidates retrieved: "
        f"{len(candidates)}"
    )

    # ========================================================
    # STEP 7 - RERANK
    # ========================================================

    print(
        "\n[7/8] Reranking candidates..."
    )

    reranked_documents = rerank(
        query=question,
        documents=candidates,
        reranker=reranker,
        top_n=5,
    )

    if not reranked_documents:
        raise ValueError(
            "Reranker returned no documents."
        )

    print(
        f"✓ Final context documents: "
        f"{len(reranked_documents)}"
    )

    # ========================================================
    # CONTEXT VALIDATION
    # ========================================================

    print(
        "\nBuilding context..."
    )

    context = build_context(
        reranked_documents
    )

    if not context.strip():
        raise ValueError(
            "Generated context is empty."
        )

    print(
        "✓ Context built successfully."
    )

    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    sources = extract_sources(
        reranked_documents
    )

    if not sources:
        raise ValueError(
            "No sources extracted."
        )

    print(
        f"✓ Sources extracted: "
        f"{len(sources)}"
    )

    # ========================================================
    # STEP 8 - GENERATE ANSWER
    # ========================================================

    print(
        "\n[8/8] Generating grounded answer..."
    )

    result = generate_answer(
        question=question,
        documents=reranked_documents,
        llm=llm,
    )

    if not result.get("answer"):
        raise ValueError(
            "LLM returned an empty answer."
        )

    print(
        "✓ Answer generated successfully."
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print_answer(
        result
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("✓ STEP 12.15 VALIDATION SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()