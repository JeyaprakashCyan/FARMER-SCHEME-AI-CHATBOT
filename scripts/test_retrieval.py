from pathlib import Path
import sys
import os

# Suppress the harmless Windows Hugging Face symlink warning.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.faiss_store import load_faiss
from src.bm25_store import load_bm25_documents, BM25Store
from src.hybrid_retriever import HybridRetriever
from src.parent_child import ParentChildStore
from src.reranker import Reranker

from src.config import (
    BM25_INDEX_PATH,
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_RRF,
    TOP_K_RERANK,
)


# ============================================================
# LOAD STORES
# ============================================================

def load_retrieval_system():

    print()
    print("=" * 80)
    print("LOADING RETRIEVAL SYSTEM")
    print("=" * 80)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print()
    print("1. Loading FAISS...")

    vector_store = load_faiss()

    print("   ✓ FAISS loaded")

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print()
    print("2. Loading BM25...")

    bm25_documents = load_bm25_documents(
        BM25_INDEX_PATH
    )

    print(
        f"   ✓ BM25 documents loaded: "
        f"{len(bm25_documents)}"
    )

    if len(bm25_documents) != 167:
        print(
            "   ⚠ WARNING: BM25 currently contains "
            f"{len(bm25_documents)} documents. "
            "Earlier ingestion reported 167 child chunks. "
            "Verify FAISS and BM25 were built from the same ingestion run."
        )

    bm25_store = BM25Store(
        bm25_documents
    )

    # --------------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------------

    print()
    print("3. Creating hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        vector_store,
        bm25_store
    )

    print("   ✓ Hybrid retriever ready")

    # --------------------------------------------------------
    # Parent store
    # --------------------------------------------------------

    print()
    print("4. Loading parent store...")

    parent_store = ParentChildStore.load()

    print(
        f"   ✓ Parent sections loaded: "
        f"{len(parent_store.parents)}"
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    print()
    print("5. Loading cross-encoder reranker...")

    reranker = Reranker()

    print("   ✓ Reranker ready")

    return (
        vector_store,
        bm25_store,
        hybrid_retriever,
        parent_store,
        reranker,
    )


# ============================================================
# PRINT DOCUMENT
# ============================================================

def print_document(
    doc,
    rank=None,
    score_type=None,
):

    metadata = doc.metadata

    print()
    print("-" * 80)

    if rank is not None:
        print(f"Rank: {rank}")

    print(
        f"Document ID     : "
        f"{metadata.get('document_id', 'N/A')}"
    )

    print(
        f"Document name   : "
        f"{metadata.get('document_name', 'N/A')}"
    )

    print(
        f"Scheme ID       : "
        f"{metadata.get('scheme_id', 'N/A')}"
    )

    print(
        f"Page            : "
        f"{metadata.get('page_number', 'N/A')}"
    )

    print(
        f"Section         : "
        f"{metadata.get('section', 'N/A')}"
    )

    print(
        f"Subsection      : "
        f"{metadata.get('subsection', 'N/A')}"
    )

    print(
        f"Chunk type      : "
        f"{metadata.get('chunk_type', 'N/A')}"
    )

    print(
        f"Parent ID       : "
        f"{metadata.get('parent_id', 'N/A')}"
    )

    print(
        f"Chunk ID        : "
        f"{metadata.get('chunk_id', 'N/A')}"
    )

    if "rrf_score" in metadata:

        print(
            f"RRF score       : "
            f"{metadata['rrf_score']:.6f}"
        )

    if "rerank_score" in metadata:

        print(
            f"Rerank score    : "
            f"{metadata['rerank_score']:.6f}"
        )

    print()

    print("CONTENT:")

    text = doc.page_content.strip()

    # Prevent terminal flooding
    if len(text) > 1500:
        text = text[:1500] + "\n...[truncated]"

    print(text)


# ============================================================
# RUN RETRIEVAL
# ============================================================

def run_retrieval(
    query,
    state,
    hybrid_retriever,
    parent_store,
    reranker,
):

    print()
    print("=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    if state:
        print(
            f"State filter: {state}"
        )
    else:
        print(
            "State filter: None"
        )

    # ========================================================
    # STEP 1 - HYBRID RETRIEVAL
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 1 - HYBRID RETRIEVAL")
    print("=" * 80)

    hybrid_docs = hybrid_retriever.retrieve(
        query,
        state=state
    )

    print(
        f"Hybrid/RRF results: "
        f"{len(hybrid_docs)}"
    )

    if not hybrid_docs:

        print()
        print(
            "❌ No documents retrieved."
        )

        return

    # --------------------------------------------------------
    # Show RRF results
    # --------------------------------------------------------

    for rank, doc in enumerate(
        hybrid_docs,
        start=1
    ):

        print_document(
            doc,
            rank=rank
        )

    # ========================================================
    # STEP 2 - PARENT EXPANSION
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 2 - PARENT EXPANSION")
    print("=" * 80)

    parents = parent_store.expand(
        hybrid_docs
    )

    print(
        f"Parent sections retrieved: "
        f"{len(parents)}"
    )

    for rank, doc in enumerate(
        parents,
        start=1
    ):

        print_document(
            doc,
            rank=rank
        )

    # ========================================================
    # STEP 3 - COMBINE CHILD + PARENT
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 3 - COMBINING CHILD + PARENT")
    print("=" * 80)

    combined = []

    seen = set()

    # Child chunks first
    for doc in hybrid_docs:

        chunk_id = doc.metadata.get(
            "chunk_id"
        )

        key = (
            "chunk",
            chunk_id
        )

        if key not in seen:

            combined.append(doc)
            seen.add(key)

    # Parent sections
    for doc in parents:

        parent_id = doc.metadata.get(
            "parent_id"
        )

        key = (
            "parent",
            parent_id
        )

        if key not in seen:

            combined.append(doc)
            seen.add(key)

    print(
        f"Combined documents: "
        f"{len(combined)}"
    )

    # ========================================================
    # STEP 3A - PREPARE RERANKER CANDIDATES
    # ========================================================

    # Parent expansion can increase the candidate count significantly.
    # Keep enough candidates for recall, but avoid sending all candidates
    # to the cross-encoder on a memory-constrained local machine.
    MAX_RERANK_CANDIDATES = 15

    # Directly retrieved chunks have an RRF score.
    # Parent documents normally do not, so they are kept after
    # directly retrieved chunks.
    def rerank_priority(doc):
        rrf_score = doc.metadata.get("rrf_score")

        if rrf_score is not None:
            return float(rrf_score)

        return 0.0

    combined_for_rerank = sorted(
        combined,
        key=rerank_priority,
        reverse=True
    )

    combined_for_rerank = combined_for_rerank[
        :MAX_RERANK_CANDIDATES
    ]

    print(
        f"Documents sent to reranker: "
        f"{len(combined_for_rerank)}"
    )

    # ========================================================
    # STEP 4 - RERANKING
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 4 - CROSS-ENCODER RERANKING")
    print("=" * 80)

    reranked = reranker.rerank(
        query,
        combined_for_rerank,
        top_k=TOP_K_RERANK
    )

    print(
        f"Final reranked results: "
        f"{len(reranked)}"
    )

    # ========================================================
    # STEP 5 - FINAL RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("STEP 5 - FINAL TOP RESULTS")
    print("=" * 80)

    for rank, doc in enumerate(
        reranked,
        start=1
    ):

        print_document(
            doc,
            rank=rank
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)

    print(
        f"Dense top K       : "
        f"{TOP_K_DENSE}"
    )

    print(
        f"BM25 top K        : "
        f"{TOP_K_BM25}"
    )

    print(
        f"RRF top K         : "
        f"{TOP_K_RRF}"
    )

    print(
        f"Final rerank K    : "
        f"{TOP_K_RERANK}"
    )

    print(
        f"Hybrid results    : "
        f"{len(hybrid_docs)}"
    )

    print(
        f"Parent results    : "
        f"{len(parents)}"
    )

    print(
        f"Final results     : "
        f"{len(reranked)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print(" FARMER SCHEME CHATBOT")
    print(" STEP 20 - RETRIEVAL TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Load system
    # --------------------------------------------------------

    (
        vector_store,
        bm25_store,
        hybrid_retriever,
        parent_store,
        reranker,
    ) = load_retrieval_system()

    # --------------------------------------------------------
    # Test questions
    # --------------------------------------------------------

    test_questions = [

        (
            "What is Agriculture Infrastructure Fund?",
            None
        ),

        (
            "Who is eligible for Agriculture Infrastructure Fund?",
            None
        ),

        (
            "What documents are required for Agriculture Infrastructure Fund?",
            None
        ),

        (
            "How much financial assistance is available under Agriculture Infrastructure Fund?",
            None
        ),

        (
            "How can I apply for Agriculture Infrastructure Fund?",
            None
        ),

        (
            "Is Agriculture Infrastructure Fund available in Karnataka?",
            "Karnataka"
        ),

        (
            "What is PM Kisan?",
            None
        ),

        (
            "Who is eligible for PM Kisan?",
            None
        ),

        (
            "What is Kisan Credit Card?",
            None
        ),

        (
            "What schemes are available for dairy farmers?",
            None
        ),

    ]

    # --------------------------------------------------------
    # Interactive menu
    # --------------------------------------------------------

    while True:

        print()
        print("=" * 80)
        print("TEST MENU")
        print("=" * 80)

        print("1. Run predefined test questions")
        print("2. Enter your own question")
        print("3. Exit")

        choice = input(
            "\nEnter choice: "
        ).strip()

        # ----------------------------------------------------
        # Predefined questions
        # ----------------------------------------------------

        if choice == "1":

            for number, (
                question,
                state
            ) in enumerate(
                test_questions,
                start=1
            ):

                print()
                print()
                print("#" * 80)

                print(
                    f"TEST QUESTION {number}"
                )

                print("#" * 80)

                try:

                    run_retrieval(
                        question,
                        state,
                        hybrid_retriever,
                        parent_store,
                        reranker,
                    )

                except Exception as e:

                    print()
                    print(
                        f"ERROR: {e}"
                    )

                input(
                    "\nPress ENTER for next question..."
                )

        # ----------------------------------------------------
        # Custom question
        # ----------------------------------------------------

        elif choice == "2":

            question = input(
                "\nEnter your question: "
            ).strip()

            if not question:

                print(
                    "Question cannot be empty."
                )

                continue

            state = input(
                "Enter state "
                "(press ENTER for none): "
            ).strip()

            if not state:
                state = None

            try:

                run_retrieval(
                    question,
                    state,
                    hybrid_retriever,
                    parent_store,
                    reranker,
                )

            except Exception as e:

                print()
                print(
                    f"ERROR: {e}"
                )

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        elif choice == "3":

            print()
            print(
                "Exiting retrieval test."
            )

            break

        else:

            print(
                "Invalid choice."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()