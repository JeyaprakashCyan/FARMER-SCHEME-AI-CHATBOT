"""
Step 12.12 - FAISS Vector Store Test
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parser.ocr_table_reconstructor import (
    PDF_PATH,
    PAGE_NUMBER,
    ReconstructionResult,
    recover_activities,
    reconstruct_cost_table,
    render_page,
)

from rag.chunking import chunk_documents
from rag.document_builder import build_langchain_documents
from rag.embeddings import (
    create_embedding_model,
)

from rag.vector_store import (
    FAISS_INDEX_DIR,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    similarity_search,
    print_search_results,
)

def main():

    print("=" * 80)
    print("STEP 12.12 - FAISS VECTOR STORE TEST")
    print("=" * 80)

    # ========================================================
    # STEP 1 - RENDER PAGE
    # ========================================================

    print("\n[1/9] Rendering AIF page...")

    page_image = render_page(
        pdf_path=PDF_PATH,
        page_number=PAGE_NUMBER,
    )

    print("✓ Page rendered successfully.")

    # ========================================================
    # STEP 2 - RECOVER ACTIVITIES
    # ========================================================

    print("\n[2/9] Recovering activities...")

    activities_result = recover_activities(
        page_image
    )

    print(
        f"✓ Activities recovered: "
        f"{len(activities_result)}"
    )

    # ========================================================
    # STEP 3 - COST TABLE
    # ========================================================

    print("\n[3/9] Reconstructing cost table...")

    cost_result = reconstruct_cost_table(
        page_image,
        PAGE_NUMBER,
    )

    print(
        f"✓ Cost norms recovered: "
        f"{len(cost_result.rows)}"
    )

    # ========================================================
    # STEP 4 - DOCUMENTS
    # ========================================================

    print("\n[4/9] Building LangChain Documents...")

    result = ReconstructionResult(
        document=Path(PDF_PATH).name,
        page_number=PAGE_NUMBER,
        activities=activities_result,
        cost_table=cost_result,
    )
    documents = build_langchain_documents(result)

    print(
        f"✓ Documents created: {len(documents)}"
    )

    # ========================================================
    # STEP 5 - CHUNKING
    # ========================================================

    print("\n[5/9] Creating chunks...")

    chunks = chunk_documents(
        documents
    )

    print(
        f"✓ Chunks created: {len(chunks)}"
    )

    # ========================================================
    # STEP 6 - EMBEDDING MODEL
    # ========================================================

    print("\n[6/9] Loading embedding model...")

    embedding_model = create_embedding_model()

    print(
        "✓ Embedding model loaded."
    )

    # ========================================================
    # STEP 7 - BUILD FAISS
    # ========================================================

    print("\n[7/9] Building FAISS index...")

    vector_store = build_faiss_index(
        chunks,
        embedding_model,
    )

    print(
        "✓ FAISS index created."
    )

    # ========================================================
    # STEP 8 - SAVE
    # ========================================================

    print("\n[8/9] Saving FAISS index...")

    save_faiss_index(
        vector_store
    )

    print(
        "✓ FAISS index saved."
    )

    index_path = Path(
        FAISS_INDEX_DIR
    )

    print(
        f"Index location: {index_path}"
    )

    # ========================================================
    # STEP 9 - LOAD AGAIN
    # ========================================================

    print("\n[9/9] Loading FAISS index again...")

    loaded_store = load_faiss_index(
        embedding_model
    )

    print(
        "✓ FAISS index loaded successfully."
    )

    # ========================================================
    # SEARCH TEST
    # ========================================================

    print("\n" + "=" * 80)
    print("SIMILARITY SEARCH TEST")
    print("=" * 80)

    query = "What are the cost norms?"

    print(
        f"\nQuery: {query}"
    )

    results = similarity_search(
        loaded_store,
        query,
        k=5,
    )

    print(
        f"Results returned: {len(results)}"
    )

    print_search_results(
        results
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not results:
        raise ValueError(
            "FAISS search returned no results."
        )

    if len(results) != 5:
        raise ValueError(
            "FAISS did not return requested top-5 results."
        )

    print()
    print("=" * 80)
    print("✓ STEP 12.12 VALIDATION SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()