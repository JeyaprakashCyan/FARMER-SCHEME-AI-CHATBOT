"""
Step 12.11 - Embedding Test

Tests:

1. AIF page rendering
2. OCR reconstruction
3. Cost table reconstruction
4. LangChain Document creation
5. Smart chunking
6. Embedding generation
7. Embedding validation
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
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

from rag.document_builder import build_langchain_documents

from rag.chunking import (
    chunk_documents,
)

from rag.embeddings import (
    create_embedding_model,
    embed_documents,
    embed_query,
    get_embedding_dimension,
    validate_embeddings,
    print_embedding_summary,
)

def main():

    print("=" * 80)
    print("STEP 12.11 - EMBEDDING TEST")
    print("=" * 80)

    # ========================================================
    # STEP 1 - RENDER PAGE
    # ========================================================

    print("\n[1/7] Rendering AIF page...")

    page_image = render_page(
        pdf_path=PDF_PATH,
        page_number=PAGE_NUMBER,
    )

    print("✓ Page rendered successfully.")

    # ========================================================
    # STEP 2 - RECOVER ACTIVITIES
    # ========================================================

    print("\n[2/7] Recovering activities...")

    activities_result = recover_activities(
        page_image
    )

    print(
        f"✓ Activities recovered: "
        f"{len(activities_result)}"
    )

    # ========================================================
    # STEP 3 - RECONSTRUCT COST TABLE
    # ========================================================

    print("\n[3/7] Reconstructing cost table...")

    cost_result = reconstruct_cost_table(
        page_image,
        PAGE_NUMBER,
    )

    print(
        f"✓ Cost norms recovered: "
        f"{len(cost_result.rows)}"
    )

    # ========================================================
    # STEP 4 - BUILD LANGCHAIN DOCUMENTS
    # ========================================================

    print("\n[4/7] Building LangChain Documents...")

    result = ReconstructionResult(
        document=Path(PDF_PATH).name,
        page_number=PAGE_NUMBER,
        activities=activities_result,
        cost_table=cost_result,
    )
    documents = build_langchain_documents(result)

    print(
        f"✓ LangChain Documents: {len(documents)}"
    )

    # ========================================================
    # STEP 5 - SMART CHUNKING
    # ========================================================

    print("\n[5/7] Creating chunks...")

    chunks = chunk_documents(documents)

    print(
        f"✓ Chunks created: {len(chunks)}"
    )

    # ========================================================
    # STEP 6 - CREATE EMBEDDING MODEL
    # ========================================================

    print("\n[6/7] Loading embedding model...")

    embedding_model = create_embedding_model()

    print("✓ Embedding model loaded.")

    # ========================================================
    # STEP 7 - GENERATE EMBEDDINGS
    # ========================================================

    print("\n[7/7] Generating embeddings...")

    embeddings = embed_documents(
        chunks,
        embedding_model,
    )

    validate_embeddings(
        chunks,
        embeddings,
    )

    print("✓ Embeddings generated successfully.")

    # ========================================================
    # SUMMARY
    # ========================================================

    print_embedding_summary(
        chunks,
        embeddings,
    )

    # ========================================================
    # QUERY EMBEDDING TEST
    # ========================================================

    print("\n" + "=" * 80)
    print("QUERY EMBEDDING TEST")
    print("=" * 80)

    test_query = "What are the cost norms?"

    query_vector = embed_query(
        test_query,
        embedding_model,
    )

    print(f"Query          : {test_query}")
    print(f"Query dimension: {len(query_vector)}")

    # ========================================================
    # DIMENSION TEST
    # ========================================================

    dimension = get_embedding_dimension(
        embedding_model
    )

    print(f"Model dimension: {dimension}")

    if dimension != len(embeddings[0]):
        raise ValueError(
            "Query and document embedding dimensions do not match."
        )

    print(
        "✓ Query/document embedding dimensions match."
    )

    # ========================================================
    # SAMPLE VECTOR
    # ========================================================

    print("\nSample embedding:")
    print(embeddings[0][:10])

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n" + "=" * 80)
    print("✓ STEP 12.11 VALIDATION SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()