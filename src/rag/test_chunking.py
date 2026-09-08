from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parser.ocr_table_reconstructor import (
    PDF_PATH,
    PAGE_NUMBER,
    render_page,
    recover_activities,
    reconstruct_cost_table,
    ReconstructionResult,
)

from rag.document_builder import (
    build_langchain_documents,
)

from rag.chunking import (
    chunk_documents,
    print_chunk_summary,
    print_chunks,
    validate_chunks,
)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "STEP 12.10 - SMART CHUNKING TEST"
    )

    print(
        "=" * 80
    )

    # ------------------------------------------------------------------------
    # Step 1: Render PDF
    # ------------------------------------------------------------------------

    image = render_page(
        PDF_PATH,
        PAGE_NUMBER,
    )

    # ------------------------------------------------------------------------
    # Step 2: Recover activities
    # ------------------------------------------------------------------------

    activities = recover_activities(
        image
    )

    # ------------------------------------------------------------------------
    # Step 3: Reconstruct cost table
    # ------------------------------------------------------------------------

    cost_table = reconstruct_cost_table(
        image,
        PAGE_NUMBER,
    )

    # ------------------------------------------------------------------------
    # Step 4: Build reconstruction result
    # ------------------------------------------------------------------------

    result = ReconstructionResult(

        document=Path(
            PDF_PATH
        ).name,

        page_number=PAGE_NUMBER,

        activities=activities,

        cost_table=cost_table,
    )

    # ------------------------------------------------------------------------
    # Step 5: Build LangChain Documents
    # ------------------------------------------------------------------------

    documents = build_langchain_documents(
        result
    )

    print(
        f"\nLangChain Documents created: "
        f"{len(documents)}"
    )

    # ------------------------------------------------------------------------
    # Step 6: Smart chunking
    # ------------------------------------------------------------------------

    chunks = chunk_documents(
        documents
    )

    # ------------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------------

    print_chunk_summary(
        documents,
        chunks,
    )

    # ------------------------------------------------------------------------
    # Step 8: Print generated chunks
    # ------------------------------------------------------------------------

    print_chunks(
        chunks
    )

    # ------------------------------------------------------------------------
    # Step 9: Validation
    # ------------------------------------------------------------------------

    errors = validate_chunks(
        documents,
        chunks,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 12.10 VALIDATION"
    )

    print(
        "=" * 80
    )

    if errors:

        print(
            "\n✗ VALIDATION FAILED"
        )

        for error in errors:

            print(
                f"✗ {error}"
            )

    else:

        print(
            "✓ Chunks generated successfully."
        )

        print(
            "✓ page_content preserved."
        )

        print(
            "✓ Source metadata preserved."
        )

        print(
            "✓ Page metadata preserved."
        )

        print(
            "✓ Content type metadata preserved."
        )

        print(
            "✓ OCR confidence metadata preserved."
        )

        print(
            "✓ Chunk index metadata added."
        )

        print(
            "✓ Atomic activity documents kept intact."
        )

        print(
            "✓ Atomic cost norm documents kept intact."
        )

        print(
            "\n✓ STEP 12.10 VALIDATION SUCCESS"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()