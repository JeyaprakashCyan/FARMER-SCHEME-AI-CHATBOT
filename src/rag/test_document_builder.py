from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from parser.ocr_table_reconstructor import (
    DEBUG_JSON,
    PDF_PATH,
    PAGE_NUMBER,
    render_page,
    recover_activities,
    reconstruct_cost_table,
    ReconstructionResult,
)

from rag.document_builder import (
    build_langchain_documents,
    print_document_summary,
    validate_langchain_documents,
)


def main():

    print("=" * 80)
    print("STEP 12.8 - TEST")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. Render AIF Page 2
    # ------------------------------------------------------------

    image = render_page(
        PDF_PATH,
        PAGE_NUMBER,
    )

    # ------------------------------------------------------------
    # 2. Reconstruct activities
    # ------------------------------------------------------------

    activities = recover_activities(
        image
    )

    # ------------------------------------------------------------
    # 3. Reconstruct cost table
    # ------------------------------------------------------------

    cost_table = reconstruct_cost_table(
        image,
        PAGE_NUMBER,
    )

    # ------------------------------------------------------------
    # 4. Create reconstruction result
    # ------------------------------------------------------------

    result = ReconstructionResult(
        document=Path(
            PDF_PATH
        ).name,
        page_number=PAGE_NUMBER,
        activities=activities,
        cost_table=cost_table,
    )

    # ------------------------------------------------------------
    # 5. Convert to LangChain Documents
    # ------------------------------------------------------------

    documents = build_langchain_documents(
        result
    )

    # ------------------------------------------------------------
    # 6. Print
    # ------------------------------------------------------------

    print_document_summary(
        documents
    )

    # ------------------------------------------------------------
    # 7. Validate
    # ------------------------------------------------------------

    errors = validate_langchain_documents(
        documents
    )

    print("\n" + "=" * 80)
    print("STEP 12.8 VALIDATION")
    print("=" * 80)

    if errors:

        print(
            "VALIDATION FAILED"
        )

        for error in errors:
            print(
                f"✗ {error}"
            )

    else:

        print(
            "✓ VALIDATION SUCCESS"
        )

        print(
            f"✓ {len(documents)} LangChain Documents created."
        )


if __name__ == "__main__":
    main()