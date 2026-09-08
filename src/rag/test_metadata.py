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
    print_document_summary,
    validate_langchain_documents,
)

from rag.metadata_schema import (
    validate_metadata,
    normalize_document_type,
)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "STEP 12.9 - METADATA ARCHITECTURE TEST"
    )

    print(
        "=" * 80
    )

    # ------------------------------------------------------------------------
    # Step 1: Render PDF page
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
    # Step 5: Build LangChain documents
    # ------------------------------------------------------------------------

    documents = build_langchain_documents(
        result
    )

    # ------------------------------------------------------------------------
    # Step 6: Print summary
    # ------------------------------------------------------------------------

    print_document_summary(
        documents
    )

    # ------------------------------------------------------------------------
    # Step 7: Validate documents
    # ------------------------------------------------------------------------

    errors = validate_langchain_documents(
        documents
    )

    # ------------------------------------------------------------------------
    # Step 8: Additional metadata checks
    # ------------------------------------------------------------------------

    metadata_errors = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        current_errors = validate_metadata(
            document.metadata
        )

        for error in current_errors:

            metadata_errors.append(
                f"Document {index}: {error}"
            )

    # ------------------------------------------------------------------------
    # Step 9: Check document count
    # ------------------------------------------------------------------------

    if len(documents) != 15:

        errors.append(
            f"Expected 15 documents, "
            f"found {len(documents)}."
        )

    # ------------------------------------------------------------------------
    # Step 10: Check canonical activity
    # ------------------------------------------------------------------------

    activity_8 = None

    for document in documents:

        if (
            document.metadata.get(
                "content_type"
            )
            == "activity"
            and document.metadata.get(
                "activity_number"
            )
            == 8
        ):

            activity_8 = document
            break

    if activity_8 is None:

        errors.append(
            "Activity #8 was not found."
        )

    else:

        expected_text = (
            "(viii) Logistics facilities"
        )

        if expected_text not in (
            activity_8.page_content
        ):

            errors.append(
                "Activity #8 canonical text "
                "is incorrect."
            )

    # ------------------------------------------------------------------------
    # Step 11: Check canonical costs
    # ------------------------------------------------------------------------

    expected_costs = [
        "145.00 lakh",
        "50.00 lakh",
        "225.00 lakh",
        "1.00 lakh/MT",
        "25.00 lakh",
    ]

    actual_costs = []

    for document in documents:

        if (
            document.metadata.get(
                "content_type"
            )
            == "cost_norm"
        ):

            text = document.page_content

            if "Cost Norm:" in text:

                cost = text.split(
                    "Cost Norm:",
                    1
                )[1].strip()

                actual_costs.append(
                    cost
                )

    if actual_costs != expected_costs:

        errors.append(
            "Canonical cost values do not match.\n"
            f"Expected: {expected_costs}\n"
            f"Actual:   {actual_costs}"
        )

    # ------------------------------------------------------------------------
    # Step 12: Check normalized document type
    # ------------------------------------------------------------------------

    document_types = {
        document.metadata.get(
            "document_type"
        )
        for document in documents
    }

    expected_document_type = (
        "agriculture_infrastructure_fund"
    )

    if document_types != {
        expected_document_type
    }:

        errors.append(
            "Document type normalization failed.\n"
            f"Expected: "
            f"{expected_document_type}\n"
            f"Actual: "
            f"{document_types}"
        )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 12.9 VALIDATION"
    )

    print(
        "=" * 80
    )

    if errors or metadata_errors:

        print(
            "\n✗ VALIDATION FAILED"
        )

        for error in errors:

            print(
                f"✗ {error}"
            )

        for error in metadata_errors:

            print(
                f"✗ {error}"
            )

    else:

        print(
            "✓ Metadata schema validation passed."
        )

        print(
            "✓ All required metadata fields present."
        )

        print(
            "✓ Metadata types are valid."
        )

        print(
            "✓ Document type normalization passed."
        )

        print(
            "✓ Activity #8 canonical text verified."
        )

        print(
            "✓ All 5 canonical cost values verified."
        )

        print(
            "✓ 15 LangChain Documents validated."
        )

        print(
            "\n✓ STEP 12.9 VALIDATION SUCCESS"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()