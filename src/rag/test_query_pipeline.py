import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.query_pipeline import (
    QueryPipeline,
    print_result,
)


def main():

    print(
        "\n=========================================="
    )

    print(
        "STEP 12.17 TEST"
    )

    print(
        "=========================================="
    )

    rag = QueryPipeline()

    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    question = (
        "What is the Agriculture Infrastructure Fund?"
    )

    result = rag.ask(
        question
    )

    print_result(
        result
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not result["answer"]:

        raise AssertionError(
            "Answer is empty."
        )

    if not result["sources"]:

        raise AssertionError(
            "No sources returned."
        )

    print(
        "\n=========================================="
    )

    print(
        "✓ STEP 12.17 VALIDATION SUCCESS"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":

    main()