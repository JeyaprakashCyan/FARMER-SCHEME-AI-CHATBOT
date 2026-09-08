"""
LangSmith observability helpers.

Provides a traced wrapper around the complete RAG query pipeline.
"""

from typing import Any, Dict

from langsmith import traceable


@traceable(
    name="Farmer Scheme RAG",
    run_type="chain",
)
def traced_query(pipeline, question: str) -> Dict[str, Any]:
    """
    Execute one complete RAG query under a LangSmith trace.

    The pipeline itself remains unchanged.
    """

    result = pipeline.ask(question)

    return {
        "question": result.get("question", question),
        "standalone_question": result.get(
            "standalone_question",
            "",
        ),
        "answer": result.get(
            "answer",
            "",
        ),
        "sources": result.get(
            "sources",
            [],
        ),
    }