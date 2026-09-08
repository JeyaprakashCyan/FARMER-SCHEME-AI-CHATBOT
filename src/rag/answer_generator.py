"""
Answer generation for the Farmer Scheme RAG chatbot.

Responsibilities:
1. Create the answer-generation LLM.
2. Build grounded context from retrieved documents.
3. Generate concise answers using ONLY retrieved context.
4. Handle factual / numeric questions carefully.
5. Extract source metadata for display.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# CONFIGURATION
# ============================================================

ANSWER_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini",
)
ANSWER_TEMPERATURE = 0.0

MAX_CONTEXT_DOCUMENTS = 5
MAX_SOURCE_RESULTS = 5


# ============================================================
# LLM
# ============================================================

def create_answer_llm():
    """
    Create the LLM used for final answer generation.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the environment or project .env file."
        )

    return ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=ANSWER_TEMPERATURE,
        api_key=api_key,
    )


# ============================================================
# DOCUMENT FORMATTING
# ============================================================

def format_document(document: Document, index: int) -> str:
    """
    Convert one retrieved Document into a source-aware context block.
    """

    metadata = document.metadata or {}

    file_name = (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file")
        or "unknown"
    )

    page = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("page_no")
        or "unknown"
    )

    content_type = (
        metadata.get("content_type")
        or metadata.get("type")
        or "unknown"
    )

    scheme_match = metadata.get("scheme_match", False)

    rerank_score = metadata.get("rerank_score")

    source_info = (
        f"[SOURCE {index}]\n"
        f"File: {file_name}\n"
        f"Page: {page}\n"
        f"Content Type: {content_type}\n"
        f"Scheme Match: {scheme_match}\n"
    )

    if rerank_score is not None:
        source_info += f"Rerank Score: {rerank_score}\n"

    source_info += f"\nContent:\n{document.page_content}"

    return source_info


def build_context(
    documents: List[Document],
    max_documents: int = MAX_CONTEXT_DOCUMENTS,
) -> str:
    """
    Build the final context sent to the LLM.
    """

    selected_documents = documents[:max_documents]

    context_parts = []

    for index, document in enumerate(selected_documents, start=1):
        context_parts.append(
            format_document(
                document=document,
                index=index,
            )
        )

    return "\n\n" + "\n\n".join(context_parts)


# ============================================================
# FACT QUESTION DETECTION
# ============================================================

def is_fact_question(question: str) -> bool:
    """
    Detect questions that usually require an exact value,
    number, percentage, amount, year, count, etc.
    """

    question_lower = question.lower()

    fact_keywords = [
        "how much",
        "how many",
        "what percentage",
        "what percent",
        "what is the amount",
        "what was the amount",
        "what is the number",
        "what was the number",
        "total number",
        "total amount",
        "number of",
        "amount of",
        "percentage of",
        "in financial year",
        "financial year",
        "fy ",
        "year ",
        "during 2024",
        "during 2025",
        "during 2026",
        "insured",
        "beneficiaries",
        "beneficiaries covered",
        "coverage",
    ]

    return any(
        keyword in question_lower
        for keyword in fact_keywords
    )


# ============================================================
# PROMPT
# ============================================================

def build_answer_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the grounded answer-generation prompt.

    Important:
    Fact questions receive additional instructions to extract
    exact values from the supplied context.
    """

    fact_question = is_fact_question(question)

    fact_instructions = ""

    if fact_question:
        fact_instructions = """
IMPORTANT — THIS IS AN EXACT FACT / NUMERIC QUESTION.

The user is asking for a specific factual value.

Follow these rules STRICTLY:

1. Identify exactly WHAT the user is asking about.
   Examples:
   - "how much livestock insured" -> livestock insured
   - "how many farmers benefited" -> farmers/beneficiaries
   - "what percentage subsidy" -> subsidy percentage
   - "what amount was provided" -> amount provided

2. Identify any TIME PERIOD specified by the user.
   For example:
   - financial year 2024-25
   - FY 2024-25
   - 2025
   - 2026

3. Find a statement in the supplied context where:
      VALUE + ENTITY + TIME PERIOD
   correspond to the user's question.

4. Use the value ONLY when it belongs to the entity being asked about.

5. NEVER combine a number from one statement with the
   entity from another statement.

6. For example, if the context contains:
      "21.01 Lakh livestock insured in financial year (2024-25)"
   and separately contains:
      "5,10.08 crore households owning livestock and/or poultry"
   and the user asks:
      "How much livestock insured in financial year (2024-25)?"

   The correct answer is:
      "21.01 Lakh livestock were insured in financial year (2024-25)."

   DO NOT answer:
      "21.01 Lakh households..."

7. Do NOT confuse:
   - livestock with households
   - farmers with beneficiaries
   - animals with households
   - amount with number of beneficiaries
   - percentage with quantity

8. If the exact value and its corresponding entity are
   explicitly present in the context, answer directly.

9. Preserve the original unit:
   lakh, crore, %, hectares, tonnes, etc.

10. Do not use outside knowledge to replace or correct
    a value from the supplied context.

11. If there are multiple possible values and the context
    does not clearly establish which one answers the question,
    say that the available documents do not provide a clear
    answer rather than guessing.
"""

    prompt = f"""
You are the answer-generation component of a Farmer Scheme
Retrieval-Augmented Generation (RAG) chatbot.

Your job is to answer the user's question using ONLY the
retrieved context supplied below.

GENERAL RULES:

1. Use only information contained in the retrieved context.
2. Do not invent facts.
3. Do not use outside/general knowledge to fill missing information.
4. If the answer is explicitly present in the context, answer it.
5. If the exact requested fact is not present or cannot be
   clearly matched to the requested entity and time period,
   say that the information could not be determined from
   the available farmer scheme documents.
6. Keep the answer concise and directly answer the question.
7. Do not discuss the retrieval process.
8. Do not mention "chunks", "embeddings", "reranking", "FAISS",
   or internal system details.
9. If a number is present in the context, preserve its original
   unit and meaning.
10. Do not confuse similar schemes or unrelated numbers.

{fact_instructions}

SOURCE CONTEXT:
============================================================

{context}

============================================================

USER QUESTION:
{question}

ANSWER:
"""

    return prompt


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def build_source_reference(document: Document) -> Dict[str, Any]:
    """
    Convert a Document into a clean source reference.
    """

    metadata = document.metadata or {}

    file_name = (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file")
        or "unknown"
    )

    page = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("page_no")
        or "unknown"
    )

    content_type = (
        metadata.get("content_type")
        or metadata.get("type")
        or "unknown"
    )

    source = {
        "file": file_name,
        "file_name": file_name,
        "page": page,
        "page_number": page,
        "content_type": content_type,
    }

    if "rerank_score" in metadata:
        source["rerank_score"] = metadata["rerank_score"]

    return source


def extract_sources(
    documents: List[Document],
    max_sources: int = MAX_SOURCE_RESULTS,
) -> List[Dict[str, Any]]:
    """
    Extract source information from the final context documents.
    """

    sources = []

    seen = set()

    for document in documents[:max_sources]:

        source = build_source_reference(document)

        key = (
            source["file_name"],
            source["page"],
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(source)

    return sources


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question: str,
    documents: List[Document],
    llm=None,
) -> Dict[str, Any]:
    """
    Generate a grounded answer from retrieved documents.

    Returns:
        {
            "answer": "...",
            "sources": [...]
        }
    """

    if not documents:
        return {
            "answer": (
                "I could not find relevant information in the "
                "available farmer scheme documents."
            ),
            "sources": [],
        }

    if llm is None:
        llm = create_answer_llm()

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        documents=documents,
        max_documents=MAX_CONTEXT_DOCUMENTS,
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_answer_prompt(
        question=question,
        context=context,
    )

    # --------------------------------------------------------
    # Call LLM
    # --------------------------------------------------------

    response = llm.invoke(prompt)

    # --------------------------------------------------------
    # Extract response text
    # --------------------------------------------------------

    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    answer = answer.strip()

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    sources = extract_sources(
        documents=documents,
        max_sources=MAX_SOURCE_RESULTS,
    )

    return {
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# PRINT HELPER
# ============================================================

def print_answer(result: Dict[str, Any]) -> None:
    """
    Pretty-print the generated answer and sources.
    """

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result.get("answer", ""))

    print("\nSOURCES")
    print("-" * 70)

    sources = result.get("sources", [])

    if not sources:
        print("No sources available.")
        return

    for index, source in enumerate(sources, start=1):

        file_name = source.get(
            "file_name",
            source.get("file", "unknown"),
        )

        page = source.get(
            "page",
            source.get("page_number", "unknown"),
        )

        print(
            f"{index}. {file_name} "
            f"(Page {page})"
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ANSWER GENERATOR TEST")
    print("=" * 70)

    test_document = Document(
        page_content=(
            "Livestock Insurance\n"
            "21.01 Lakh\n"
            "Livestock insured in financial year (2024-25)"
        ),
        metadata={
            "file_name": "Livestock Insurance - An introduction.pdf",
            "page": 1,
            "content_type": "paragraph",
            "scheme_match": True,
        },
    )

    test_question = (
        "How much livestock insured in financial year (2024-25)?"
    )

    print("\nQuestion:")
    print(test_question)

    print("\nFact question detected:")
    print(is_fact_question(test_question))

    print("\nCreating LLM...")

    llm = create_answer_llm()

    result = generate_answer(
        question=test_question,
        documents=[test_document],
        llm=llm,
    )

    print_answer(result)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)