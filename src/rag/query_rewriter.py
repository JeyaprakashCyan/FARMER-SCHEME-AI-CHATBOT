"""
Context-Aware Query Rewriter
Step 12.21.3

Purpose:
    Convert follow-up questions into standalone questions using
    conversation history before retrieval.

Example:

    User:
        What is PM-KISAN?

    Follow-up:
        What are its benefits?

    Rewritten:
        What are the benefits of PM-KISAN?

The rewritten question is used for retrieval.
The original user question is preserved for conversation memory.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# CONFIGURATION
# ============================================================

REWRITER_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini",
)
REWRITER_TEMPERATURE = 0.0

ENABLE_QUERY_REWRITING = True


# ============================================================
# CREATE LLM
# ============================================================

def create_query_rewriter():
    """
    Create the LLM used for context-aware query rewriting.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found in the environment or project .env file."
        )

    return ChatOpenAI(
        model=REWRITER_MODEL,
        temperature=REWRITER_TEMPERATURE,
        api_key=api_key,
    )


# ============================================================
# FORMAT HISTORY
# ============================================================

def format_conversation_history(history) -> str:
    """
    Convert conversation history into a readable text format.

    Supports the ConversationMemory format:
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    """

    if not history:
        return ""

    lines = []

    for message in history:
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if role == "user":
            lines.append(f"User: {content}")

        elif role == "assistant":
            lines.append(f"Assistant: {content}")

        else:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ============================================================
# REWRITE PROMPT
# ============================================================

def build_rewrite_prompt(
    question: str,
    conversation_history: str,
) -> str:
    """
    Build the prompt for standalone-question rewriting.
    """

    return f"""
You are a query rewriting component for a farmer scheme
question-answering system.

Your task is to rewrite the user's latest question into a
standalone question that can be understood without the
conversation history.

IMPORTANT RULES:

1. Rewrite ONLY when the latest question depends on previous
   conversation context.

2. Resolve pronouns and references such as:
   - it
   - its
   - this scheme
   - that scheme
   - they
   - them
   - this
   - that

3. Preserve the exact scheme name when it is known from the
   conversation.

4. Do NOT answer the question.

5. Do NOT add facts that are not present in the conversation.

6. Do NOT change the user's actual intent.

7. If the question is already standalone, return it unchanged.

8. Return ONLY the rewritten standalone question.

Conversation history:
--------------------
{conversation_history}
--------------------

Latest user question:
--------------------
{question}
--------------------

Standalone question:
"""


# ============================================================
# REWRITE QUESTION
# ============================================================

def rewrite_question(
    question: str,
    history=None,
    llm=None,
) -> str:
    """
    Rewrite a question using conversation history.

    If there is no history, the original question is returned.

    Parameters
    ----------
    question:
        Latest user question.

    history:
        Conversation history from ConversationMemory.

    llm:
        Optional pre-created ChatOpenAI instance.

    Returns
    -------
    str
        Standalone question.
    """

    question = question.strip()

    if not question:
        return question

    # No need to call another LLM on the first question.
    if not history:
        return question

    if not ENABLE_QUERY_REWRITING:
        return question

    if llm is None:
        llm = create_query_rewriter()

    conversation_history = format_conversation_history(history)

    prompt = build_rewrite_prompt(
        question=question,
        conversation_history=conversation_history,
    )

    response = llm.invoke(prompt)

    rewritten_question = response.content.strip()

    # Safety fallback
    if not rewritten_question:
        return question

    return rewritten_question


# ============================================================
# TEST
# ============================================================

def main():

    print("=" * 60)
    print("QUERY REWRITER TEST")
    print("=" * 60)

    history = [
        {
            "role": "user",
            "content": "What is PM-KISAN?",
        },
        {
            "role": "assistant",
            "content": (
                "PM-KISAN, or Pradhan Mantri Kisan Samman Nidhi, "
                "is a Central Sector Scheme that provides income "
                "support to eligible landholding farmer families."
            ),
        },
    ]

    question = "What are its benefits?"

    print("\nOriginal question:")
    print(question)

    print("\nConversation history:")
    print(format_conversation_history(history))

    rewritten = rewrite_question(
        question=question,
        history=history,
    )

    print("\nRewritten question:")
    print(rewritten)

    print("\n" + "=" * 60)

    if rewritten:
        print("✓ QUERY REWRITER TEST PASSED")
    else:
        print("✗ QUERY REWRITER TEST FAILED")


if __name__ == "__main__":
    main()

