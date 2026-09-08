"""
Farmer Scheme AI
================

Streamlit UI for the Farmer Scheme RAG application.

Features
--------
- Clean native Streamlit chat interface
- Existing QueryPipeline integration
- Conversation memory
- Query rewriting
- FAISS retrieval
- BGE reranking
- Source display
- LangSmith tracing
- Robust result extraction
- Error handling
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langsmith import traceable


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAG_DIR = PROJECT_ROOT / "src" / "rag"

if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# IMPORT RAG PIPELINE
# ============================================================

from query_pipeline import QueryPipeline


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Farmer Scheme AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIGHT CSS ONLY
# ============================================================
#
# IMPORTANT:
# No HTML cards, divs or spans are used.
# The UI itself is built completely with Streamlit components.
#
# This prevents the previous problem where HTML appeared as
# literal code on the screen.
# ============================================================

st.markdown(
    """
<style>

    .stApp {
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(34, 197, 94, 0.07),
                transparent 30%
            ),
            radial-gradient(
                circle at 92% 0%,
                rgba(59, 130, 246, 0.06),
                transparent 30%
            );
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 12px;
        min-height: 42px;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None


if "session_id" not in st.session_state:

    st.session_state.session_id = (
        "streamlit-"
        + uuid.uuid4().hex[:12]
    )


if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# CREATE PIPELINE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def create_pipeline(
    session_id: str,
):
    """
    Create the RAG pipeline once per Streamlit session.
    """

    return QueryPipeline(
        session_id=session_id
    )


# ============================================================
# LANGSMITH TRACING
# ============================================================

@traceable(
    name="Farmer Scheme RAG",
    run_type="chain",
)
def traced_query(
    pipeline,
    question: str,
):
    """
    Execute the complete RAG pipeline.

    IMPORTANT:
    Return the COMPLETE result from pipeline.ask().
    Do not reconstruct the dictionary here because that
    can accidentally remove fields returned by QueryPipeline.
    """

    result = pipeline.ask(
        question
    )

    return result


# ============================================================
# RESULT EXTRACTION
# ============================================================

def extract_answer(result):
    """
    Extract the generated answer from the pipeline result.

    Supports the current expected key:
        answer

    And also common alternatives:
        final_answer
        response
        content
    """

    if result is None:
        return ""

    # --------------------------------------------------------
    # Dictionary result
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        possible_keys = [
            "answer",
            "final_answer",
            "response",
            "content",
            "output",
        ]

        for key in possible_keys:

            value = result.get(key)

            if value is not None:

                if isinstance(
                    value,
                    str,
                ) and value.strip():

                    return value.strip()

                if not isinstance(
                    value,
                    str,
                ):

                    return str(value)

    # --------------------------------------------------------
    # Object result
    # --------------------------------------------------------

    possible_attributes = [
        "answer",
        "final_answer",
        "response",
        "content",
        "output",
    ]

    for attribute in possible_attributes:

        if hasattr(
            result,
            attribute,
        ):

            value = getattr(
                result,
                attribute,
            )

            if value is not None:

                if isinstance(
                    value,
                    str,
                ) and value.strip():

                    return value.strip()

                return str(value)

    # --------------------------------------------------------
    # String result
    # --------------------------------------------------------

    if isinstance(
        result,
        str,
    ):

        return result.strip()

    return ""


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_sources(result):
    """
    Extract source information safely.
    """

    if result is None:
        return []

    if isinstance(
        result,
        dict,
    ):

        sources = (
            result.get("sources")
            or result.get("source_documents")
            or result.get("references")
            or []
        )

        if isinstance(
            sources,
            list,
        ):

            return sources

    return []


# ============================================================
# STANDALONE QUERY EXTRACTION
# ============================================================

def extract_standalone_question(
    result
):
    """
    Extract rewritten retrieval question if available.
    """

    if result is None:
        return ""

    if isinstance(
        result,
        dict,
    ):

        return (
            result.get(
                "standalone_question"
            )
            or result.get(
                "rewritten_question"
            )
            or result.get(
                "retrieval_query"
            )
            or ""
        )

    return ""


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    sources
):
    """
    Display source documents using native Streamlit UI.
    """

    if not sources:
        return

    with st.expander(
        f"📚 Sources ({len(sources)})"
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            # -----------------------------------------------
            # Dictionary source
            # -----------------------------------------------

            if isinstance(
                source,
                dict,
            ):

                file_name = (
                    source.get(
                        "file_name"
                    )
                    or source.get(
                        "file"
                    )
                    or source.get(
                        "source"
                    )
                    or "Unknown document"
                )

                page = (
                    source.get(
                        "page"
                    )
                    or source.get(
                        "page_number"
                    )
                    or "Unknown"
                )

                content_type = (
                    source.get(
                        "content_type"
                    )
                    or "Unknown"
                )

                rerank_score = source.get(
                    "rerank_score"
                )

                st.markdown(
                    f"**📄 {file_name}**"
                )

                metadata_text = (
                    f"Page {page}  •  "
                    f"Type: {content_type}"
                )

                if rerank_score is not None:

                    try:

                        metadata_text += (
                            f"  •  "
                            f"Score: {float(rerank_score):.4f}"
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        pass

                st.caption(
                    metadata_text
                )

            # -----------------------------------------------
            # String source
            # -----------------------------------------------

            else:

                st.markdown(
                    f"**{index}.** {source}"
                )

            if index < len(sources):

                st.divider()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌾 Farmer Scheme AI"
)

st.caption(
    "Your intelligent assistant for Indian agricultural "
    "schemes, farmer benefits, eligibility, financial "
    "support and government programmes."
)


# ============================================================
# SYSTEM METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(
    4
)

with col1:

    st.metric(
        "Knowledge Base",
        "26 PDFs",
    )

with col2:

    st.metric(
        "Vector Search",
        "FAISS",
    )

with col3:

    st.metric(
        "Reranker",
        "BGE",
    )

with col4:

    st.metric(
        "Tracing",
        "LangSmith",
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🌾 Farmer Scheme AI"
    )

    st.caption(
        "Ask questions about agricultural schemes "
        "and get answers grounded in the project "
        "knowledge base."
    )

    st.divider()

    st.subheader(
        "💡 Try asking"
    )

    example_questions = [
        "What is PM-KISAN?",
        "What are the benefits of PM-KISAN?",
        "Who is eligible for AIF?",
        "How does KCC work?",
        "What is natural farming?",
        "What is the Rashtriya Gokul Mission?",
    ]

    for index, example in enumerate(
        example_questions
    ):

        if st.button(
            example,
            key=f"example_{index}",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                example
            )

            st.rerun()


    st.divider()

    st.subheader(
        "⚙️ Chat controls"
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        # Clear QueryPipeline memory
        if (
            st.session_state.pipeline
            is not None
        ):

            try:

                st.session_state.pipeline.memory.clear_session(
                    st.session_state.pipeline.session_id
                )

            except Exception:

                pass

        # Clear Streamlit messages
        st.session_state.messages = []

        st.rerun()


    st.divider()

    st.subheader(
        "🔬 RAG pipeline"
    )

    st.caption(
        "🧠 Conversation Memory"
    )

    st.caption(
        "✍️ Query Rewriting"
    )

    st.caption(
        "🔎 FAISS Retrieval"
    )

    st.caption(
        "🎯 BGE Reranking"
    )

    st.caption(
        "📊 Relevance Threshold"
    )

    st.caption(
        "🛡️ Quality Control"
    )

    st.caption(
        "🤖 LLM Answer Generation"
    )

    st.caption(
        "📚 Source References"
    )

    st.divider()

    st.caption(
        f"Session: {st.session_state.session_id}"
    )


# ============================================================
# INITIALIZE RAG PIPELINE
# ============================================================

if st.session_state.pipeline is None:

    with st.spinner(
        "🌾 Loading Farmer Scheme AI..."
    ):

        try:

            st.session_state.pipeline = (
                create_pipeline(
                    st.session_state.session_id
                )
            )

            st.session_state.pipeline_error = None

        except Exception as error:

            st.session_state.pipeline_error = (
                str(error)
            )


# ============================================================
# PIPELINE ERROR
# ============================================================

if st.session_state.pipeline_error:

    st.error(
        "Unable to initialize the RAG pipeline."
    )

    st.subheader(
        "Technical details"
    )

    st.code(
        st.session_state.pipeline_error
    )

    st.stop()


pipeline = (
    st.session_state.pipeline
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.subheader(
        "👋 Welcome"
    )

    welcome_col1, welcome_col2, welcome_col3 = (
        st.columns(3)
    )

    with welcome_col1:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🌱 Agricultural Schemes"
            )

            st.write(
                "Explore PM-KISAN, AIF, KCC, "
                "natural farming and many other "
                "government programmes."
            )

    with welcome_col2:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🎯 Grounded Answers"
            )

            st.write(
                "Answers are generated from the "
                "indexed government scheme documents "
                "in the knowledge base."
            )

    with welcome_col3:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 💬 Ask Follow-ups"
            )

            st.write(
                "Continue the conversation naturally "
                "with context-aware follow-up questions."
            )

    st.write("")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    role = message.get(
        "role",
        "assistant",
    )

    avatar = (
        "🧑‍🌾"
        if role == "user"
        else "🌾"
    )

    with st.chat_message(
        role,
        avatar=avatar,
    ):

        st.markdown(
            message.get(
                "content",
                "",
            )
        )

        if role == "assistant":

            display_sources(
                message.get(
                    "sources",
                    [],
                )
            )


# ============================================================
# PENDING SIDEBAR QUESTION
# ============================================================

pending_question = (
    st.session_state.pending_question
)

st.session_state.pending_question = None


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask about an agricultural scheme..."
)


# Sidebar question has priority only when
# no typed question was submitted.

question = (
    user_question
    if user_question
    else pending_question
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.stop()


    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑‍🌾",
    ):

        st.markdown(
            question
        )


    # ========================================================
    # ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="🌾",
    ):

        try:

            # ------------------------------------------------
            # RUN RAG
            # ------------------------------------------------

            with st.spinner(
                "🔎 Searching the scheme knowledge base..."
            ):

                result = traced_query(
                    pipeline,
                    question,
                )


            # ------------------------------------------------
            # EXTRACT ANSWER
            # ------------------------------------------------

            answer = extract_answer(
                result
            )


            # ------------------------------------------------
            # EXTRACT SOURCES
            # ------------------------------------------------

            sources = extract_sources(
                result
            )


            # ------------------------------------------------
            # EXTRACT RETRIEVAL QUERY
            # ------------------------------------------------

            standalone_question = (
                extract_standalone_question(
                    result
                )
            )


            # ------------------------------------------------
            # CRITICAL SAFETY CHECK
            # ------------------------------------------------

            if not answer:

                st.error(
                    "The RAG pipeline completed, "
                    "but no answer text was returned."
                )

                # Show actual returned structure
                # temporarily for debugging.

                with st.expander(
                    "🔧 Debug: pipeline result"
                ):

                    st.write(
                        result
                    )

                answer = (
                    "I received the retrieval result, "
                    "but the answer text was empty."
                )


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.markdown(
                answer
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            display_sources(
                sources
            )


            # ------------------------------------------------
            # RETRIEVAL DETAILS
            # ------------------------------------------------

            with st.expander(
                "🔍 Retrieval details"
            ):

                st.caption(
                    "Original question"
                )

                st.code(
                    question
                )

                if standalone_question:

                    st.caption(
                        "Rewritten retrieval query"
                    )

                    st.code(
                        standalone_question
                    )

                # Useful development information
                if isinstance(
                    result,
                    dict,
                ):

                    st.caption(
                        "Pipeline result fields"
                    )

                    st.write(
                        list(
                            result.keys()
                        )
                    )


            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )


        except Exception as error:

            # ------------------------------------------------
            # ERROR
            # ------------------------------------------------

            st.error(
                "Sorry, I couldn't process "
                "your question."
            )

            with st.expander(
                "🔧 Technical details"
            ):

                st.exception(
                    error
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Sorry, I encountered an "
                        "error while processing "
                        "your question."
                    ),
                    "sources": [],
                }
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌾 Farmer Scheme AI  •  "
    "RAG + FAISS + BGE Reranker + "
    "Conversation Memory + LangSmith"
)