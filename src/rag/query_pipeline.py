"""
Query Pipeline
--------------
Production RAG query pipeline for the Farmer Scheme Chatbot.

Pipeline:

User Question
    ↓
Conversation Memory
    ↓
Context-aware Query Rewriting
    ↓
Scheme Detection
    ↓
Fact / Year Detection
    ↓
FAISS Retrieval
    ↓
Fact/Year Expanded Retrieval
    ↓
Merge + Deduplicate
    ↓
Scheme Prioritization
    ↓
BGE Reranking
    ↓
Fact/Year Boost
    ↓
Relevance Threshold
    ↓
Quality Control
    ↓
LLM Answer Generation
    ↓
Conversation Memory
    ↓
Answer + Sources
"""

import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document


# ============================================================
# LOCAL IMPORTS
# ============================================================

from conversation_memory import (
    ConversationMemory,
)

from query_rewriter import (
    create_query_rewriter,
    rewrite_question,
)

from scheme_router import (
    detect_scheme,
    prioritize_scheme_documents,
    apply_scheme_boost,
    is_fact_or_year_question,
    build_fact_query_variants,
)

from embeddings import (
    create_embedding_model,
)

from vector_store import (
    load_faiss_index,
)

from retriever import (
    retrieve_documents,
)

from reranker import (
    create_reranker,
    rerank,
)

from answer_generator import (
    generate_answer,
)


# ============================================================
# CONFIGURATION
# ============================================================

RETRIEVAL_K = 20

# Number of documents sent to the reranker.
RERANK_TOP_N = 5

# Minimum reranker score.
RERANK_SCORE_THRESHOLD = 0.10

# Minimum number of documents required for answer generation.
MIN_CONTEXT_DOCS = 1

# Conversation memory.
MEMORY_MAX_TURNS = 5

# Default CLI session.
SESSION_ID = "default-session"

# Query rewriting.
ENABLE_QUERY_REWRITING = True

# Fact/year retrieval.
ENABLE_FACT_YEAR_RETRIEVAL = True

# Number of additional fact retrieval queries.
MAX_FACT_QUERY_VARIANTS = 3


# ============================================================
# FACT/YEAR BOOST SETTINGS
# ============================================================

# These are intentionally small boosts.
#
# We are NOT changing the global reranker threshold.
# We are simply helping fact-bearing documents survive
# when the question clearly targets a specific scheme/year.

FACT_SCHEME_BOOST = 0.20
FACT_YEAR_BOOST = 0.10
FACT_KEYWORD_BOOST = 0.05


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for keyword matching.

    Example:

        "2024-25" -> "2024 25"
        "PM-KISAN" -> "pm kisan"
    """

    if not text:
        return ""

    text = str(text).lower()

    # Replace separators with spaces.
    text = re.sub(
        r"[-_/]+",
        " ",
        text,
    )

    # Remove punctuation.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Collapse multiple spaces.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# QUERY PIPELINE CLASS
# ============================================================

class QueryPipeline:

    def __init__(
        self,
        session_id: str = SESSION_ID,
    ):
        """
        Initialize all RAG components.
        """

        self.session_id = session_id

        # ----------------------------------------------------
        # Conversation Memory
        # ----------------------------------------------------

        self.memory = ConversationMemory(
            max_turns=MEMORY_MAX_TURNS
        )

        # ----------------------------------------------------
        # Embedding Model
        # ----------------------------------------------------

        print()
        print("Loading embedding model...")

        self.embedding_model = (
            create_embedding_model()
        )

        # ----------------------------------------------------
        # FAISS Vector Store
        # ----------------------------------------------------

        print("Loading FAISS vector store...")

        self.vector_store = (
            load_faiss_index(
                self.embedding_model
            )
        )

        # ----------------------------------------------------
        # Reranker
        # ----------------------------------------------------

        print("Loading reranker...")

        self.reranker = (
            create_reranker()
        )

        # ----------------------------------------------------
        # Query Rewriter
        # ----------------------------------------------------

        print("Loading query rewriter...")

        self.query_rewriter = (
            create_query_rewriter()
        )

        print()
        print("✓ Query pipeline initialized")


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    def remove_duplicates(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Remove duplicate documents.

        Document identity is based on:

            file
            page
            content
        """

        if not documents:
            return []

        unique_documents = []

        seen = set()

        for document in documents:

            metadata = (
                document.metadata or {}
            )

            file_name = (
                metadata.get("file_name")
                or metadata.get("source")
                or metadata.get("filename")
                or metadata.get("file")
                or ""
            )

            page = (
                metadata.get("page")
                or metadata.get("page_number")
                or metadata.get("page_no")
                or ""
            )

            content = (
                document.page_content
                or ""
            )

            identity = (
                str(file_name),
                str(page),
                normalize_text(content),
            )

            if identity in seen:
                continue

            seen.add(identity)

            unique_documents.append(
                document
            )

        return unique_documents


    # ========================================================
    # RELEVANCE THRESHOLD
    # ========================================================

    def apply_relevance_threshold(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Keep documents whose effective reranker score
        meets the configured threshold.

        The effective score can include the targeted
        fact/year boost.
        """

        if not documents:
            return []

        filtered = []

        for document in documents:

            metadata = (
                document.metadata or {}
            )

            rerank_score = metadata.get(
                "rerank_score"
            )

            # If score is unavailable, keep the document.
            if rerank_score is None:

                filtered.append(
                    document
                )

                continue

            try:

                rerank_score = float(
                    rerank_score
                )

            except (
                TypeError,
                ValueError,
            ):

                filtered.append(
                    document
                )

                continue

            effective_score = metadata.get(
                "effective_rerank_score",
                rerank_score,
            )

            try:

                effective_score = float(
                    effective_score
                )

            except (
                TypeError,
                ValueError,
            ):

                effective_score = (
                    rerank_score
                )

            if (
                effective_score
                >= RERANK_SCORE_THRESHOLD
            ):

                filtered.append(
                    document
                )

        return filtered


    # ========================================================
    # QUALITY CONTROL
    # ========================================================

    def quality_control(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Final context cleanup.

        Removes:

        - empty documents
        - extremely short documents
        - duplicate content
        """

        if not documents:
            return []

        cleaned = []

        seen_content = set()

        for document in documents:

            content = (
                document.page_content
                or ""
            ).strip()

            # Empty document.
            if not content:
                continue

            # Ignore useless tiny content.
            if len(content) < 20:
                continue

            normalized_content = (
                normalize_text(content)
            )

            if normalized_content in seen_content:
                continue

            seen_content.add(
                normalized_content
            )

            cleaned.append(
                document
            )

        return cleaned


    # ========================================================
    # MEMORY
    # ========================================================

    def get_history(self):
        """
        Return conversation history.
        """

        return self.memory.get_history(
            self.session_id
        )


    # ========================================================
    # QUERY REWRITING
    # ========================================================

    def get_standalone_question(
        self,
        question: str,
    ) -> str:
        """
        Convert a follow-up question into a standalone question.

        Example:

        User:
            What is PM-KISAN?

        Follow-up:
            What are its benefits?

        Rewritten:
            What are the benefits of PM-KISAN?
        """

        if not ENABLE_QUERY_REWRITING:
            return question

        history = self.get_history()

        if not history:
            return question

        try:

            rewritten = rewrite_question(
                question=question,
                history=history,
                llm=self.query_rewriter,
            )

            if rewritten:

                rewritten = (
                    rewritten.strip()
                )

                if rewritten:
                    return rewritten

        except Exception as exc:

            print(
                f"⚠ Query rewriting failed: {exc}"
            )

        # Fallback to original question.
        return question


    # ========================================================
    # BASE RETRIEVAL
    # ========================================================

    def retrieve_base_documents(
        self,
        query: str,
    ) -> List[Document]:
        """
        Standard FAISS retrieval.
        """

        documents = retrieve_documents(
            vector_store=self.vector_store,
            query=query,
            k=RETRIEVAL_K,
        )

        return documents or []


    # ========================================================
    # FACT/YEAR RETRIEVAL
    # ========================================================

    def retrieve_fact_documents(
        self,
        question: str,
        scheme: Optional[Dict],
    ) -> List[Document]:
        """
        Perform additional retrieval for fact/year questions.

        Example:

            How much livestock insured
            in financial year 2024-25?

        Additional semantic queries are generated so that
        the fact-bearing page has more opportunities to enter
        the reranking stage.
        """

        if not ENABLE_FACT_YEAR_RETRIEVAL:
            return []

        if not is_fact_or_year_question(
            question
        ):
            return []

        variants = build_fact_query_variants(
            question=question,
            scheme=scheme,
        )

        if not variants:
            return []

        # Original question is already handled by
        # base retrieval.
        extra_variants = variants[
            1:
        ]

        # Limit additional retrieval.
        extra_variants = extra_variants[
            :MAX_FACT_QUERY_VARIANTS
        ]

        if not extra_variants:
            return []

        all_documents = []

        print()
        print("FACT/YEAR RETRIEVAL BOOST")
        print("-" * 60)

        for index, variant in enumerate(
            extra_variants,
            start=1,
        ):

            print(
                f"Extra query {index}: "
                f"{variant}"
            )

            try:

                documents = (
                    retrieve_documents(
                        vector_store=self.vector_store,
                        query=variant,
                        k=RETRIEVAL_K,
                    )
                )

                if documents:

                    all_documents.extend(
                        documents
                    )

            except Exception as exc:

                print(
                    f"⚠ Extra retrieval failed: "
                    f"{exc}"
                )

        print(
            f"Extra documents retrieved: "
            f"{len(all_documents)}"
        )

        return all_documents


    # ========================================================
    # FACT/YEAR BOOST
    # ========================================================

    def apply_fact_year_boost(
        self,
        documents: List[Document],
        question: str,
        scheme: Optional[Dict],
    ) -> List[Document]:
        """
        Apply a targeted boost for fact/year questions.

        Boosts are based on:

        1. Exact scheme document
        2. Requested year appearing in content
        3. Fact-related words appearing in content

        This does NOT hard-code any answer.
        """

        if not documents:
            return documents

        if not is_fact_or_year_question(
            question
        ):
            return documents

        normalized_question = (
            normalize_text(question)
        )

        # ----------------------------------------------------
        # Extract requested years
        # ----------------------------------------------------

        requested_years = re.findall(
            r"\b20\d{2}\s*[-/]\s*(?:20)?\d{2}\b",
            normalized_question,
        )

        requested_years = [
            normalize_text(year)
            for year in requested_years
        ]

        # ----------------------------------------------------
        # Fact-related terms
        # ----------------------------------------------------

        fact_terms = [
            "how much",
            "how many",
            "number",
            "total",
            "amount",
            "quantity",
            "percentage",
            "percent",
            "rate",
            "insured",
            "insurance",
            "coverage",
            "covered",
            "beneficiaries",
            "beneficiary",
            "financial year",
        ]

        # ----------------------------------------------------
        # Process documents
        # ----------------------------------------------------

        for document in documents:

            metadata = (
                document.metadata or {}
            )

            content = normalize_text(
                document.page_content or ""
            )

            # Original reranker score.
            base_score = metadata.get(
                "rerank_score",
                0.0,
            )

            try:

                base_score = float(
                    base_score
                )

            except (
                TypeError,
                ValueError,
            ):

                base_score = 0.0

            boost = 0.0

            # ------------------------------------------------
            # Scheme boost
            # ------------------------------------------------

            scheme_match = bool(
                metadata.get(
                    "scheme_match",
                    False,
                )
            )

            if (
                scheme
                and scheme_match
            ):

                boost += FACT_SCHEME_BOOST

            # ------------------------------------------------
            # Year boost
            # ------------------------------------------------

            year_match = False

            content_compact = (
                content.replace(
                    " ",
                    "",
                )
            )

            for requested_year in requested_years:

                year_compact = (
                    requested_year.replace(
                        " ",
                        "",
                    )
                )

                if (
                    requested_year in content
                    or year_compact
                    in content_compact
                ):

                    year_match = True
                    break

            if year_match:

                boost += FACT_YEAR_BOOST

            # ------------------------------------------------
            # Fact keyword boost
            # ------------------------------------------------

            keyword_matches = 0

            for term in fact_terms:

                if term in content:

                    keyword_matches += 1

            if keyword_matches > 0:

                boost += (
                    FACT_KEYWORD_BOOST
                )

            # ------------------------------------------------
            # Store diagnostics
            # ------------------------------------------------

            metadata[
                "fact_year_boost"
            ] = round(
                boost,
                6,
            )

            metadata[
                "year_match"
            ] = year_match

            metadata[
                "fact_keyword_matches"
            ] = keyword_matches

            metadata[
                "effective_rerank_score"
            ] = round(
                base_score + boost,
                6,
            )

        # ----------------------------------------------------
        # Sort by effective score
        # ----------------------------------------------------

        documents.sort(
            key=lambda document: float(
                document.metadata.get(
                    "effective_rerank_score",
                    document.metadata.get(
                        "rerank_score",
                        0.0,
                    ),
                )
            ),
            reverse=True,
        )

        return documents


    # ========================================================
    # SCHEME-AWARE RETRIEVAL
    # ========================================================

    def retrieve_documents(
        self,
        standalone_question: str,
    ) -> Tuple[
        List[Document],
        Optional[Dict],
    ]:
        """
        Complete retrieval stage.

        Returns:

            documents
            detected scheme
        """

        # ----------------------------------------------------
        # Detect scheme
        # ----------------------------------------------------

        scheme = detect_scheme(
            standalone_question
        )

        print()
        print("SCHEME ROUTING")
        print("-" * 60)

        if scheme:

            print(
                f"Detected scheme : "
                f"{scheme['name']}"
            )

            print(
                f"Target file     : "
                f"{scheme['file_name']}"
            )

            print(
                f"Matched alias   : "
                f"{scheme['matched_alias']}"
            )

        else:

            print(
                "Detected scheme : None"
            )

        # ----------------------------------------------------
        # Fact/year detection
        # ----------------------------------------------------

        fact_query = (
            is_fact_or_year_question(
                standalone_question
            )
        )

        print(
            f"Fact/year query  : "
            f"{fact_query}"
        )

        # ----------------------------------------------------
        # Base retrieval
        # ----------------------------------------------------

        print()
        print("BASE RETRIEVAL")
        print("-" * 60)

        base_documents = (
            self.retrieve_base_documents(
                standalone_question
            )
        )

        print(
            f"Base documents retrieved: "
            f"{len(base_documents)}"
        )

        # ----------------------------------------------------
        # Fact/year expanded retrieval
        # ----------------------------------------------------

        fact_documents = (
            self.retrieve_fact_documents(
                question=standalone_question,
                scheme=scheme,
            )
        )

        # ----------------------------------------------------
        # Merge all candidates
        # ----------------------------------------------------

        documents = (
            base_documents
            + fact_documents
        )

        print()
        print(
            f"Documents before "
            f"deduplication: "
            f"{len(documents)}"
        )

        # ----------------------------------------------------
        # Deduplicate
        # ----------------------------------------------------

        documents = (
            self.remove_duplicates(
                documents
            )
        )

        print(
            f"Documents after "
            f"deduplication: "
            f"{len(documents)}"
        )

        # ----------------------------------------------------
        # Apply scheme metadata
        # ----------------------------------------------------

        documents = apply_scheme_boost(
            documents,
            scheme,
        )

        # ----------------------------------------------------
        # Prioritize exact scheme documents
        # ----------------------------------------------------

        documents = (
            prioritize_scheme_documents(
                documents,
                scheme,
            )
        )

        scheme_count = sum(
            1
            for document in documents
            if document.metadata.get(
                "scheme_match",
                False,
            )
        )

        print(
            f"Scheme-matching documents: "
            f"{scheme_count}"
        )

        return documents, scheme


    # ========================================================
    # DISPLAY DOCUMENTS
    # ========================================================

    def display_documents(
        self,
        documents: List[Document],
        title: str,
    ) -> None:
        """
        Print document retrieval/reranking diagnostics.
        """

        print()
        print("=" * 70)
        print(title)
        print("=" * 70)

        if not documents:

            print("No documents.")

            return

        for index, document in enumerate(
            documents,
            start=1,
        ):

            metadata = (
                document.metadata or {}
            )

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

            rerank_score = metadata.get(
                "rerank_score"
            )

            effective_score = metadata.get(
                "effective_rerank_score"
            )

            scheme_match = metadata.get(
                "scheme_match",
                False,
            )

            year_match = metadata.get(
                "year_match",
                False,
            )

            fact_boost = metadata.get(
                "fact_year_boost"
            )

            keyword_matches = metadata.get(
                "fact_keyword_matches"
            )

            print()
            print(
                f"{index}. {file_name}"
            )

            print(
                f"   Page         : {page}"
            )

            print(
                f"   Content type : "
                f"{content_type}"
            )

            print(
                f"   Scheme match : "
                f"{scheme_match}"
            )

            if rerank_score is not None:

                try:

                    print(
                        f"   Rerank score : "
                        f"{float(rerank_score):.6f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    print(
                        f"   Rerank score : "
                        f"{rerank_score}"
                    )

            if effective_score is not None:

                try:

                    print(
                        f"   Effective    : "
                        f"{float(effective_score):.6f}"
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    print(
                        f"   Effective    : "
                        f"{effective_score}"
                    )

            if fact_boost is not None:

                print(
                    f"   Fact boost   : "
                    f"{fact_boost}"
                )

            if year_match:

                print(
                    "   Year match   : True"
                )

            if keyword_matches is not None:

                print(
                    f"   Fact keywords: "
                    f"{keyword_matches}"
                )

            preview = (
                document.page_content
                or ""
            ).replace(
                "\n",
                " ",
            )

            if len(preview) > 250:

                preview = (
                    preview[:250]
                    + "..."
                )

            print(
                f"   Content      : "
                f"{preview}"
            )


    # ========================================================
    # RERANK DOCUMENTS
    # ========================================================

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Document]:
        """
        Rerank documents using the project's
        CrossEncoder reranking helper.

        IMPORTANT:

        self.reranker is a raw CrossEncoder.

        CrossEncoder itself does NOT have:
            .rerank()

        The project's reranker.py provides:
            rerank()

        which internally calls CrossEncoder.predict()
        and returns proper LangChain Documents containing
        rerank_score metadata.
        """

        if not documents:
            return []

        print()
        print("RERANKING")
        print("-" * 60)

        print(
            f"Input documents : "
            f"{len(documents)}"
        )

        print(
            f"Top N           : "
            f"{RERANK_TOP_N}"
        )

        # ----------------------------------------------------
        # IMPORTANT FIX
        # ----------------------------------------------------
        #
        # WRONG:
        #
        # self.reranker.rerank(...)
        #
        # CrossEncoder does not provide .rerank().
        #
        # CORRECT:
        #
        # rerank(
        #     query=...,
        #     documents=...,
        #     reranker=self.reranker,
        # )
        #
        # ----------------------------------------------------

        reranked = rerank(
            query=query,
            documents=documents,
            reranker=self.reranker,
            top_n=RERANK_TOP_N,
        )

        return reranked


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    def save_conversation(
        self,
        question: str,
        answer: str,
    ) -> None:
        """
        Save user question + assistant answer.
        """

        self.memory.add_turn(
            session_id=self.session_id,
            user_message=question,
            assistant_message=answer,
        )


    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
    ) -> Dict:
        """
        Run the complete RAG pipeline.

        Returns:

            {
                "answer": ...,
                "sources": ...,
                "standalone_question": ...,
                "scheme": ...,
                "documents": ...
            }
        """

        question = (
            question or ""
        ).strip()

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:

            return {
                "answer": (
                    "Please enter a question."
                ),
                "sources": [],
                "standalone_question": "",
                "scheme": None,
                "documents": [],
            }

        # ----------------------------------------------------
        # New query
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("NEW QUERY")
        print("=" * 70)

        print(
            f"Session ID: "
            f"{self.session_id}"
        )

        if self.memory.has_history(
            self.session_id
        ):

            print(
                "Conversation history: "
                "available"
            )

        else:

            print(
                "Conversation history: "
                "empty"
            )

        # ----------------------------------------------------
        # STEP 1 — Query rewriting
        # ----------------------------------------------------

        standalone_question = (
            self.get_standalone_question(
                question
            )
        )

        print()
        print("QUERY")

        print(
            f"Original   : "
            f"{question}"
        )

        print(
            f"Standalone : "
            f"{standalone_question}"
        )

        # ----------------------------------------------------
        # STEP 2 — Retrieval
        # ----------------------------------------------------

        (
            retrieved_documents,
            scheme,
        ) = self.retrieve_documents(
            standalone_question
        )

        # ----------------------------------------------------
        # No retrieval result
        # ----------------------------------------------------

        if not retrieved_documents:

            answer = (
                "I could not find enough "
                "information in the available "
                "farmer scheme documents to "
                "answer this question."
            )

            self.save_conversation(
                question,
                answer,
            )

            return {
                "answer": answer,
                "sources": [],
                "standalone_question":
                    standalone_question,
                "scheme": scheme,
                "documents": [],
            }

        # ----------------------------------------------------
        # STEP 3 — Reranking
        # ----------------------------------------------------

        reranked_documents = (
            self.rerank_documents(
                query=standalone_question,
                documents=retrieved_documents,
            )
        )

        self.display_documents(
            reranked_documents,
            "RERANKED DOCUMENTS",
        )

        # ----------------------------------------------------
        # STEP 4 — Fact/year boost
        # ----------------------------------------------------

        boosted_documents = (
            self.apply_fact_year_boost(
                documents=reranked_documents,
                question=standalone_question,
                scheme=scheme,
            )
        )

        self.display_documents(
            boosted_documents,
            "AFTER FACT/YEAR BOOST",
        )

        # ----------------------------------------------------
        # STEP 5 — Threshold
        # ----------------------------------------------------

        final_documents = (
            self.apply_relevance_threshold(
                boosted_documents
            )
        )

        print()
        print(
            f"Documents after threshold: "
            f"{len(final_documents)}"
        )

        # ----------------------------------------------------
        # STEP 6 — Quality control
        # ----------------------------------------------------

        final_documents = (
            self.quality_control(
                final_documents
            )
        )

        print(
            f"Documents after quality "
            f"control: "
            f"{len(final_documents)}"
        )

        # ----------------------------------------------------
        # STEP 7 — Final context
        # ----------------------------------------------------

        self.display_documents(
            final_documents,
            "FINAL CONTEXT",
        )

        # ----------------------------------------------------
        # Insufficient context
        # ----------------------------------------------------

        if (
            len(final_documents)
            < MIN_CONTEXT_DOCS
        ):

            answer = (
                "I could not find enough "
                "information in the available "
                "farmer scheme documents to "
                "answer this question."
            )

            self.save_conversation(
                question,
                answer,
            )

            return {
                "answer": answer,
                "sources": [],
                "standalone_question":
                    standalone_question,
                "scheme": scheme,
                "documents":
                    final_documents,
            }

        # ----------------------------------------------------
        # STEP 8 — LLM ANSWER GENERATION
        # ----------------------------------------------------

        print()
        print("GENERATING ANSWER")
        print("-" * 60)

        try:

            result = generate_answer(
                question=question,
                documents=final_documents,
            )

        except TypeError:

            # Compatibility fallback.
            result = generate_answer(
                question,
                final_documents,
            )

        # ----------------------------------------------------
        # Extract answer + sources
        # ----------------------------------------------------

        answer = ""

        sources = []

        if isinstance(
            result,
            dict,
        ):

            answer = (
                result.get("answer")
                or result.get("final_answer")
                or result.get("response")
                or result.get("content")
                or result.get("output")
                or ""
            )

            sources = (
                result.get("sources")
                or []
            )

        elif isinstance(
            result,
            str,
        ):

            answer = result

        else:

            answer = str(result)

        answer = (
            answer or ""
        ).strip()

        # ----------------------------------------------------
        # Empty answer fallback
        # ----------------------------------------------------

        if not answer:

            answer = (
                "I could not generate a "
                "reliable answer from the "
                "available farmer scheme "
                "documents."
            )

        # ----------------------------------------------------
        # STEP 9 — Save memory
        # ----------------------------------------------------

        self.save_conversation(
            question,
            answer,
        )

        # ----------------------------------------------------
        # STEP 10 — Display answer
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(answer)

        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            print()
            print("SOURCES")
            print("-" * 60)

            for index, source in enumerate(
                sources,
                start=1,
            ):

                if isinstance(
                    source,
                    dict,
                ):

                    file_name = (
                        source.get("file_name")
                        or source.get("file")
                        or "unknown"
                    )

                    page = (
                        source.get("page")
                        or source.get("page_number")
                        or "unknown"
                    )

                    print(
                        f"{index}. "
                        f"{file_name} "
                        f"(Page {page})"
                    )

                else:

                    print(
                        f"{index}. "
                        f"{source}"
                    )

        print("=" * 70)

        # ----------------------------------------------------
        # Return COMPLETE result
        # ----------------------------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "standalone_question":
                standalone_question,
            "scheme": scheme,
            "documents":
                final_documents,
        }


    # ========================================================
    # PRINT MEMORY
    # ========================================================

    def print_memory(self) -> None:
        """
        Display current conversation memory.
        """

        print()

        self.memory.print_memory(
            self.session_id
        )


    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(self) -> None:
        """
        Clear current session memory.
        """

        self.memory.clear_session(
            self.session_id
        )

        print(
            "✓ Conversation memory cleared."
        )


# ============================================================
# CLI
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FARMER SCHEME CHATBOT")
    print("=" * 70)

    # --------------------------------------------------------
    # Initialize pipeline
    # --------------------------------------------------------

    pipeline = QueryPipeline(
        session_id=SESSION_ID
    )

    print()
    print(
        "Type your question."
    )

    print(
        "Commands: memory | clear | exit"
    )

    print()

    # --------------------------------------------------------
    # Chat loop
    # --------------------------------------------------------

    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            print(
                "Goodbye!"
            )

            break

        # ----------------------------------------------------
        # Empty input
        # ----------------------------------------------------

        if not question:
            continue

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Goodbye!"
            )

            break

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        if question.lower() == "memory":

            pipeline.print_memory()

            continue

        # ----------------------------------------------------
        # Clear
        # ----------------------------------------------------

        if question.lower() == "clear":

            pipeline.clear_memory()

            continue

        # ----------------------------------------------------
        # Process question
        # ----------------------------------------------------

        try:

            pipeline.ask(
                question
            )

        except Exception as exc:

            print()
            print(
                "✗ Query failed:"
            )

            print(
                str(exc)
            )

            print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()