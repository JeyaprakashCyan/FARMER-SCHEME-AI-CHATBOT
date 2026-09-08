from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from src.config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:
    """
    Cross-encoder reranker for final retrieval candidates.

    Designed to be memory-safe for local Windows development.
    """

    # Maximum number of documents sent to the cross-encoder
    MAX_CANDIDATES = 15

    # Number of query/document pairs processed at once
    BATCH_SIZE = 4

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        max_candidates: int = MAX_CANDIDATES,
        batch_size: int = BATCH_SIZE,
    ):
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        """
        Load the CrossEncoder only when required.
        """

        if self._model is None:

            print()
            print("=" * 80)
            print("LOADING CROSS-ENCODER MODEL")
            print("=" * 80)

            print(f"Model       : {self.model_name}")
            print(f"Max length  : 512")
            print(f"Batch size  : {self.batch_size}")

            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
            )

            print("✓ Cross-encoder loaded successfully.")

        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = TOP_K_RERANK,
    ) -> List[Document]:

        if not documents:
            return []

        # ---------------------------------------------------------
        # Remove duplicate documents
        # ---------------------------------------------------------

        unique_documents = []
        seen = set()

        for document in documents:

            chunk_id = document.metadata.get("chunk_id")
            parent_id = document.metadata.get("parent_id")

            key = (
                chunk_id
                or parent_id
                or document.page_content[:200]
            )

            if key in seen:
                continue

            seen.add(key)
            unique_documents.append(document)

        print(
            f"Reranker input before limit : "
            f"{len(unique_documents)}"
        )

        # ---------------------------------------------------------
        # Limit candidates
        # ---------------------------------------------------------

        candidates = unique_documents[: self.max_candidates]

        print(
            f"Reranker candidates         : "
            f"{len(candidates)}"
        )

        # ---------------------------------------------------------
        # Prepare query/document pairs
        # ---------------------------------------------------------

        pairs = [
            (
                query,
                document.page_content
            )
            for document in candidates
        ]

        # ---------------------------------------------------------
        # Run cross encoder
        # ---------------------------------------------------------

        model = self._get_model()

        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # ---------------------------------------------------------
        # Attach scores
        # ---------------------------------------------------------

        ranked = sorted(
            zip(scores, candidates),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        # ---------------------------------------------------------
        # Return top K
        # ---------------------------------------------------------

        results = []

        for score, document in ranked[:top_k]:

            document.metadata["rerank_score"] = float(score)

            results.append(document)

        return results