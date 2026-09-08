from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_K_RRF,
)


class HybridRetriever:

    def __init__(
        self,
        vector_store,
        bm25_store,
    ):

        self.vector_store = vector_store
        self.bm25_store = bm25_store

    # ========================================================
    # MAIN RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        state: str | None = None,
    ) -> List[Document]:

        print()
        print("Running FAISS search...")

        dense_docs = self._dense_search(
            query,
            state
        )

        print(
            f"FAISS results: {len(dense_docs)}"
        )

        print()
        print("Running BM25 search...")

        bm25_docs = self._bm25_search(
            query,
            state
        )

        print(
            f"BM25 results: {len(bm25_docs)}"
        )

        print()
        print("Running RRF fusion...")

        results = self._rrf(
            dense_docs,
            bm25_docs
        )

        print(
            f"RRF results: {len(results)}"
        )

        return results

    # ========================================================
    # FAISS
    # ========================================================

    def _dense_search(
        self,
        query,
        state=None,
    ):

        docs = self.vector_store.similarity_search(
            query,
            k=TOP_K_DENSE
        )

        results = []

        for doc in docs:

            if self._allowed(
                doc,
                state
            ):

                results.append(doc)

        return results

    # ========================================================
    # BM25
    # ========================================================

    def _bm25_search(
        self,
        query,
        state=None,
    ):

        return self.bm25_store.search(
            query,
            top_k=TOP_K_BM25,
            state=state
        )

    # ========================================================
    # RRF
    # ========================================================

    def _rrf(
        self,
        dense_docs,
        bm25_docs,
    ):

        RRF_K = 60

        scores = {}
        documents = {}

        # ----------------------------------------------------
        # FAISS ranking
        # ----------------------------------------------------

        for rank, doc in enumerate(
            dense_docs,
            start=1
        ):

            chunk_id = doc.metadata.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (RRF_K + rank)
            )

            documents[chunk_id] = doc

        # ----------------------------------------------------
        # BM25 ranking
        # ----------------------------------------------------

        for rank, doc in enumerate(
            bm25_docs,
            start=1
        ):

            chunk_id = doc.metadata.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (RRF_K + rank)
            )

            documents[chunk_id] = doc

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for chunk_id, score in ranked:

            doc = documents[chunk_id]

            doc.metadata[
                "rrf_score"
            ] = float(score)

            results.append(doc)

            if len(results) >= TOP_K_RRF:
                break

        return results

    # ========================================================
    # STATE FILTER
    # ========================================================

    @staticmethod
    def _allowed(
        doc,
        state
    ):

        if not state:
            return True

        metadata = doc.metadata

        scope = metadata.get(
            "applicable_scope",
            "India"
        )

        document_state = metadata.get(
            "state"
        )

        # ----------------------------------------------------
        # India-wide documents are always allowed
        # ----------------------------------------------------

        if (
            scope
            and scope.lower() == "india"
        ):

            return True

        # ----------------------------------------------------
        # State-specific documents
        # ----------------------------------------------------

        if (
            scope
            and scope.lower() == "state"
            and document_state
            and document_state.lower()
            == state.lower()
        ):

            return True

        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the hybrid FAISS and BM25 retriever module."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Confirm that the HybridRetriever class is importable",
    )
    args = parser.parse_args()
    if args.status:
        print("HybridRetriever import OK")
        print("Dense source: FAISS")
        print("Lexical source: BM25")
        print("Fusion: reciprocal rank fusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())