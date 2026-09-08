"""
Step 12.12 - FAISS Vector Store

Pipeline:

Chunked Documents
        ↓
Embedding Model
        ↓
FAISS
        ↓
Persistent Vector Store
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from embeddings import create_embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "vector_store" / "faiss_index"


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from LangChain Documents.
    """

    if not documents:
        raise ValueError(
            "Cannot build FAISS index from empty documents."
        )

    vector_store = FAISS.from_documents(
        documents,
        embedding_model,
    )

    return vector_store


# ============================================================
# SAVE FAISS INDEX
# ============================================================

def save_faiss_index(
    vector_store: FAISS,
    index_dir: Path = FAISS_INDEX_DIR,
) -> None:
    """
    Persist FAISS index to disk.
    """

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(index_dir)
    )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_faiss_index(
    embedding_model: HuggingFaceEmbeddings,
    index_dir: Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Load a previously persisted FAISS index.
    """

    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_dir}"
        )

    vector_store = FAISS.load_local(
        str(index_dir),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vector_store


# ============================================================
# SEARCH
# ============================================================

def similarity_search(
    vector_store: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Perform similarity search.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than zero."
        )

    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results


# ============================================================
# PRINT SEARCH RESULTS
# ============================================================

def print_search_results(
    results: List[Document],
) -> None:

    print()
    print("=" * 80)
    print("FAISS SEARCH RESULTS")
    print("=" * 80)

    for index, document in enumerate(results, start=1):

        print()
        print(f"Result #{index}")
        print("-" * 80)

        print(
            f"Content Type : "
            f"{document.metadata.get('content_type')}"
        )

        print(
            f"Source       : "
            f"{document.metadata.get('source')}"
        )

        print(
            f"Page         : "
            f"{document.metadata.get('page')}"
        )

        print(
            f"Content      : "
            f"{document.page_content}"
        )