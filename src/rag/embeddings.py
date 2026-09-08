"""
Step 12.11 - Embeddings

Converts LangChain Documents into vector embeddings.

Pipeline:
    Chunked Documents
        ↓
    Embedding Model
        ↓
    Vector Embeddings
        ↓
    Ready for FAISS
"""

from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

EMBEDDING_DEVICE = "cpu"

NORMALIZE_EMBEDDINGS = True


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return the Hugging Face embedding model.
    """

    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": NORMALIZE_EMBEDDINGS
        },
    )

    return model


# ============================================================
# EMBED DOCUMENTS
# ============================================================

def embed_documents(
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> List[List[float]]:
    """
    Convert document text into embeddings.

    Returns:
        List of embedding vectors.
    """

    if not documents:
        raise ValueError("No documents supplied for embedding.")

    texts = [
        document.page_content
        for document in documents
    ]

    embeddings = embedding_model.embed_documents(texts)

    return embeddings


# ============================================================
# EMBED SINGLE QUERY
# ============================================================

def embed_query(
    query: str,
    embedding_model: HuggingFaceEmbeddings,
) -> List[float]:
    """
    Convert a user query into an embedding vector.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    return embedding_model.embed_query(query)


# ============================================================
# GET EMBEDDING DIMENSION
# ============================================================

def get_embedding_dimension(
    embedding_model: HuggingFaceEmbeddings,
) -> int:
    """
    Determine embedding vector dimension.
    """

    test_vector = embedding_model.embed_query("dimension test")

    return len(test_vector)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(
    documents: List[Document],
    embeddings: List[List[float]],
) -> None:
    """
    Validate that embeddings correspond correctly
    to the supplied documents.
    """

    if len(documents) != len(embeddings):
        raise ValueError(
            f"Document count ({len(documents)}) does not match "
            f"embedding count ({len(embeddings)})."
        )

    if not embeddings:
        raise ValueError("No embeddings generated.")

    dimension = len(embeddings[0])

    if dimension == 0:
        raise ValueError("Embedding dimension is zero.")

    for index, vector in enumerate(embeddings):

        if len(vector) != dimension:
            raise ValueError(
                f"Embedding dimension mismatch at index {index}."
            )

        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError(
                f"Invalid embedding values at index {index}."
            )


# ============================================================
# PRINT EMBEDDING SUMMARY
# ============================================================

def print_embedding_summary(
    documents: List[Document],
    embeddings: List[List[float]],
) -> None:

    print()
    print("=" * 80)
    print("STEP 12.11 - EMBEDDING SUMMARY")
    print("=" * 80)

    print(f"Embedding Model : {EMBEDDING_MODEL_NAME}")
    print(f"Device          : {EMBEDDING_DEVICE}")
    print(f"Normalized      : {NORMALIZE_EMBEDDINGS}")

    print(f"Documents       : {len(documents)}")

    if embeddings:
        print(f"Vector Dimension: {len(embeddings[0])}")

    print("=" * 80)