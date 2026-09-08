import os

from dotenv import load_dotenv

load_dotenv()


EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3"
)


FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "storage/faiss"
)

BM25_INDEX_PATH = os.getenv(
    "BM25_INDEX_PATH",
    "storage/bm25"
)


TOP_K_DENSE = int(
    os.getenv(
        "TOP_K_DENSE",
        "30"
    )
)

TOP_K_BM25 = int(
    os.getenv(
        "TOP_K_BM25",
        "30"
    )
)

TOP_K_RRF = int(
    os.getenv(
        "TOP_K_RRF",
        "20"
    )
)

TOP_K_RERANK = int(
    os.getenv(
        "TOP_K_RERANK",
        "8"
    )
)