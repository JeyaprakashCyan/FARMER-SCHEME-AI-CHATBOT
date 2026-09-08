import argparse
import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH
)


def _index_path() -> Path:
    path = Path(FAISS_INDEX_PATH)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_faiss(documents):

    if not documents:
        raise ValueError(
            "No documents supplied to FAISS."
        )

    index_path = _index_path()
    index_path.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Creating FAISS index "
        f"with {len(documents)} chunks..."
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = (
        FAISS.from_documents(
            documents,
            embeddings
        )
    )

    vector_store.save_local(
        str(index_path)
    )

    print(
        f"FAISS index saved to "
        f"{index_path}"
    )

    return vector_store


def load_faiss():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    return FAISS.load_local(
        str(_index_path()),
        embeddings,
        allow_dangerous_deserialization=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or load the local FAISS index."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show whether the configured FAISS index exists",
    )
    args = parser.parse_args()

    index_path = _index_path()
    index_files = sorted(index_path.glob("index.*"))
    print(f"FAISS index directory: {index_path}")
    print(f"Index exists: {index_path.is_dir()}")
    print(f"Index files: {len(index_files)}")
    for index_file in index_files:
        print(f"- {index_file.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())