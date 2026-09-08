import argparse
import json
import re
import sys

from pathlib import Path

from rank_bm25 import BM25Okapi

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BM25_PATH = PROJECT_ROOT / "storage" / "bm25"
DEFAULT_DEBUG_PATH = PROJECT_ROOT / "storage" / "debug"


def _resolve_store_path(path) -> Path:
    store_path = Path(path)
    if not store_path.is_absolute():
        store_path = PROJECT_ROOT / store_path
    return store_path


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

def tokenize(text):

    if not text:
        return []

    return re.findall(
        r"[A-Za-z0-9]+",
        text.lower()
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

def save_bm25_documents(
    documents,
    path
):

    store_path = _resolve_store_path(path)
    store_path.mkdir(
        parents=True,
        exist_ok=True
    )

    data = []

    for doc in documents:

        data.append({

            "page_content":
                doc.page_content,

            "metadata":
                doc.metadata

        })

    output_file = (
        store_path
        / "documents.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"BM25 documents saved: "
        f"{len(data)}"
    )


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

def load_bm25_documents(
    path
):

    input_file = _resolve_store_path(path) / "documents.json"

    if not input_file.exists():

        raise FileNotFoundError(
            f"BM25 file not found: "
            f"{input_file}"
        )

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return [

        Document(
            page_content=item[
                "page_content"
            ],

            metadata=item[
                "metadata"
            ]
        )

        for item in data
    ]


def load_documents_from_debug(
    debug_path: Path
) -> list[Document]:
    documents = []

    for chunk_file in sorted(debug_path.glob("*_chunks.json")):
        with chunk_file.open(encoding="utf-8") as file:
            data = json.load(file)

        for item in data.get("children", []):
            documents.append(
                Document(
                    page_content=item.get("page_content", ""),
                    metadata=item.get("metadata", {}),
                )
            )

    return documents


# ---------------------------------------------------------
# BM25 Store
# ---------------------------------------------------------

class BM25Store:

    def __init__(
        self,
        documents
    ):

        self.documents = documents

        corpus = [

            tokenize(
                doc.page_content
            )

            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            corpus
        )

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search(
        self,
        query,
        top_k=30,
        state=None
    ):

        tokens = tokenize(
            query
        )

        if not tokens:
            return []

        scores = self.bm25.get_scores(
            tokens
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for index, score in ranked:

            doc = self.documents[index]

            if not self._allowed(
                doc,
                state
            ):
                continue

            results.append(
                doc
            )

            if len(results) >= top_k:
                break

        return results

    # -----------------------------------------------------
    # State filtering
    # -----------------------------------------------------

    @staticmethod
    def _allowed(
        doc,
        state
    ):

        if not state:
            return True

        scope = doc.metadata.get(
            "applicable_scope",
            "India"
        )

        document_state = (
            doc.metadata.get(
                "state"
            )
        )

        # India-wide documents
        if scope.lower() == "india":
            return True

        # State-specific documents
        if (
            scope.lower() == "state"
            and document_state
            and document_state.lower()
            == state.lower()
        ):
            return True

        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or search the local BM25 document store."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Search query; omit it to show store status",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_BM25_PATH,
        help="BM25 store directory (default: storage/bm25)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum search results (default: 5)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the BM25 store from storage/debug chunk files",
    )
    parser.add_argument(
        "--debug-path",
        type=Path,
        default=DEFAULT_DEBUG_PATH,
        help="Directory containing *_chunks.json files",
    )
    args = parser.parse_args()
    store_path = args.path
    if not store_path.is_absolute():
        store_path = PROJECT_ROOT / store_path

    debug_path = args.debug_path
    if not debug_path.is_absolute():
        debug_path = PROJECT_ROOT / debug_path

    if args.build:
        documents = load_documents_from_debug(debug_path)
        if not documents:
            print(f"No child chunks found in {debug_path}")
            return 1
        save_bm25_documents(documents, store_path)

    input_file = store_path / "documents.json"
    if not input_file.exists():
        print(f"BM25 store not found: {input_file}")
        print("Run with --build after ingestion to create it.")
        return 1

    documents = load_bm25_documents(store_path)
    print(f"BM25 store: {store_path}")
    print(f"Documents: {len(documents)}")

    if args.query:
        query = " ".join(args.query)
        results = BM25Store(documents).search(query, top_k=args.top_k)
        print(f"Results for: {query}")
        for index, document in enumerate(results, start=1):
            print(
                f"{index}. "
                f"{document.metadata.get('document_name', 'Unknown')} | "
                f"{document.page_content[:160].replace(chr(10), ' ')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())