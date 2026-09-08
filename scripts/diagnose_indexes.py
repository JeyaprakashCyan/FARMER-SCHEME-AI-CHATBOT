from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.faiss_store import load_faiss
from src.bm25_store import load_bm25_documents
from src.config import FAISS_INDEX_PATH, BM25_INDEX_PATH


def get_chunk_id(doc):
    return doc.metadata.get("chunk_id")


def main():

    print()
    print("=" * 80)
    print("INDEX CONSISTENCY DIAGNOSTIC")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load BM25
    # ---------------------------------------------------------

    print("\nLoading BM25...")

    bm25_documents = load_bm25_documents(BM25_INDEX_PATH)

    bm25_ids = [
        get_chunk_id(doc)
        for doc in bm25_documents
        if get_chunk_id(doc)
    ]

    print(f"BM25 documents       : {len(bm25_documents)}")
    print(f"BM25 chunk IDs       : {len(bm25_ids)}")
    print(f"BM25 unique chunk IDs: {len(set(bm25_ids))}")

    # ---------------------------------------------------------
    # Load FAISS
    # ---------------------------------------------------------

    print("\nLoading FAISS...")

    vector_store = load_faiss()

    faiss_documents = list(
        vector_store.docstore._dict.values()
    )

    faiss_ids = [
        get_chunk_id(doc)
        for doc in faiss_documents
        if get_chunk_id(doc)
    ]

    print(f"FAISS documents       : {len(faiss_documents)}")
    print(f"FAISS chunk IDs       : {len(faiss_ids)}")
    print(f"FAISS unique chunk IDs: {len(set(faiss_ids))}")

    # ---------------------------------------------------------
    # Compare
    # ---------------------------------------------------------

    bm25_set = set(bm25_ids)
    faiss_set = set(faiss_ids)

    bm25_only = bm25_set - faiss_set
    faiss_only = faiss_set - bm25_set

    duplicate_bm25 = len(bm25_ids) - len(bm25_set)
    duplicate_faiss = len(faiss_ids) - len(faiss_set)

    print()
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print(f"BM25 only chunk IDs   : {len(bm25_only)}")
    print(f"FAISS only chunk IDs  : {len(faiss_only)}")
    print(f"Duplicate BM25 IDs    : {duplicate_bm25}")
    print(f"Duplicate FAISS IDs   : {duplicate_faiss}")

    # ---------------------------------------------------------
    # BM25-only details
    # ---------------------------------------------------------

    if bm25_only:

        print()
        print("=" * 80)
        print("BM25-ONLY CHUNKS")
        print("=" * 80)

        for chunk_id in sorted(bm25_only):

            matching = [
                doc
                for doc in bm25_documents
                if get_chunk_id(doc) == chunk_id
            ]

            for doc in matching:

                print()
                print(f"Chunk ID     : {chunk_id}")
                print(
                    f"Document     : "
                    f"{doc.metadata.get('document_name')}"
                )
                print(
                    f"Page         : "
                    f"{doc.metadata.get('page_number')}"
                )
                print(
                    f"Parent ID    : "
                    f"{doc.metadata.get('parent_id')}"
                )
                print(
                    f"Text preview : "
                    f"{doc.page_content[:200].replace(chr(10), ' ')}"
                )

    # ---------------------------------------------------------
    # FAISS-only details
    # ---------------------------------------------------------

    if faiss_only:

        print()
        print("=" * 80)
        print("FAISS-ONLY CHUNKS")
        print("=" * 80)

        for chunk_id in sorted(faiss_only):

            matching = [
                doc
                for doc in faiss_documents
                if get_chunk_id(doc) == chunk_id
            ]

            for doc in matching:

                print()
                print(f"Chunk ID     : {chunk_id}")
                print(
                    f"Document     : "
                    f"{doc.metadata.get('document_name')}"
                )
                print(
                    f"Page         : "
                    f"{doc.metadata.get('page_number')}"
                )
                print(
                    f"Parent ID    : "
                    f"{doc.metadata.get('parent_id')}"
                )
                print(
                    f"Text preview : "
                    f"{doc.page_content[:200].replace(chr(10), ' ')}"
                )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print()
    print("=" * 80)

    if (
        len(bm25_documents) == len(faiss_documents)
        and not bm25_only
        and not faiss_only
        and duplicate_bm25 == 0
        and duplicate_faiss == 0
    ):

        print("SUCCESS: FAISS and BM25 are perfectly synchronized.")

    else:

        print("WARNING: FAISS and BM25 are NOT synchronized.")

    print("=" * 80)


if __name__ == "__main__":
    main()