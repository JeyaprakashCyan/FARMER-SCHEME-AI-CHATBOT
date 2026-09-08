from pathlib import Path
import json
import sys
import traceback

# ---------------------------------------------------------
# Make project root available when running this script
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
from src.faiss_store import create_faiss
from src.bm25_store import save_bm25_documents
from src.parent_child import ParentChildStore
from src.config import BM25_INDEX_PATH, FAISS_INDEX_PATH
from src.metadata import DocumentMetadata
from src.parser.chunker import create_child_documents, create_parent_documents
from src.parser.pdf_parser import parse_pdf

PDF_DIR = PROJECT_ROOT / "data" / "pdf"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
DEBUG_DIR = PROJECT_ROOT / "storage" / "debug"
PARENT_STORE_DIR = PROJECT_ROOT / "storage" / "parent_store"


def load_metadata_for_pdf(pdf_path: Path) -> DocumentMetadata:
    metadata_path = METADATA_DIR / f"{pdf_path.stem}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata JSON missing: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as file:
        return DocumentMetadata.model_validate(json.load(file))


def save_debug_outputs(document_id, parsed_document, parents, children):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    with (DEBUG_DIR / f"{document_id}_ocr.txt").open("w", encoding="utf-8") as file:
        for page in parsed_document.pages:
            file.write(f"\n{'=' * 80}\nPAGE {page.page_number}\n")
            file.write(f"Extraction: {page.extraction_method}\n{'=' * 80}\n\n")
            file.write(f"{page.text}\n")
    with (DEBUG_DIR / f"{document_id}_chunks.json").open("w", encoding="utf-8") as file:
        json.dump({
            "parents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in parents],
            "children": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in children],
        }, file, ensure_ascii=False, indent=2)


def ingest_all_documents(enable_ocr=True):
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDFs found in {PDF_DIR}")

    all_parents = []
    all_children = []
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        metadata = load_metadata_for_pdf(pdf_path)
        parsed_document = parse_pdf(pdf_path, enable_ocr=enable_ocr)
        base_metadata = metadata.model_dump()
        parents = []
        for page in parsed_document.pages:
            if page.text.strip():
                parents.extend(create_parent_documents(page, base_metadata))
        children = create_child_documents(parents)
        save_debug_outputs(metadata.document_id, parsed_document, parents, children)
        all_parents.extend(parents)
        all_children.extend(children)
        print(f"Processed {pdf_path.name}: {len(parents)} parents, {len(children)} children")

    return {"parents": all_parents, "children": all_children}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def validate_documents(documents, document_type):
    """
    Remove invalid/empty LangChain documents.
    """

    valid = []
    invalid = []

    for doc in documents:

        if doc is None:
            invalid.append(doc)
            continue

        text = (doc.page_content or "").strip()

        if len(text) < 30:
            invalid.append(doc)
            continue

        valid.append(doc)

    print()
    print(f"{document_type} validation")
    print("-" * 60)
    print(f"Total:   {len(documents)}")
    print(f"Valid:   {len(valid)}")
    print(f"Invalid: {len(invalid)}")

    return valid


def show_chunk_statistics(children):

    print()
    print("=" * 80)
    print("CHUNK STATISTICS")
    print("=" * 80)

    document_ids = set()
    parent_ids = set()

    for doc in children:

        metadata = doc.metadata

        document_id = metadata.get("document_id")
        parent_id = metadata.get("parent_id")

        if document_id:
            document_ids.add(document_id)

        if parent_id:
            parent_ids.add(parent_id)

    print(f"Child chunks      : {len(children)}")
    print(f"Unique documents  : {len(document_ids)}")
    print(f"Unique parents    : {len(parent_ids)}")


def verify_faiss():

    faiss_path = Path(FAISS_INDEX_PATH)
    if not faiss_path.is_absolute():
        faiss_path = PROJECT_ROOT / faiss_path

    print()
    print("=" * 80)
    print("FAISS VERIFICATION")
    print("=" * 80)

    print(f"FAISS directory: {faiss_path}")
    print(f"Directory exists: {faiss_path.exists()}")

    if not faiss_path.exists():
        print("ERROR: FAISS directory was not created.")
        return False

    files = list(faiss_path.iterdir())

    print(f"Files found: {len(files)}")

    for file in files:
        print(f"  - {file.name}")

    index_file = faiss_path / "index.faiss"
    pickle_file = faiss_path / "index.pkl"

    print()

    print(f"index.faiss exists: {index_file.exists()}")
    print(f"index.pkl exists:   {pickle_file.exists()}")

    if index_file.exists() and pickle_file.exists():

        print()
        print("SUCCESS: FAISS index created successfully.")

        print()
        print("FAISS file sizes:")

        print(
            f"index.faiss : "
            f"{index_file.stat().st_size:,} bytes"
        )

        print(
            f"index.pkl   : "
            f"{pickle_file.stat().st_size:,} bytes"
        )

        return True

    print()
    print("ERROR: FAISS files are missing.")

    return False


def verify_bm25():

    bm25_path = Path(BM25_INDEX_PATH)
    if not bm25_path.is_absolute():
        bm25_path = PROJECT_ROOT / bm25_path

    print()
    print("=" * 80)
    print("BM25 VERIFICATION")
    print("=" * 80)

    print(f"BM25 directory: {bm25_path}")
    print(f"Directory exists: {bm25_path.exists()}")

    if not bm25_path.exists():
        print("ERROR: BM25 directory does not exist.")
        return False

    files = list(bm25_path.iterdir())

    print(f"Files found: {len(files)}")

    for file in files:
        print(f"  - {file.name}")

    documents_file = bm25_path / "documents.json"

    if documents_file.exists():

        print()
        print("SUCCESS: BM25 documents created.")

        print(
            f"documents.json size: "
            f"{documents_file.stat().st_size:,} bytes"
        )

        return True

    print()
    print("ERROR: documents.json not found.")

    return False


# ---------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------

def main():

    print()
    print("=" * 80)
    print(" FARMER SCHEME CHATBOT")
    print(" COMPLETE RAG INGESTION")
    print("=" * 80)

    print()
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"FAISS path         : {FAISS_INDEX_PATH}")
    print(f"BM25 path          : {BM25_INDEX_PATH}")

    # -----------------------------------------------------
    # STEP 1
    # Parse PDFs + metadata + OCR
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 1 - PROCESSING ALL PDF DOCUMENTS")
    print("=" * 80)

    try:

        result = ingest_all_documents()

    except Exception as e:

        print()
        print("=" * 80)
        print("INGESTION FAILED")
        print("=" * 80)

        print(f"Error: {e}")

        traceback.print_exc()

        return

    # -----------------------------------------------------
    # STEP 2
    # Get parent / child documents
    # -----------------------------------------------------

    parents = result.get("parents", [])
    children = result.get("children", [])

    print()
    print("=" * 80)
    print("STEP 2 - INGESTION RESULT")
    print("=" * 80)

    print(f"Parent sections : {len(parents)}")
    print(f"Child chunks    : {len(children)}")

    # -----------------------------------------------------
    # STEP 3
    # Validate
    # -----------------------------------------------------

    parents = validate_documents(
        parents,
        "PARENT DOCUMENTS"
    )

    children = validate_documents(
        children,
        "CHILD DOCUMENTS"
    )

    if not children:

        print()
        print("=" * 80)
        print("ERROR")
        print("=" * 80)

        print(
            "No valid child chunks were generated."
        )

        print(
            "FAISS cannot be created without child chunks."
        )

        return

    show_chunk_statistics(children)

    # -----------------------------------------------------
    # STEP 4
    # Save parent store
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 3 - SAVING PARENT STORE")
    print("=" * 80)

    try:

        parent_store = ParentChildStore(parents)

        parent_store.save()

        print()
        print("Parent store saved successfully.")

    except Exception as e:

        print()
        print("ERROR SAVING PARENT STORE")
        print(e)

        traceback.print_exc()

        return

    # -----------------------------------------------------
    # STEP 5
    # Create FAISS
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 4 - CREATING FAISS INDEX")
    print("=" * 80)

    print()
    print(
        f"Creating embeddings for {len(children)} child chunks..."
    )

    try:

        vector_store = create_faiss(children)

        print()
        print("FAISS creation completed.")

    except Exception as e:

        print()
        print("=" * 80)
        print("FAISS CREATION FAILED")
        print("=" * 80)

        print(f"Error: {e}")

        traceback.print_exc()

        return

    # -----------------------------------------------------
    # STEP 6
    # Verify FAISS
    # -----------------------------------------------------

    faiss_ok = verify_faiss()

    if not faiss_ok:

        print()
        print(
            "FAISS verification failed."
        )

        return

    # -----------------------------------------------------
    # STEP 7
    # Create BM25
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 5 - CREATING BM25 INDEX")
    print("=" * 80)

    try:

        save_bm25_documents(
            children,
            BM25_INDEX_PATH
        )

        print()
        print("BM25 creation completed.")

    except Exception as e:

        print()
        print("=" * 80)
        print("BM25 CREATION FAILED")
        print("=" * 80)

        print(f"Error: {e}")

        traceback.print_exc()

        return

    # -----------------------------------------------------
    # STEP 8
    # Verify BM25
    # -----------------------------------------------------

    bm25_ok = verify_bm25()

    # -----------------------------------------------------
    # STEP 9
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("INGESTION SUMMARY")
    print("=" * 80)

    print(f"Documents processed : 26")
    print(f"Parent sections     : {len(parents)}")
    print(f"Child chunks        : {len(children)}")
    print(f"FAISS created       : {'YES' if faiss_ok else 'NO'}")
    print(f"BM25 created        : {'YES' if bm25_ok else 'NO'}")

    print()

    if faiss_ok and bm25_ok:

        print("=" * 80)
        print(" SUCCESS - RAG INDEXING COMPLETED")
        print("=" * 80)

        print()
        print("Created:")
        print("  storage/faiss/index.faiss")
        print("  storage/faiss/index.pkl")
        print("  storage/bm25/documents.json")
        print("  storage/parent_store/parents.json")

        print()
        print("You can now proceed to retrieval testing.")

    else:

        print("=" * 80)
        print(" INGESTION INCOMPLETE")
        print("=" * 80)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()