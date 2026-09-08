from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ParentChildStore:

    def __init__(
        self,
        parent_documents: List[Document]
    ):

        self.parents: Dict[str, Document] = {}

        for document in parent_documents:

            parent_id = document.metadata.get(
                "parent_id"
            )

            if parent_id:

                self.parents[parent_id] = document

    def get_parent(
        self,
        parent_id: str
    ) -> Document | None:

        return self.parents.get(
            parent_id
        )

    def expand(
        self,
        child_documents: List[Document]
    ) -> List[Document]:

        results = []

        seen = set()

        for child in child_documents:

            parent_id = child.metadata.get(
                "parent_id"
            )

            if not parent_id:
                continue

            if parent_id in seen:
                continue

            parent = self.get_parent(
                parent_id
            )

            if parent:

                results.append(parent)

                seen.add(parent_id)

        return results

    def save(self, path=None):
        output_dir = Path(path) if path else Path(__file__).resolve().parents[1] / "storage" / "parent_store"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "parents.json"
        data = [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in self.parents.values()
        ]
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return output_file

    @classmethod
    def load(cls, path=None):
        input_dir = Path(path) if path else Path(__file__).resolve().parents[1] / "storage" / "parent_store"
        input_file = input_dir / "parents.json"
        if not input_file.exists():
            raise FileNotFoundError(f"Parent store not found: {input_file}")
        with input_file.open(encoding="utf-8") as file:
            data = json.load(file)
        documents = [
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in data
        ]
        return cls(documents)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate parent-child document expansion."
    )
    parser.add_argument(
        "text_file",
        nargs="?",
        type=Path,
        help="Text file to process; reads standard input when omitted",
    )
    parser.add_argument(
        "--document-id",
        default="cli-document",
        help="Document identifier stored in metadata",
    )
    args = parser.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        return 0

    from src.parser.chunker import create_child_documents, create_parent_documents
    from src.parser.pdf_parser import ParsedPage

    page = ParsedPage(
        page_number=1,
        text=text,
        extraction_method="text",
    )
    parents = create_parent_documents(
        page,
        {"document_id": args.document_id},
    )
    children = create_child_documents(parents)
    store = ParentChildStore(parents)
    expanded = store.expand(children)

    print(f"Parents: {len(parents)}")
    print(f"Children: {len(children)}")
    print(f"Expanded parents: {len(expanded)}")
    if children and expanded:
        print("Parent-child expansion: OK")
    elif not parents and not children:
        print("No sections detected in the input.")
    else:
        print("Parent-child expansion: FAILED", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())