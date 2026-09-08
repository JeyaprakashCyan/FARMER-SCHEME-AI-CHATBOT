from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.parser.section_detector import (
    SectionBlock,
    detect_sections
)

from src.parser.table_parser import (
    extract_table_like_blocks
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHILD_CHUNK_SIZE = 1800
CHILD_CHUNK_OVERLAP = 250


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_id(value: str) -> str:

    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------
# Parent section creation
# ---------------------------------------------------------

def create_parent_documents(
    parsed_page,
    base_metadata: dict
) -> List[Document]:

    sections = detect_sections(
        parsed_page.page_number,
        parsed_page.text
    )

    parents = []

    for index, section in enumerate(
        sections
    ):

        parent_key = (
            f"{base_metadata['document_id']}"
            f"_{parsed_page.page_number}"
            f"_{index}"
            f"_{section.section}"
            f"_{section.subsection or ''}"
        )

        parent_id = make_id(parent_key)

        tables = extract_table_like_blocks(
            section.text
        )

        content_parts = []

        content_parts.append(
            f"Section: {section.section}"
        )

        if section.subsection:

            content_parts.append(
                f"Subsection: "
                f"{section.subsection}"
            )

        content_parts.append(
            section.text
        )

        if tables:

            content_parts.append(
                "\nNormalized table data:\n"
                + "\n\n".join(tables)
            )

        content = "\n\n".join(
            content_parts
        )

        metadata = {
            **base_metadata,

            "page_number":
                parsed_page.page_number,

            "section":
                section.section,

            "subsection":
                section.subsection,

            "parent_id":
                parent_id,

            "chunk_type":
                "parent",

            "extraction_method":
                parsed_page.extraction_method,

            "image_count":
                parsed_page.image_count
        }

        parents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return parents


# ---------------------------------------------------------
# Child chunk creation
# ---------------------------------------------------------

def create_child_documents(
    parent_documents: List[Document]
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    children = []

    for parent in parent_documents:

        split_docs = splitter.split_documents(
            [parent]
        )

        for index, child in enumerate(
            split_docs
        ):

            parent_id = parent.metadata[
                "parent_id"
            ]

            raw_id = (
                f"{parent_id}"
                f"_child_{index}"
            )

            child_id = make_id(raw_id)

            child.metadata.update({

                "parent_id":
                    parent_id,

                "chunk_id":
                    child_id,

                "chunk_index":
                    index,

                "chunk_type":
                    "child"
            })

            children.append(child)

    return children


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create parent sections and child chunks from text."
    )
    parser.add_argument(
        "text_file",
        nargs="?",
        type=Path,
        help="Text file to chunk; reads standard input when omitted",
    )
    parser.add_argument(
        "--document-id",
        default="cli-document",
        help="Document identifier stored in chunk metadata",
    )
    parser.add_argument(
        "--page-number",
        type=int,
        default=1,
        help="Page number stored in chunk metadata (default: 1)",
    )
    args = parser.parse_args()

    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        return 0

    from src.parser.pdf_parser import ParsedPage

    page = ParsedPage(
        page_number=args.page_number,
        text=text,
        extraction_method="text",
    )
    parents = create_parent_documents(
        page,
        {"document_id": args.document_id},
    )
    children = create_child_documents(parents)

    print(f"Created {len(parents)} parent documents.")
    print(f"Created {len(children)} child chunks.")
    for index, child in enumerate(children, start=1):
        print(
            f"Child {index}: "
            f"id={child.metadata['chunk_id']}, "
            f"characters={len(child.page_content)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())