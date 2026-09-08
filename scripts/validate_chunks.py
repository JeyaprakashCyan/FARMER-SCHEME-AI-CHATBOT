import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = PROJECT_ROOT / "storage" / "debug"

REQUIRED_METADATA = [
    "document_id",
    "document_name",
    "scheme_id",
    "page_number",
    "section",
    "parent_id",
    "chunk_id",
    "chunk_type",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate debug parent and child chunk JSON files."
    )
    parser.add_argument(
        "debug_dir",
        nargs="?",
        type=Path,
        default=DEBUG_DIR,
        help="Directory containing *_chunks.json files",
    )
    args = parser.parse_args()
    debug_dir = args.debug_dir
    if not debug_dir.is_absolute():
        debug_dir = PROJECT_ROOT / debug_dir

    files = list(
        debug_dir.glob(
            "*_chunks.json"
        )
    )

    if not files:

        raise RuntimeError(
            f"No chunk debug files found in {debug_dir}."
        )

    total_children = 0

    errors = []

    for file_path in files:

        print(
            f"\nChecking: "
            f"{file_path.name}"
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        children = data.get(
            "children",
            []
        )

        total_children += len(
            children
        )

        for index, child in enumerate(
            children
        ):

            metadata = child.get(
                "metadata",
                {}
            )

            content = child.get(
                "page_content",
                ""
            )

            for field in REQUIRED_METADATA:

                if field not in metadata:

                    errors.append(
                        f"{file_path.name} "
                        f"child {index}: "
                        f"missing {field}"
                    )

            if not content.strip():

                errors.append(
                    f"{file_path.name} "
                    f"child {index}: "
                    f"empty content"
                )

            if metadata.get(
                "chunk_type"
            ) != "child":

                errors.append(
                    f"{file_path.name} "
                    f"child {index}: "
                    f"wrong chunk_type"
                )

    print("\n" + "=" * 80)

    print(
        f"Total children: "
        f"{total_children}"
    )

    print(
        f"Errors: "
        f"{len(errors)}"
    )

    print("=" * 80)

    if errors:

        for error in errors[:100]:

            print(
                "ERROR:",
                error
            )

        raise RuntimeError(
            "Chunk validation failed."
        )

    print(
        "✓ Chunk validation passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())