import json
from pathlib import Path

from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parents[1]

from src.metadata import DocumentMetadata


def main() -> int:
    metadata_dir = ROOT_DIR / "data" / "metadata"
    metadata_files = sorted(metadata_dir.glob("*.json"))

    if not metadata_files:
        print(f"No metadata files found in {metadata_dir}", file=sys.stderr)
        return 1

    errors = 0
    for metadata_file in metadata_files:
        try:
            with metadata_file.open(encoding="utf-8") as file:
                DocumentMetadata.model_validate(json.load(file))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            errors += 1
            print(f"INVALID {metadata_file.name}: {error}", file=sys.stderr)

    if errors:
        print(f"{errors} of {len(metadata_files)} metadata files failed validation.")
        return 1

    print(f"Validated {len(metadata_files)} metadata files successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())