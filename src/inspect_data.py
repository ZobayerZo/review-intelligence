
import gzip
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "Electronics Reviews": RAW_DIR / "reviews" / "Electronics.jsonl.gz",
    "Beauty Reviews": RAW_DIR / "reviews" / "Beauty_and_Personal_Care.jsonl.gz",
    "Home Reviews": RAW_DIR / "reviews" / "Home_and_Kitchen.jsonl.gz",
    "Electronics Metadata": RAW_DIR / "metadata" / "meta_Electronics.jsonl.gz",
    "Beauty Metadata": RAW_DIR / "metadata" / "meta_Beauty_and_Personal_Care.jsonl.gz",
    "Home Metadata": RAW_DIR / "metadata" / "meta_Home_and_Kitchen.jsonl.gz",
}


def inspect_file(name, filepath):
    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return

    size_gb = filepath.stat().st_size / (1024 ** 3)
    print(f"Compressed size: {size_gb:.2f} GB")

    try:
        with gzip.open(filepath, "rt", encoding="utf-8") as file:
            for i, line in enumerate(file):
                record = json.loads(line)

                print(f"\nRecord {i + 1}")
                print("Available columns:", list(record.keys()))

                if i == 0:
                    print("\nSample values:")

                    for key, value in record.items():
                        preview = str(value)[:150]
                        print(f"  {key}: {preview}")

                if i == 2:
                    break

    except Exception as error:
        print(f"ERROR while reading file: {error}")


def main():
    print("AMAZON REVIEWS 2023 - DATASET INSPECTION")

    for name, filepath in FILES.items():
        inspect_file(name, filepath)

    print("\n" + "=" * 65)
    print("INITIAL DATASET INSPECTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
