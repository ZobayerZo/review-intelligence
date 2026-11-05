
import gzip
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "reviews"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "pilot"

SAMPLE_SIZE = 10_000

DATASETS = {
    "Electronics": "Electronics.jsonl.gz",
    "Beauty": "Beauty_and_Personal_Care.jsonl.gz",
    "Home_Kitchen": "Home_and_Kitchen.jsonl.gz",
}


# --------------------------------------------------
# 2. READ COMPRESSED JSONL FILE
# --------------------------------------------------

def read_reviews(filepath, sample_size):

    records = []
    malformed_lines = 0

    print(f"\nReading: {filepath.name}")

    with gzip.open(filepath, "rt", encoding="utf-8") as file:

        for line in file:

            try:
                record = json.loads(line)

                records.append(record)

            except json.JSONDecodeError:
                malformed_lines += 1

            if len(records) >= sample_size:
                break

    print(f"Records collected: {len(records):,}")
    print(f"Malformed lines skipped: {malformed_lines:,}")

    return pd.DataFrame(records), malformed_lines


# --------------------------------------------------
# 3. CLEAN AND TRANSFORM DATA
# --------------------------------------------------

def clean_reviews(df, category):

    initial_count = len(df)

    # Retain relevant columns
    columns = [
        "rating",
        "title",
        "text",
        "asin",
        "parent_asin",
        "timestamp",
        "helpful_vote",
        "verified_purchase",
    ]

    df = df[columns].copy()

    # Clean text fields
    for column in ["title", "text"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    # Remove records without review text
    df = df[
        (df["title"] != "") | (df["text"] != "")
    ].copy()

    # Convert rating into numeric format
    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    # Retain valid ratings
    df = df[
        df["rating"].between(1, 5)
    ].copy()

    # Convert timestamp into numeric format
    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(subset=["timestamp"]).copy()

    # Convert Unix milliseconds into datetime
    df["review_date"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["review_date"]).copy()

    # Generate monthly grouping
    df["review_month"] = (
        df["review_date"].dt.strftime("%Y-%m")
    )

    # Handle helpful votes
    df["helpful_vote"] = pd.to_numeric(
        df["helpful_vote"], errors="coerce"
    ).fillna(0).clip(lower=0).astype("int64")

    # Handle verified purchases
    df["verified_purchase"] = (
        df["verified_purchase"].fillna(False).astype(bool)
    )

    # Combine review title and body
    df["combined_text"] = (
        df["title"] + ". " + df["text"]
    ).str.strip(" .")

    # Generate three-class sentiment labels
    df["sentiment"] = pd.cut(
        df["rating"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        include_lowest=True
    )

    # Add category
    df["category"] = category

    # Remove exact review duplicates
    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=["parent_asin", "timestamp", "text"]
    ).copy()

    duplicates_removed = (
        before_deduplication - len(df)
    )

    # Mark the sampling method
    df["sampling_method"] = "first_10000_smoke_test"

    # Reset index
    df = df.reset_index(drop=True)

    print(f"\nCleaning results: {category}")
    print(f"Initial records: {initial_count:,}")
    print(f"Final records: {len(df):,}")
    print(f"Duplicates removed: {duplicates_removed:,}")

    return df, duplicates_removed


# --------------------------------------------------
# 4. PROCESS EACH CATEGORY
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_datasets = []
    summary = {}

    for category, filename in DATASETS.items():

        filepath = RAW_DIR / filename

        if not filepath.exists():
            print(f"\nFile not found: {filepath}")
            continue

        raw_df, malformed = read_reviews(
            filepath, SAMPLE_SIZE
        )

        if raw_df.empty:
            print(f"No records available for {category}")
            continue

        clean_df, duplicates = clean_reviews(
            raw_df, category
        )

        # Save category-specific Parquet
        output_file = (
            OUTPUT_DIR / f"{category.lower()}_pilot.parquet"
        )

        clean_df.to_parquet(
            output_file,
            index=False,
            compression="snappy"
        )

        processed_datasets.append(clean_df)

        summary[category] = {
            "records_saved": len(clean_df),
            "duplicates_removed": duplicates,
            "malformed_lines": malformed,
            "output_file": str(output_file),
        }

        print(f"Saved: {output_file}")

        del raw_df
        del clean_df

    # Save combined dataset
    if processed_datasets:

        combined_df = pd.concat(
            processed_datasets,
            ignore_index=True
        )

        combined_file = (
            OUTPUT_DIR / "all_categories_pilot.parquet"
        )

        combined_df.to_parquet(
            combined_file,
            index=False,
            compression="snappy"
        )

        summary["TOTAL"] = {
            "records_saved": len(combined_df),
            "categories": len(processed_datasets),
            "output_file": str(combined_file),
        }

        print("\nCombined dataset saved successfully.")

        del combined_df

    # Save processing summary
    summary_file = OUTPUT_DIR / "ingestion_summary.json"

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("DATA INGESTION COMPLETED")
    print("=" * 55)

    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
