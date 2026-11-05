
import gzip
import json
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "reviews"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "representative"

SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

DATASETS = {
    "Electronics": "Electronics.jsonl.gz",
    "Beauty": "Beauty_and_Personal_Care.jsonl.gz",
    "Home_Kitchen": "Home_and_Kitchen.jsonl.gz",
}


def is_valid_review(record):
    """Check whether a review has the fields needed for analysis."""
    try:
        rating = float(record.get("rating"))
        timestamp = float(record.get("timestamp"))
    except (TypeError, ValueError):
        return False

    title = str(record.get("title") or "").strip()
    text = str(record.get("text") or "").strip()

    return (
        1 <= rating <= 5
        and timestamp > 0
        and (title or text)
    )


def reservoir_sample(filepath, sample_size, seed):
    """
    Uniformly sample reviews from the entire compressed file
    without loading the complete dataset into memory.
    """
    rng = random.Random(seed)

    reservoir = []
    eligible_count = 0
    total_lines = 0
    malformed_lines = 0

    print(f"\nScanning: {filepath.name}")
    print("Reservoir sampling started...")

    with gzip.open(filepath, "rt", encoding="utf-8") as file:
        for line in file:
            total_lines += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not is_valid_review(record):
                continue

            eligible_count += 1

            # Retain only the fields needed for this project.
            selected_record = {
                "rating": record.get("rating"),
                "title": record.get("title"),
                "text": record.get("text"),
                "asin": record.get("asin"),
                "parent_asin": record.get("parent_asin"),
                "timestamp": record.get("timestamp"),
                "helpful_vote": record.get("helpful_vote"),
                "verified_purchase": record.get("verified_purchase"),
            }

            if len(reservoir) < sample_size:
                reservoir.append(selected_record)
            else:
                position = rng.randrange(eligible_count)

                if position < sample_size:
                    reservoir[position] = selected_record

            if total_lines % 1_000_000 == 0:
                print(
                    f"Lines scanned: {total_lines:,} | "
                    f"Eligible reviews: {eligible_count:,} | "
                    f"Sample retained: {len(reservoir):,}"
                )

    print("\nScanning completed.")
    print(f"Total lines scanned: {total_lines:,}")
    print(f"Eligible reviews: {eligible_count:,}")
    print(f"Malformed lines: {malformed_lines:,}")
    print(f"Reviews sampled: {len(reservoir):,}")

    return reservoir, {
        "total_lines_scanned": total_lines,
        "eligible_reviews": eligible_count,
        "malformed_lines": malformed_lines,
        "sampled_reviews": len(reservoir),
    }


def clean_reviews(records, category):
    df = pd.DataFrame(records)

    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    df["helpful_vote"] = pd.to_numeric(
        df["helpful_vote"], errors="coerce"
    ).fillna(0).clip(lower=0).astype("int64")

    for column in ["title", "text"]:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["verified_purchase"] = (
        df["verified_purchase"]
        .fillna(False)
        .astype(bool)
    )

    df = df.dropna(
        subset=["rating", "timestamp", "parent_asin"]
    ).copy()

    df = df[df["rating"].between(1, 5)].copy()

    df["review_date"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["review_date"]).copy()

    df = df[
        (df["title"] != "") | (df["text"] != "")
    ].copy()

    df["review_month"] = (
        df["review_date"].dt.strftime("%Y-%m")
    )

    df["combined_text"] = (
        df["title"] + ". " + df["text"]
    ).str.strip(" .")

    df["sentiment"] = pd.cut(
        df["rating"],
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        include_lowest=True,
    )

    df["category"] = category

    before_deduplication = len(df)

    df = df.drop_duplicates(
        subset=["parent_asin", "timestamp", "text"]
    ).copy()

    duplicates_removed = before_deduplication - len(df)

    df["sampling_method"] = (
        f"reservoir_random_seed_{RANDOM_SEED}"
    )

    df = df.reset_index(drop=True)

    print(f"\nCleaning results: {category}")
    print(f"Records after cleaning: {len(df):,}")
    print(f"Duplicates removed: {duplicates_removed:,}")

    return df, duplicates_removed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_datasets = []
    summary = {}

    for index, (category, filename) in enumerate(DATASETS.items()):
        filepath = RAW_DIR / filename

        if not filepath.exists():
            print(f"\nFile not found: {filepath}")
            continue

        records, scan_stats = reservoir_sample(
            filepath,
            SAMPLE_SIZE,
            RANDOM_SEED + index,
        )

        if not records:
            print(f"No valid reviews found for {category}")
            continue

        df, duplicates = clean_reviews(records, category)

        output_file = (
            OUTPUT_DIR / f"{category.lower()}_sample.parquet"
        )

        df.to_parquet(
            output_file,
            index=False,
            compression="snappy",
        )

        print(f"Saved: {output_file}")

        processed_datasets.append(df)

        summary[category] = {
            **scan_stats,
            "records_saved": len(df),
            "duplicates_removed": duplicates,
            "output_file": str(output_file),
        }

        del records
        del df

    if processed_datasets:
        combined_df = pd.concat(
            processed_datasets,
            ignore_index=True,
        )

        combined_file = (
            OUTPUT_DIR / "all_categories_sample.parquet"
        )

        combined_df.to_parquet(
            combined_file,
            index=False,
            compression="snappy",
        )

        summary["TOTAL"] = {
            "records_saved": len(combined_df),
            "categories": len(processed_datasets),
            "output_file": str(combined_file),
        }

        print("\nCombined dataset saved successfully.")
        del combined_df

    summary_file = OUTPUT_DIR / "sampling_summary.json"

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("REPRESENTATIVE SAMPLING COMPLETED")
    print("=" * 55)
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()
