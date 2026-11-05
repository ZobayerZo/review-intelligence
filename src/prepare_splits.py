
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "representative"
    / "all_categories_sample.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

RANDOM_SEED = 42


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading representative dataset...")
    df = pd.read_parquet(INPUT_FILE)

    print(f"Dataset size: {len(df):,}")

    # Combine category and sentiment for stratification.
    df["stratify_key"] = (
        df["category"].astype(str)
        + "_"
        + df["sentiment"].astype(str)
    )

    print("\nOriginal sentiment distribution:")
    print(pd.crosstab(df["category"], df["sentiment"]))

    # First split: 70% training, 30% remaining.
    train_df, remaining_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["stratify_key"],
    )

    # Second split: divide remaining data equally.
    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=remaining_df["stratify_key"],
    )

    # Remove helper column.
    for dataset in [train_df, val_df, test_df]:
        dataset.drop(columns=["stratify_key"], inplace=True)

    # Save datasets.
    datasets = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for name, dataset in datasets.items():
        output_file = OUTPUT_DIR / f"{name}.parquet"

        dataset.to_parquet(
            output_file,
            index=False,
            compression="snappy",
        )

        print(f"\nSaved {name}: {len(dataset):,} records")

    # Build split summary.
    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratio": "70:15:15",
        "total_records": len(df),
        "splits": {},
    }

    for name, dataset in datasets.items():
        category_distribution = (
            dataset["category"].value_counts().to_dict()
        )

        sentiment_distribution = (
            dataset["sentiment"]
            .value_counts()
            .to_dict()
        )

        summary["splits"][name] = {
            "records": len(dataset),
            "categories": category_distribution,
            "sentiments": {
                str(k): int(v)
                for k, v in sentiment_distribution.items()
            },
        }

        print(f"\n{name.upper()} DISTRIBUTION")
        print("Categories:")
        print(dataset["category"].value_counts())

        print("\nSentiments:")
        print(dataset["sentiment"].value_counts())

        print("\nCategory vs sentiment:")
        print(pd.crosstab(
            dataset["category"],
            dataset["sentiment"],
        ))

    summary_file = OUTPUT_DIR / "split_summary.json"

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 55)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 55)
    print(f"Training:   {len(train_df):,}")
    print(f"Validation: {len(val_df):,}")
    print(f"Test:       {len(test_df):,}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
