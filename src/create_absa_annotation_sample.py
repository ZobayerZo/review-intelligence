
import os
import pandas as pd


# Dataset location
DATA_PATH = (
    "data/processed/representative/"
    "all_categories_sample.parquet"
)

# Output location
OUTPUT_PATH = "reports/absa_annotation_sample.csv"

# Number of reviews per category
SAMPLES_PER_CATEGORY = 15

# Reproducible random sampling
RANDOM_STATE = 42


def main():

    df = pd.read_parquet(DATA_PATH)

    sampled_reviews = []

    for category in sorted(df["category"].unique()):

        category_df = df[
            (df["category"] == category)
            & (df["text"].notna())
            & (df["text"].str.len() >= 40)
        ]

        sample_size = min(
            SAMPLES_PER_CATEGORY,
            len(category_df)
        )

        sample = category_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        sampled_reviews.append(sample)

    annotation_df = pd.concat(
        sampled_reviews,
        ignore_index=True
    )

    annotation_df = annotation_df[
        ["category", "rating", "text"]
    ].copy()

    # Assign unique annotation IDs
    annotation_df.insert(
        0,
        "review_id",
        [
            f"ABSA_{i:03d}"
            for i in range(1, len(annotation_df) + 1)
        ]
    )

    # Empty columns for manual annotation
    annotation_df["gold_aspects"] = ""
    annotation_df["annotation_notes"] = ""

    # Create output directory
    os.makedirs("reports", exist_ok=True)

    annotation_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Annotation dataset created successfully.")
    print(f"Total reviews: {len(annotation_df)}")

    print("\nCategory distribution:")
    print(annotation_df["category"].value_counts())

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
