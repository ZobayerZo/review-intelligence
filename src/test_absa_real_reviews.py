
import pandas as pd
from pathlib import Path

from absa_sentiment import analyze_review


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "representative"
    / "all_categories_sample.parquet"
)

print("Loading Amazon review dataset...")

df = pd.read_parquet(DATA_PATH)

print("Total reviews:", len(df))
print("Available columns:", df.columns.tolist())


# --------------------------------------------------
# 2. SELECT REAL REVIEWS
# --------------------------------------------------

categories = [
    "Electronics",
    "Beauty",
    "Home_Kitchen"
]

sampled_reviews = []

for category in categories:

    category_df = df[
        (df["category"] == category)
        & (df["text"].notna())
        & (df["text"].str.len() > 50)
    ]

    sample = category_df.sample(
        n=2,
        random_state=42
    )

    sampled_reviews.append(sample)

test_df = pd.concat(sampled_reviews)

print("\nSelected reviews:", len(test_df))


# --------------------------------------------------
# 3. RUN ABSA
# --------------------------------------------------

for index, (_, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    print("\n" + "=" * 70)
    print(f"REVIEW {index}")
    print("=" * 70)

    print("Category:", row["category"])
    print("Rating:", row["rating"])
    print("\nReview:")
    print(row["text"][:1000])

    results = analyze_review(
        text=row["text"],
        category=row["category"]
    )

    print("\nASPECT-LEVEL SENTIMENT RESULTS")
    print("-" * 50)

    if not results:
        print("No aspects detected.")
        continue

    for result in results:

        print(
            f"{result['aspect']} | "
            f"{result['sentiment']} | "
            f"{result['confidence']:.2%}"
        )

print("\nReal review testing completed.")
