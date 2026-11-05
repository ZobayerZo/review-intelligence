
import pandas as pd

from pathlib import Path
from absa_sentiment import analyze_review


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
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
    / "absa"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = OUTPUT_DIR / "absa_pilot_results.parquet"
SUMMARY_PATH = OUTPUT_DIR / "absa_pilot_summary.csv"


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

CATEGORIES = [
    "Electronics",
    "Beauty",
    "Home_Kitchen"
]

SAMPLES_PER_CATEGORY = 100
RANDOM_STATE = 42


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

print("Loading Amazon review dataset...")

df = pd.read_parquet(INPUT_PATH)

print("Total available reviews:", len(df))


# --------------------------------------------------
# 4. SELECT PILOT REVIEWS
# --------------------------------------------------

samples = []

for category in CATEGORIES:

    category_df = df[
        (df["category"] == category)
        & (df["text"].notna())
        & (df["text"].str.len() > 50)
    ]

    sample = category_df.sample(
        n=min(SAMPLES_PER_CATEGORY, len(category_df)),
        random_state=RANDOM_STATE
    ).copy()

    samples.append(sample)

test_df = pd.concat(samples)

print("\nSelected reviews:", len(test_df))

print("\nCategory distribution:")
print(test_df["category"].value_counts())


# --------------------------------------------------
# 5. RUN ABSA
# --------------------------------------------------

results = []

summary = []

total_reviews = len(test_df)

print("\nStarting ABSA processing...")


for position, (source_index, row) in enumerate(
    test_df.iterrows(),
    start=1
):

    review_text = row["text"]
    category = row["category"]

    review_id = f"ABSA_{source_index}"

    try:

        predictions = analyze_review(
            text=review_text,
            category=category
        )

        # Save review-level processing information
        summary.append({
            "review_id": review_id,
            "category": category,
            "rating": row["rating"],
            "aspect_count": len(predictions),
            "has_aspects": len(predictions) > 0
        })

        # Save aspect-level predictions
        for prediction in predictions:

            results.append({
                "review_id": review_id,
                "category": category,
                "rating": row["rating"],
                "title": row.get("title", ""),
                "review_text": review_text,
                "asin": row.get("asin", None),
                "parent_asin": row.get("parent_asin", None),
                "aspect": prediction["aspect"],
                "normalized_aspect": prediction[
                    "normalized_aspect"
                ],
                "aspect_sentiment": prediction["sentiment"],
                "confidence": prediction["confidence"],
                "source_sentence": prediction["sentence"],
                "extraction_source": prediction["source"]
            })

    except Exception as error:

        print(
            f"\nError processing {review_id}: {error}"
        )

        summary.append({
            "review_id": review_id,
            "category": category,
            "rating": row["rating"],
            "aspect_count": 0,
            "has_aspects": False
        })

    if position % 10 == 0 or position == total_reviews:

        print(
            f"Processed {position}/{total_reviews} reviews | "
            f"Aspect predictions: {len(results)}"
        )


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
summary_df = pd.DataFrame(summary)

if not results_df.empty:

    results_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("\nABSA results saved:", OUTPUT_PATH)

else:

    print("\nNo aspect predictions were generated.")


summary_df.to_csv(
    SUMMARY_PATH,
    index=False
)

print("Review summary saved:", SUMMARY_PATH)


# --------------------------------------------------
# 7. DISPLAY FINAL STATISTICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("ABSA PILOT SUMMARY")
print("=" * 60)

print("Total reviews processed:", len(summary_df))
print("Reviews with aspects:", summary_df["has_aspects"].sum())
print("Reviews without aspects:", (~summary_df["has_aspects"]).sum())
print("Total aspect predictions:", len(results_df))

if not results_df.empty:

    print("\nAspect sentiment distribution:")
    print(results_df["aspect_sentiment"].value_counts())

    print("\nCategory-wise aspect predictions:")
    print(
        results_df.groupby("category").size()
    )

print("\nPilot processing completed.")
