
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. LOAD PILOT DATA
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "absa"
)

RESULTS_PATH = DATA_DIR / "absa_pilot_results.parquet"
SUMMARY_PATH = DATA_DIR / "absa_pilot_summary.csv"

df = pd.read_parquet(RESULTS_PATH)
reviews = pd.read_csv(SUMMARY_PATH)


# --------------------------------------------------
# 2. BASIC STATISTICS
# --------------------------------------------------

print("\n" + "=" * 65)
print("ABSA PILOT ANALYTICS REPORT")
print("=" * 65)

print("\nTotal reviews:", len(reviews))
print("Reviews with aspects:", reviews["has_aspects"].sum())
print("Reviews without aspects:", (~reviews["has_aspects"]).sum())
print("Total aspect predictions:", len(df))

print(
    "Average aspects per processed review:",
    round(len(df) / len(reviews), 2)
)


# --------------------------------------------------
# 3. CATEGORY-WISE PERFORMANCE
# --------------------------------------------------

print("\nCATEGORY-WISE REVIEW COVERAGE")
print("-" * 65)

category_summary = reviews.groupby("category").agg(
    total_reviews=("review_id", "count"),
    reviews_with_aspects=("has_aspects", "sum"),
    total_aspects=("aspect_count", "sum")
).reset_index()

category_summary["detection_rate"] = (
    category_summary["reviews_with_aspects"]
    / category_summary["total_reviews"]
    * 100
).round(2)

print(category_summary.to_string(index=False))


# --------------------------------------------------
# 4. SENTIMENT DISTRIBUTION
# --------------------------------------------------

print("\nASPECT SENTIMENT DISTRIBUTION")
print("-" * 65)

sentiment_summary = (
    df["aspect_sentiment"]
    .value_counts()
    .to_frame("count")
)

sentiment_summary["percentage"] = (
    sentiment_summary["count"]
    / len(df)
    * 100
).round(2)

print(sentiment_summary)


# --------------------------------------------------
# 5. SENTIMENT BY CATEGORY
# --------------------------------------------------

print("\nSENTIMENT BY CATEGORY")
print("-" * 65)

sentiment_by_category = pd.crosstab(
    df["category"],
    df["aspect_sentiment"]
)

print(sentiment_by_category)


# --------------------------------------------------
# 6. MOST FREQUENT ASPECTS
# --------------------------------------------------

print("\nTOP 20 MOST FREQUENT ASPECTS")
print("-" * 65)

top_aspects = (
    df.groupby(["category", "normalized_aspect"])
    .agg(
        mentions=("review_id", "count"),
        average_confidence=("confidence", "mean")
    )
    .reset_index()
    .sort_values(
        ["mentions", "category"],
        ascending=[False, True]
    )
)

top_aspects["average_confidence"] = (
    top_aspects["average_confidence"].round(4)
)

print(top_aspects.head(20).to_string(index=False))


# --------------------------------------------------
# 7. CONFIDENCE ANALYSIS
# --------------------------------------------------

print("\nCONFIDENCE BY SENTIMENT")
print("-" * 65)

confidence_summary = df.groupby(
    "aspect_sentiment"
)["confidence"].agg(
    ["count", "mean", "min", "max"]
).round(4)

print(confidence_summary)


# --------------------------------------------------
# 8. LOW-CONFIDENCE PREDICTIONS
# --------------------------------------------------

print("\n20 LOWEST-CONFIDENCE PREDICTIONS")
print("-" * 65)

low_confidence = df.sort_values(
    "confidence"
).head(20)

for _, row in low_confidence.iterrows():

    print("\nCategory:", row["category"])
    print("Aspect:", row["aspect"])
    print("Sentiment:", row["aspect_sentiment"])
    print("Confidence:", row["confidence"])
    print("Sentence:", row["source_sentence"])


# --------------------------------------------------
# 9. SAVE ANALYTICS
# --------------------------------------------------

category_summary.to_csv(
    DATA_DIR / "absa_category_summary.csv",
    index=False
)

top_aspects.to_csv(
    DATA_DIR / "absa_top_aspects.csv",
    index=False
)

print("\nAnalytics files saved successfully.")
print("Output directory:", DATA_DIR)
