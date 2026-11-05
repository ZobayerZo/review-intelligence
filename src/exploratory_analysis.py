
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot"
    / "all_categories_pilot.parquet"
)

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_FILE)

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nVerified purchase distribution:")
print(df["verified_purchase"].value_counts())

print("\nReview date range:")
print(df["review_date"].min(), "to", df["review_date"].max())

# Review length
df["review_length"] = df["combined_text"].str.split().str.len()

print("\nReview length statistics:")
print(df["review_length"].describe())

# Monthly review counts
monthly_reviews = (
    df.groupby("review_month")
    .size()
    .sort_index()
)

print("\nMonthly review counts:")
print(monthly_reviews)

# 1. Rating distribution
plt.figure(figsize=(8, 5))
df["rating"].value_counts().sort_index().plot(kind="bar")
plt.title("Review Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "rating_distribution.png", dpi=300)
plt.close()

# 2. Sentiment distribution
plt.figure(figsize=(8, 5))
df["sentiment"].value_counts().reindex(
    ["negative", "neutral", "positive"]
).plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "sentiment_distribution.png", dpi=300)
plt.close()

# 3. Sentiment by category
sentiment_category = pd.crosstab(
    df["category"], df["sentiment"], normalize="index"
) * 100

sentiment_category.plot(kind="bar", figsize=(9, 5))
plt.title("Sentiment Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Percentage of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "sentiment_by_category.png", dpi=300)
plt.close()

# 4. Review length distribution
plt.figure(figsize=(8, 5))
df["review_length"].clip(upper=500).hist(bins=50)
plt.title("Review Length Distribution")
plt.xlabel("Number of Words (capped at 500)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "review_length_distribution.png", dpi=300)
plt.close()

# 5. Monthly review trend
plt.figure(figsize=(12, 5))
monthly_reviews.plot(kind="line", marker="o")
plt.title("Monthly Review Volume")
plt.xlabel("Month")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "monthly_review_trend.png", dpi=300)
plt.close()

# 6. Helpful vote distribution
plt.figure(figsize=(8, 5))
df["helpful_vote"].clip(upper=50).hist(bins=25)
plt.title("Helpful Vote Distribution")
plt.xlabel("Helpful Votes (capped at 50)")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "helpful_vote_distribution.png", dpi=300)
plt.close()

# Save numerical summary
summary = {
    "total_reviews": int(len(df)),
    "categories": df["category"].value_counts().to_dict(),
    "ratings": {
        str(k): int(v)
        for k, v in df["rating"].value_counts().sort_index().items()
    },
    "sentiments": {
        str(k): int(v)
        for k, v in df["sentiment"].value_counts().items()
    },
    "average_review_length": float(df["review_length"].mean()),
    "median_review_length": float(df["review_length"].median()),
    "average_helpful_votes": float(df["helpful_vote"].mean()),
    "verified_purchase_percentage": float(
        df["verified_purchase"].mean() * 100
    ),
    "date_range": {
        "start": str(df["review_date"].min()),
        "end": str(df["review_date"].max()),
    },
}

summary_file = (
    PROJECT_ROOT / "reports" / "eda_summary.json"
)

with open(summary_file, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\n" + "=" * 55)
print("EDA COMPLETED")
print("=" * 55)
print(f"Charts saved in: {FIGURE_DIR}")
print(f"Summary saved in: {summary_file}")
