
import pandas as pd
import re
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT / "data" / "processed" / "representative"
    / "all_categories_sample.parquet"
)

ABSA_DIR = ROOT / "data" / "processed" / "absa"

RESULTS_PATH = ABSA_DIR / "absa_pilot_results.parquet"
SUMMARY_PATH = ABSA_DIR / "absa_pilot_summary.csv"

OUTPUT_RESULTS = (
    ABSA_DIR / "absa_pilot_results_enriched.parquet"
)

OUTPUT_SUMMARY = (
    ABSA_DIR / "absa_pilot_summary_enriched.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading datasets...")

source = pd.read_parquet(SOURCE_PATH)
results = pd.read_parquet(RESULTS_PATH)
summary = pd.read_csv(SUMMARY_PATH)

print("Original reviews:", len(source))
print("Aspect predictions:", len(results))
print("Review summaries:", len(summary))


# ============================================================
# 3. PREPARE SOURCE METADATA
# ============================================================

metadata_columns = [
    "asin",
    "parent_asin",
    "review_date",
    "review_month",
    "helpful_vote",
    "verified_purchase",
    "timestamp",
    "sampling_method"
]

missing = [
    column for column in metadata_columns
    if column not in source.columns
]

if missing:
    raise ValueError(f"Missing source fields: {missing}")

metadata = source[metadata_columns].copy()
metadata["source_index"] = source.index

if metadata["source_index"].duplicated().any():
    raise ValueError("Source index is not unique.")


# ============================================================
# 4. RECOVER ORIGINAL SOURCE INDEX
# ============================================================

def get_source_index(review_id):

    match = re.fullmatch(r"ABSA_(\d+)", str(review_id))

    if not match:
        raise ValueError(f"Invalid review ID: {review_id}")

    return int(match.group(1))


results["source_index"] = (
    results["review_id"].apply(get_source_index)
)

summary["source_index"] = (
    summary["review_id"].apply(get_source_index)
)


# ============================================================
# 5. CHECK SOURCE MAPPING
# ============================================================

valid_indices = set(metadata["source_index"])

for name, frame in [
    ("Aspect results", results),
    ("Review summary", summary)
]:

    missing_indices = (
        set(frame["source_index"]) - valid_indices
    )

    if missing_indices:
        raise ValueError(
            f"{name}: {len(missing_indices)} unmatched source IDs."
        )

print("Review ID mapping: PASSED")


# ============================================================
# 6. ENRICH BOTH DATASETS
# ============================================================

enriched_results = results.merge(
    metadata,
    on="source_index",
    how="left",
    validate="many_to_one"
)

enriched_summary = summary.merge(
    metadata,
    on="source_index",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 7. VALIDATE CATEGORY AND RATING
# ============================================================

source_check = source[
    ["category", "rating"]
].copy()

source_check["source_index"] = source.index

validation = enriched_summary.merge(
    source_check,
    on="source_index",
    how="left",
    suffixes=("_absa", "_source")
)

assert (
    validation["category_absa"]
    == validation["category_source"]
).all(), "Category mismatch detected."

assert (
    validation["rating_absa"]
    == validation["rating_source"]
).all(), "Rating mismatch detected."

print("Category validation: PASSED")
print("Rating validation: PASSED")


# ============================================================
# 8. REMOVE INTERNAL MAPPING COLUMN
# ============================================================

enriched_results = enriched_results.drop(
    columns=["source_index"]
)

enriched_summary = enriched_summary.drop(
    columns=["source_index"]
)


# ============================================================
# 9. SAVE
# ============================================================

enriched_results.to_parquet(
    OUTPUT_RESULTS,
    index=False
)

enriched_summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print("\nENRICHMENT COMPLETED")
print("=" * 55)

print("Aspect records:", len(enriched_results))
print("Review records:", len(enriched_summary))

print("\nReview summary now includes:")
print(enriched_summary.columns.tolist())

print("\nUnique products in all reviews:",
      enriched_summary["parent_asin"].nunique())

print("\nTop products by review count:")

print(
    enriched_summary.groupby("parent_asin")
    .agg(
        reviews=("review_id", "nunique"),
        category=("category", "first")
    )
    .sort_values("reviews", ascending=False)
    .head(10)
    .to_string()
)

print("\nUpdated files saved successfully.")
