
import pandas as pd
from aspect_extraction import extract_aspects


DATA_PATH = "data/processed/representative/all_categories_sample.parquet"

df = pd.read_parquet(DATA_PATH)

sample = (
    df[df["text"].str.len() > 50]
    .sample(n=15, random_state=42)
)

print("Testing improved aspect extraction\n")

for index, (_, row) in enumerate(sample.iterrows(), start=1):

    review = row["text"]
    category = row["category"]

    aspects = extract_aspects(
        review,
        category=category
    )

    print("=" * 80)
    print(f"REVIEW {index}")
    print(f"Category: {category}")
    print(f"Rating: {row['rating']}")
    print(f"Review: {review[:500]}")

    print("\nExtracted aspects:")

    if aspects:
        for aspect in aspects:
            label = (
                "Known aspect"
                if aspect["known_aspect"]
                else "Discovered aspect"
            )

            print(f"- {aspect['aspect']} [{label}]")
    else:
        print("- No aspects extracted")

    print()
