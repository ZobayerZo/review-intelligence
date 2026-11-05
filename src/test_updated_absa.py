import sys
from pathlib import Path

# Add project root to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.absa_sentiment import analyze_review


review = (
    "The camera quality is excellent, and the display "
    "is very bright. However, the battery life is "
    "disappointing, and the phone takes too long to charge."
)

results = analyze_review(review, "Electronics")

print("\nABSA PIPELINE TEST")
print("=" * 60)

for item in results:
    print(
        f"Aspect: {item['aspect']}\n"
        f"Sentiment: {item['sentiment']}\n"
        f"Confidence: {item['confidence']:.4f}\n"
        f"Sentence: {item['sentence']}\n"
        + "-" * 60
    )