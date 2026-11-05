
from transformers import pipeline

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

print("Loading pretrained ABSA model...")
print("The first run may take some time to download the model.")

classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    device=-1  # CPU inference
)

reviews = [
    {
        "text": "The battery life is terrible, but the camera quality is excellent.",
        "aspect": "battery life"
    },
    {
        "text": "The battery life is terrible, but the camera quality is excellent.",
        "aspect": "camera quality"
    },
    {
        "text": "The screen is bright and beautiful, but the speakers sound awful.",
        "aspect": "speakers"
    }
]

print("\nABSA TEST RESULTS")
print("=" * 60)

for item in reviews:
    result = classifier(
        item["text"],
        text_pair=item["aspect"]
    )

    print(f"\nReview: {item['text']}")
    print(f"Aspect: {item['aspect']}")
    print(f"Prediction: {result}")
