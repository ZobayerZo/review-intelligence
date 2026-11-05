
from transformers import pipeline

try:
    from .aspect_extraction import (
        extract_aspects,
        clean_review,
        nlp
    )
except ImportError:
    from aspect_extraction import (
        extract_aspects,
        clean_review,
        nlp
    )


# --------------------------------------------------
# 1. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"

_classifier = None


# --------------------------------------------------
# 2. LOAD PRETRAINED ABSA MODEL
# --------------------------------------------------

def load_absa_model():

    global _classifier

    if _classifier is None:

        print("Loading pretrained ABSA model...")

        _classifier = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            device=-1
        )

        print("ABSA model loaded successfully.")

    return _classifier


# --------------------------------------------------
# 3. FIND SENTENCE CONTAINING AN ASPECT
# --------------------------------------------------

def get_aspect_sentence(
    text: str,
    aspect_start: int,
    aspect_end: int
) -> str:

    doc = nlp(text)

    for sentence in doc.sents:

        if (
            sentence.start_char < aspect_end
            and sentence.end_char > aspect_start
        ):
            return sentence.text.strip()

    return text


# --------------------------------------------------
# 4. PREDICT ASPECT-LEVEL SENTIMENT
# --------------------------------------------------

def predict_aspect_sentiment(
    sentence: str,
    aspect: str,
    classifier=None
) -> dict:

    if classifier is None:
        classifier = load_absa_model()

    prediction = classifier(
        sentence,
        text_pair=aspect
    )[0]

    return {
        "sentiment": prediction["label"].capitalize(),
        "confidence": round(
            float(prediction["score"]),
            4
        )
    }


# --------------------------------------------------
# 5. COMPLETE ABSA PIPELINE
# --------------------------------------------------

def analyze_review(
    text: str,
    category: str
) -> list[dict]:

    cleaned_text = clean_review(text)

    if not cleaned_text:
        return []

    # Extract product aspects
    extracted_aspects = extract_aspects(
        cleaned_text,
        category=category
    )

    if not extracted_aspects:
        return []

    # Load model only when needed
    classifier = load_absa_model()

    results = []

    for aspect_info in extracted_aspects:

        aspect = aspect_info["aspect"]

        # Find the sentence containing the aspect
        sentence = get_aspect_sentence(
            cleaned_text,
            aspect_info["start"],
            aspect_info["end"]
        )

        # Predict sentiment toward this aspect
        prediction = predict_aspect_sentiment(
            sentence,
            aspect,
            classifier
        )

        results.append({
            "aspect": aspect,
            "normalized_aspect": aspect_info[
                "normalized_aspect"
            ],
            "sentiment": prediction["sentiment"],
            "confidence": prediction["confidence"],
            "sentence": sentence,
            "source": aspect_info["source"]
        })

    return results


# --------------------------------------------------
# 6. TEST THE COMPLETE PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    sample_review = (
        "The camera quality is excellent, but the "
        "battery life is disappointing. "
        "The display is very sharp."
    )

    print("\nORIGINAL REVIEW")
    print("-" * 60)
    print(sample_review)

    print("\nABSA RESULTS")
    print("-" * 60)

    results = analyze_review(
        sample_review,
        category="Electronics"
    )

    for result in results:

        print(f"\nAspect: {result['aspect']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Source: {result['source']}")
        print(f"Sentence: {result['sentence']}")

    print("\nTotal aspects analyzed:", len(results))
