
import re
import spacy

from bs4 import BeautifulSoup
from spacy.matcher import PhraseMatcher


# --------------------------------------------------
# 1. LOAD NLP MODEL
# --------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# 2. CATEGORY-SPECIFIC ASPECT PHRASES
# --------------------------------------------------

CATEGORY_ASPECTS = {
    "Electronics": [
        "battery", "battery life", "battery performance",
        "camera", "camera quality", "screen", "screen protector",
        "touch screen", "display", "sound", "sound quality",
        "audio", "speaker", "speakers", "connection",
        "signal", "internet speed", "resolution", "monitor",
        "charger", "charging speed", "cable", "keyboard",
        "mouse", "performance", "software", "hub",
        "build quality", "picture quality", "image quality",
        "price", "value for money", "SD card slot",
        "card slot", "storage capacity", "cloud storage",
        "onboard storage", "installation", "setup",
        "network connectivity", "battery capacity"
    ],
    "Beauty": [
        "skin", "sensitive skin", "lips", "dry lips",
        "lashes", "eyelashes", "color", "colors",
        "color quality", "moisturizer", "fragrance",
        "scent", "smell", "texture", "application",
        "coverage", "hydration", "hair", "brush",
        "dry shampoo", "shampoo", "lip balm",
        "skin tone", "dry skin", "oily skin",
        "moisture", "moisturizing performance",
        "lasting power", "price", "value for money"
    ],
    "Home_Kitchen": [
        "battery", "battery life", "vacuum",
        "suction", "suction power", "brushes",
        "fabric", "curtain", "material", "size",
        "design", "quality", "instructions",
        "app", "temperature", "water", "timer",
        "charger", "cord", "bottle", "frame",
        "storage", "cleaning", "mapping",
        "performance", "price", "build quality",
        "noise level", "ease of use", "charging time",
        "heat settings", "temperature control",
        "footrest", "lid", "holding capacity",
        "weight", "durability", "stability"
    ]
}


# --------------------------------------------------
# 3. ALLOWED NOUN HEADS
# --------------------------------------------------

ASPECT_HEADS = {
    "Electronics": {
        "battery", "camera", "screen", "display",
        "sound", "audio", "speaker", "connection",
        "signal", "speed", "resolution", "monitor",
        "charger", "charging", "cable", "keyboard",
        "mouse", "performance", "software", "hub",
        "quality", "picture", "image", "price",
        "port", "button", "volume", "slot",
        "capacity", "storage", "installation",
        "setup", "connectivity", "memory"
    },
    "Beauty": {
        "skin", "lip", "lash", "eyelash", "color",
        "moisturizer", "fragrance", "scent", "texture",
        "application", "coverage", "hydration",
        "hair", "brush", "tanner", "price",
        "quality", "formula", "cream", "serum",
        "shade", "smell", "finish", "shampoo",
        "balm", "moisture"
    },
    "Home_Kitchen": {
        "battery", "vacuum", "suction", "brush",
        "fabric", "curtain", "material", "size",
        "design", "quality", "instruction", "app",
        "temperature", "water", "timer", "charger",
        "cord", "bottle", "frame", "storage",
        "cleaning", "mapping", "performance",
        "price", "noise", "power", "handle",
        "lid", "container", "filter", "motor",
        "ring", "screen", "feature", "footrest",
        "capacity", "weight", "durability",
        "stability"
    }
}


GENERIC_ASPECTS = {
    "device", "product", "item", "thing",
    "stuff", "one", "ones", "purchase",
    "customer", "company", "business",
    "portfolio", "password", "router"
}


# --------------------------------------------------
# 4. BUILD PHRASE MATCHER
# --------------------------------------------------

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

for category, phrases in CATEGORY_ASPECTS.items():
    matcher.add(
        category,
        [nlp.make_doc(phrase) for phrase in phrases]
    )


# --------------------------------------------------
# 5. CLEAN REVIEW TEXT
# --------------------------------------------------

def clean_review(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(
        text, "html.parser"
    ).get_text(" ")

    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# 6. NORMALIZE ASPECT
# --------------------------------------------------

def normalize_aspect(aspect: str) -> str:

    aspect = aspect.lower().strip()

    aspect = re.sub(r"\s+", " ", aspect)

    return aspect


# --------------------------------------------------
# 7. EXTRACT ASPECTS
# --------------------------------------------------

def extract_aspects(
    text: str,
    category: str = None
) -> list[dict]:

    text = clean_review(text)

    if not text:
        return []

    if category not in CATEGORY_ASPECTS:
        raise ValueError(
            f"Unsupported category: {category}"
        )

    doc = nlp(text)
    candidates = []

    # A. Vocabulary matches
    for match_id, start, end in matcher(doc):

        match_category = nlp.vocab.strings[match_id]

        if match_category != category:
            continue

        span = doc[start:end]

        candidates.append({
            "aspect": span.text,
            "normalized_aspect": normalize_aspect(span.text),
            "start": span.start_char,
            "end": span.end_char,
            "known_aspect": True,
            "source": "vocabulary"
        })

    # B. Full noun phrase discovery
    allowed_heads = ASPECT_HEADS[category]

    for chunk in doc.noun_chunks:

        if chunk.root.lemma_.lower() not in allowed_heads:
            continue

        tokens = [
            token for token in chunk
            if token.pos_ in {
                "NOUN", "PROPN", "ADJ", "NUM"
            }
            and token.dep_ not in {"det", "poss"}
        ]

        if not tokens:
            continue

        # Avoid selecting non-contiguous tokens
        start_token = tokens[0]
        end_token = tokens[-1]

        span = doc[start_token.i:end_token.i + 1]

        aspect = span.text.strip()

        normalized = normalize_aspect(aspect)

        if normalized in GENERIC_ASPECTS:
            continue

        if len(aspect) > 60:
            continue

        candidates.append({
            "aspect": aspect,
            "normalized_aspect": normalized,
            "start": span.start_char,
            "end": span.end_char,
            "known_aspect": False,
            "source": "nlp_discovery"
        })

    # C. Prefer longer phrases at overlapping positions
    candidates.sort(
        key=lambda item: (
            -(item["end"] - item["start"]),
            not item["known_aspect"],
            item["start"]
        )
    )

    selected = []

    for candidate in candidates:

        if candidate["normalized_aspect"] in GENERIC_ASPECTS:
            continue

        overlaps = any(
            candidate["start"] < existing["end"]
            and candidate["end"] > existing["start"]
            for existing in selected
        )

        if overlaps:
            continue

        selected.append(candidate)

    # D. Keep only one occurrence of each normalized aspect
    unique_aspects = []
    seen = set()

    for candidate in sorted(
        selected, key=lambda item: item["start"]
    ):

        normalized = candidate["normalized_aspect"]

        if normalized in seen:
            continue

        seen.add(normalized)
        unique_aspects.append(candidate)

    return unique_aspects


# --------------------------------------------------
# 8. LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    examples = [
        (
            "Electronics",
            "The SD card slot is missing and the "
            "storage capacity is disappointing."
        ),
        (
            "Beauty",
            "The dry shampoo smells wonderful "
            "and my hair feels soft."
        ),
        (
            "Home_Kitchen",
            "The footrest works great and the "
            "lid is surprisingly sturdy."
        )
    ]

    for category, review in examples:

        print("\nCATEGORY:", category)
        print("REVIEW:", review)

        aspects = extract_aspects(review, category)

        for aspect in aspects:
            print(
                " -",
                aspect["aspect"],
                "|",
                aspect["source"]
            )
