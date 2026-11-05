import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extraction import extract_aspects


TEST_CASES = [
    {
        "category": "Electronics",
        "review": "The battery life is poor, but the camera quality is excellent.",
        "expected": {"battery life", "camera quality"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Electronics",
        "review": "The display is bright, and the charging cable is too short.",
        "expected": {"display", "charging cable"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Electronics",
        "review": "The phone takes too long to charge, even though its screen looks great.",
        "expected": {"charging time", "screen"},
        "unexpected": set(),
    },
    {
        "category": "Beauty",
        "review": "The dry shampoo smells wonderful, and my hair feels soft.",
        "expected": {"dry shampoo", "hair"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Beauty",
        "review": "The moisturizer has a smooth texture and a pleasant fragrance.",
        "expected": {"moisturizer", "texture", "fragrance"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Beauty",
        "review": "This lip balm provides good hydration and the color lasts all day.",
        "expected": {"lip balm", "hydration", "color"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Home_Kitchen",
        "review": "The footrest works great, and the lid is surprisingly sturdy.",
        "expected": {"footrest", "lid"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Home_Kitchen",
        "review": "The vacuum has strong suction power, but the noise level is high.",
        "expected": {"vacuum", "suction power", "noise level"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Home_Kitchen",
        "review": "The container is durable, and the storage space is useful.",
        "expected": {"container", "storage"},
        "unexpected": {"charging time"},
    },
    {
        "category": "Electronics",
        "review": "I charge my phone every night before going to bed.",
        "expected": set(),
        "unexpected": {"charging time"},
    },
]


def run_tests():
    passed = 0
    failed = 0
    failures = []

    print("\nABSA ASPECT EXTRACTION REGRESSION TEST")
    print("=" * 72)

    for number, test in enumerate(TEST_CASES, start=1):
        predictions = extract_aspects(
            test["review"],
            test["category"]
        )

        detected = {
            item["normalized_aspect"]
            for item in predictions
        }

        missing = test["expected"] - detected
        unexpected = test["unexpected"] & detected

        success = not missing and not unexpected

        print(f"\nTest {number}: {test['category']}")
        print(f"Review: {test['review']}")
        print(f"Detected: {sorted(detected)}")

        if success:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL")
            failed += 1

            if missing:
                print(f"Missing expected aspects: {sorted(missing)}")
            if unexpected:
                print(f"Unexpected aspects: {sorted(unexpected)}")

            failures.append({
                "test": number,
                "category": test["category"],
                "missing": sorted(missing),
                "unexpected": sorted(unexpected),
            })

    print("\n" + "=" * 72)
    print(f"TOTAL: {len(TEST_CASES)}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")

    if failures:
        print("\nFailure details:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)

    print("\nAll regression tests passed.")


if __name__ == "__main__":
    run_tests()
