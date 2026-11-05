
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_FILE = PROJECT_ROOT / "models" / "baseline_tfidf_logreg.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

LABELS = ["negative", "neutral", "positive"]


def main():
    print("Loading selected baseline model...")
    saved_model = joblib.load(MODEL_FILE)

    vectorizer = saved_model["vectorizer"]
    model = saved_model["model"]

    print("Loading untouched test dataset...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    X_text = test_df["combined_text"].fillna("")
    y_true = test_df["sentiment"].astype(str)

    print(f"Test records: {len(test_df):,}")

    print("\nTransforming test reviews...")
    X_test = vectorizer.transform(X_text)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    # Overall evaluation
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted")
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS")
    print("=" * 55)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    # Category-wise evaluation
    category_results = {}

    print("\n" + "=" * 55)
    print("PERFORMANCE BY CATEGORY")
    print("=" * 55)

    for category in sorted(test_df["category"].unique()):
        mask = test_df["category"] == category

        category_true = y_true[mask]
        category_pred = pd.Series(
            y_pred, index=test_df.index
        )[mask]

        category_metrics = {
            "records": int(mask.sum()),
            "accuracy": float(
                accuracy_score(category_true, category_pred)
            ),
            "macro_f1": float(
                f1_score(
                    category_true,
                    category_pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    category_true,
                    category_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

        category_results[category] = category_metrics

        print(f"\n{category}")
        for name, value in category_metrics.items():
            print(f"{name}: {value}")

    # Confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Baseline Model - Test Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(3), LABELS, rotation=30)
    plt.yticks(range(3), LABELS)

    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")

    for i in range(3):
        for j in range(3):
            plt.text(
                j, i, f"{matrix[i, j]:,}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "baseline_test_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    # Save complete evaluation
    results = {
        "model": "TF-IDF + Logistic Regression",
        "configuration": "class_weight=balanced",
        "test_records": len(test_df),
        "overall_metrics": metrics,
        "classification_report": report,
        "category_metrics": category_results,
    }

    output_file = REPORT_DIR / "baseline_test_results.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 55)
    print("TEST EVALUATION COMPLETED")
    print("=" * 55)
    print(f"Results saved: {output_file}")
    print("Confusion matrix saved.")


if __name__ == "__main__":
    main()
