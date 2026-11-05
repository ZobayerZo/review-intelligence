
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

LABELS = ["negative", "neutral", "positive"]


def evaluate_model(model, X, y, model_name):
    predictions = model.predict(X)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(y, predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y, predictions, average="weighted")
        ),
        "macro_precision": float(
            precision_score(
                y, predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y, predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    report = classification_report(
        y,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=LABELS,
    )

    print(f"\n{'=' * 55}")
    print(model_name)
    print("=" * 55)

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    return metrics, report, matrix


def main():
    print("Loading datasets...")

    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "validation.parquet")

    X_train_text = train_df["combined_text"].fillna("")
    y_train = train_df["sentiment"].astype(str)

    X_val_text = val_df["combined_text"].fillna("")
    y_val = val_df["sentiment"].astype(str)

    print(f"Training records: {len(train_df):,}")
    print(f"Validation records: {len(val_df):,}")

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=150_000,
        sublinear_tf=True,
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    print(f"Training matrix shape: {X_train.shape}")
    print(f"Validation matrix shape: {X_val.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")

    results = {}

    best_model = None
    best_metrics = None
    best_report = None
    best_matrix = None

    configurations = {
        "unweighted": None,
        "balanced": "balanced",
    }

    for name, class_weight in configurations.items():

        print(f"\nTraining Logistic Regression: {name}")

        model = LogisticRegression(
            C=1.0,
            max_iter=500,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )

        model.fit(X_train, y_train)

        metrics, report, matrix = evaluate_model(
            model,
            X_val,
            y_val,
            name,
        )

        results[name] = metrics

        if (
            best_metrics is None
            or metrics["macro_f1"] > best_metrics["macro_f1"]
        ):
            best_model = model
            best_metrics = metrics
            best_report = report
            best_matrix = matrix

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    comparison = pd.DataFrame(results).T
    print(comparison)

    print(f"\nSelected model: {best_metrics['model']}")

    # Save selected model and vectorizer.
    model_file = MODEL_DIR / "baseline_tfidf_logreg.joblib"

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": best_model,
            "labels": LABELS,
        },
        model_file,
    )

    # Save validation metrics.
    metrics_file = REPORT_DIR / "baseline_validation_metrics.json"

    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "comparison": results,
                "selected_model": best_metrics["model"],
                "selected_metrics": best_metrics,
                "classification_report": best_report,
            },
            file,
            indent=4,
        )

    # Save confusion matrix.
    plt.figure(figsize=(7, 6))

    plt.imshow(best_matrix, interpolation="nearest")
    plt.title(
        f"Validation Confusion Matrix: {best_metrics['model']}"
    )
    plt.colorbar()

    plt.xticks(range(3), LABELS, rotation=30)
    plt.yticks(range(3), LABELS)

    plt.xlabel("Predicted sentiment")
    plt.ylabel("Actual sentiment")

    for i in range(3):
        for j in range(3):
            plt.text(
                j,
                i,
                f"{best_matrix[i, j]:,}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "baseline_confusion_matrix.png",
        dpi=300,
    )
    plt.close()

    print("\n" + "=" * 55)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 55)
    print(f"Model saved: {model_file}")
    print(f"Metrics saved: {metrics_file}")
    print("Confusion matrix saved.")


if __name__ == "__main__":
    main()
