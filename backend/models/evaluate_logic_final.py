import os
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = r"C:\Users\Rasika\BiasLess"

DATA_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "logic_edu_clean.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "logic_tfidf_svm_improved.pkl"
)


def main():

    print("Loading final LOGIC model...")

    model = joblib.load(
        MODEL_FILE
    )

    df = pd.read_csv(
        DATA_FILE
    )

    test_df = df[
        df["split"] == "TEST"
    ].copy()

    X_test = test_df["text"]
    y_test = test_df["label"]

    print()
    print("=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    print(f"Test samples: {len(test_df)}")

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    labels = sorted(
        y_test.unique()
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("Confusion Matrix:")
    print()

    print("Labels:")
    print(labels)

    print()

    print(cm)

    print()
    print("=" * 60)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()