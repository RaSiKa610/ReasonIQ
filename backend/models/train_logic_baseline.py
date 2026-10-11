import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "logic_tfidf_logistic.pkl"
)


def main():

    print("Loading LOGIC dataset...")

    df = pd.read_csv(DATA_FILE)

    train_df = df[df["split"] == "TRAIN"].copy()
    dev_df = df[df["split"] == "DEV"].copy()
    test_df = df[df["split"] == "TEST"].copy()

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_dev = dev_df["text"]
    y_dev = dev_df["label"]

    print()
    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(dev_df)}")
    print(f"Test samples:       {len(test_df)}")

    print("\nTraining labels:")
    print(y_train.value_counts())

    print()
    print("=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            )
        )
    ])

    print("Training TF-IDF + Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    print()
    print("=" * 60)
    print("VALIDATION EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_dev)

    accuracy = accuracy_score(
        y_dev,
        y_pred
    )

    print(f"\nValidation Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_dev,
            y_pred,
            digits=4
        )
    )

    labels = sorted(
        y_train.unique()
    )

    cm = confusion_matrix(
        y_dev,
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
    print("SAVING MODEL")
    print("=" * 60)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(f"Model saved to:")
    print(MODEL_FILE)

    print()
    print("DONE.")


if __name__ == "__main__":
    main()