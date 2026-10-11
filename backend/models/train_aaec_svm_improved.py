import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "aaec_components_clean.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "aaec_tfidf_svm_improved.pkl"
)


def main():

    print("Loading AAEC dataset...")

    df = pd.read_csv(DATA_FILE)

    train_df = df[df["split"] == "TRAIN"].copy()
    test_df = df[df["split"] == "TEST"].copy()

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    print()
    print("=" * 60)
    print("DATASET")
    print("=" * 60)

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples:  {len(test_df)}")

    print()
    print("=" * 60)
    print("BUILDING IMPROVED MODEL")
    print("=" * 60)

    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                max_features=50000
            )
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                lowercase=True,
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True
            )
        )
    ])

    model = Pipeline([
        (
            "features",
            features
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced"
            )
        )
    ])

    print("Training Word TF-IDF + Character TF-IDF + Linear SVM...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    labels = [
        "MajorClaim",
        "Claim",
        "Premise"
    ]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print("Confusion Matrix:")
    print()

    print("              Predicted")
    print("              MajorClaim  Claim  Premise")

    print(
        f"MajorClaim    {cm[0][0]:10d}  "
        f"{cm[0][1]:6d}  "
        f"{cm[0][2]:7d}"
    )

    print(
        f"Claim         {cm[1][0]:10d}  "
        f"{cm[1][1]:6d}  "
        f"{cm[1][2]:7d}"
    )

    print(
        f"Premise       {cm[2][0]:10d}  "
        f"{cm[2][1]:6d}  "
        f"{cm[2][2]:7d}"
    )

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