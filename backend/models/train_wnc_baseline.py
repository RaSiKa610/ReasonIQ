import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

BASE_DIR = r"C:\Users\Rasika\BiasLess"

DATA_FILE = os.path.join(
    BASE_DIR,
    "datasets",
    "processed",
    "wnc_binary_clean.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "wnc_tfidf_logistic.pkl"
)


def main():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Reading: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    train = df[df["split"] == "train"]
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]

    X_train = train["text"]
    y_train = train["label"]

    X_dev = dev["text"]
    y_dev = dev["label"]

    X_test = test["text"]
    y_test = test["label"]

    print("\nDataset")
    print("-------")
    print(f"Train: {len(train):,}")
    print(f"Dev:   {len(dev):,}")
    print(f"Test:  {len(test):,}")

    # --------------------------------------------------
    # TF-IDF
    # --------------------------------------------------

    print("\nTraining TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_dev_tfidf = vectorizer.transform(X_dev)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Feature count: {X_train_tfidf.shape[1]:,}")

    # --------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    model.fit(X_train_tfidf, y_train)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print("\nValidation Results")
    print("------------------")

    dev_pred = model.predict(X_dev_tfidf)

    dev_accuracy = accuracy_score(
        y_dev,
        dev_pred
    )

    dev_precision, dev_recall, dev_f1, _ = (
        precision_recall_fscore_support(
            y_dev,
            dev_pred,
            average="macro",
            zero_division=0
        )
    )

    print(f"Accuracy:  {dev_accuracy:.4f}")
    print(f"Precision: {dev_precision:.4f}")
    print(f"Recall:    {dev_recall:.4f}")
    print(f"Macro F1:  {dev_f1:.4f}")

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    print("\nTest Results")
    print("------------")

    test_pred = model.predict(X_test_tfidf)

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_precision, test_recall, test_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            test_pred,
            average="macro",
            zero_division=0
        )
    )

    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"Macro F1:  {test_f1:.4f}")

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "Neutral",
                "Biased"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # Save model + vectorizer
    # --------------------------------------------------

    import joblib

    artifact = {
        "vectorizer": vectorizer,
        "model": model
    }

    joblib.dump(
        artifact,
        MODEL_FILE
    )

    print("\nModel saved:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()