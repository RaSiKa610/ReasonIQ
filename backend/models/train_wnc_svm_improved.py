import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    "wnc_tfidf_svm_improved.pkl"
)


def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    return accuracy, precision, recall, f1


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
    # WORD + CHARACTER TF-IDF
    # --------------------------------------------------

    print("\nBuilding Word + Character TF-IDF...")

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=50000
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=30000
    )

    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])

    print("Fitting vectorizer on training data...")

    X_train_features = vectorizer.fit_transform(
        X_train
    )

    print(
        f"Total feature count: "
        f"{X_train_features.shape[1]:,}"
    )

    print("Transforming development data...")

    X_dev_features = vectorizer.transform(
        X_dev
    )

    print("Transforming test data...")

    X_test_features = vectorizer.transform(
        X_test
    )

    # --------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------

    print("\nTraining Linear SVM...")

    model = LinearSVC(
        C=1.0,
        class_weight="balanced"
    )

    model.fit(
        X_train_features,
        y_train
    )

    # --------------------------------------------------
    # DEVELOPMENT EVALUATION
    # --------------------------------------------------

    print("\nValidation Results")
    print("------------------")

    dev_pred = model.predict(
        X_dev_features
    )

    (
        dev_accuracy,
        dev_precision,
        dev_recall,
        dev_f1
    ) = calculate_metrics(
        y_dev,
        dev_pred
    )

    print(
        f"Accuracy:  {dev_accuracy:.4f}"
    )

    print(
        f"Precision: {dev_precision:.4f}"
    )

    print(
        f"Recall:    {dev_recall:.4f}"
    )

    print(
        f"Macro F1:  {dev_f1:.4f}"
    )

    # --------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------

    print("\nTest Results")
    print("------------")

    test_pred = model.predict(
        X_test_features
    )

    (
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    ) = calculate_metrics(
        y_test,
        test_pred
    )

    print(
        f"Accuracy:  {test_accuracy:.4f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall:    {test_recall:.4f}"
    )

    print(
        f"Macro F1:  {test_f1:.4f}"
    )

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "Neutral",
                "Bias-associated"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

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