import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================
# Paths
# =========================

TRAIN_PATH = "../datasets/processed/logic/edu_train_clean.csv"
DEV_PATH = "../datasets/processed/logic/edu_dev_clean.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# Load datasets
# =========================

train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)

X_train = train_df["source_article"]
y_train = train_df["updated_label"]

X_dev = dev_df["source_article"]
y_dev = dev_df["updated_label"]


print("=" * 60)
print("BIASLESS FALLACY CLASSIFIER")
print("=" * 60)

print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_dev)}")
print(f"Number of classes: {y_train.nunique()}")

print("\nClasses:")
for label in sorted(y_train.unique()):
    print(f" - {label}")


# =========================
# Build ML pipeline
# =========================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])


# =========================
# Train
# =========================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

model.fit(X_train, y_train)

print("Training completed.")


# =========================
# Validation prediction
# =========================

y_pred = model.predict(X_dev)


# =========================
# Evaluation
# =========================

accuracy = accuracy_score(y_dev, y_pred)

precision = precision_score(
    y_dev,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_dev,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_dev,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_dev,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")
print(f"Weighted F1    : {weighted_f1:.4f}")


# =========================
# Detailed report
# =========================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_dev,
        y_pred,
        zero_division=0
    )
)


# =========================
# Save model
# =========================

model_path = os.path.join(
    MODEL_DIR,
    "fallacy_classifier.pkl"
)

import joblib

joblib.dump(model, model_path)

print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"\nSaved to: {model_path}")