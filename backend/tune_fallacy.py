import pandas as pd
import itertools
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# Paths
# =========================

TRAIN_PATH = "../datasets/processed/logic/edu_train_clean.csv"
DEV_PATH = "../datasets/processed/logic/edu_dev_clean.csv"


# =========================
# Load data
# =========================

train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)

X_train = train_df["source_article"]
y_train = train_df["updated_label"]

X_dev = dev_df["source_article"]
y_dev = dev_df["updated_label"]


# =========================
# Experiments
# =========================

ngram_ranges = [
    (1, 1),
    (1, 2),
    (1, 3)
]

C_values = [
    0.5,
    1.0,
    2.0,
    5.0
]


results = []


print("=" * 75)
print("BIASLESS - TF-IDF + LINEAR SVM HYPERPARAMETER TUNING")
print("=" * 75)

print(f"\nTraining samples : {len(X_train)}")
print(f"Validation samples : {len(X_dev)}")
print(f"Total experiments : {len(ngram_ranges) * len(C_values)}")


# =========================
# Run experiments
# =========================

experiment_number = 0

for ngram_range, C in itertools.product(
    ngram_ranges,
    C_values
):

    experiment_number += 1

    print("\n" + "-" * 75)
    print(
        f"Experiment {experiment_number}/12"
    )
    print(
        f"n-gram range = {ngram_range}, C = {C}"
    )

    start_time = time.time()

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LinearSVC(
                class_weight="balanced",
                C=C
            )
        )
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)

    accuracy = accuracy_score(
        y_dev,
        y_pred
    )

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

    elapsed = time.time() - start_time

    results.append({
        "ngram_range": str(ngram_range),
        "C": C,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    })

    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"Time        : {elapsed:.2f} seconds")


# =========================
# Results table
# =========================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 75)
print("FINAL RANKING")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================
# Best configuration
# =========================

best = results_df.iloc[0]

print("\n" + "=" * 75)
print("BEST CONFIGURATION")
print("=" * 75)

print(
    f"\nn-gram range : {best['ngram_range']}"
)

print(
    f"C            : {best['C']}"
)

print(
    f"Accuracy     : {best['accuracy']:.4f}"
)

print(
    f"Macro F1     : {best['macro_f1']:.4f}"
)

print(
    f"Weighted F1  : {best['weighted_f1']:.4f}"
)


# =========================
# Save results
# =========================

output_path = "../models/fallacy_tuning_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nResults saved to: {output_path}"
)