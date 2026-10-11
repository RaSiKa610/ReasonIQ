import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================
# Paths
# =========================

MODEL_PATH = "../models/fallacy_svm.pkl"
TEST_PATH = "../datasets/processed/logic/edu_test_clean.csv"

RESULTS_DIR = "../results/fallacy"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# Load model
# =========================

print("=" * 70)
print("BIASLESS - FINAL LOGIC TEST EVALUATION")
print("=" * 70)

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# =========================
# Load test data
# =========================

test_df = pd.read_csv(TEST_PATH)

X_test = test_df["source_article"]
y_test = test_df["updated_label"]


print(f"\nTest samples: {len(test_df)}")
print(f"Classes: {y_test.nunique()}")


# =========================
# Prediction
# =========================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

print("Predictions completed.")


# =========================
# Overall metrics
# =========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================
# Print results
# =========================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"\nAccuracy        : {accuracy:.4f}")
print(f"Macro Precision : {precision:.4f}")
print(f"Macro Recall    : {recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")


# =========================
# Classification report
# =========================

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# =========================
# Confusion matrix
# =========================

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)


# =========================
# Save metrics
# =========================

metrics_path = os.path.join(
    RESULTS_DIR,
    "final_test_metrics.txt"
)

with open(metrics_path, "w", encoding="utf-8") as f:

    f.write("BIASLESS - FINAL LOGIC TEST EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Test samples: {len(test_df)}\n")
    f.write(f"Number of classes: {y_test.nunique()}\n\n")

    f.write("MODEL CONFIGURATION\n")
    f.write("-" * 70 + "\n")
    f.write("TF-IDF word n-grams: (1,2)\n")
    f.write("Classifier: Linear SVM\n")
    f.write("C: 0.5\n")
    f.write("class_weight: balanced\n\n")

    f.write("OVERALL METRICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro Precision: {precision:.4f}\n")
    f.write(f"Macro Recall: {recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 70 + "\n")
    f.write(report)


# =========================
# Save confusion matrix
# =========================

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_path)


# =========================
# Save predictions
# =========================

predictions_df = pd.DataFrame({
    "text": X_test,
    "actual_label": y_test,
    "predicted_label": y_pred
})

predictions_path = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# =========================
# Finished
# =========================

print("\n" + "=" * 70)
print("RESULT FILES SAVED")
print("=" * 70)

print(f"\nMetrics:")
print(metrics_path)

print(f"\nConfusion matrix:")
print(cm_path)

print(f"\nPredictions:")
print(predictions_path)

print("\nFinal LOGIC evaluation completed.")