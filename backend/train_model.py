
import json
import re
from pathlib import Path

import joblib
import pandas as pd

from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

# --------------------------------------
# 1. Configuration
# --------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

LABELS = ["World", "Sports", "Business", "Technology"]

RANDOM_STATE = 42
SAMPLES_PER_CLASS = 5000


# --------------------------------------
# 2. Text preprocessing
# --------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------
# 3. Load dataset
# --------------------------------------

print("Loading AG News dataset...")

dataset = load_dataset("fancyzhx/ag_news")

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# Balanced subset: 5000 per category
train_df = (
    train_df.groupby("label", group_keys=False)
    .sample(
        n=SAMPLES_PER_CLASS,
        random_state=RANDOM_STATE
    )
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))


# --------------------------------------
# 4. Preprocess text
# --------------------------------------

print("Cleaning text...")

X_train = train_df["text"].apply(clean_text)
X_test = test_df["text"].apply(clean_text)

y_train = train_df["label"]
y_test = test_df["label"]


# --------------------------------------
# 5. Build ML pipeline
# --------------------------------------

print("Building TF-IDF + Logistic Regression pipeline...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])


# --------------------------------------
# 6. Train model
# --------------------------------------

print("Training model...")

model.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------
# 7. Evaluate model
# --------------------------------------

print("Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro"
)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2, 3],
    target_names=LABELS,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2, 3]
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=LABELS,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(matrix)


# --------------------------------------
# 8. Save model and metrics
# --------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "news_classifier.joblib"
)

metrics = {
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "training_samples": len(train_df),
    "testing_samples": len(test_df),
    "labels": LABELS,
    "classification_report": report,
    "confusion_matrix": matrix.tolist()
}

with open(
    MODEL_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nModel and metrics saved successfully!")


# --------------------------------------
# 9. Test prediction
# --------------------------------------

sample = (
    "The technology company announced a new "
    "artificial intelligence processor."
)

prediction = model.predict([clean_text(sample)])[0]

print("\nSample text:", sample)
print("Predicted category:", LABELS[prediction])
