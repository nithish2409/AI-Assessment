
import re
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "news_classifier.joblib"

LABELS = ["World", "Sports", "Business", "Technology"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NLPService:

    def __init__(self):
        self.model = None

        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)

    def classify(self, text):
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        cleaned = clean_text(text)

        if not cleaned:
            raise ValueError("Text is empty after preprocessing.")

        vectorizer = self.model.named_steps["tfidf"]
        vector = vectorizer.transform([cleaned])    

        if vector.nnz == 0:
            raise ValueError(
                "No recognized vocabulary terms found in the text."
            )
        prediction = int(self.model.predict([cleaned])[0])

        probabilities = self.model.predict_proba([cleaned])[0]

        class_probabilities = {
            LABELS[int(label)]: round(float(prob), 4)
            for label, prob in zip(
                self.model.classes_,
                probabilities
            )
        }

        return {
            "category": LABELS[prediction],
            "confidence": round(
                class_probabilities[LABELS[prediction]], 4
            ),
            "probabilities": class_probabilities
        }


nlp_service = NLPService()
