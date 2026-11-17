
import re
from collections import Counter

import numpy as np

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import cosine_similarity

from app.services.nlp_service import nlp_service, LABELS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FeatureService:

    @staticmethod
    def require_model():
        if nlp_service.model is None:
            raise RuntimeError(
                "Trained model is not loaded."
            )
        return nlp_service.model

   
    @staticmethod
    def validate_text(text):
        cleaned = clean_text(text)

        if not cleaned:
            raise ValueError(
                "No valid text remains after preprocessing."
            )

        return cleaned

    # ----------------------------------
    # 1. TF-IDF
    # ----------------------------------

    def tfidf(self, text, top_n=10):
        model = self.require_model()
        cleaned = self.validate_text(text)

        vectorizer = model.named_steps["tfidf"]

        matrix = vectorizer.transform([cleaned])
        names = vectorizer.get_feature_names_out()

        scores = matrix.toarray()[0]

        indices = np.argsort(scores)[::-1]
        indices = [
            i for i in indices if scores[i] > 0
        ][:top_n]

        return {
            "method": "TF-IDF",
            "features": [
                {
                    "term": names[i],
                    "score": round(float(scores[i]), 4)
                }
                for i in indices
            ]
        }

    # ----------------------------------
    # 2. Bag of Words
    # ----------------------------------

    def bow(self, text, top_n=10):
        cleaned = self.validate_text(text)

        vectorizer = CountVectorizer(
            stop_words="english"
        )

        try:
            matrix = vectorizer.fit_transform(
                [cleaned]
            )
        except ValueError:
            return {
                "method": "Bag of Words",
                "features": []
            }

        names = vectorizer.get_feature_names_out()
        counts = matrix.toarray()[0]

        indices = np.argsort(counts)[::-1][:top_n]

        return {
            "method": "Bag of Words",
            "features": [
                {
                    "term": names[i],
                    "count": int(counts[i])
                }
                for i in indices
            ]
        }

    # ----------------------------------
    # 3. N-gram analysis
    # ----------------------------------

    def ngrams(self, text, n=2, top_n=10):
        cleaned = self.validate_text(text)

        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            stop_words="english"
        )

        try:
            matrix = vectorizer.fit_transform(
                [cleaned]
            )
        except ValueError:
            return {
                "method": "N-gram Analysis",
                "n": n,
                "features": []
            }

        names = vectorizer.get_feature_names_out()
        counts = matrix.toarray()[0]

        indices = np.argsort(counts)[::-1][:top_n]

        return {
            "method": "N-gram Analysis",
            "n": n,
            "features": [
                {
                    "phrase": names[i],
                    "count": int(counts[i])
                }
                for i in indices
            ]
        }

    # ----------------------------------
    # 4. Keyword extraction
    # ----------------------------------

    
    def keywords(self, text, top_n=10):
        model = self.require_model()
        cleaned = self.validate_text(text)

        vectorizer = model.named_steps["tfidf"]

        matrix = vectorizer.transform([cleaned])
        names = vectorizer.get_feature_names_out()

        row = matrix.getrow(0)

        ranked = sorted(
            zip(row.indices, row.data),
            key=lambda item: item[1],
            reverse=True
        )[:top_n]

        return {
            "method": "Keyword Extraction",
            "keywords": [
                {
                    "keyword": str(names[i]),
                    "score": round(float(score), 4)
                }
                for i, score in ranked
            ]
        }


    # ----------------------------------
    # 5. Text similarity
    # ----------------------------------

    def similarity(self, text1, text2):
        model = self.require_model()

        first = self.validate_text(text1)
        second = self.validate_text(text2)

        vectorizer = model.named_steps["tfidf"]

        vectors = vectorizer.transform(
            [first, second]
        )


        if vectors[0].nnz == 0 or vectors[1].nnz == 0:
            raise ValueError(
                "Both texts must contain recognized vocabulary terms."
            )


        score = cosine_similarity(
            vectors[0],
            vectors[1]
        )[0][0]

        return {
            "method": "Cosine Similarity",
            "similarity_score": round(
                float(score), 4
            ),
            "similarity_percentage": round(
                float(score) * 100, 2
            )
        }

    # ----------------------------------
    # 6. Feature selection
    # ----------------------------------

    def feature_selection(
        self,
        category="Technology",
        top_n=10
    ):
        model = self.require_model()

        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        if category not in LABELS:
            raise ValueError(
                "Invalid news category."
            )

        names = vectorizer.get_feature_names_out()

        class_id = LABELS.index(category)
        class_position = list(
            classifier.classes_
        ).index(class_id)

        coefficients = classifier.coef_[
            class_position
        ]

        indices = np.argsort(
            coefficients
        )[::-1][:top_n]

        return {
            "method": "Feature Selection",
            "category": category,
            "features": [
                {
                    "term": names[i],
                    "importance": round(
                        float(coefficients[i]), 4
                    )
                }
                for i in indices
            ]
        }


feature_service = FeatureService()
