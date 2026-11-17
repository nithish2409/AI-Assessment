
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics():
    response = client.get("/api/nlp/metrics")
    assert response.status_code == 200

    metrics = response.json()
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert metrics["testing_samples"] == 7600


def test_classification():
    response = client.post(
        "/api/nlp/classify",
        json={
            "text": (
                "The football team won the championship "
                "after scoring three goals."
            )
        }
    )

    assert response.status_code == 200

    data = response.json()
    assert data["category"] in [
        "World", "Sports", "Business", "Technology"
    ]
    assert 0 <= data["confidence"] <= 1

    assert sum(data["probabilities"].values()) == pytest.approx(
        1.0, abs=0.001
    )


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "!!!???",
])
def test_invalid_text(text):
    response = client.post(
        "/api/nlp/classify",
        json={"text": text}
    )
    assert response.status_code == 422


def test_very_long_text():
    response = client.post(
        "/api/nlp/classify",
        json={"text": "technology " * 3000}
    )
    assert response.status_code == 422


def test_invalid_ngram_size():
    response = client.post(
        "/api/nlp/ngrams",
        json={
            "text": "machine learning is useful",
            "n": 5
        }
    )
    assert response.status_code == 422


def test_invalid_category():
    response = client.post(
        "/api/nlp/features",
        json={"category": "Unknown"}
    )
    assert response.status_code == 422


def test_identical_text_similarity():
    response = client.post(
        "/api/nlp/similarity",
        json={
            "text1": "Technology companies develop software.",
            "text2": "Technology companies develop software."
        }
    )

    assert response.status_code == 200
    assert response.json()["similarity_score"] == pytest.approx(
        1.0, abs=0.0001
    )


def test_combined_analysis():
    response = client.post(
        "/api/nlp/analyze",
        json={
            "text": (
                "Researchers developed artificial intelligence "
                "software for autonomous vehicles."
            ),
            "top_n": 5
        }
    )

    assert response.status_code == 200

    data = response.json()

    required = [
        "classification",
        "tfidf",
        "bow",
        "ngrams",
        "keywords",
        "feature_selection"
    ]

    assert all(key in data for key in required)
