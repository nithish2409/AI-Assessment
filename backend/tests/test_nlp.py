
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_status():
    response = client.get("/api/nlp/status")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_tfidf():
    response = client.post(
        "/api/nlp/tfidf",
        json={
            "text": "Artificial intelligence and machine learning",
            "top_n": 5
        }
    )
    assert response.status_code == 200
    assert "features" in response.json()


def test_bow():
    response = client.post(
        "/api/nlp/bow",
        json={
            "text": "Python Python Python data",
            "top_n": 5
        }
    )
    assert response.status_code == 200

    features = response.json()["features"]
    counts = {
        item["term"]: item["count"]
        for item in features
    }
    assert counts["python"] == 3


def test_ngrams():
    response = client.post(
        "/api/nlp/ngrams",
        json={
            "text": "machine learning machine learning",
            "n": 2,
            "top_n": 5
        }
    )
    assert response.status_code == 200

    features = response.json()["features"]
    assert any(
        item["phrase"] == "machine learning"
        and item["count"] == 2
        for item in features
    )


def test_keywords():
    response = client.post(
        "/api/nlp/keywords",
        json={
            "text": "Technology companies develop artificial intelligence",
            "top_n": 5
        }
    )
    assert response.status_code == 200
    assert "keywords" in response.json()


def test_similarity():
    text = "Machine learning improves software"

    response = client.post(
        "/api/nlp/similarity",
        json={
            "text1": text,
            "text2": text
        }
    )
    assert response.status_code == 200
    assert response.json()["similarity_score"] == 1.0


def test_feature_selection():
    response = client.post(
        "/api/nlp/features",
        json={
            "category": "Technology",
            "top_n": 10
        }
    )
    assert response.status_code == 200
    assert len(response.json()["features"]) == 10


def test_combined_analysis():
    response = client.post(
        "/api/nlp/analyze",
        json={
            "text": "Microsoft released new AI software",
            "top_n": 5
        }
    )
    assert response.status_code == 200

    result = response.json()

    assert "classification" in result
    assert "tfidf" in result
    assert "bow" in result
    assert "ngrams" in result
    assert "keywords" in result
    assert "feature_selection" in result
