
from fastapi import APIRouter, HTTPException

from app.schemas.nlp import (
    TextRequest,
    ClassificationResponse,
    FeatureRequest,
    NgramRequest,
    SimilarityRequest,
    FeatureSelectionRequest,
)

from app.services.nlp_service import nlp_service
from app.services.feature_service import feature_service

router = APIRouter()

import json
from pathlib import Path



# --------------------------------------
# Shared error handling
# --------------------------------------

def execute_feature(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )


# --------------------------------------
# NLP Status
# --------------------------------------

@router.get("/status")
def nlp_status():
    return {
        "status": (
            "ready"
            if nlp_service.model is not None
            else "model_not_loaded"
        )
    }



@router.get("/metrics")
def get_model_metrics():
    metrics_path = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "metrics.json"
    )

    if not metrics_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model metrics are not available."
        )

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=503,
            detail="Unable to read model metrics."
        )



# --------------------------------------
# News Classification
# --------------------------------------

@router.post(
    "/classify",
    response_model=ClassificationResponse
)
def classify_text(request: TextRequest):
    return execute_feature(
        nlp_service.classify,
        request.text
    )


# --------------------------------------
# 1. TF-IDF
# --------------------------------------

@router.post("/tfidf")
def extract_tfidf(request: FeatureRequest):
    return execute_feature(
        feature_service.tfidf,
        request.text,
        request.top_n
    )


# --------------------------------------
# 2. Bag of Words
# --------------------------------------

@router.post("/bow")
def extract_bow(request: FeatureRequest):
    return execute_feature(
        feature_service.bow,
        request.text,
        request.top_n
    )


# --------------------------------------
# 3. N-grams
# --------------------------------------

@router.post("/ngrams")
def extract_ngrams(request: NgramRequest):
    return execute_feature(
        feature_service.ngrams,
        request.text,
        request.n,
        request.top_n
    )


# --------------------------------------
# 4. Keyword Extraction
# --------------------------------------

@router.post("/keywords")
def extract_keywords(request: FeatureRequest):
    return execute_feature(
        feature_service.keywords,
        request.text,
        request.top_n
    )


# --------------------------------------
# 5. Text Similarity
# --------------------------------------

@router.post("/similarity")
def compare_texts(request: SimilarityRequest):
    return execute_feature(
        feature_service.similarity,
        request.text1,
        request.text2
    )


# --------------------------------------
# 6. Feature Selection
# --------------------------------------

@router.post("/features")
def select_features(
    request: FeatureSelectionRequest
):
    return execute_feature(
        feature_service.feature_selection,
        request.category,
        request.top_n
    )


# --------------------------------------
# Combined NLP Analysis
# --------------------------------------

@router.post("/analyze")
def analyze_text(request: FeatureRequest):

    def run_analysis():
        classification = nlp_service.classify(
            request.text
        )

        return {
            "classification": classification,

            "tfidf": feature_service.tfidf(
                request.text, request.top_n
            ),

            "bow": feature_service.bow(
                request.text, request.top_n
            ),

            "ngrams": feature_service.ngrams(
                request.text, 2, request.top_n
            ),

            "keywords": feature_service.keywords(
                request.text, request.top_n
            ),

            "feature_selection":
                feature_service.feature_selection(
                    classification["category"],
                    request.top_n
                )
        }

    return execute_feature(run_analysis)
