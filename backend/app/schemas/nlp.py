
from typing import Literal
from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 20000


def validate_text(value: str) -> str:
    value = value.strip()

    if not value:
        raise ValueError("Text cannot be empty.")

    if len(value) > MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text must not exceed {MAX_TEXT_LENGTH} characters."
        )

    if not any(char.isalnum() for char in value):
        raise ValueError(
            "Text must contain letters or numbers."
        )

    return value


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def check_text(cls, value):
        return validate_text(value)


class ClassificationResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]


class FeatureRequest(TextRequest):
    top_n: int = Field(default=10, ge=1, le=50)


class NgramRequest(FeatureRequest):
    n: int = Field(default=2, ge=1, le=3)


class SimilarityRequest(BaseModel):
    text1: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    text2: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)

    @field_validator("text1", "text2")
    @classmethod
    def check_text(cls, value):
        return validate_text(value)


class FeatureSelectionRequest(BaseModel):
    category: Literal[
        "World",
        "Sports",
        "Business",
        "Technology"
    ] = "Technology"

    top_n: int = Field(default=10, ge=1, le=50)
