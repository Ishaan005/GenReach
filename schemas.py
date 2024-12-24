from pydantic import BaseModel, field_validator
from typing import Optional, Dict

class ContentQuery(BaseModel):
    content: str
    query: str
    weights: Optional[Dict[str, float]]

    @field_validator("content")
    def validate_content(cls, value):
        if not value.strip():
            raise ValueError("Content cannot be empty")
        if len(value.split()) < 10:
            raise ValueError("Content should have at least 10 words")
        return value

    @field_validator("query")
    def validate_query(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be empty")
        if len(value.split()) < 2:
            raise ValueError("Query should have at least 2 words")
        return value

    @field_validator("weights")
    def validate_weights(cls, value):
        default_weights = {
            "word_count_score": 0.2,
            "relevance_score": 0.2,
            "subjective_score": 0.1,
            "readability_score": 0.1,
            "structure_score": 0.1,
            "semantic_score": 0.1,
            "ner_score": 0.1,
            "diversity_score": 0.1
        }
        if value is None:
            return default_weights
        total = sum(value.values())
        if total != 1:
            raise ValueError(f"Sum of weights should be 1, but got {total}")
        return {**default_weights, **value}