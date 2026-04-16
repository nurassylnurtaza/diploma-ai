from __future__ import annotations

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    text: str = Field(min_length=1)
    model_version: str = Field(default="baseline")
    threshold: float = Field(default=0.5, gt=0.0, le=1.0)


class Explanation(BaseModel):
    key_phrases: list[str] = Field(default_factory=list)
    top_sentences: list[str] = Field(default_factory=list)


class InferResponse(BaseModel):
    label: str
    score: float
    confidence: float
    explanation: Explanation

