from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model_store import ModelStore, load_env_default_version, load_env_models_dir
from .schemas import Explanation, InferRequest, InferResponse


def split_csv(value: str) -> list[str]:
    parts = [p.strip() for p in (value or "").split(",")]
    return [p for p in parts if p]


def heuristic_infer(text: str) -> tuple[float, float]:
    # Deterministic fallback: map length -> score in [0..1]
    length_score = min(1.0, max(0.0, len(text) / 240.0))
    score = round(float(length_score), 4)
    confidence = 0.75
    return score, confidence


def score_to_label(score: float, threshold: float) -> str:
    if score < threshold:
        return "low"
    if score < 0.8:
        return "medium"
    return "high"


def build_explanation(text: str) -> Explanation:
    words = [w for w in text.replace("\n", " ").split(" ") if w]
    key_phrases = words[: min(5, len(words))]
    top_sentences = [s.strip() for s in text.split(".") if s.strip()][:2] or [text[:160]]
    return Explanation(key_phrases=key_phrases, top_sentences=top_sentences)


models = ModelStore(models_dir=load_env_models_dir(), default_version=load_env_default_version())

app = FastAPI(title="MoodInsight AI", version="0.1.0")

cors_origins = split_csv(os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://[::1]:5173"))
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
def _startup() -> None:
    models.try_warmup()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    text = req.text.strip()
    threshold = float(req.threshold)
    model_version = (req.model_version or "").strip() or models.default_version

    out = models.infer(text=text, model_version=model_version)
    if out is None:
        score, confidence = heuristic_infer(text)
    else:
        score, confidence = out.score, out.confidence

    score = float(max(0.0, min(1.0, score)))
    confidence = float(max(0.0, min(1.0, confidence)))

    label = score_to_label(score, threshold)
    explanation = build_explanation(text)

    return InferResponse(label=label, score=score, confidence=confidence, explanation=explanation)

