from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class ModelOutput:
    score: float
    confidence: float


DEFAULT_CLASS_RISK_SCORE: dict[str, float] = {
    # Default mapping for the included mental-health multiclass model.
    # Produces a continuous score in [0..1] that is later mapped to low/medium/high.
    "Normal": 0.0,
    "Anxiety": 0.6,
    "Depression": 0.85,
    "Suicidal": 1.0,
}


class ModelStore:
    def __init__(self, models_dir: str, default_version: str) -> None:
        self._models_dir = Path(models_dir)
        self._default_version = default_version
        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}

    @property
    def default_version(self) -> str:
        return self._default_version

    def try_warmup(self) -> None:
        # Best-effort warmup; service still works without model files.
        try:
            self.get_model(self._default_version)
        except Exception:
            return

    def model_path(self, version: str) -> Path:
        version = (version or "").strip() or self._default_version
        return self._models_dir / f"{version}.joblib"

    def get_model(self, version: str) -> Any | None:
        version = (version or "").strip() or self._default_version
        with self._lock:
            if version in self._models:
                return self._models[version]

            path = self.model_path(version)
            if not path.exists():
                self._models[version] = None
                return None

            m = joblib.load(path)
            self._models[version] = m
            return m

    def infer(self, text: str, model_version: str) -> ModelOutput | None:
        model = self.get_model(model_version)
        if model is None:
            return None

        classes = self._safe_classes_list(getattr(model, "classes_", None))

        # sklearn-style APIs (best compatibility).
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            proba = np.asarray(proba, dtype=float)
            if proba.ndim != 1 or proba.size == 0:
                return None

            if classes and len(classes) == int(proba.size):
                score = float(self._risk_score_from_classes(classes, proba))
            else:
                score = float(self._pick_positive_probability(model, proba))
            confidence = float(np.max(proba))
            return ModelOutput(score=score, confidence=confidence)

        if hasattr(model, "decision_function"):
            margins = np.asarray(model.decision_function([text]), dtype=float)

            # Binary case: margin scalar -> sigmoid.
            if margins.ndim == 1 and margins.size == 1:
                margin = float(margins.reshape(-1)[0])
                score = float(1.0 / (1.0 + np.exp(-margin)))
                confidence = float(max(score, 1.0 - score))
                return ModelOutput(score=score, confidence=confidence)

            # Multiclass: margins vector -> softmax (not calibrated, but usable for confidence/weights).
            margins = margins.reshape(1, -1) if margins.ndim == 1 else margins
            if margins.ndim != 2 or margins.shape[0] != 1 or margins.shape[1] == 0:
                return None

            probs = self._softmax(margins[0])
            confidence = float(np.max(probs))

            if classes and len(classes) == int(probs.size):
                score = float(self._risk_score_from_classes(classes, probs))
            else:
                # Without class labels, treat "non-first class" as risk-like.
                score = float(1.0 - probs[0])

            return ModelOutput(score=score, confidence=confidence)

        if hasattr(model, "predict"):
            pred = model.predict([text])[0]
            # If model only predicts a class, return a coarse score.
            # You can refine this once you know your classes.
            pred_s = str(pred)
            if pred_s in DEFAULT_CLASS_RISK_SCORE:
                score = float(DEFAULT_CLASS_RISK_SCORE[pred_s])
                confidence = 1.0
            else:
                score = 1.0 if pred_s in {"1", "true", "True", "pos", "positive", "high"} else 0.0
                confidence = 1.0
            return ModelOutput(score=score, confidence=confidence)

        return None

    def _pick_positive_probability(self, model: Any, proba: np.ndarray) -> float:
        # Try to select a "positive" class index if possible; otherwise fall back to the last column.
        classes = getattr(model, "classes_", None)
        if classes is None:
            return float(proba[-1])

        try:
            classes_list = [str(c) for c in list(classes)]
        except Exception:
            return float(proba[-1])

        for key in ("1", "pos", "positive", "high"):
            if key in classes_list:
                return float(proba[classes_list.index(key)])

        return float(proba[-1])

    def _safe_classes_list(self, classes: Any) -> list[str] | None:
        if classes is None:
            return None
        try:
            out = [str(c) for c in list(classes)]
            return out if out else None
        except Exception:
            return None

    def _risk_score_from_classes(self, classes: list[str], probs: np.ndarray) -> float:
        weights = np.asarray([DEFAULT_CLASS_RISK_SCORE.get(c, 0.5) for c in classes], dtype=float)
        score = float(np.dot(probs.astype(float), weights))
        # Clip to [0..1]
        return float(max(0.0, min(1.0, score)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - np.max(x)
        ex = np.exp(x)
        s = ex.sum()
        if s <= 0:
            return np.ones_like(ex) / float(ex.size)
        return ex / s


def load_env_models_dir() -> str:
    return os.getenv("MODELS_DIR", "./models").strip() or "./models"


def load_env_default_version() -> str:
    return os.getenv("DEFAULT_MODEL_VERSION", "baseline").strip() or "baseline"

