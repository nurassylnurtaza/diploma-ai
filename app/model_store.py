import os
import torch
import threading
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = ["Anxiety", "Depression", "Normal", "Suicidal"]

RISK_MAP = {
    "Normal": 0.0,
    "Anxiety": 0.6,
    "Depression": 0.85,
    "Suicidal": 1.0,
}


@dataclass
class ModelOutput:
    label: str
    score: float
    confidence: float


class ModelStore:
    def __init__(self):
        self.models_dir = Path(os.getenv("MODELS_DIR", "./models"))
        self.default_version = os.getenv("DEFAULT_MODEL_VERSION", "mentalbert-v1")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load_model(self):
        with self._lock:
            if self._model is not None:
                return

            path = self.models_dir / self.default_version

            if not path.exists():
                raise ValueError(f"Model not found at {path}")

            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path)

            self._model.to(self.device)
            self._model.eval()

    def predict(self, text: str):
        try:
            self.load_model()

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            idx = int(probs.argmax())
            label = LABELS[idx]

            return {
                "label": label,
                "confidence": float(probs[idx]),
                "score": float(RISK_MAP[label])
            }

        except Exception as e:
            print("🔥 ERROR:", str(e))
            raise e