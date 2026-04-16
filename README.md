# FastAPI text analyzer microservice

This service exposes a simple HTTP API that accepts raw text and returns an analysis result (label/score/confidence + short explanation).

It is designed to work with the Go backend in this workspace:

- Go expects `POST /infer` with `{ text, model_version, threshold }`
- Service responds with `{ label, score, confidence, explanation: { key_phrases, top_sentences } }`

## Run locally (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8081
```

## Run with Docker

```bash
docker build -t moodinsight-ai .
docker run --rm -p 8081:8081 moodinsight-ai
```

## Environment variables

- `PORT`: server port (default `8081`)
- `CORS_ORIGINS`: comma-separated allowed origins (default allows Vite dev server `http://localhost:5173`)
- `MODELS_DIR`: directory with `*.joblib` models (default `./models`)
- `DEFAULT_MODEL_VERSION`: default model version (default `baseline`)

## Model files (joblib)

Put your sklearn pipeline (recommended) into:

```
models/baseline.joblib
```

The service will attempt to load `models/{model_version}.joblib`. If the file is missing or loading fails, it falls back to a deterministic heuristic (so the API still works out-of-the-box).

