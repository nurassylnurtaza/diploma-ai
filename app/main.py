from fastapi import FastAPI
from pydantic import BaseModel

from .model_store import ModelStore


app = FastAPI(title="Mental Health Classifier API")

model_store = ModelStore()


class Request(BaseModel):
    text: str


@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    result = model_store.predict(req.text)

    return {
        "label": result["label"],
        "score": result["score"],
        "confidence": result["confidence"]
    }