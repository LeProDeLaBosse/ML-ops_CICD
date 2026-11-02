"""Service de prédiction du risque de défaut de crédit (credit-api).

Lancement local (depuis la racine du dépôt) :
    uvicorn src.api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import json
import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config

MODEL_PATH = os.getenv("MODEL_PATH", str(config.MODEL_PATH))
model = joblib.load(MODEL_PATH)
MODEL_VERSION = json.loads(config.METADATA_PATH.read_text())["model_version"]

app = FastAPI(title="credit-api", version="1.0.0")


class CreditRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    income: float = Field(ge=0)
    loan_amount: float = Field(gt=0)
    employment_years: float = Field(ge=0, le=60)
    num_credit_lines: int = Field(ge=0, le=30)
    region: Literal["North", "South", "East", "West"]


class CreditResponse(BaseModel):
    default_probability: float
    decision: Literal["accept", "review"]
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    features = pd.DataFrame([request.model_dump()])[config.FEATURES]
    probability = float(model.predict_proba(features)[0, 1])
    return CreditResponse(
        default_probability=round(probability, 4),
        decision="review" if probability >= config.DECISION_THRESHOLD else "accept",
        model_version=MODEL_VERSION,
    )
