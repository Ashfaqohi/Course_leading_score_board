"""FastAPI service for the saved lead-conversion model."""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


MODEL_PATH = Path(__file__).with_name("pipeline.bin")
CATEGORICAL_FEATURES = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
]


class Lead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lead_source: str
    industry: str
    employment_status: str
    location: str
    number_of_courses_viewed: int
    annual_income: float
    interaction_count: int
    lead_score: float


class Prediction(BaseModel):
    conversion_probability: float
    predicted_class: int
    converted: bool


with MODEL_PATH.open("rb") as model_file:
    artifact: Any = pickle.load(model_file)

if isinstance(artifact, dict):
    model = artifact["model"]
    feature_columns = artifact.get("columns", artifact.get("feature_columns"))
    if feature_columns is None:
        raise ValueError("The model artifact does not include feature columns.")
else:
    model = artifact
    feature_columns = None

app = FastAPI(title="Course Lead Conversion API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(lead: Lead) -> Prediction:
    lead_frame = pd.DataFrame([lead.model_dump()])

    if feature_columns is not None:
        model_input = pd.get_dummies(
            lead_frame,
            columns=CATEGORICAL_FEATURES,
        ).reindex(columns=feature_columns, fill_value=0)
        classes = model.classes_
    else:
        model_input = lead_frame
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = model.steps[-1][1].classes_

    probabilities = model.predict_proba(model_input)[0]
    conversion_probability = float(probabilities[list(classes).index(1)])
    predicted_class = int(classes[probabilities.argmax()])

    return Prediction(
        conversion_probability=conversion_probability,
        predicted_class=predicted_class,
        converted=predicted_class == 1,
    )
