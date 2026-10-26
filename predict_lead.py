import pickle
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


FEATURES = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
    "number_of_courses_viewed",
    "annual_income",
    "interaction_count",
    "lead_score",
]

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}


def predict_lead(lead_data: Mapping[str, Any], model_path: Path | None = None) -> dict[str, Any]:
    """Predict the conversion probability for one lead.

    The function supports both artifact formats used in this project:
    a fitted sklearn pipeline, or a dictionary containing a model and the
    one-hot encoded feature columns.
    """
    missing_fields = [feature for feature in FEATURES if feature not in lead_data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    model_path = model_path or Path(__file__).with_name("pipeline.bin")
    with model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    lead_frame = pd.DataFrame([dict(lead_data)], columns=FEATURES)

    # The saved artifact may be a fitted pipeline or a model plus one-hot columns.
    if isinstance(artifact, dict):
        model = artifact["model"]
        columns = artifact.get("columns", artifact.get("feature_columns"))
        if columns is None:
            raise ValueError("The model artifact does not include feature columns.")
        model_input = pd.get_dummies(
            lead_frame,
            columns=["lead_source", "industry", "employment_status", "location"],
        ).reindex(columns=columns, fill_value=0)
        probabilities = model.predict_proba(model_input)[0]
        classes = model.classes_
    else:
        probabilities = artifact.predict_proba(lead_frame)[0]
        classes = getattr(artifact, "classes_", None)
        if classes is None:
            classes = artifact.steps[-1][1].classes_

    # The target uses 1 for converted; choose that class explicitly.
    conversion_index = list(classes).index(1)
    conversion_probability = float(probabilities[conversion_index])
    predicted_class = int(classes[probabilities.argmax()])

    return {
        "conversion_probability": conversion_probability,
        "predicted_class": predicted_class,
        "converted": predicted_class == 1,
    }


def main() -> None:
    result = predict_lead(lead)
    print(f"Conversion probability: {result['conversion_probability']:.3f}")



if __name__ == "__main__":
    main()
