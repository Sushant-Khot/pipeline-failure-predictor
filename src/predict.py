from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

from src.preprocessing import build_feature_frame

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "pipeline_failure_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"


def load_model_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with PREPROCESSOR_PATH.open("rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    with FEATURE_NAMES_PATH.open("rb") as feature_file:
        feature_names = pickle.load(feature_file)
    return model, preprocessor, feature_names


def predict_pipeline_failure(record: Dict[str, Any]) -> Dict[str, Any]:
    frame = build_feature_frame(record)
    model, preprocessor, feature_names = load_model_artifacts()

    transformed = preprocessor.transform(frame)
    probability = model.predict_proba(transformed)[:, 1][0]
    prediction = int(model.predict(transformed)[0])

    return {
        "prediction": prediction,
        "probability": float(probability),
        "risk_level": "High risk" if probability >= 0.5 else "Low risk",
        "feature_names": feature_names,
    }
