"""UI-support analytics: dataset KPIs, real feature-importance risk drivers,
a documented health-score heuristic, and condition-based recommendations.

Nothing here changes the trained model, the preprocessing pipeline, or the
prediction logic in predict.py. Everything is either:
  (a) read directly from the real dataset / real model artifacts, or
  (b) an explicitly documented UI-level heuristic (health score, risk
      category, recommendations) layered on top of the real prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.predict import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    load_model_artifacts,
)
from src.preprocessing import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "pipeline_runs.csv"
METRICS_PATH = MODEL_PATH.parent / "metrics.json"
TARGET_COLUMN = "pipeline_failed"

RISK_THRESHOLDS = [
    (0.25, "LOW"),
    (0.50, "MEDIUM"),
    (0.75, "HIGH"),
    (1.01, "CRITICAL"),
]


def get_risk_category(probability: float) -> str:
    """UI-level risk banding on top of the model's probability output.
    Thresholds are a display convention, not part of the trained model.
    """
    for cutoff, label in RISK_THRESHOLDS:
        if probability < cutoff:
            return label
    return "CRITICAL"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_dataset_predictions() -> pd.DataFrame:
    """Runs the real trained model over the historical dataset once, so KPI
    cards and analytics charts reflect actual model behavior instead of
    invented numbers.
    """
    df = load_dataset()
    model, preprocessor, _ = load_model_artifacts()
    transformed = preprocessor.transform(df[FEATURE_COLUMNS])
    df = df.copy()
    df["predicted_probability"] = model.predict_proba(transformed)[:, 1]
    df["predicted_failure"] = model.predict(transformed)
    return df


def compute_kpis(df_with_preds: pd.DataFrame) -> Dict[str, Any]:
    total_runs = len(df_with_preds)
    actual_failure_rate = df_with_preds[TARGET_COLUMN].mean() if total_runs else 0.0
    high_risk_runs = int((df_with_preds["predicted_probability"] >= 0.5).sum())
    avg_execution_time = df_with_preds["execution_time_min"].mean() if total_runs else 0.0
    return {
        "total_runs": total_runs,
        "failure_rate": float(actual_failure_rate),
        "high_risk_runs": high_risk_runs,
        "avg_execution_time": float(avg_execution_time),
    }


def _humanize(raw_name: str) -> str:
    name = raw_name.split("__", 1)[-1]
    name = name.replace("_", " ").strip()
    return name.title()


def get_risk_drivers(top_n: int = 6) -> List[Dict[str, Any]]:
    """Real, model-derived feature importances (RandomForest Gini importance),
    aggregated from one-hot-encoded columns back to their source feature.
    Not fabricated — this reads model.feature_importances_ directly.
    """
    model, preprocessor, _ = load_model_artifacts()
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importances = model.feature_importances_
    grouped: Dict[str, float] = {}
    for raw_name, importance in zip(feature_names, importances):
        base = raw_name.split("__", 1)[-1]
        for cat_col in ("pipeline_name", "day_of_week"):
            if base.startswith(cat_col + "_"):
                base = cat_col
                break
        grouped[base] = grouped.get(base, 0.0) + float(importance)

    ranked = sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    max_value = ranked[0][1] if ranked else 1.0
    return [
        {
            "feature": _humanize(name),
            "importance": value,
            "relative": value / max_value if max_value else 0.0,
        }
        for name, value in ranked
    ]


def compute_health_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Documented UI-level heuristic (NOT a model output). Combines
    operational signals the user entered into a single 0-100 score so the
    dashboard has a quick-glance health indicator alongside the ML
    prediction. Weights are fixed and transparent, listed inline below.
    """
    score = 100.0

    cpu = payload.get("cpu_utilization", 0.0)
    if cpu > 85:
        score -= 20
    elif cpu > 70:
        score -= 10

    memory = payload.get("memory_utilization", 0.0)
    if memory > 85:
        score -= 20
    elif memory > 70:
        score -= 10

    expected = payload.get("expected_time_min", 0) or 1
    execution = payload.get("execution_time_min", 0.0)
    deviation_ratio = execution / expected
    if deviation_ratio > 1.5:
        score -= 20
    elif deviation_ratio > 1.2:
        score -= 10

    retries = payload.get("retry_count", 0)
    score -= min(retries * 5, 15)

    previous_failures = payload.get("previous_failure_count", 0)
    score -= min(previous_failures * 5, 15)

    source_delay = payload.get("source_delay_min", 0.0)
    if source_delay > 15:
        score -= 10
    elif source_delay > 10:
        score -= 5

    score = max(0.0, min(100.0, score))

    if score >= 90:
        category = "Healthy"
    elif score >= 70:
        category = "Monitor"
    elif score >= 50:
        category = "Warning"
    else:
        category = "Critical"

    return {"score": round(score, 1), "category": category}


def build_recommendations(payload: Dict[str, Any], risk_drivers: List[Dict[str, Any]]) -> List[str]:
    """Condition-based recommendations derived from the actual submitted
    values (and, for input volume, the actual historical dataset average).
    No invented operational facts.
    """
    recs: List[str] = []

    memory = payload.get("memory_utilization", 0.0)
    if memory > 80:
        recs.append(f"Investigate elevated memory utilization (currently {memory:.1f}%).")

    cpu = payload.get("cpu_utilization", 0.0)
    if cpu > 80:
        recs.append(f"Investigate elevated CPU utilization (currently {cpu:.1f}%).")

    expected = payload.get("expected_time_min", 0) or 1
    execution = payload.get("execution_time_min", 0.0)
    if execution > expected * 1.2:
        pct = (execution / expected - 1) * 100
        recs.append(f"Review runtime deviation — execution exceeded expected time by {pct:.0f}%.")

    source_delay = payload.get("source_delay_min", 0.0)
    if source_delay > 10:
        recs.append(f"Verify source-file arrival timing — observed delay is {source_delay:.1f} minutes.")

    try:
        avg_input = load_dataset()["input_records"].mean()
        input_records = payload.get("input_records", 0)
        if avg_input and input_records > avg_input * 1.5:
            recs.append(
                f"Check whether today's input volume is abnormal ({input_records:,} vs. "
                f"a historical average of {avg_input:,.0f} records)."
            )
    except Exception:
        pass

    if payload.get("retry_count", 0) > 0:
        recs.append(f"Investigate retry causes — {payload['retry_count']} retries recorded for this run.")

    if payload.get("previous_failure_count", 0) > 0:
        recs.append(
            f"Review recent failure history — {payload['previous_failure_count']} prior "
            "failures recorded for this pipeline."
        )

    if risk_drivers:
        top_two = ", ".join(d["feature"] for d in risk_drivers[:2])
        recs.append(f"Prioritize mitigation on the model's top risk drivers: {top_two}.")

    if not recs:
        recs.append("No elevated risk signals detected in the submitted operational metrics.")

    return recs


def load_model_metrics() -> Dict[str, Any] | None:
    """Loads metrics.json written by train.py. Returns None if training
    hasn't been re-run since this feature was added — the UI must show an
    explicit 'not yet computed' state rather than fabricate numbers.
    """
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open("r") as f:
        return json.load(f)
