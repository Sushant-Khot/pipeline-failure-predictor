from __future__ import annotations

from typing import Any, Dict

import pandas as pd


FEATURE_COLUMNS = [
    "pipeline_name",
    "input_records",
    "output_records",
    "data_size_mb",
    "execution_time_min",
    "expected_time_min",
    "cpu_utilization",
    "memory_utilization",
    "source_delay_min",
    "retry_count",
    "previous_run_duration",
    "previous_failure_count",
    "worker_count",
    "day_of_week",
    "hour",
]


def build_feature_frame(record: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([record])
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return df[FEATURE_COLUMNS]
