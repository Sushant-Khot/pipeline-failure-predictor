from src.predict import predict_pipeline_failure
from src.preprocessing import build_feature_frame


def test_build_feature_frame_keeps_expected_columns():
    sample = {
        "pipeline_name": "sales_daily",
        "input_records": 1000000,
        "output_records": 980000,
        "data_size_mb": 1800.0,
        "execution_time_min": 35.0,
        "expected_time_min": 30,
        "cpu_utilization": 80.0,
        "memory_utilization": 92.5,
        "source_delay_min": 8.4,
        "retry_count": 2,
        "previous_run_duration": 30.5,
        "previous_failure_count": 1,
        "worker_count": 6,
        "day_of_week": "Friday",
        "hour": 5,
    }

    frame = build_feature_frame(sample)
    assert list(frame.columns) == [
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


def test_predict_pipeline_failure_returns_probability_and_label():
    sample = {
        "pipeline_name": "sales_daily",
        "input_records": 1000000,
        "output_records": 980000,
        "data_size_mb": 1800.0,
        "execution_time_min": 35.0,
        "expected_time_min": 30,
        "cpu_utilization": 80.0,
        "memory_utilization": 92.5,
        "source_delay_min": 8.4,
        "retry_count": 2,
        "previous_run_duration": 30.5,
        "previous_failure_count": 1,
        "worker_count": 6,
        "day_of_week": "Friday",
        "hour": 5,
    }

    result = predict_pipeline_failure(sample)
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0
