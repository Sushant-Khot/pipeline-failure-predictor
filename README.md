# Pipeline Failure Predictor

This project predicts whether a data pipeline run is likely to fail based on operational telemetry such as runtime, CPU and memory usage, retries, delays, and historical failure signals.

## Project structure

- `data/pipeline_runs.csv` — training dataset
- `notebooks/pipeline_failure_prediction.ipynb` — exploratory notebook with modeling experiments
- `src/preprocessing.py` — feature schema helper
- `src/train.py` — model training script
- `src/predict.py` — prediction entry point using saved model artifacts
- `app/app.py` — Streamlit web app
- `model/` — trained model artifacts and feature metadata
- `app/` — Streamlit web application

## Setup

1. Create and activate a virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python src/train.py
   ```
4. Run the app:
   ```bash
   streamlit run app/app.py
   ```

## Example prediction request

```python
from src.predict import predict_pipeline_failure

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

print(predict_pipeline_failure(sample))
```
