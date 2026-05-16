import os
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-demand-forecast/.env")

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
MLFLOW_PATH  = f"file://{PROJECT_ROOT}/mlruns"

# Initialise FastAPI app
app = FastAPI(
    title="Demand Forecast API",
    description="Predicts next day product demand using ML",
    version="1.0.0"
)

# Load model at startup
mlflow.set_tracking_uri(MLFLOW_PATH)
# Load directly from experiment runs instead of registry
runs_path = f"{PROJECT_ROOT}/mlruns"
client = mlflow.MlflowClient(tracking_uri=MLFLOW_PATH)

# Get best run automatically
experiment = client.get_experiment_by_name("demand_forecasting")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
    max_results=1
)

best_run_id = runs[0].info.run_id
model_uri   = f"{MLFLOW_PATH}/{experiment.experiment_id}/{best_run_id}/artifacts/model"
model       = mlflow.pyfunc.load_model(model_uri)
print(f"Loaded model from run: {best_run_id}")
print("Model loaded successfully")


# ── Request/Response schemas ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    qty_lag_1:           float
    qty_lag_7:           float
    qty_lag_30:          float
    qty_rolling_avg_7:   float
    qty_rolling_avg_30:  float
    qty_rolling_std_7:   float
    daily_revenue:       float

class BatchPredictRequest(BaseModel):
    records: List[PredictRequest]

class PredictResponse(BaseModel):
    predicted_demand: float
    model_version:    str = "1"

class BatchPredictResponse(BaseModel):
    predictions:   List[float]
    record_count:  int
    model_version: str = "1"


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "model": "demand_forecast_model", "version": "1"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        data = pd.DataFrame([request.dict()])
        prediction = model.predict(data)[0]
        prediction = max(0, round(float(prediction), 2))
        return PredictResponse(predicted_demand=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    try:
        data = pd.DataFrame([r.dict() for r in request.records])
        predictions = model.predict(data)
        predictions = [max(0, round(float(p), 2)) for p in predictions]
        return BatchPredictResponse(
            predictions=predictions,
            record_count=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))