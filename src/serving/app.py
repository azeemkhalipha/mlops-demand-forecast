import sys
import os

# Works both locally and inside Docker
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
from model_loader import model_loader

FEATURE_COLS = [
    "qty_lag_1", "qty_lag_7", "qty_lag_30",
    "qty_rolling_avg_7", "qty_rolling_avg_30",
    "qty_rolling_std_7", "daily_revenue"
]

class PredictionRequest(BaseModel):
    qty_lag_1:           float
    qty_lag_7:           float
    qty_lag_30:          float
    qty_rolling_avg_7:   float
    qty_rolling_avg_30:  float
    qty_rolling_std_7:   float
    daily_revenue:       float

class PredictionResponse(BaseModel):
    predicted_quantity: float
    model_version:      str
    status:             str

class BatchPredictionRequest(BaseModel):
    inputs: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    count:       int
    status:      str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — loading model...")
    model_loader.load()
    yield
    print("Shutting down...")

app = FastAPI(
    title="Demand Forecast API",
    description="MLOps demand forecasting model serving API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status":        "healthy",
        "model_loaded":  model_loader.is_loaded,
        "model_version": model_loader.model_version or "none"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features   = {col: getattr(request, col) for col in FEATURE_COLS}
        prediction = model_loader.predict(features)
        return PredictionResponse(
            predicted_quantity=prediction,
            model_version=model_loader.model_version,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest):
    try:
        features_list = [
            {col: getattr(item, col) for col in FEATURE_COLS}
            for item in request.inputs
        ]
        predictions = model_loader.batch_predict(features_list)
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Demand Forecast API is running",
        "docs":    "/docs",
        "health":  "/health"
    }
