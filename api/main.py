"""
Customer Churn Prediction — FastAPI Application
Endpoints:
  GET  /health         → health check
  GET  /model/info     → model metadata
  POST /predict        → single prediction
  POST /predict/batch  → batch prediction
  GET  /metrics        → basic metrics
"""
import os, pickle, time, logging
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="End-to-End MLOps Pipeline — Predict customer churn probability",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(BASE, "..", "models", "best_model.pkl")
SCALER_PATH  = os.path.join(BASE, "..", "models", "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE, "..", "models", "feature_columns.pkl")

model = scaler = feature_columns = None
_start  = time.time()
_preds  = 0

@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_columns
    try:
        with open(MODEL_PATH,   "rb") as f: model           = pickle.load(f)
        with open(SCALER_PATH,  "rb") as f: scaler          = pickle.load(f)
        with open(COLUMNS_PATH, "rb") as f: feature_columns = pickle.load(f)
        logger.info(f"✅ Model loaded: {type(model).__name__}")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not found: {e} — run notebooks 01-04 first.")

# ── Schemas ──────────────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    gender:           int   = Field(..., ge=0, le=1,   example=1)
    SeniorCitizen:    int   = Field(..., ge=0, le=1,   example=0)
    Partner:          int   = Field(..., ge=0, le=1,   example=1)
    Dependents:       int   = Field(..., ge=0, le=1,   example=0)
    tenure:           int   = Field(..., ge=0, le=72,  example=2)
    PhoneService:     int   = Field(..., ge=0, le=1,   example=1)
    PaperlessBilling: int   = Field(..., ge=0, le=1,   example=1)
    MonthlyCharges:   float = Field(..., ge=0, le=200, example=85.5)
    TotalCharges:     float = Field(..., ge=0,         example=171.0)

class BatchInput(BaseModel):
    customers: List[CustomerInput]

class PredictionOut(BaseModel):
    churn_prediction:  int
    churn_probability: float
    risk_level:        str
    recommendation:    str

class BatchOut(BaseModel):
    total:       int
    predictions: List[PredictionOut]

# ── Helper ───────────────────────────────────────────────────────────────────
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
RECS = {
    "HIGH":   "Immediate retention action — offer discount or loyalty bonus.",
    "MEDIUM": "Monitor customer — consider proactive outreach.",
    "LOW":    "Customer appears stable — continue regular engagement.",
}

def _predict(c: CustomerInput) -> PredictionOut:
    global _preds
    if model is None:
        raise HTTPException(503, "Model not loaded. Run training notebooks first.")
    row = pd.DataFrame([c.dict()])
    row[NUM_COLS] = scaler.transform(row[NUM_COLS])
    if feature_columns:
        for col in feature_columns:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_columns]
    prob = float(model.predict_proba(row)[0, 1])
    pred = int(prob >= 0.40)
    risk = "HIGH" if prob > 0.70 else "MEDIUM" if prob > 0.40 else "LOW"
    _preds += 1
    return PredictionOut(churn_prediction=pred, churn_probability=round(prob, 4),
                         risk_level=risk, recommendation=RECS[risk])

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "healthy", "model_loaded": model is not None,
            "uptime_seconds": round(time.time() - _start, 1)}

@app.get("/model/info", tags=["System"])
def model_info():
    if not model: raise HTTPException(503, "Model not loaded.")
    return {"model_type": type(model).__name__,
            "n_features": len(feature_columns) if feature_columns else "unknown",
            "features": feature_columns}

@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(c: CustomerInput):
    """Predict churn for a single customer."""
    logger.info(f"Single predict | tenure={c.tenure} | monthly={c.MonthlyCharges}")
    return _predict(c)

@app.post("/predict/batch", response_model=BatchOut, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """Predict churn for multiple customers."""
    logger.info(f"Batch predict | n={len(batch.customers)}")
    preds = [_predict(c) for c in batch.customers]
    return BatchOut(total=len(preds), predictions=preds)

@app.get("/metrics", tags=["System"])
def metrics():
    return {"uptime_seconds": round(time.time() - _start, 1),
            "total_predictions": _preds, "model_loaded": int(model is not None)}
