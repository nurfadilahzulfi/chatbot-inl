"""
CPO LSTM Price Prediction - FastAPI
====================================
Endpoint REST API untuk prediksi harga CPO (Crude Palm Oil)
menggunakan model LSTM dengan PyTorch.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import io
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime

from schemas import (
    TrainRequest, TrainResponse,
    PredictRequest, PredictResponse,
    ForecastResponse, MetricsResponse,
    ModelInfoResponse, HealthResponse
)
from model import CPO_LSTM
from pipeline import CPOPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("cpo_api")

# ── Global pipeline state ─────────────────────────────────────────────────────
pipeline: Optional[CPOPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pre-trained model on startup if available."""
    global pipeline
    model_path = os.getenv("MODEL_PATH", "cpo_lstm_model.pth")
    if os.path.exists(model_path):
        try:
            pipeline = CPOPipeline()
            pipeline.load_model(model_path)
            logger.info(f"✅ Pre-trained model loaded from {model_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load model: {e}")
    else:
        logger.info("ℹ️  No pre-trained model found. Use /train to train a new model.")
    yield
    logger.info("🛑 Shutting down CPO LSTM API")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CPO LSTM Price Prediction API",
    description=(
        "REST API untuk prediksi harga **Crude Palm Oil (CPO)** menggunakan "
        "model **LSTM Deep Learning** dengan PyTorch.\n\n"
        "### Alur Penggunaan\n"
        "1. `POST /train` — Upload CSV dan latih model baru\n"
        "2. `GET  /forecast` — Dapatkan prediksi 7 hari ke depan\n"
        "3. `GET  /metrics` — Lihat performa model di test set\n"
        "4. `GET  /model-info` — Lihat konfigurasi model aktif"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Cek status API dan apakah model sudah siap digunakan."""
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None and pipeline.is_trained,
        timestamp=datetime.utcnow().isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_model(
    file: UploadFile = File(..., description="File CSV data harga CPO"),
    config: str = None,   # JSON string opsional untuk override CONFIG
):
    """
    Upload file CSV dan latih model LSTM dari awal.

    **Format CSV yang diharapkan:**
    ```
    Tanggal,Terakhir,Pembukaan,Tertinggi,Terendah,Vol.,Perubahan%
    ```
    atau kolom English:
    ```
    Date,Close,Open,High,Low,Volume,Pct_Change
    ```

    **Config (opsional)** — kirim sebagai JSON string di field `config`,
    contoh: `{"epochs": 50, "lstm_hidden_size": 64}`
    """
    global pipeline

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Hanya file .csv yang diterima.")

    # Parse optional config override
    override = {}
    if config:
        try:
            override = json.loads(config)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Config bukan JSON yang valid.")

    # Read CSV bytes
    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")

    logger.info(f"📁 File diterima: {file.filename}, shape={df_raw.shape}")

    # Build & train pipeline
    try:
        pipeline = CPOPipeline(config_override=override)
        result = pipeline.train(df_raw)
    except Exception as e:
        logger.exception("Training gagal")
        raise HTTPException(status_code=500, detail=f"Training gagal: {e}")

    # Optionally save model
    save_path = os.getenv("MODEL_PATH", "cpo_lstm_model.pth")
    pipeline.save_model(save_path)
    logger.info(f"💾 Model disimpan ke {save_path}")

    return TrainResponse(
        message="Model berhasil dilatih.",
        epochs_trained=result["epochs_trained"],
        best_val_loss=result["best_val_loss"],
        train_samples=result["train_samples"],
        val_samples=result["val_samples"],
        test_samples=result["test_samples"],
        input_features=result["input_features"],
        config_used=pipeline.config,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Forecast — 7-day autoregressive
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/forecast", response_model=ForecastResponse, tags=["Prediction"])
def get_forecast(
    horizon: int = 7,
    mc_samples: int = 100,
):
    """
    Dapatkan prediksi harga CPO untuk `horizon` hari ke depan
    menggunakan autoregressive inference + MC Dropout.

    - **horizon**: jumlah hari prediksi (1–30, default 7)
    - **mc_samples**: jumlah sampel Monte Carlo untuk interval kepercayaan
    """
    _require_trained()
    if not (1 <= horizon <= 30):
        raise HTTPException(status_code=400, detail="horizon harus antara 1 dan 30.")
    if not (10 <= mc_samples <= 500):
        raise HTTPException(status_code=400, detail="mc_samples harus antara 10 dan 500.")

    try:
        result = pipeline.forecast(horizon=horizon, n_mc_samples=mc_samples)
    except Exception as e:
        logger.exception("Forecast gagal")
        raise HTTPException(status_code=500, detail=f"Forecast gagal: {e}")

    return ForecastResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
#  Predict — single-step from custom window
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_from_csv(
    file: UploadFile = File(..., description="CSV berisi setidaknya sequence_length baris terbaru"),
):
    """
    Prediksi 1-step ke depan dari data CSV yang diunggah.
    CSV harus berisi minimal `sequence_length` baris (default 60) data historis.
    Format kolom sama seperti di `/train`.
    """
    _require_trained()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Hanya file .csv yang diterima.")

    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {e}")

    try:
        result = pipeline.predict_next(df_raw)
    except Exception as e:
        logger.exception("Predict gagal")
        raise HTTPException(status_code=500, detail=f"Predict gagal: {e}")

    return PredictResponse(**result)


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
def get_metrics():
    """
    Dapatkan metrik performa model di validation set dan test set:
    MAE, RMSE, MAPE, R², Directional Accuracy.
    """
    _require_trained()
    if pipeline.metrics is None:
        raise HTTPException(status_code=404, detail="Metrik belum tersedia. Latih model terlebih dahulu.")
    return MetricsResponse(**pipeline.metrics)


# ─────────────────────────────────────────────────────────────────────────────
#  Model Info
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Lihat arsitektur dan konfigurasi model yang sedang aktif."""
    _require_trained()
    return ModelInfoResponse(
        architecture="Stacked LSTM + Attention",
        hidden_size=pipeline.config["lstm_hidden_size"],
        num_layers=pipeline.config["lstm_num_layers"],
        bidirectional=pipeline.config["bidirectional"],
        use_attention=pipeline.config["use_attention"],
        sequence_length=pipeline.config["sequence_length"],
        forecast_horizon=pipeline.config["forecast_horizon"],
        input_features=pipeline.n_features,
        total_parameters=pipeline.total_params,
        config=pipeline.config,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def _require_trained():
    if pipeline is None or not pipeline.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model belum dilatih. Gunakan POST /train terlebih dahulu.",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
