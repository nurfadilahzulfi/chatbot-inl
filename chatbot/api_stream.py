"""
api_stream.py — Chatbot + Forecast API (Unified)
==================================================
Menggabungkan endpoint chatbot (SmartChatbot) dengan
CPOPipeline (pipeline.py) yang sudah production-ready.

Startup flow:
  1. SmartChatbot dimuat → FAQ handler, data CSV, price_data
  2. CPOPipeline.train() → latih LSTM dari data CSV yang sama
  3. Semua endpoint siap digunakan
"""

import os
import sys
import time
import logging

# Suppress TF warnings dari sentence_transformers
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

# === PATH SETUP ===
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd
from datetime import timedelta
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import modul chatbot & pipeline baru
from main import SmartChatbot
from pipeline import CPOPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("cpo_chatbot_api")

# ── Global state ──────────────────────────────────────────────────────────────
bot: Optional[SmartChatbot]  = None
cpo_pipeline: Optional[CPOPipeline] = None
# Cache prediksi — dihitung SEKALI setelah training, dipakai semua endpoint
# Ini memastikan grafik dashboard dan chatbot menampilkan hasil yang SAMA
_forecast_cache: Optional[dict] = None   # hasil forecast(horizon=7)
_chart_cache:    Optional[dict] = None   # hasil yang sudah diformat untuk chart
# Cache data RF production — di-fetch dari rf_api.py (port 3001) saat startup
_rf_cache:       Optional[dict] = None   # hasil /rf-analysis dari port 3001


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot, cpo_pipeline

    logger.info("Memulai sistem CPO Chatbot + LSTM Pipeline...")

    # 1. Inisialisasi chatbot (load CSV, FAQ, dll)
    try:
        bot = SmartChatbot()
        logger.info("SmartChatbot siap.")
    except Exception as e:
        logger.error(f"Gagal inisialisasi chatbot: {e}")
        bot = None

    # 2. Train CPOPipeline dari RAW CSV (bukan build_ohlc_dataframe yang sudah di-parse)
    #    PENTING: pipeline._load_and_clean() harus menerima data mentah agar
    #    parsing angka Indonesia tidak dilakukan dua kali (double-parse bug)
    if bot and bot.price_data:
        try:
            logger.info("Memulai training CPOPipeline (pipeline.py)...")

            # Baca CSV langsung — pipeline akan parse sendiri via _load_and_clean()
            import pandas as pd
            from main import FILE_CSV
            try:
                df_raw = pd.read_csv(FILE_CSV, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df_raw = pd.read_csv(FILE_CSV, encoding='latin-1')
            logger.info(f"Raw CSV dimuat: {df_raw.shape[0]} baris, kolom: {list(df_raw.columns)}")


            cpo_pipeline = CPOPipeline()
            cpo_pipeline.train(df_raw)

            # Inject pipeline ke chatbot
            bot.set_pipeline(cpo_pipeline)

            # ── PRE-COMPUTE FORECAST CACHE ────────────────────────────────────
            # Hitung SEKALI, simpan hasilnya → grafik & chatbot pakai hasil SAMA
            global _forecast_cache, _chart_cache
            logger.info("Pre-computing forecast cache (100 MC samples)...")
            _forecast_cache = cpo_pipeline.forecast(horizon=7, n_mc_samples=100)
            _chart_cache    = _build_chart_payload(_forecast_cache)
            logger.info(f"Forecast cache siap. "
                        f"Last known: ${_forecast_cache['last_known_price']:.2f}, "
                        f"Day-1 pred: ${_forecast_cache['forecasts'][0]['predicted_price']:.2f}")
            logger.info("CPOPipeline training selesai & di-inject ke chatbot.")
        except Exception as e:
            logger.error(f"CPOPipeline training gagal: {e}")
            import traceback; traceback.print_exc()
            cpo_pipeline = None
    else:
        logger.warning("Tidak ada data harga — pipeline tidak aktif.")


    # ── FETCH RF PRODUCTION DATA ──────────────────────────────────────────────
    # Ambil data dari RF API dan inject ke chatbot agar chatbot
    # bisa menjawab pertanyaan produksi RBDPO via intent PRODUCTION.
    # RF API harus sudah running sebelum api_stream.py dijalankan.
    # Host & port dikontrol via env var RF_API_HOST / RF_API_PORT.
    global _rf_cache
    _rf_host = os.environ.get("RF_API_HOST", "localhost")
    _rf_port = os.environ.get("RF_API_PORT", "3001")
    _rf_url  = f"http://{_rf_host}:{_rf_port}/rf-analysis"
    try:
        import requests as _req
        logger.info(f"Mengambil data RF production dari {_rf_url} ...")
        _r = _req.get(_rf_url, timeout=60)
        if _r.status_code == 200:
            _rf_cache = _r.json()
            n_bulan   = len(_rf_cache.get("historis", []))
            if bot:
                bot.set_rf_data(_rf_cache)
            logger.info(
                f"RF data berhasil dimuat: {n_bulan} bulan historis. "
                f"Chatbot siap menjawab pertanyaan produksi RBDPO."
            )
        else:
            logger.warning(
                f"RF API merespons HTTP {_r.status_code}. "
                f"Intent PRODUCTION tidak aktif."
            )
    except Exception as _e:
        logger.warning(
            f"RF API tidak dapat dijangkau ({_rf_url}): {_e}. "
            f"Jalankan rf_api.py terlebih dahulu untuk mengaktifkan "
            f"intent produksi pada chatbot."
        )

    yield
    logger.info("Server shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CPO Chatbot & Forecast API",
    description=(
        "API gabungan chatbot INL dan prediksi harga CPO (LSTM).\n\n"
        "### Endpoints utama\n"
        "- `POST /chat` — Chatbot tanya-jawab (streaming)\n"
        "- `GET  /forecast-data` — Data grafik prediksi 7 hari\n"
        "- `GET  /forecast` — Detail forecast + confidence interval\n"
        "- `GET  /metrics` — Metrik performa model\n"
        "- `GET  /status` — Status sistem\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Build chart payload dari forecast result
# Dipanggil SEKALI saat startup, hasilnya di-cache
# ─────────────────────────────────────────────────────────────────────────────
def _build_chart_payload(forecast_result: dict) -> dict:
    """Konversi forecast() result → format Chart.js untuk /forecast-data.

    Sekarang juga menyertakan data backtest (prediksi vs aktual pada
    periode historis) agar grafik bisa menampilkan overlay kedua garis.
    """
    if not bot or not bot.price_data:
        return {"status": "no_data", "categories": [], "actual": [], "prediction": []}

    forecasts    = forecast_result["forecasts"]
    predictions  = [f["predicted_price"] for f in forecasts]
    dates_future = [f["date"]             for f in forecasts]
    lower_90     = [f["lower_90"]         for f in forecasts]
    upper_90     = [f["upper_90"]         for f in forecasts]
    lower_50     = [f["lower_50"]         for f in forecasts]
    upper_50     = [f["upper_50"]         for f in forecasts]

    # Tampilkan semua data aktual dari tahun 2025 ke atas
    from datetime import datetime
    cutoff_date = datetime(2025, 1, 1)
    last_n        = [item for item in bot.price_data if item['date'] >= cutoff_date]
    if not last_n:
        # Fallback ke 30 hari terakhir jika belum ada data 2025
        last_n = bot.price_data[-30:]
    dates_actual  = [item['date_str'] for item in last_n]
    prices_actual = [item['price']    for item in last_n]
    n_actual      = len(prices_actual)
    n_pred        = len(predictions)

    final_categories = dates_actual + dates_future
    final_actual     = prices_actual + [None] * n_pred

    # Bridge: sambungkan garis aktual ke prediksi
    bridge = [None] * (n_actual - 1) + [prices_actual[-1]]
    final_prediction = bridge + predictions
    final_lower_90   = bridge + lower_90
    final_upper_90   = bridge + upper_90
    final_lower_50   = bridge + lower_50
    final_upper_50   = bridge + upper_50

    # ── Backtest overlay: prediksi model di area historis ──────────────────
    # Buat array backtest_predicted yang sejajar dgn final_categories
    # Hanya isi di tanggal yang ada di backtest, sisanya None
    backtest_overlay = [None] * len(final_categories)
    if cpo_pipeline and cpo_pipeline.backtest_dates:
        bt_map = dict(zip(cpo_pipeline.backtest_dates, cpo_pipeline.backtest_predicted))
        for i, dt in enumerate(final_categories):
            if dt in bt_map:
                backtest_overlay[i] = bt_map[dt]

    # Data backtest mentah (untuk grafik terpisah jika dibutuhkan)
    backtest_raw = {}
    if cpo_pipeline and cpo_pipeline.backtest_dates:
        backtest_raw = {
            "dates"     : cpo_pipeline.backtest_dates,
            "actual"    : cpo_pipeline.backtest_actual,
            "predicted" : cpo_pipeline.backtest_predicted,
        }

    return {
        "status"     : "ok",
        "categories" : final_categories,
        "actual"     : final_actual,
        "prediction" : final_prediction,
        "lower_90"   : final_lower_90,
        "upper_90"   : final_upper_90,
        "lower_50"   : final_lower_50,
        "upper_50"   : final_upper_50,
        "last_known" : forecast_result["last_known_price"],
        "mc_samples" : forecast_result["mc_samples_used"],
        # Backtest overlay — prediksi model pada tanggal historis
        "backtest_overlay" : backtest_overlay,
        "backtest"         : backtest_raw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHATBOT (STREAMING)
# ─────────────────────────────────────────────────────────────────────────────
def _stream_text(text: str):
    yield ""
    for word in text.split():
        yield word + " "
        time.sleep(0.03)


@app.post("/chat", tags=["Chatbot"])
async def chat_stream(req: ChatRequest):
    """Tanya-jawab dengan Sobat INL (streaming per kata)."""
    if not bot:
        return StreamingResponse(_stream_text("Server Error: Bot tidak siap."), media_type="text/plain")
    try:
        # Inject cached forecast ke chatbot agar hasilnya SAMA dengan grafik
        answer = bot.get_response(req.question, cached_forecast=_forecast_cache)
        return StreamingResponse(_stream_text(answer), media_type="text/plain")
    except Exception as e:
        return StreamingResponse(_stream_text(f"Error: {str(e)}"), media_type="text/plain")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FORECAST DATA (untuk grafik dashboard) — pakai cache
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/forecast-data", tags=["Forecast"])
def get_forecast_data():
    """
    Data JSON untuk grafik Chart.js di dashboard.
    Menggunakan hasil cache yang dihitung sekali saat startup
    → grafik dan chatbot selalu menampilkan angka yang SAMA.
    """
    if not bot or not bot.price_data:
        return {"status": "no_data", "categories": [], "actual": [], "prediction": []}
    if not cpo_pipeline or not cpo_pipeline.is_trained:
        status = "training" if cpo_pipeline else "unavailable"
        return {"status": status, "categories": [], "actual": [], "prediction": []}
    if _chart_cache is None:
        return {"status": "computing", "categories": [], "actual": [], "prediction": []}
    return _chart_cache


# ─────────────────────────────────────────────────────────────────────────────
# 3. FORECAST DETAIL — pakai cache
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/forecast", tags=["Forecast"])
def get_forecast():
    """
    Prediksi harga CPO lengkap dengan confidence interval.
    Menggunakan cache yang sama dengan grafik dashboard.
    """
    if not cpo_pipeline or not cpo_pipeline.is_trained:
        return JSONResponse(status_code=503, content={"detail": "Model belum siap."})
    if _forecast_cache is None:
        return JSONResponse(status_code=503, content={"detail": "Forecast cache belum siap."})
    return _forecast_cache


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/metrics", tags=["Evaluation"])
def get_metrics():
    """Metrik performa model: MAE, RMSE, MAPE, R², Directional Accuracy."""
    if not cpo_pipeline or not cpo_pipeline.is_trained:
        return JSONResponse(status_code=503, content={"detail": "Model belum siap."})
    if cpo_pipeline.metrics is None:
        return JSONResponse(status_code=404, content={"detail": "Metrik belum tersedia."})
    return cpo_pipeline.metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. STATUS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/status", tags=["System"])
def get_status():
    """Cek status seluruh sistem."""
    chatbot_ok  = bot is not None
    pipeline_ok = cpo_pipeline is not None and cpo_pipeline.is_trained
    rf_ok       = _rf_cache is not None

    return {
        "chatbot_ready"   : chatbot_ok,
        "pipeline_ready"  : pipeline_ok,
        "rf_data_ready"   : rf_ok,
        "rf_bulan_data"   : len(_rf_cache.get("historis", [])) if rf_ok else 0,
        "data_rows"       : len(bot.price_data) if chatbot_ok else 0,
        "input_features"  : cpo_pipeline.n_features if pipeline_ok else 0,
        "total_params"    : cpo_pipeline.total_params if pipeline_ok else 0,
        "metrics_available": cpo_pipeline.metrics is not None if pipeline_ok else False,
        "message"         : (
            "Sistem siap (LSTM + RF Production)"
            if (chatbot_ok and pipeline_ok and rf_ok)
            else "Sistem siap (RF Production tidak aktif)"
            if (chatbot_ok and pipeline_ok)
            else "Sistem belum sepenuhnya siap"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api_stream:app", host="0.0.0.0", port=3000, reload=False)