# rf_api.py
import traceback
from datetime import datetime
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from rf_data_loader import load_and_preprocess
from rf_model import run_loocv, run_feature_importance, get_loocv_predictions
from rf_schemas import (
    RFAnalysisResponse,
    ProductionHistoryResponse,
    BulanData,
    FeatureItem,
    EvaluasiMetrik,
)

app = FastAPI(
    title="RF Production Analysis API",
    description="Analisis Produksi RBDPO dengan Random Forest — PT INL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _build_dataset_list(df: pd.DataFrame) -> list[BulanData]:
    result = []
    for _, row in df.iterrows():
        result.append(BulanData(
            bulan               = row["bulan"].strftime("%Y-%m"),
            realisasi_produksi  = round(float(row["realisasi_produksi"]), 2),
            target_produksi     = round(float(row["target_produksi"]), 2),
            stok_cpo            = round(float(row["stok_rata2"]), 2),
            is_imputed          = bool(row["is_imputed"]),
        ))
    return result


def _build_feature_list(fi_raw: list) -> list[FeatureItem]:
    return [
        FeatureItem(
            fitur            = item["fitur"],
            kontribusi_pct   = item["kontribusi_pct"],
            permutation      = item["permutation"],
            permutation_std  = item["permutation_std"],
        )
        for item in fi_raw
    ]


# ─────────────────────────────────────────────
# CACHE & STARTUP
# ─────────────────────────────────────────────

_CACHE = {
    "timestamp": 0,
    "df": None,
    "fi_raw": None,
    "y_pred_loocv": None,
    "metrik": None
}
CACHE_TTL_SEC = 3600

def _get_cached_data():
    now = time.time()
    if _CACHE["df"] is not None and (now - _CACHE["timestamp"] < CACHE_TTL_SEC):
        return _CACHE["df"], _CACHE["fi_raw"], _CACHE["y_pred_loocv"], _CACHE["metrik"]
    
    print("\n [CACHE MISS] Mengambil data dan melatih model...")
    df = load_and_preprocess()
    fi_raw = run_feature_importance(df)
    _, y_pred, metrik = run_loocv(df)
    
    _CACHE["df"] = df
    _CACHE["fi_raw"] = fi_raw
    _CACHE["y_pred_loocv"] = y_pred.tolist()
    _CACHE["metrik"] = metrik
    _CACHE["timestamp"] = now
    
    return _CACHE["df"], _CACHE["fi_raw"], _CACHE["y_pred_loocv"], _CACHE["metrik"]

@app.on_event("startup")
def startup_event():
    print("Pre-warming cache...")
    try:
        _get_cached_data()
        print("Cache pre-warmed.")
    except Exception as e:
        print("Gagal pre-warm cache:", e)


# ─────────────────────────────────────────────
# ENDPOINT 1: Analisis Lengkap RF
# ─────────────────────────────────────────────

@app.get("/rf-analysis", response_model=RFAnalysisResponse)
def rf_analysis():
    """
    Endpoint utama.
    Mengembalikan:
    - Dataset lengkap (21 bulan, dengan flag imputasi)
    - Feature importance (MDI + Permutation)
    - Metrik evaluasi LOOCV (MAE, RMSE, R², MAPE)
    """
    try:
        print("\n [/rf-analysis] Request masuk...")
        df, fi_raw, _, metrik = _get_cached_data()

        return RFAnalysisResponse(
            status             = "success",
            dataset            = _build_dataset_list(df),
            feature_importance = _build_feature_list(fi_raw),
            evaluasi           = EvaluasiMetrik(**metrik),
            last_updated       = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────
# ENDPOINT 2: Data Historis + Prediksi LOOCV
# ─────────────────────────────────────────────

@app.get("/rf-production-history", response_model=ProductionHistoryResponse)
def rf_production_history():
    """
    Endpoint untuk grafik di dashboard.
    Mengembalikan:
    - Sumbu X: label bulan
    - Realisasi aktual
    - Target RKAP
    - Prediksi LOOCV (untuk perbandingan visual)
    - Stok CPO rata-rata
    - Flag is_imputed per bulan
    """
    try:
        print("\n [/rf-production-history] Request masuk...")
        df, _, y_pred_loocv, _ = _get_cached_data()

        categories     = df["bulan"].dt.strftime("%b %Y").tolist()
        realisasi      = [round(float(v), 2) for v in df["realisasi_produksi"]]
        target         = [round(float(v), 2) for v in df["target_produksi"]]
        stok           = [round(float(v), 2) for v in df["stok_rata2"]]
        prediksi_loocv = [round(float(v), 2) for v in y_pred_loocv]
        is_imputed     = df["is_imputed"].tolist()

        return ProductionHistoryResponse(
            status          = "success",
            categories      = categories,
            realisasi       = realisasi,
            target          = target,
            prediksi_loocv  = prediksi_loocv,
            stok_cpo        = stok,
            is_imputed      = is_imputed,
            last_updated    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────
# ENDPOINT 3: Feature Importance Saja (Ringan)
# ─────────────────────────────────────────────

@app.get("/rf-feature-importance")
def rf_feature_importance():
    """
    Endpoint ringan — hanya kembalikan feature importance.
    Cocok untuk widget kecil di dashboard tanpa load data besar.
    """
    try:
        print("\n [/rf-feature-importance] Request masuk...")
        df, fi_raw, _, _ = _get_cached_data()

        return {
            "status"            : "success",
            "feature_importance": fi_raw,
            "jumlah_data"       : len(df),
            "jumlah_imputasi"   : int(df["is_imputed"].sum()),
            "last_updated"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────
# ENDPOINT 4: Health Check
# ─────────────────────────────────────────────

@app.get("/rf-health")
def rf_health():
    """Cek apakah API aktif dan API produksi bisa dijangkau."""
    import requests as req
    api_ok = False
    try:
        r = req.get("http://103.176.66.42:9009/api/laporan-prod", timeout=5)
        api_ok = r.status_code == 200
    except:
        pass

    return {
        "status"          : "online",
        "api_produksi"    : "reachable" if api_ok else "unreachable",
        "timestamp"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "rf_api:app",
        host="0.0.0.0",
        port=3001,          # Port berbeda dari api_stream.py (3000)
        reload=False,
    )