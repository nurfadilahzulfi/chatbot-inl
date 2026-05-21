# rf_api.py
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Lock

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rf_data_loader import load_and_preprocess
from rf_model import run_loocv, run_feature_importance
from rf_schemas import (
    RFAnalysisResponse,
    RFFeatureImportanceResponse,
    RFHistorisResponse,
    BulanHistoris,
    FeatureImportanceItem,
    EvaluasiModel,
)


# ─────────────────────────────────────────────
# CACHE GLOBAL
# ─────────────────────────────────────────────
# Hasil komputasi (LOOCV + Feature Importance) disimpan di memori.
# TTL = 2 jam. Setelah itu, request berikutnya akan hitung ulang.

_cache: dict = {}
_cache_lock  = Lock()
CACHE_TTL    = 7200   # detik (2 jam)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_or_build_cache() -> tuple:
    """
    Kembalikan (df, fi_raw, y_pred, metrik) dari cache jika masih valid.
    Jika cache kedaluwarsa atau kosong, hitung ulang dan simpan.
    Thread-safe via Lock.
    """
    with _cache_lock:
        now = time.time()
        age = now - _cache.get("ts", 0)

        if _cache and age < CACHE_TTL:
            print(f"  ⚡ Cache HIT (umur: {int(age)}s) — langsung kirim hasil")
            return (
                _cache["df"],
                _cache["fi_raw"],
                _cache["y_pred"],
                _cache["metrik"],
            )

        # Cache kosong / kedaluwarsa → hitung ulang
        print(f"\n{'='*55}")
        print(f"MEMBANGUN CACHE — {_now()}")
        print(f"{'='*55}")

        t0 = time.time()

        df = load_and_preprocess()
        print(f"Data loaded         : {len(df)} bulan")

        fi_raw = run_feature_importance(df)
        print(f"Feature importance  : selesai")

        _, y_pred, metrik = run_loocv(df)
        print(f"LOOCV selesai       : R²={metrik['r2']}  MAPE={metrik['mape']}%")

        elapsed = round(time.time() - t0, 1)
        _cache.update({
            "df":     df,
            "fi_raw": fi_raw,
            "y_pred": y_pred,
            "metrik": metrik,
            "ts":     now,
        })

        exp_time = datetime.fromtimestamp(now + CACHE_TTL).strftime("%H:%M:%S")
        print(f"Cache tersimpan     : valid hingga {exp_time} (waktu komputasi: {elapsed}s)")
        print(f"{'='*55}\n")

        return df, fi_raw, y_pred, metrik


# ─────────────────────────────────────────────
# WARM-UP SAAT STARTUP
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-compute cache saat server pertama kali dijalankan."""
    import asyncio
    print("\n[STARTUP] Pre-computing RF model — harap tunggu...")
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _get_or_build_cache)
        print("[STARTUP] Cache warm — semua endpoint siap digunakan!\n")
    except Exception as e:
        print(f"[STARTUP] Gagal pre-compute cache: {e}\n")
    yield


# ─────────────────────────────────────────────
# INISIALISASI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "RF Production Analysis API",
    description = "Analisis Feature Importance & Historis Produksi RBDPO — PT INL",
    version     = "2.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────
# HELPER BUILDER
# ─────────────────────────────────────────────

def _build_evaluasi(metrik: dict) -> EvaluasiModel:
    return EvaluasiModel(**metrik)


def _build_fi_list(fi_raw: list) -> list[FeatureImportanceItem]:
    return [FeatureImportanceItem(**item) for item in fi_raw]


def _build_historis(df, y_pred: np.ndarray) -> list[BulanHistoris]:
    result = []
    y_true = df["realisasi_rbdpo"].values

    for i, row in df.iterrows():
        aktual = float(y_true[i])
        pred   = float(y_pred[i])
        error  = (
            abs(aktual - pred) / (aktual + 1e-9) * 100
            if aktual > 0 else 0.0
        )
        result.append(BulanHistoris(
            bulan           = row["bulan"].strftime("%Y-%m"),
            bulan_label     = row["bulan"].strftime("%b %Y"),
            realisasi       = round(aktual, 2),
            target_rkap     = round(float(row["target_rkap"]), 2),
            cpo_consume     = round(float(row["cpo_consume"]), 2),
            hari_olah       = round(float(row["hari_olah"]), 1),
            yield_rbdpo     = round(float(row["yield_rbdpo"]), 4),
            stok_rata2      = round(float(row["stok_rata2"]), 2),
            prediksi_loocv  = round(pred, 2),
            error_pct       = round(error, 2),
            is_imputed      = bool(row["stok_imputed"]),
        ))
    return result


# ─────────────────────────────────────────────
# ENDPOINT 1: Analisis Lengkap
# ─────────────────────────────────────────────

@app.get(
    "/rf-analysis",
    response_model = RFAnalysisResponse,
    summary        = "Analisis RF Lengkap",
    description    = "Feature importance + historis + evaluasi LOOCV sekaligus. Hasil di-cache 2 jam.",
)
def rf_analysis():
    try:
        print(f"\n[/rf-analysis] {_now()}")
        df, fi_raw, y_pred, metrik = _get_or_build_cache()

        payload = RFAnalysisResponse(
            status             = "success",
            last_updated       = _now(),
            evaluasi           = _build_evaluasi(metrik),
            feature_importance = _build_fi_list(fi_raw),
            historis           = _build_historis(df, y_pred),
        )
        print(f"Response dikirim ke client — {_now()}")
        return payload
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────
# ENDPOINT 2: Feature Importance Saja
# ─────────────────────────────────────────────

@app.get(
    "/rf-feature-importance",
    response_model = RFFeatureImportanceResponse,
    summary        = "Feature Importance",
    description    = "Kembalikan peringkat variabel berpengaruh. Hasil dari cache.",
)
def rf_feature_importance():
    try:
        print(f"\n[/rf-feature-importance] {_now()}")
        df, fi_raw, _, _ = _get_or_build_cache()

        payload = RFFeatureImportanceResponse(
            status             = "success",
            last_updated       = _now(),
            jumlah_data        = len(df),
            feature_importance = _build_fi_list(fi_raw),
        )
        print(f"Response dikirim ke client — {_now()}")
        return payload
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────
# ENDPOINT 3: Data Historis untuk Grafik
# ─────────────────────────────────────────────

@app.get(
    "/rf-production-history",
    response_model = RFHistorisResponse,
    summary        = "Data Historis Produksi",
    description    = "Data siap pakai untuk grafik ApexCharts di Vue.js dashboard. Hasil dari cache.",
)
def rf_production_history():
    try:
        print(f"\n[/rf-production-history] {_now()}")
        df, _, y_pred, metrik = _get_or_build_cache()

        payload = RFHistorisResponse(
            status         = "success",
            last_updated   = _now(),
            categories     = df["bulan"].dt.strftime("%b %Y").tolist(),
            realisasi      = [round(float(v), 2) for v in df["realisasi_rbdpo"]],
            target_rkap    = [round(float(v), 2) for v in df["target_rkap"]],
            cpo_consume    = [round(float(v), 2) for v in df["cpo_consume"]],
            hari_olah      = [round(float(v), 1) for v in df["hari_olah"]],
            prediksi_loocv = [round(float(v), 2) for v in y_pred],
            is_imputed     = df["stok_imputed"].tolist(),
            evaluasi       = _build_evaluasi(metrik),
        )
        print(f"Response dikirim ke client — {_now()}")
        return payload
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error.")


# ─────────────────────────────────────────────
# ENDPOINT 4: Health Check
# ─────────────────────────────────────────────

@app.get(
    "/rf-health",
    summary     = "Health Check",
    description = "Cek status API, cache, dan koneksi ke server produksi.",
)
def rf_health():
    import requests as req

    api_ok     = False
    cache_info = {
        "ready": bool(_cache),
        "age_seconds": int(time.time() - _cache.get("ts", 0)) if _cache else None,
        "ttl_seconds": CACHE_TTL,
        "expires_in": max(0, int(CACHE_TTL - (time.time() - _cache.get("ts", 0)))) if _cache else 0,
    }

    try:
        r = req.get("http://103.176.66.42:9009/api/laporan-prod", timeout=5)
        api_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status"       : "online",
        "api_produksi" : "reachable" if api_ok else "unreachable",
        "cache"        : cache_info,
        "endpoints"    : [
            "/rf-health",
            "/rf-feature-importance",
            "/rf-production-history",
            "/rf-analysis",
            "/rf-invalidate-cache",
        ],
        "docs"         : "/docs",
        "timestamp"    : _now(),
    }


# ─────────────────────────────────────────────
# ENDPOINT 5: Invalidate Cache (manual refresh)
# ─────────────────────────────────────────────

@app.post(
    "/rf-invalidate-cache",
    summary     = "Invalidate Cache",
    description = "Paksa hitung ulang model di request berikutnya. Gunakan jika ada data baru.",
)
def rf_invalidate_cache():
    with _cache_lock:
        _cache.clear()
    print(f"\n[/rf-invalidate-cache] Cache dihapus — {_now()}")
    return {
        "status"  : "success",
        "message" : "Cache dihapus. Request berikutnya akan hitung ulang model.",
        "timestamp": _now(),
    }


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "rf_api:app",
        host    = "0.0.0.0",
        port    = 3001,     # Berbeda dari chatbot CPO (port 3000)
        reload  = False,
    )