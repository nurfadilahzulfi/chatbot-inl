# rf_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI FITUR
# ─────────────────────────────────────────────

FITUR = [
    "stok_rata2",       # Rata-rata stok CPO harian
    "stok_max",         # Stok CPO maksimum
    "stok_hari_aktif",  # Hari stok aktif
    "target_rkap",      # Target produksi RKAP
    "cpo_consume",      # Total CPO dikonsumsi
    "hari_olah",        # Hari pabrik beroperasi
    "yield_rbdpo",      # Yield RBDPO (%)
    "pfad_total",       # Produk samping PFAD
    "cpo_per_hari",     # Intensitas produksi per hari
    "bulan_ke",         # Bulan dalam setahun (1–12)
    "kuartal",          # Kuartal (1–4)
    "realisasi_prev",   # Realisasi bulan sebelumnya
    "cpo_prev",         # CPO consume bulan sebelumnya
    "hari_olah_prev",   # Hari olah bulan sebelumnya
]

NAMA_FITUR = [
    "Stok CPO (rata2)", "Stok CPO (maks)", "Hari Stok Aktif", "Target RKAP",
    "CPO Dikonsumsi", "Hari Olah", "Yield RBDPO (%)", "PFAD Total", "CPO per Hari",
    "Bulan ke-", "Kuartal", "Realisasi Prev", "CPO Prev", "Hari Olah Prev",
]

LABEL = "realisasi_rbdpo"

RF_PARAMS = dict(
    n_estimators  = 500,
    max_features  = "sqrt",
    random_state  = 42,
    n_jobs        = -1,
)


def _build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(**RF_PARAMS)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _interpretasi_r2(r2: float) -> str:
    if r2 >= 0.75: return "Sangat Baik — model menjelaskan >75% variansi realisasi"
    if r2 >= 0.50: return "Baik — model menjelaskan >50% variansi realisasi"
    if r2 >= 0.30: return "Cukup — model menjelaskan >30% variansi realisasi"
    if r2 >= 0.00: return "Lemah — model sedikit lebih baik dari prediksi rata-rata"
    return "Perlu peningkatan data — R² negatif"


def _interpretasi_mape(mape: float) -> str:
    if mape <= 10:  return "Sangat Akurat (MAPE ≤ 10%)"
    if mape <= 20:  return "Akurat (MAPE ≤ 20%)"
    if mape <= 30:  return "Cukup Akurat (MAPE ≤ 30%)"
    if mape <= 50:  return "Kurang Akurat (MAPE ≤ 50%)"
    return "Tidak Akurat (MAPE > 50%) — perlu data lebih banyak"


# ─────────────────────────────────────────────
# FUNGSI UTAMA
# ─────────────────────────────────────────────

def run_loocv(df: pd.DataFrame) -> tuple:
    """
    Jalankan Leave-One-Out Cross Validation.
    Return: (y_true, y_pred, metrik_dict)
    """
    X = df[FITUR].values
    y = df[LABEL].values

    loo          = LeaveOneOut()
    y_pred_list  = np.zeros(len(y))
    y_true_list  = y.copy()

    for tr_idx, te_idx in loo.split(X):
        m = _build_rf()
        m.fit(X[tr_idx], y[tr_idx])
        y_pred_list[te_idx] = m.predict(X[te_idx])

    mae  = float(mean_absolute_error(y_true_list, y_pred_list))
    rmse = float(np.sqrt(mean_squared_error(y_true_list, y_pred_list)))
    r2   = float(r2_score(y_true_list, y_pred_list))

    # MAPE hanya pada bulan produksi aktif (realisasi > 0)
    mask = y_true_list > 0
    mape = float(
        np.mean(np.abs(
            (y_true_list[mask] - y_pred_list[mask]) / (y_true_list[mask] + 1e-9)
        )) * 100
    ) if mask.sum() > 0 else 0.0

    metrik = {
        "mae"               : round(mae,  2),
        "rmse"              : round(rmse, 2),
        "r2"                : round(r2,   4),
        "mape"              : round(mape, 2),
        "jumlah_data"       : int(len(df)),
        "jumlah_imputed"    : int(df["stok_imputed"].sum()),
        "interpretasi_r2"   : _interpretasi_r2(r2),
        "interpretasi_mape" : _interpretasi_mape(mape),
    }

    return y_true_list, y_pred_list, metrik


def run_feature_importance(df: pd.DataFrame) -> list:
    """
    Train RF dengan seluruh data, hitung MDI + Permutation Importance.
    Return: list dict per fitur, diurutkan MDI descending.
    """
    X = df[FITUR].values
    y = df[LABEL].values

    model = _build_rf()
    model.fit(X, y)

    mdi  = model.feature_importances_
    perm = permutation_importance(
        model, X, y,
        n_repeats   = 50,
        random_state= 42,
        n_jobs      = -1,
    )

    results = []
    for i, nama in enumerate(NAMA_FITUR):
        results.append({
            "fitur"           : nama,
            "mdi_pct"         : round(float(mdi[i]) * 100, 2),
            "permutation"     : round(float(perm.importances_mean[i]), 6),
            "permutation_std" : round(float(perm.importances_std[i]),  6),
        })

    results.sort(key=lambda x: x["mdi_pct"], reverse=True)
    for i, r in enumerate(results):
        r["peringkat"] = i + 1

    return results


def get_loocv_predictions(df: pd.DataFrame) -> np.ndarray:
    """Kembalikan array prediksi LOOCV untuk seluruh dataset."""
    _, y_pred, _ = run_loocv(df)
    return y_pred