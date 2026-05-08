# rf_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

FITUR = [
    "stok_rata2", "stok_max", "stok_hari_aktif", "stok_std",
    "target_produksi", "realisasi_prev", "bulan_ke", "kuartal"
]

NAMA_FITUR = [
    "Stok CPO (rata-rata)", "Stok CPO (maks)", "Hari Stok Aktif", "Stok CPO (std)",
    "Target Produksi", "Realisasi Bulan Lalu", "Bulan ke-", "Kuartal"
]

LABEL = "realisasi_produksi"


def _build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )


def run_loocv(df: pd.DataFrame):
    """
    Jalankan Leave-One-Out Cross Validation.
    Kembalikan (y_true, y_pred, metrik_dict).
    """
    X = df[FITUR].values
    y = df[LABEL].values

    loo = LeaveOneOut()
    y_pred_list, y_true_list = [], []

    for tr_idx, te_idx in loo.split(X):
        model = _build_rf()
        model.fit(X[tr_idx], y[tr_idx])
        y_pred_list.append(model.predict(X[te_idx])[0])
        y_true_list.append(y[te_idx][0])

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))

    # MAPE hanya pada bulan yang realisasi > 0
    mask = y_true > 0
    mape = float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-9))) * 100
    ) if mask.sum() > 0 else 0.0

    metrik = {
        "mae"             : round(mae,  2),
        "rmse"            : round(rmse, 2),
        "r2"              : round(r2,   4),
        "mape"            : round(mape, 2),
        "jumlah_data"     : int(len(df)),
        "jumlah_imputasi" : int(df["is_imputed"].sum()),
    }

    return y_true, y_pred, metrik


def run_feature_importance(df: pd.DataFrame) -> list:
    """
    Train RF dengan seluruh data, hitung MDI + Permutation Importance.
    Kembalikan list dict per fitur, diurutkan MDI descending.
    """
    X = df[FITUR].values
    y = df[LABEL].values

    model = _build_rf()
    model.fit(X, y)

    # MDI
    mdi = model.feature_importances_

    # Permutation
    perm = permutation_importance(
        model, X, y,
        n_repeats=50,
        random_state=42,
        n_jobs=-1
    )

    results = []
    for i, nama in enumerate(NAMA_FITUR):
        results.append({
            "fitur"           : nama,
            "kontribusi_pct"  : round(float(mdi[i]) * 100, 2),
            "permutation"     : round(float(perm.importances_mean[i]), 6),
            "permutation_std" : round(float(perm.importances_std[i]),  6),
        })

    # Urutkan berdasarkan MDI descending
    results.sort(key=lambda x: x["kontribusi_pct"], reverse=True)
    return results


def get_loocv_predictions(df: pd.DataFrame) -> list:
    """
    Kembalikan prediksi LOOCV per baris untuk ditampilkan di grafik.
    """
    _, y_pred, _ = run_loocv(df)
    return y_pred.tolist()