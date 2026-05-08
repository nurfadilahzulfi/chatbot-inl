# rf_data_loader.py
import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import traceback

BASE_URL   = "http://103.193.145.61:9009/api"
URL_STOK   = f"{BASE_URL}/stock-cpo"
URL_LAPORAN= f"{BASE_URL}/laporan-prod"
URL_TARGET = f"{BASE_URL}/target-prod"

TIMEOUT = 120

def _fetch(url: str, nama: str) -> list:
    """Ambil data dari satu endpoint API."""
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        records = data["data"] if isinstance(data, dict) and "data" in data else data
        print(f"   {nama}: {len(records)} records")
        return records
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Gagal konek ke API ({nama}). Pastikan server aktif.")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"API timeout ({nama}).")
    except Exception as e:
        raise RuntimeError(f"Error fetch {nama}: {e}")


def _build_realisasi(records: list) -> dict:
    """Agregasi RBDPO bahan_olah per bulan."""
    result = defaultdict(float)
    for x in records:
        try:
            item = x.get("item_produksi", {})
            if item.get("name") == "RBDPO" and item.get("kategori") == "bahan_olah":
                bulan = x["tanggal"][:7]
                result[bulan] += float(x["qty"])
        except:
            continue
    return dict(result)


def _build_target(records: list) -> dict:
    """Agregasi RKAP per bulan."""
    result = defaultdict(float)
    for x in records:
        try:
            if x.get("uraian", {}).get("nama") == "RKAP":
                bulan = x["tanggal"][:7]
                result[bulan] += float(x["value"])
        except:
            continue
    return dict(result)


def _build_stok(records: list) -> dict:
    """Agregasi stok CPO harian → fitur bulanan."""
    raw = defaultdict(list)
    for x in records:
        try:
            bulan = x["tanggal"][:7]
            qty   = float(x["qty"])
            raw[bulan].append(qty)
        except:
            continue

    result = {}
    for bulan, values in raw.items():
        arr = np.array(values)
        result[bulan] = {
            "stok_rata2"      : float(np.mean(arr)),
            "stok_max"        : float(np.max(arr)),
            "stok_min"        : float(np.min(arr)),
            "stok_std"        : float(np.std(arr)),
            "stok_hari_aktif" : int(np.sum(arr > 0)),
        }
    return result


def load_and_preprocess() -> pd.DataFrame:
    """
    Pipeline lengkap:
    1. Fetch 3 API
    2. Agregasi per bulan
    3. Outer join + imputasi stok
    4. Tambah fitur engineering
    Kembalikan DataFrame siap pakai.
    """
    print(" Mengambil data dari API...")

    # 1. Fetch
    raw_laporan = _fetch(URL_LAPORAN, "Laporan Produksi")
    raw_target  = _fetch(URL_TARGET,  "Target Produksi")
    raw_stok    = _fetch(URL_STOK,    "Stok CPO")

    # 2. Agregasi
    realisasi_d = _build_realisasi(raw_laporan)
    target_d    = _build_target(raw_target)
    stok_d      = _build_stok(raw_stok)

    # 3. Bangun DataFrame base (inner join realisasi & target)
    bulan_base = sorted(set(realisasi_d) & set(target_d))
    if not bulan_base:
        raise RuntimeError("Tidak ada bulan yang overlap antara realisasi dan target.")

    rows = []
    for b in bulan_base:
        row = {
            "bulan"              : b,
            "realisasi_produksi" : realisasi_d[b],
            "target_produksi"    : target_d[b],
        }
        # Stok: ada → pakai data asli, tidak ada → NaN (akan diimputasi)
        if b in stok_d:
            row.update(stok_d[b])
            row["is_imputed"] = False
        else:
            row.update({
                "stok_rata2"      : np.nan,
                "stok_max"        : np.nan,
                "stok_min"        : np.nan,
                "stok_std"        : np.nan,
                "stok_hari_aktif" : np.nan,
            })
            row["is_imputed"] = True
        rows.append(row)

    df = pd.DataFrame(rows)
    df["bulan"] = pd.to_datetime(df["bulan"])
    df = df.sort_values("bulan").reset_index(drop=True)

    # 4. Imputasi interpolasi linear untuk kolom stok
    stok_cols = ["stok_rata2", "stok_max", "stok_min", "stok_std", "stok_hari_aktif"]
    for col in stok_cols:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    df["stok_std"] = df["stok_std"].fillna(0)

    # 5. Feature engineering tambahan
    df["bulan_ke"]       = df["bulan"].dt.month
    df["kuartal"]        = df["bulan"].dt.quarter
    df["realisasi_prev"] = df["realisasi_produksi"].shift(1).fillna(0)

    n_total    = len(df)
    n_imputed  = int(df["is_imputed"].sum())
    print(f"   Dataset siap: {n_total} bulan ({n_imputed} diimputasi, {n_total - n_imputed} asli)")

    return df