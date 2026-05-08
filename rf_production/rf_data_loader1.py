# rf_data_loader.py
# Versi: baca dari file JSON lokal (tanpa API)

import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# PATH FILE JSON
# File JSON berada di folder 'data' yang
# sejajar dengan folder 'rf_production'
#
# Struktur folder yang diharapkan:
#   project/
#   ├── data/
#   │   ├── laporan produksi.json
#   │   ├── target produksi.json
#   │   └── stok cpo.json
#   └── rf_production/
#       ├── rf_api.py
#       ├── rf_data_loader.py   ← file ini
#       ├── rf_model.py
#       └── rf_schemas.py
# ─────────────────────────────────────────────

# Folder rf_production (lokasi file ini)
RF_DIR   = os.path.dirname(os.path.abspath(__file__))

# Naik satu level → folder induk project
BASE_DIR = os.path.dirname(RF_DIR)

PATH_LAPORAN = os.path.join(BASE_DIR, "data", "laporan produksi.json")
PATH_TARGET  = os.path.join(BASE_DIR, "data", "target produksi.json")
PATH_STOK    = os.path.join(BASE_DIR, "data", "stok cpo.json")


# ─────────────────────────────────────────────
# HELPER: Baca JSON lokal
# ─────────────────────────────────────────────

def _load_json(path: str, nama: str) -> list:
    """Baca file JSON lokal, kembalikan list records."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"File '{nama}' tidak ditemukan di: {path}\n"
            f"Pastikan file ada di folder 'data/' di samping rf_data_loader.py"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Handle format {data: [...]} maupun [...]
        records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        print(f"  {nama:20s} → {len(records):>5} records")
        return records
    except json.JSONDecodeError as e:
        raise RuntimeError(f"File JSON '{nama}' rusak atau tidak valid: {e}")


# ─────────────────────────────────────────────
# HELPER: Agregasi per sumber data
# ─────────────────────────────────────────────

def _build_realisasi(records: list) -> dict:
    """Agregasi RBDPO bahan_olah per bulan → realisasi produksi."""
    result = defaultdict(float)
    for x in records:
        try:
            item = x.get("item_produksi", {})
            if item.get("name") == "RBDPO" and item.get("kategori") == "bahan_olah":
                bulan = x["tanggal"][:7]
                result[bulan] += float(x["qty"])
        except Exception:
            continue
    return dict(result)


def _build_target(records: list) -> dict:
    """Agregasi RKAP per bulan → target produksi."""
    result = defaultdict(float)
    for x in records:
        try:
            if x.get("uraian", {}).get("nama") == "RKAP":
                bulan = x["tanggal"][:7]
                result[bulan] += float(x["value"])
        except Exception:
            continue
    return dict(result)


def _build_stok(records: list) -> dict:
    """Agregasi stok CPO harian → fitur bulanan (rata2, max, min, std, hari aktif)."""
    raw = defaultdict(list)
    for x in records:
        try:
            bulan = x["tanggal"][:7]
            qty   = float(x["qty"])
            raw[bulan].append(qty)
        except Exception:
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


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def load_and_preprocess() -> pd.DataFrame:
    """
    Pipeline lengkap:
    1. Baca 3 file JSON lokal
    2. Agregasi per bulan
    3. Gabungkan (left join) + imputasi stok yang kosong
    4. Feature engineering (bulan_ke, kuartal, realisasi_prev)
    Kembalikan DataFrame siap pakai untuk model RF.
    """
    print("Membaca file JSON lokal...\n")

    # 1. Baca file
    raw_laporan = _load_json(PATH_LAPORAN, "Laporan Produksi")
    raw_target  = _load_json(PATH_TARGET,  "Target Produksi")
    raw_stok    = _load_json(PATH_STOK,    "Stok CPO")

    # 2. Agregasi
    realisasi_d = _build_realisasi(raw_laporan)
    target_d    = _build_target(raw_target)
    stok_d      = _build_stok(raw_stok)

    print(f"\n  Bulan tersedia:")
    print(f"    Realisasi : {len(realisasi_d)} bulan "
          f"({sorted(realisasi_d)[0]} s/d {sorted(realisasi_d)[-1]})")
    print(f"    Target    : {len(target_d)} bulan "
          f"({sorted(target_d)[0]} s/d {sorted(target_d)[-1]})")
    print(f"    Stok CPO  : {len(stok_d)} bulan "
          f"({sorted(stok_d)[0]} s/d {sorted(stok_d)[-1]})")

    # 3. Base dataset = inner join realisasi & target
    bulan_base = sorted(set(realisasi_d) & set(target_d))
    if not bulan_base:
        raise RuntimeError(
            "Tidak ada bulan yang overlap antara realisasi dan target. "
            "Periksa isi file JSON."
        )

    rows = []
    for b in bulan_base:
        row = {
            "bulan"              : b,
            "realisasi_produksi" : realisasi_d[b],
            "target_produksi"    : target_d[b],
        }
        # Stok tersedia → pakai data asli, tidak ada → NaN (diimputasi nanti)
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

    # 4. Imputasi interpolasi linear untuk kolom stok yang kosong
    stok_cols = ["stok_rata2", "stok_max", "stok_min", "stok_std", "stok_hari_aktif"]
    for col in stok_cols:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
    df["stok_std"] = df["stok_std"].fillna(0)

    # 5. Feature engineering tambahan
    df["bulan_ke"]       = df["bulan"].dt.month       # 1–12
    df["kuartal"]        = df["bulan"].dt.quarter     # 1–4
    df["realisasi_prev"] = df["realisasi_produksi"].shift(1).fillna(0)

    n_total   = len(df)
    n_imputed = int(df["is_imputed"].sum())
    n_asli    = n_total - n_imputed

    print(f"\n Dataset siap: {n_total} bulan total")
    print(f"     - Data asli    : {n_asli} bulan (stok tersedia)")
    print(f"     - Diimputasi   : {n_imputed} bulan (stok diinterpolasi)")
    print(f"     - Rentang      : {df['bulan'].min().strftime('%b %Y')} "
          f"s/d {df['bulan'].max().strftime('%b %Y')}")

    return df