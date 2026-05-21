# rf_data_loader.py
#
# Strategi sumber data (prioritas urut):
#   1. API production  → http://103.176.66.42:9009  (data 2024–sekarang)
#   2. File JSON lokal → folder data/               (fallback jika API mati)
#   3. File Excel      → folder data/excel/         (data historis 2021–2023)
#
# Struktur folder yang diharapkan:
#   project/
#   ├── data/
#   │   ├── laporan_produksi.json
#   │   ├── stok_cpo.json
#   │   └── target_produksi.json
#   ├── data/excel/
#   │   ├── 01_DAILY_REPORT_JANUARI_2021.xlsx
#   │   ├── 02_DAILY_REPORT_FEBRUARI_2021.xlsx
#   │   └── ... (semua file Excel 2021–2023)
#   └── rf_production/
#       ├── rf_api.py
#       ├── rf_data_loader.py   ← file ini
#       ├── rf_model.py
#       └── rf_schemas.py

import os
import re
import json
import glob
import requests
import numpy as np
import pandas as pd
from collections import defaultdict

# ─────────────────────────────────────────────
# PATH KONFIGURASI
# ─────────────────────────────────────────────

# Folder rf_production (lokasi file ini)
_RF_DIR  = os.path.dirname(os.path.abspath(__file__))
# Folder induk project
_BASE    = os.path.dirname(_RF_DIR)

# API production
API_BASE    = "http://103.193.145.61:9009/api"
URL_LAPORAN = f"{API_BASE}/laporan-prod"
URL_STOK    = f"{API_BASE}/stock-cpo"
URL_TARGET  = f"{API_BASE}/target-prod"
TIMEOUT     = 30

# Fallback JSON lokal
PATH_JSON_LAPORAN = os.path.join(_BASE, "data", "laporan_produksi.json")
PATH_JSON_STOK    = os.path.join(_BASE, "data", "stok_cpo.json")
PATH_JSON_TARGET  = os.path.join(_BASE, "data", "target_produksi.json")

# Folder Excel historis 2021–2023
EXCEL_DIR = os.path.join(_BASE, "data", "excel")

# ─────────────────────────────────────────────
# LAYER 1: FETCH DARI API PRODUCTION
# ─────────────────────────────────────────────

def _fetch_api(url: str, nama: str) -> list | None:
    """
    Coba ambil data dari API production.
    Return list records jika berhasil, None jika gagal.
    """
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data    = resp.json()
        records = data["data"] if isinstance(data, dict) and "data" in data else data
        print(f"[API] {nama:20s} → {len(records):>5} records")
        return records
    except Exception as e:
        print(f"[API] {nama} gagal: {e}")
        return None


# ─────────────────────────────────────────────
# LAYER 2: FALLBACK JSON LOKAL
# ─────────────────────────────────────────────

def _load_json(path: str, nama: str) -> list | None:
    """
    Baca file JSON lokal sebagai fallback.
    Return list records jika file ada, None jika tidak.
    """
    if not os.path.exists(path):
        print(f"[JSON] {nama} tidak ditemukan di: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        print(f"[JSON] {nama:20s} → {len(records):>5} records")
        return records
    except Exception as e:
        print(f"[JSON] {nama} error: {e}")
        return None


def _get_records(url: str, json_path: str, nama: str) -> list:
    """
    Coba API dulu, fallback ke JSON lokal.
    Raise RuntimeError jika keduanya gagal.
    """
    records = _fetch_api(url, nama)
    if records is None:
        print(f"Fallback ke JSON lokal untuk {nama}...")
        records = _load_json(json_path, nama)
    if records is None:
        raise RuntimeError(
            f"Tidak dapat memuat {nama}. "
            f"API tidak bisa dijangkau dan file JSON lokal tidak ditemukan.\n"
            f"  Pastikan:\n"
            f"  - Server {API_BASE} aktif, ATAU\n"
            f"  - File ada di {json_path}"
        )
    return records


# ─────────────────────────────────────────────
# LAYER 3: PARSER EXCEL 2021–2023
# ─────────────────────────────────────────────

_BULAN_MAP = {
    "januari":1,"februari":2,"maret":3,"april":4,"mei":5,"juni":6,
    "juli":7,"agustus":8,"september":9,"oktober":10,"november":11,"desember":12,
    "january":1,"february":2,"march":3,"may":5,"june":6,"july":7,
    "august":8,"october":10,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,
    "aug":8,"agu":8,"sep":9,"oct":10,"okt":10,"nov":11,"dec":12,"des":12,
}


def _parse_periode(filename: str):
    """Ekstrak (year, month) dari nama file Excel."""
    fname = os.path.basename(filename).lower()
    year_match = re.search(r"(202\d)", fname)
    year  = int(year_match.group(1)) if year_match else None
    month = None
    for nama, num in sorted(_BULAN_MAP.items(), key=lambda x: -len(x[0])):
        if len(nama) > 2 and nama in fname:
            month = num
            break
    return year, month


def _find_sheet(xl) -> str:
    """Pilih sheet utama, hindari sheet pelengkap."""
    skip = ["sheet", "trace", "utility", "summary", "yearly", "v.1"]
    for name in xl.sheet_names:
        if not any(x in name.lower() for x in skip):
            return name
    return xl.sheet_names[0]


def _find_row(df: pd.DataFrame, keywords, max_row: int = 120) -> int | None:
    """Cari baris pertama yang mengandung salah satu keyword."""
    if isinstance(keywords, str):
        keywords = [keywords]
    for r in range(min(df.shape[0], max_row)):
        for c in range(min(df.shape[1], 6)):
            cell = str(df.iloc[r, c]).strip().lower()
            if any(kw.lower() in cell for kw in keywords):
                return r
    return None


def _get_daily(df: pd.DataFrame, row: int, col_start: int = 5) -> list:
    """Ambil nilai harian dari baris tertentu (kolom 5–35)."""
    vals = []
    for c in range(col_start, col_start + 31):
        if c >= df.shape[1]:
            break
        try:
            v = float(df.iloc[row, c])
            vals.append(0.0 if np.isnan(v) else v)
        except Exception:
            vals.append(0.0)
    return vals


def _get_total(df: pd.DataFrame, row: int, col_start: int = 5) -> float:
    """
    Ambil nilai TOTAL dari baris tertentu.
    Cari kolom total (toleransi 5% dari sum harian).
    """
    daily    = _get_daily(df, row, col_start)
    expected = sum(daily)
    if expected == 0:
        return 0.0
    # Cari kolom total setelah kolom 31 harian
    for c in range(col_start + 31, col_start + 51):
        if c >= df.shape[1]:
            break
        try:
            v = float(df.iloc[row, c])
            if not np.isnan(v) and abs(v - expected) / (expected + 1) < 0.05:
                return v
        except Exception:
            continue
    return expected  # fallback ke sum harian


def _extract_excel_month(filepath: str) -> dict | None:
    """
    Ekstrak data bulanan dari satu file Excel Daily Report.
    Return dict atau None jika gagal / tahun >= 2024 (sudah ada di JSON).
    """
    year, month = _parse_periode(filepath)
    if not year or not month:
        return None
    # Lewati 2024+ karena sudah tercakup di JSON/API
    if year >= 2024:
        return None

    try:
        xl  = pd.ExcelFile(filepath)
        sht = _find_sheet(xl)
        df  = xl.parse(sht, header=None)
        CS  = 5  # kolom start — konsisten semua tahun

        row_rbdpo = _find_row(df, ["RbdPO"])
        row_cpo   = _find_row(df, ["CPO Consume Total", "CPO Consume Poram", "CPO Consume"])
        row_rd    = _find_row(df, ["Running Days"])
        row_ffa   = _find_row(df, ["- FFA", "% FFA", "FFA"])
        row_pfad  = _find_row(df, ["PFAD"])

        rbdpo_total = _get_total(df, row_rbdpo, CS) if row_rbdpo is not None else 0.0
        cpo_total   = _get_total(df, row_cpo,   CS) if row_cpo   is not None else 0.0
        pfad_total  = _get_total(df, row_pfad,  CS) if row_pfad  is not None else 0.0

        # Hari olah: Running Days × 3 lini produksi
        rd_vals   = _get_daily(df, row_rd, CS) if row_rd is not None else []
        hari_olah = int(sum(1 for v in rd_vals if v == 1.0)) * 3

        # Rata-rata FFA (filter nilai wajar 1–8%)
        ffa_vals = (
            [v for v in _get_daily(df, row_ffa, CS) if 1 < v < 8]
            if row_ffa is not None else []
        )
        ffa_avg = float(np.mean(ffa_vals)) if ffa_vals else 0.0

        yield_pct = (rbdpo_total / cpo_total * 100) if cpo_total > 0 else 0.0

        return {
            "bulan"           : f"{year}-{month:02d}",
            "realisasi_rbdpo" : round(rbdpo_total, 2),
            "cpo_consume"     : round(cpo_total,   2),
            "hari_olah"       : hari_olah,
            "pfad_total"      : round(pfad_total,  2),
            "yield_rbdpo"     : round(yield_pct,   4),
            "ffa_avg"         : round(ffa_avg,     4),
            "target_rkap"     : np.nan,    # diisi nanti dari target_d
            "stok_rata2"      : np.nan,    # diisi nanti dari imputasi
            "stok_max"        : np.nan,
            "stok_std"        : np.nan,
            "stok_hari_aktif" : np.nan,
            "stok_imputed"    : True,
            "sumber"          : "excel",
        }
    except Exception as e:
        print(f"[Excel] {os.path.basename(filepath)}: {e}")
        return None


def _load_excel_historis(excel_dir: str) -> pd.DataFrame:
    """
    Baca semua file Excel di folder excel_dir.
    Return DataFrame kosong jika folder tidak ada atau tidak ada file.
    """
    if not os.path.isdir(excel_dir):
        print(f"Folder Excel tidak ditemukan: {excel_dir} — dilewati")
        return pd.DataFrame()

    excel_files = sorted(glob.glob(os.path.join(excel_dir, "*.xlsx")))
    if not excel_files:
        print(f"Tidak ada file .xlsx di {excel_dir} — dilewati")
        return pd.DataFrame()

    print(f"Memproses {len(excel_files)} file Excel...")
    rows = []
    for fpath in excel_files:
        result = _extract_excel_month(fpath)
        if result:
            rows.append(result)
            print(f"{result['bulan']} | "
                  f"RBDPO: {result['realisasi_rbdpo']:>15,.0f} | "
                  f"Hari: {result['hari_olah']:>4}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["bulan"] = pd.to_datetime(df["bulan"])
    df = df.sort_values("bulan").drop_duplicates("bulan").reset_index(drop=True)
    print(f"[Excel] {len(df)} bulan historis "
          f"({df['bulan'].min().strftime('%b %Y')} – "
          f"{df['bulan'].max().strftime('%b %Y')})")
    return df


# ─────────────────────────────────────────────
# AGREGASI JSON/API
# ─────────────────────────────────────────────

def _build_laporan(records: list):
    realisasi_d = defaultdict(float)
    cpo_d       = defaultdict(float)
    hari_olah_d = defaultdict(float)
    pfad_d      = defaultdict(float)

    for x in records:
        item = x.get("item_produksi", {})
        name = item.get("name", "")
        kat  = item.get("kategori", "")
        b    = x.get("tanggal", "")[:7]
        try:
            qty = float(x["qty"])
        except Exception:
            continue
        if   name == "RBDPO" and kat == "produk_hasil": realisasi_d[b] += qty
        elif name == "CPO"   and kat == "bahan_olah"  : cpo_d[b]       += qty
        elif name == "Hari Olah"                      : hari_olah_d[b] += qty
        elif name == "PFAD"  and kat == "produk_hasil": pfad_d[b]       += qty

    return dict(realisasi_d), dict(cpo_d), dict(hari_olah_d), dict(pfad_d)


def _build_target(records: list) -> dict:
    result = defaultdict(float)
    for x in records:
        try:
            if x.get("uraian", {}).get("nama") == "RKAP":
                result[x["tanggal"][:7]] += float(x["value"])
        except Exception:
            continue
    # Fix anomali satuan (58,900 → 58,900,000)
    for b in list(result):
        if result[b] < 1_000_000:
            result[b] *= 1000
    return dict(result)


def _build_stok(records: list) -> dict:
    raw = defaultdict(list)
    for x in records:
        try:
            raw[x["tanggal"][:7]].append(float(x["qty"]))
        except Exception:
            continue
    result = {}
    for b, vals in raw.items():
        arr = np.array(vals)
        result[b] = {
            "stok_rata2"      : float(np.mean(arr)),
            "stok_max"        : float(np.max(arr)),
            "stok_std"        : float(np.std(arr)),
            "stok_hari_aktif" : int(np.sum(arr > 0)),
        }
    return result


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def load_and_preprocess() -> pd.DataFrame:
    """
    Pipeline lengkap dengan 3 sumber data:
    1. Fetch laporan, stok, target dari API production
       (fallback ke JSON lokal jika API mati)
    2. Baca Excel historis 2021–2023 dari folder data/excel/
    3. Gabungkan, imputasi stok, feature engineering
    Return: DataFrame siap pakai untuk model RF
    """
    print("\n" + "="*55)
    print("📡 MEMUAT DATA")
    print("="*55)

    # ── Sumber 1 & 2: API / JSON lokal ──────────────────────────────────
    print("\n[1] Laporan Produksi, Stok CPO, Target RKAP")
    raw_laporan = _get_records(URL_LAPORAN, PATH_JSON_LAPORAN, "Laporan Produksi")
    raw_stok    = _get_records(URL_STOK,    PATH_JSON_STOK,    "Stok CPO")
    raw_target  = _get_records(URL_TARGET,  PATH_JSON_TARGET,  "Target RKAP")

    realisasi_d, cpo_d, hari_olah_d, pfad_d = _build_laporan(raw_laporan)
    target_d = _build_target(raw_target)
    stok_d   = _build_stok(raw_stok)

    # Bangun DataFrame base dari JSON/API (2024–sekarang)
    bulan_json = sorted(set(realisasi_d) & set(target_d))
    rows_json  = []
    for b in bulan_json:
        row = {
            "bulan"           : b,
            "realisasi_rbdpo" : realisasi_d[b],
            "cpo_consume"     : cpo_d.get(b, np.nan),
            "hari_olah"       : hari_olah_d.get(b, np.nan),
            "pfad_total"      : pfad_d.get(b, np.nan),
            "yield_rbdpo"     : 0.0,
            "ffa_avg"         : 0.0,
            "target_rkap"     : target_d[b],
            "stok_imputed"    : b not in stok_d,
            "sumber"          : "api",
        }
        if b in stok_d:
            row.update(stok_d[b])
        else:
            row.update({
                "stok_rata2": np.nan, "stok_max": np.nan,
                "stok_std"  : np.nan, "stok_hari_aktif": np.nan,
            })
        rows_json.append(row)

    df_api = pd.DataFrame(rows_json)
    df_api["bulan"] = pd.to_datetime(df_api["bulan"])

    # ── Sumber 3: Excel historis ─────────────────────────────────────────
    print(f"\n[2] Excel Historis 2021–2023 dari: {EXCEL_DIR}")
    df_excel = _load_excel_historis(EXCEL_DIR)

    # ── Gabungkan ─────────────────────────────────────────────────────────
    if not df_excel.empty:
        # Isi target_rkap Excel dengan median dari data API
        median_target = float(df_api["target_rkap"].median())
        df_excel["target_rkap"] = median_target

        df_all = pd.concat([df_excel, df_api], ignore_index=True)
        print(f"\nGabungan: {len(df_excel)} bln Excel + {len(df_api)} bln API")
    else:
        df_all = df_api.copy()
        print(f"\nHanya data API: {len(df_api)} bulan")

    df_all = (
        df_all
        .sort_values("bulan")
        .drop_duplicates("bulan")   # API menang jika ada duplikat
        .reset_index(drop=True)
    )

    # ── Imputasi stok dengan interpolasi linear ───────────────────────────
    for col in ["stok_rata2", "stok_max", "stok_std", "stok_hari_aktif"]:
        df_all[col] = df_all[col].interpolate(method="linear", limit_direction="both")
    df_all["stok_std"] = df_all["stok_std"].fillna(0)

    # Isi NaN kolom operasional dengan median
    for col in ["cpo_consume", "hari_olah", "pfad_total", "target_rkap", "ffa_avg"]:
        df_all[col] = df_all[col].fillna(df_all[col].median())

    # ── Feature engineering ───────────────────────────────────────────────
    df_all["yield_rbdpo"] = np.where(
        df_all["cpo_consume"] > 0,
        df_all["realisasi_rbdpo"] / df_all["cpo_consume"] * 100, 0
    )
    df_all["cpo_per_hari"]   = df_all["cpo_consume"] / (df_all["hari_olah"] + 1)
    df_all["bulan_ke"]       = df_all["bulan"].dt.month
    df_all["kuartal"]        = df_all["bulan"].dt.quarter
    df_all["realisasi_prev"] = df_all["realisasi_rbdpo"].shift(1).fillna(0)
    df_all["cpo_prev"]       = df_all["cpo_consume"].shift(1).fillna(
                                    df_all["cpo_consume"].median())
    df_all["hari_olah_prev"] = df_all["hari_olah"].shift(1).fillna(
                                    df_all["hari_olah"].median())

    # ── Summary ───────────────────────────────────────────────────────────
    n_total   = len(df_all)
    n_excel   = int((df_all["sumber"] == "excel").sum())
    n_api     = int((df_all["sumber"] == "api").sum())
    n_imputed = int(df_all["stok_imputed"].sum())

    print(f"\n{'='*55}")
    print(f"DATASET SIAP")
    print(f"{'='*55}")
    print(f"   Total bulan    : {n_total}")
    print(f"   Dari Excel     : {n_excel} bulan (2021–2023)")
    print(f"   Dari API       : {n_api} bulan (2024–sekarang)")
    print(f"   Stok imputasi  : {n_imputed} bulan")
    print(f"   Rentang        : "
          f"{df_all['bulan'].min().strftime('%b %Y')} – "
          f"{df_all['bulan'].max().strftime('%b %Y')}")
    print(f"{'='*55}\n")

    return df_all