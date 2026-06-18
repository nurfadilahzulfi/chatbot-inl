"""
production_analyzer.py — Modul analitik produksi RBDPO untuk SmartChatbot
==========================================================================
Mengolah data hasil RF API (port 3001) dan mem-format context string
yang siap digunakan oleh LLM Ollama untuk menjawab pertanyaan produksi.

Peran modul ini sejajar dengan price_analytics.py, tapi untuk data produksi.

Data yang bisa dijawab:
  - Realisasi produksi RBDPO per bulan
  - Perbandingan vs target RKAP
  - Detail operasional (hari olah, CPO consume, yield, stok)
  - Prediksi LOOCV model RF
  - Akurasi model (MAE, RMSE, MAPE, R²)
  - Feature importance (faktor paling berpengaruh)
"""

import os
import re
import requests
import logging
from typing import Optional

logger = logging.getLogger("production_analyzer")

# ── URL RF API ────────────────────────────────────────────────────────────────
# Host & port dibaca dari environment variable agar fleksibel.
# Jika RF API berjalan di mesin lain, cukup set:
#   RF_API_HOST=192.168.1.10  (sebelum menjalankan api_stream.py)
# Default fallback: localhost (cocok jika kedua server di mesin yang sama)
_RF_HOST = os.environ.get("RF_API_HOST", "localhost")
_RF_PORT = os.environ.get("RF_API_PORT", "3001")
RF_API_URL = f"http://{_RF_HOST}:{_RF_PORT}/rf-analysis"
RF_TIMEOUT = 60   # detik — startup bisa lambat karena LOOCV

logger.info(f"RF API target: {RF_API_URL}")

# ── Mapping nama bulan Indonesia → angka ─────────────────────────────────────
_BULAN_MAP = {
    "januari": 1,  "jan": 1,
    "februari": 2, "feb": 2,
    "maret": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "mei": 5,
    "juni": 6,     "jun": 6,
    "juli": 7,     "jul": 7,
    "agustus": 8,  "agu": 8, "ags": 8,
    "september": 9,"sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11,"nov": 11,
    "desember": 12,"des": 12,
}

# ── Helper format angka ───────────────────────────────────────────────────────

def _fmt_ton(value: float) -> str:
    """Format angka ke ribuan ton, dua desimal."""
    return f"{value:,.2f} ton"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_hari(value: float) -> str:
    return f"{value:.0f} hari"


# ── Fetch dari RF API ─────────────────────────────────────────────────────────

def fetch_rf_data() -> Optional[dict]:
    """
    Ambil data analisis lengkap dari RF API (port 3001).
    Return dict hasil /rf-analysis, atau None jika gagal.
    """
    try:
        resp = requests.get(RF_API_URL, timeout=RF_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            f"RF data berhasil dimuat: "
            f"{len(data.get('historis', []))} bulan historis."
        )
        return data
    except requests.exceptions.ConnectionError:
        logger.warning(
            "RF API tidak dapat dijangkau (localhost:3001). "
            "Pastikan rf_api.py sudah running."
        )
    except requests.exceptions.Timeout:
        logger.warning(
            f"RF API timeout setelah {RF_TIMEOUT}s. "
            "LOOCV mungkin masih berjalan."
        )
    except Exception as e:
        logger.warning(f"Gagal fetch RF data: {e}")
    return None


# ── Kelas Utama ───────────────────────────────────────────────────────────────

class ProductionAnalyzer:
    """
    Mengolah data produksi RBDPO dari RF API dan mem-format context string
    untuk dikirim ke LLM Ollama sebagai konteks jawaban.

    Inisialisasi:
        analyzer = ProductionAnalyzer(rf_data)
        context  = analyzer.analyze(query)

    rf_data adalah dict hasil /rf-analysis dari rf_api.py.
    """

    def __init__(self, rf_data: dict):
        self._historis    = rf_data.get("historis", [])          # list BulanHistoris
        self._evaluasi    = rf_data.get("evaluasi", {})          # dict EvaluasiModel
        self._fi          = rf_data.get("feature_importance", [])# list FeatureImportanceItem
        self._last_updated = rf_data.get("last_updated", "-")

    # ── Dispatcher utama ─────────────────────────────────────────────────────

    def analyze(self, query: str) -> str:
        """
        Pilih fungsi yang tepat berdasarkan query, kembalikan context string
        siap pakai untuk LLM.
        """
        q = query.lower()

        # Feature importance / faktor berpengaruh
        if any(kw in q for kw in [
            "faktor", "feature importance", "paling berpengaruh",
            "variabel", "pengaruh", "determinan", "apa yang mempengaruhi",
            "feature", "importance",
        ]):
            return self.get_feature_importance()

        # Akurasi / performa model
        if any(kw in q for kw in [
            "akurasi", "performa model", "mape", "mae", "rmse", "r2", "r²",
            "seberapa akurat", "error model", "evaluasi model",
            "keandalan", "presisi model",
        ]):
            return self.get_model_performance()

        # Tren / rekap keseluruhan
        if any(kw in q for kw in [
            "tertinggi", "terbesar", "terbaik", "terendah", "terkecil",
            "paling tinggi", "paling rendah", "rekor", "maksimum", "minimum",
        ]):
            return self.get_best_worst(q)

        # Bulan spesifik
        month_ctx = self._extract_month_query(q)
        if month_ctx:
            return self.get_month_detail(*month_ctx)

        # Target / RKAP
        if any(kw in q for kw in ["target", "rkap", "capaian", "tercapai", "pencapaian"]):
            return self.get_target_summary()

        # Default: ringkasan terbaru
        return self.get_summary()

    # ── Ringkasan produksi terkini ────────────────────────────────────────────

    def get_summary(self) -> str:
        """Ringkasan produksi bulan terakhir dan beberapa bulan sebelumnya."""
        if not self._historis:
            return "Data produksi RBDPO belum tersedia."

        # Ambil 3 bulan terakhir
        recent = self._historis[-3:]
        lines = []
        for item in recent:
            capaian_pct = (
                item["realisasi"] / item["target_rkap"] * 100
                if item.get("target_rkap", 0) > 0 else 0
            )
            status = "✅ TERCAPAI" if item["realisasi"] >= item["target_rkap"] else "❌ BELUM TERCAPAI"
            lines.append(
                f"📅 **{item['bulan_label']}**\n"
                f"  • Realisasi  : {_fmt_ton(item['realisasi'])}\n"
                f"  • Target RKAP: {_fmt_ton(item['target_rkap'])}\n"
                f"  • Capaian    : {_fmt_pct(capaian_pct)} — {status}\n"
                f"  • Hari Olah  : {_fmt_hari(item['hari_olah'])}\n"
                f"  • Yield RBDPO: {_fmt_pct(item['yield_rbdpo'] * 100 if item['yield_rbdpo'] < 10 else item['yield_rbdpo'])}\n"
                f"  • Prediksi RF: {_fmt_ton(item['prediksi_loocv'])} (error: {_fmt_pct(item['error_pct'])})"
            )

        total_bulan = len(self._historis)
        return (
            f"📊 **DATA PRODUKSI RBDPO — {total_bulan} BULAN TERSEDIA**\n"
            f"(Update terakhir: {self._last_updated})\n\n"
            + "\n\n".join(lines)
        )

    # ── Detail bulan spesifik ─────────────────────────────────────────────────

    def get_month_detail(self, year: int, month: int) -> str:
        """Detail produksi untuk tahun dan bulan tertentu."""
        if not self._historis:
            return "Data produksi belum tersedia."

        target_key = f"{year}-{month:02d}"
        matched = next(
            (h for h in self._historis if h["bulan"] == target_key), None
        )

        if not matched:
            # Coba cari yang paling dekat
            available = [h["bulan_label"] for h in self._historis]
            return (
                f"Data produksi untuk {month:02d}/{year} tidak ditemukan.\n"
                f"Data tersedia: {', '.join(available[-6:])}"
            )

        item = matched
        capaian_pct = (
            item["realisasi"] / item["target_rkap"] * 100
            if item.get("target_rkap", 0) > 0 else 0
        )
        selisih = item["realisasi"] - item["target_rkap"]
        status  = "✅ TERCAPAI" if selisih >= 0 else "❌ BELUM TERCAPAI"
        selisih_str = (
            f"+{_fmt_ton(selisih)}" if selisih >= 0 else f"-{_fmt_ton(abs(selisih))}"
        )
        yield_val = item["yield_rbdpo"]
        if yield_val < 10:   # jika disimpan dalam bentuk 0.xx → kalikan 100
            yield_val *= 100

        return (
            f"📦 **DETAIL PRODUKSI — {item['bulan_label']}**\n\n"
            f"REALISASI vs TARGET:\n"
            f"  • Realisasi RBDPO  : {_fmt_ton(item['realisasi'])}\n"
            f"  • Target RKAP      : {_fmt_ton(item['target_rkap'])}\n"
            f"  • Capaian          : {_fmt_pct(capaian_pct)} ({selisih_str}) — {status}\n\n"
            f"OPERASIONAL:\n"
            f"  • Hari Olah        : {_fmt_hari(item['hari_olah'])}\n"
            f"  • CPO Dikonsumsi   : {_fmt_ton(item['cpo_consume'])}\n"
            f"  • Yield RBDPO      : {_fmt_pct(yield_val)}\n"
            f"  • Stok CPO (rata²) : {_fmt_ton(item['stok_rata2'])}\n\n"
            f"PREDIKSI MODEL RF:\n"
            f"  • Prediksi LOOCV   : {_fmt_ton(item['prediksi_loocv'])}\n"
            f"  • Error prediksi   : {_fmt_pct(item['error_pct'])}"
            + ("\n  • ⚠️ Data stok diimputasi (estimasi)" if item.get("is_imputed") else "")
        )

    # ── Bulan tertinggi / terendah ────────────────────────────────────────────

    def get_best_worst(self, query: str = "") -> str:
        """Tampilkan bulan dengan produksi tertinggi dan terendah."""
        if not self._historis:
            return "Data produksi belum tersedia."

        # Filter hanya bulan yang ada produksi (realisasi > 0)
        aktif = [h for h in self._historis if h["realisasi"] > 0]
        if not aktif:
            return "Tidak ada data produksi aktif."

        terbaik   = max(aktif, key=lambda x: x["realisasi"])
        terburuk  = min(aktif, key=lambda x: x["realisasi"])
        rata_rata = sum(h["realisasi"] for h in aktif) / len(aktif)

        # Deteksi pertanyaan spesifik tertinggi atau terendah saja
        q = query.lower()
        if any(kw in q for kw in ["tertinggi", "terbesar", "terbaik", "maksimum", "rekor", "paling tinggi"]):
            capaian = terbaik["realisasi"] / terbaik["target_rkap"] * 100 if terbaik.get("target_rkap", 0) > 0 else 0
            return (
                f"🏆 **PRODUKSI RBDPO TERTINGGI**\n"
                f"  • Bulan  : {terbaik['bulan_label']}\n"
                f"  • Volume : {_fmt_ton(terbaik['realisasi'])}\n"
                f"  • Target : {_fmt_ton(terbaik['target_rkap'])} ({_fmt_pct(capaian)})\n"
                f"  • Hari Olah: {_fmt_hari(terbaik['hari_olah'])}"
            )

        if any(kw in q for kw in ["terendah", "terkecil", "minimum", "paling rendah"]):
            capaian = terburuk["realisasi"] / terburuk["target_rkap"] * 100 if terburuk.get("target_rkap", 0) > 0 else 0
            return (
                f"📉 **PRODUKSI RBDPO TERENDAH**\n"
                f"  • Bulan  : {terburuk['bulan_label']}\n"
                f"  • Volume : {_fmt_ton(terburuk['realisasi'])}\n"
                f"  • Target : {_fmt_ton(terburuk['target_rkap'])} ({_fmt_pct(capaian)})\n"
                f"  • Hari Olah: {_fmt_hari(terburuk['hari_olah'])}"
            )

        # Tampilkan keduanya + rata-rata
        return (
            f"📊 **REKAP PRODUKSI RBDPO ({len(aktif)} bulan aktif)**\n\n"
            f"🏆 Tertinggi — {terbaik['bulan_label']}: {_fmt_ton(terbaik['realisasi'])}\n"
            f"📉 Terendah  — {terburuk['bulan_label']}: {_fmt_ton(terburuk['realisasi'])}\n"
            f"⚖️ Rata-rata : {_fmt_ton(rata_rata)}"
        )

    # ── Target RKAP ───────────────────────────────────────────────────────────

    def get_target_summary(self) -> str:
        """Ringkasan pencapaian target RKAP per bulan."""
        if not self._historis:
            return "Data produksi belum tersedia."

        total     = len(self._historis)
        tercapai  = sum(1 for h in self._historis if h["realisasi"] >= h["target_rkap"] and h["realisasi"] > 0)
        pct_capai = tercapai / total * 100 if total > 0 else 0

        # 3 bulan terkini
        recent_lines = []
        for item in self._historis[-3:]:
            capaian_pct = (
                item["realisasi"] / item["target_rkap"] * 100
                if item.get("target_rkap", 0) > 0 else 0
            )
            icon = "✅" if item["realisasi"] >= item["target_rkap"] else "❌"
            recent_lines.append(
                f"  {icon} {item['bulan_label']}: "
                f"{_fmt_ton(item['realisasi'])} / {_fmt_ton(item['target_rkap'])} "
                f"= {_fmt_pct(capaian_pct)}"
            )

        return (
            f"🎯 **PENCAPAIAN TARGET RKAP PRODUKSI RBDPO**\n\n"
            f"Dari {total} bulan data:\n"
            f"  • Target tercapai : {tercapai} bulan ({_fmt_pct(pct_capai)})\n"
            f"  • Tidak tercapai  : {total - tercapai} bulan\n\n"
            f"3 Bulan Terakhir:\n"
            + "\n".join(recent_lines)
        )

    # ── Akurasi model ─────────────────────────────────────────────────────────

    def get_model_performance(self) -> str:
        """Ringkasan performa model Random Forest (dari evaluasi LOOCV)."""
        if not self._evaluasi:
            return "Data evaluasi model belum tersedia."

        e = self._evaluasi
        return (
            f"🤖 **PERFORMA MODEL RANDOM FOREST — PRODUKSI RBDPO**\n"
            f"(Validasi: Leave-One-Out Cross Validation / LOOCV)\n\n"
            f"METRIK AKURASI:\n"
            f"  • MAE  : {_fmt_ton(e.get('mae', 0))} (rata-rata selisih absolut)\n"
            f"  • RMSE : {_fmt_ton(e.get('rmse', 0))} (akar rata-rata kuadrat error)\n"
            f"  • MAPE : {_fmt_pct(e.get('mape', 0))} (rata-rata error persentase)\n"
            f"  • R²   : {e.get('r2', 0):.4f} (koefisien determinasi)\n\n"
            f"INTERPRETASI:\n"
            f"  • R²  : {e.get('interpretasi_r2', '-')}\n"
            f"  • MAPE: {e.get('interpretasi_mape', '-')}\n\n"
            f"DATA TRAINING:\n"
            f"  • Jumlah data  : {e.get('jumlah_data', 0)} bulan\n"
            f"  • Data imputed : {e.get('jumlah_imputed', 0)} bulan (stok diestimasi)"
        )

    # ── Feature importance ────────────────────────────────────────────────────

    def get_feature_importance(self) -> str:
        """Top feature importance dari model RF."""
        if not self._fi:
            return "Data feature importance belum tersedia."

        # Ambil top 7 fitur
        top = self._fi[:7]
        lines = []
        for item in top:
            bar = "█" * int(item["mdi_pct"] / 3)  # visual bar sederhana
            lines.append(
                f"  {item['peringkat']}. {item['fitur']:<22} "
                f"{_fmt_pct(item['mdi_pct'])} MDI  {bar}"
            )

        return (
            f"🌲 **FAKTOR PALING BERPENGARUH TERHADAP PRODUKSI RBDPO**\n"
            f"(Random Forest — MDI Importance + Permutation Importance)\n\n"
            + "\n".join(lines)
            + f"\n\n💡 Semakin besar persentase, semakin besar pengaruh fitur tersebut\n"
              f"   terhadap prediksi realisasi produksi RBDPO."
        )

    # ── Helper: ekstrak bulan & tahun dari query ──────────────────────────────

    def _extract_month_query(self, query: str) -> Optional[tuple]:
        """
        Coba ekstrak (year, month) dari query.
        Mengenali format: 'Maret 2024', '03/2024', '2024-03', 'bulan ini', dll.
        Return tuple (year, month) atau None jika tidak ditemukan.
        """
        from datetime import datetime
        now = datetime.now()

        # "bulan ini"
        if "bulan ini" in query:
            return (now.year, now.month)

        # "bulan lalu"
        if "bulan lalu" in query or "bulan kemarin" in query:
            month = now.month - 1 if now.month > 1 else 12
            year  = now.year if now.month > 1 else now.year - 1
            return (year, month)

        # Format "Nama_Bulan YYYY" — contoh: "maret 2024", "jan 2024"
        for bulan_str, bulan_num in _BULAN_MAP.items():
            pattern = rf"\b{re.escape(bulan_str)}\b.*?(20\d{{2}})"
            m = re.search(pattern, query)
            if not m:
                # Coba urutan terbalik: "2024 maret"
                pattern2 = rf"(20\d{{2}}).*?\b{re.escape(bulan_str)}\b"
                m = re.search(pattern2, query)
            if m:
                year = int(m.group(1))
                return (year, bulan_num)

        # Format "MM/YYYY" atau "MM-YYYY"
        m = re.search(r"\b(0?[1-9]|1[0-2])[/\-](20\d{2})\b", query)
        if m:
            return (int(m.group(2)), int(m.group(1)))

        # Format "YYYY-MM"
        m = re.search(r"\b(20\d{2})-(0?[1-9]|1[0-2])\b", query)
        if m:
            return (int(m.group(1)), int(m.group(2)))

        return None
