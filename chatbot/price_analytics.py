import re
import numpy as np
from datetime import datetime, timedelta

# Mapping bulan
MONTHS_ID = {
    "januari": 1, "jan": 1, "january": 1, "01": 1, "1": 1,
    "februari": 2, "feb": 2, "february": 2, "02": 2, "2": 2,
    "maret": 3, "mar": 3, "march": 3, "03": 3, "3": 3,
    "april": 4, "apr": 4, "04": 4, "4": 4,
    "mei": 5, "may": 5, "05": 5, "5": 5,
    "juni": 6, "jun": 6, "june": 6, "06": 6, "6": 6,
    "juli": 7, "jul": 7, "july": 7, "07": 7, "7": 7,
    "agustus": 8, "agu": 8, "ags": 8, "aug": 8, "august": 8, "08": 8, "8": 8,
    "september": 9, "sep": 9, "09": 9, "9": 9,
    "oktober": 10, "okt": 10, "oct": 10, "october": 10, "10": 10,
    "november": 11, "nov": 11, "11": 11,
    "desember": 12, "des": 12, "dec": 12, "december": 12, "12": 12
}


def _fmt_usd(amount: float) -> str:
    """Format harga ke notasi USD/MT, dua desimal."""
    return f"USD {amount:,.2f}/MT"


class PriceAnalyzer:
    def __init__(self, price_data):
        # Sort data berdasarkan tanggal agar urutan benar
        self.price_data = sorted(price_data, key=lambda x: x['date']) if price_data else []

    def get_yearly_recap(self, n_years):
        """Rekap Data Tahunan (Untuk main.py)"""
        if not self.price_data:
            return "Database kosong."
        current_year = self.price_data[-1]['date'].year
        start_year   = current_year - n_years + 1

        lines = []
        all_p = []

        for y in range(start_year, current_year + 1):
            subset = [d['price'] for d in self.price_data if d['date'].year == y]
            if subset:
                all_p.extend(subset)
                avg   = np.mean(subset)
                max_p = np.max(subset)
                min_p = np.min(subset)
                lines.append(
                    f"- **{y}**: Rata-rata {_fmt_usd(avg)} "
                    f"(Min: {_fmt_usd(min_p)} | Max: {_fmt_usd(max_p)})"
                )
            else:
                lines.append(f"- **{y}**: Data tidak tersedia")

        if not all_p:
            return "Data historis tidak ditemukan untuk rentang tahun tersebut."

        return (
            f"📅 **REKAP {n_years} TAHUN TERAKHIR**\n"
            f"{chr(10).join(lines)}\n\n"
            f"🏆 **Statistik Global ({start_year}-{current_year})**:\n"
            f"• Tertinggi: {_fmt_usd(np.max(all_p))}\n"
            f"• Terendah : {_fmt_usd(np.min(all_p))}"
        )

    def analyze(self, query):
        """
        Analisis Fleksibel (Hari, Minggu, Bulan, Tahun)
        """
        query = query.lower()
        if not self.price_data:
            return "Maaf, database harga kosong."

        # Setup Variabel
        start_date, end_date, label = None, None, ""
        last_data_date = self.price_data[-1]['date']

        # --- 0. DETEKSI TAHUN JAMAK (PRIORITAS UTAMA) ---
        year_range_match = re.search(r"(\d+)\s*tahun", query)
        if year_range_match and any(
                x in query for x in ["terakhir", "rekap", "rangkum", "lalu"]):
            return self.get_yearly_recap(int(year_range_match.group(1)))

        # --- 1. DETEKSI HARI RELATIF ---
        day_match = re.search(r"(\d+)\s*hari", query)

        if day_match:
            n_days    = int(day_match.group(1))
            label     = f"{n_days} Hari Terakhir"
            end_date  = last_data_date
            start_date = last_data_date - timedelta(days=n_days - 1)

        elif "seminggu" in query or "1 minggu" in query:
            label     = "Seminggu Terakhir"
            end_date  = last_data_date
            start_date = last_data_date - timedelta(days=6)

        elif "kemarin" in query:
            label     = "Kemarin (H-1 Data)"
            target    = last_data_date - timedelta(days=1)
            start_date = target
            end_date   = target

        # --- 2. DETEKSI BULAN / TAHUN SPESIFIK ---
        elif True:
            specific_year = re.search(r"\b(20\d{2})\b", query)
            found_month   = next(
                (val for name, val in MONTHS_ID.items() if name in query), None
            )

            if found_month and specific_year:
                y     = int(specific_year.group(1))
                label = (f"Bulan "
                         f"{list(MONTHS_ID.keys())[list(MONTHS_ID.values()).index(found_month)].capitalize()} "
                         f"{y}")
                start_date = datetime(y, found_month, 1)
                next_m = found_month + 1 if found_month < 12 else 1
                next_y = y if found_month < 12 else y + 1
                end_date   = datetime(next_y, next_m, 1) - timedelta(days=1)

            elif specific_year and "tahun" in query:
                y          = int(specific_year.group(1))
                label      = f"Tahun {y}"
                start_date = datetime(y, 1, 1)
                end_date   = datetime(y, 12, 31)

        # --- 3. FALLBACK DEFAULT ---
        if start_date is None:
            label      = "30 Hari Terakhir"
            end_date   = last_data_date
            start_date = last_data_date - timedelta(days=30)

        # --- EKSEKUSI FILTER DATA ---
        subset = [d for d in self.price_data if start_date <= d['date'] <= end_date]

        if not subset:
            return f"Maaf, tidak ada data harga untuk periode {label}."

        prices = [d['price'] for d in subset]
        avg_p  = np.mean(prices)
        max_p  = np.max(prices)
        min_p  = np.min(prices)

        max_date_str = next((d['date_str'] for d in subset if d['price'] == max_p), "?")
        min_date_str = next((d['date_str'] for d in subset if d['price'] == min_p), "?")

        # --- FORMAT JAWABAN SESUAI PERTANYAAN ---
        if any(x in query for x in ["tertinggi", "max", "mahal", "paling tinggi"]):
            return (
                f"📈 **Harga Tertinggi ({label})**\n"
                f"{_fmt_usd(max_p)}\n"
                f"(Tercatat pada tanggal {max_date_str})"
            )

        elif any(x in query for x in ["terendah", "min", "murah", "paling rendah"]):
            return (
                f"📉 **Harga Terendah ({label})**\n"
                f"{_fmt_usd(min_p)}\n"
                f"(Tercatat pada tanggal {min_date_str})"
            )

        elif "rata" in query or "average" in query:
            return (
                f"⚖️ **Rata-rata Harga ({label})**\n"
                f"{_fmt_usd(avg_p)}"
            )

        # Jawaban Umum
        trend = "NAIK 📈" if prices[-1] > prices[0] else "TURUN 📉"
        return (
            f"📊 **Statistik {label}**\n"
            f"- Rata-rata : {_fmt_usd(avg_p)}\n"
            f"- Tertinggi : {_fmt_usd(max_p)} ({max_date_str})\n"
            f"- Terendah  : {_fmt_usd(min_p)} ({min_date_str})\n"
            f"- Tren      : {trend}"
        )