import os
import re
import requests
import sys
from datetime import datetime, timedelta
import statistics

# === SETUP PATH ABSOLUT ===
# BASE_DIR = folder chatbot/, ROOT_DIR = root project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

# Tambahkan chatbot/ ke sys.path agar import antar modul di folder ini tetap berjalan
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# === PATH FILE ===
FILE_CSV = os.path.join(ROOT_DIR, "data", "Data Historis Minyak Sawit AS Berjangka.csv")
FILE_TXT = os.path.join(ROOT_DIR, "data", "faq.txt")
# MODEL_PATH dihapus — model dilatih langsung dalam runtime (train-in-runtime)

# === IMPORT KONFIGURASI ===
try:
    from config_keywords import SYSTEM_PERSONA, STRICT_PROMPT, SMALL_TALK, INTENT_KEYWORDS
    from faq_handler import FAQHandler
    from forecast_handler import CPOForecaster
    from price_analytics import PriceAnalyzer
except ImportError as e:
    print(f"ERROR: Gagal memuat modul pendukung: {e}")
    sys.exit(1)

# === SETUP OLLAMA ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-coder:480b-cloud"


def parse_indonesian_number(s: str) -> float:
    """
    Konversi angka format Indonesia ke float.
    Contoh: '1.140,25' -> 1140.25, '993,25' -> 993.25
    """
    s = s.strip().replace('"', '')
    # Titik sebagai pemisah ribuan, koma sebagai desimal
    s = s.replace('.', '').replace(',', '.')
    return float(s)


class SmartChatbot:
    def __init__(self):
        print(f"Membangun Kecerdasan Sobat INL (Model: {MODEL_NAME})...")
        self.price_data = []
        self._load_database()

        self.faq_handler = FAQHandler(FILE_TXT)

        # pipeline di-inject dari api_stream.py setelah CPOPipeline.train() selesai
        # (lihat SmartChatbot.set_pipeline)
        self.pipeline = None

        self.analyzer = PriceAnalyzer(self.price_data)

    def set_pipeline(self, pipeline):
        """Inject CPOPipeline setelah training selesai di api_stream.py."""
        self.pipeline = pipeline
        print("CPOPipeline berhasil di-inject ke SmartChatbot.")


    def _load_database(self):
        """Membaca CSV Data Historis Minyak Sawit AS Berjangka."""
        print("Sedang membaca database CSV...")
        if not os.path.exists(FILE_CSV):
            print(f"File CSV tidak ditemukan: {FILE_CSV}")
            return

        try:
            import csv

            with open(FILE_CSV, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tgl_str   = row.get("Tanggal",   "").strip().strip('"')
                        close_str = row.get("Terakhir",  "").strip().strip('"')
                        open_str  = row.get("Pembukaan", "").strip().strip('"')
                        high_str  = row.get("Tertinggi", "").strip().strip('"')
                        low_str   = row.get("Terendah",  "").strip().strip('"')
                        pct_str   = row.get("Perubahan%","").strip().strip('"')

                        if not tgl_str or not close_str:
                            continue

                        # Format tanggal: DD/MM/YYYY
                        dt    = datetime.strptime(tgl_str, "%d/%m/%Y")
                        close = parse_indonesian_number(close_str)
                        open_ = parse_indonesian_number(open_str)  if open_str else close
                        high  = parse_indonesian_number(high_str)  if high_str else close
                        low   = parse_indonesian_number(low_str)   if low_str  else close
                        pct   = parse_indonesian_number(
                            pct_str.replace('%', '')) if pct_str else 0.0

                        self.price_data.append({
                            "date"    : dt,
                            "price"   : close,
                            "date_str": dt.strftime("%Y-%m-%d"),
                            # OHLC untuk feature engineering model LSTM
                            "open"    : open_,
                            "high"    : high,
                            "low"     : low,
                            "pct"     : pct,
                        })

                    except Exception:
                        continue

            self.price_data.sort(key=lambda x: x['date'])
            print(f"Berhasil memuat {len(self.price_data)} baris data harga.")

            if self.price_data:
                self.min_date = self.price_data[0]['date']
                self.max_date = self.price_data[-1]['date']
                print(f"Range Data: {self.min_date.year} s/d {self.max_date.year}")
                print(f"Harga terakhir: USD {self.price_data[-1]['price']:.2f}/MT "
                      f"({self.price_data[-1]['date_str']})")

        except Exception as e:
            print(f"Gagal membaca dataset CSV: {e}")

    def preprocess_prices(self):
        """Kembalikan DataFrame harga Close yang sudah di-resample & interpolasi."""
        import pandas as pd

        df = pd.DataFrame(self.price_data)
        df = df[['date', 'price']]
        df.set_index('date', inplace=True)

        df = df.groupby(df.index).mean()
        df = df.asfreq('D')
        df['price'] = df['price'].interpolate()

        return df

    def build_ohlc_dataframe(self) -> 'pd.DataFrame':
        """Buat DataFrame OHLC lengkap untuk input ke CPOForecaster."""
        import pandas as pd

        df = pd.DataFrame(self.price_data)
        df = df.rename(columns={
            'date' : 'Date',
            'price': 'Close',
            'open' : 'Open',
            'high' : 'High',
            'low'  : 'Low',
            'pct'  : 'Pct_Change',
        })
        # Pastikan kolom yang dibutuhkan ada
        for col in ['Open', 'High', 'Low', 'Pct_Change']:
            if col not in df.columns:
                df[col] = df['Close']
        df = df.sort_values('Date').reset_index(drop=True)
        return df

    def _check_keyword_match(self, text, category):
        keywords = INTENT_KEYWORDS.get(category, [])
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        return False

    def _classify_intent(self, query):
        q_clean = query.lower()
        if self._check_keyword_match(q_clean, "FORECAST"):   return "FORECAST"
        if self._check_keyword_match(q_clean, "INFO_COMPANY"): return "INFO_COMPANY"

        # Jika ada pola tanggal spesifik, kita anggap GENERAL agar mencari harga harian dulu
        if re.search(r"(\d{1,2})[\s\-/]", q_clean):
            return "GENERAL"

        if self._check_keyword_match(q_clean, "ANALYSIS"):   return "ANALYSIS"
        if self._check_keyword_match(q_clean, "PRICE_QUERY"): return "GENERAL"
        return "GENERAL"

    def format_date_indo(self, dt_obj):
        months = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                  "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
        return f"{dt_obj.day} {months[dt_obj.month]} {dt_obj.year}"

    def analyze_period(self, text):
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        target_year = int(year_match.group(0)) if year_match else None

        month_map = {
            "januari": 1, "jan": 1, "februari": 2, "feb": 2, "maret": 3, "mar": 3,
            "april": 4, "apr": 4, "mei": 5, "may": 5, "juni": 6, "jun": 6,
            "juli": 7, "jul": 7, "agustus": 8, "agu": 8, "agust": 8,
            "september": 9, "sep": 9, "oktober": 10, "okt": 10,
            "november": 11, "nov": 11, "desember": 12, "des": 12
        }
        target_month = None
        target_month_name = ""
        for name, num in month_map.items():
            if re.search(rf"\b{name}\b", text):
                target_month = num
                target_month_name = name.capitalize()
                break

        if not target_year:
            return None

        filtered = [p for p in self.price_data if p['date'].year == target_year]
        period_label = f"Tahun {target_year}"
        if target_month:
            filtered = [p for p in filtered if p['date'].month == target_month]
            period_label = f"Bulan {target_month_name} {target_year}"

        if not filtered:
            return None

        prices = [p['price'] for p in filtered]
        max_price = max(prices)
        min_price = min(prices)
        avg_price = statistics.mean(prices)
        max_date = next(p['date'] for p in filtered if p['price'] == max_price)
        min_date = next(p['date'] for p in filtered if p['price'] == min_price)

        return f"""
        STATISTIK PERIODE ({period_label}):
        - Harga Tertinggi: USD {max_price:.2f}/MT (pada {self.format_date_indo(max_date)})
        - Harga Terendah: USD {min_price:.2f}/MT (pada {self.format_date_indo(min_date)})
        - Rata-rata     : USD {avg_price:.2f}/MT
        - Jumlah Data   : {len(prices)} hari transaksi
        """

    def find_price_by_date(self, text):
        # Regex untuk berbagai format tanggal
        pattern = (r"(\d{1,2})[\s\-/]"
                   r"(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|"
                   r"jan|feb|mar|apr|mei|jun|jul|agu|sep|okt|nov|des|\d{1,2})[\s\-/](\d{4})")
        match = re.search(pattern, text.lower())
        if match:
            day, m_str, year = match.groups()
            month_map = {
                "januari":1,"februari":2,"maret":3,"april":4,"mei":5,"juni":6,
                "juli":7,"agustus":8,"september":9,"oktober":10,"november":11,"desember":12,
                "jan":1,"feb":2,"mar":3,"apr":4,"mei":5,"jun":6,
                "jul":7,"agu":8,"sep":9,"okt":10,"nov":11,"des":12
            }
            month = month_map.get(m_str, int(m_str) if m_str.isdigit() else 1)
            target_date_str = f"{year}-{int(month):02d}-{int(day):02d}"

            for entry in self.price_data:
                if entry['date_str'] == target_date_str:
                    return (f"KONFIRMASI: Harga CPO pada {self.format_date_indo(entry['date'])} "
                            f"ADALAH USD {entry['price']:.2f}/MT.")
            return (f"INFO: Data harga spesifik untuk tanggal {day} {m_str} {year} "
                    f"tidak ada di database.")
        return None

    def _call_ollama(self, prompt):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=300000)
            return (r.json().get("response", "Maaf, gagal merespon.")
                    if r.status_code == 200 else f"Error: {r.status_code}")
        except Exception as e:
            return f"Koneksi AI terputus: {str(e)}"

    def get_response(self, raw_q: str, cached_forecast: dict = None) -> str:
        q_clean = raw_q.lower()

        # 1. SMALL TALK
        for cat, data in SMALL_TALK.items():
            for kw in data["keywords"]:
                if re.search(rf"\b{re.escape(kw)}\b", q_clean):
                    return data["response"]

        # 2. PRIORITAS UTAMA: Cari Data Tanggal Spesifik
        specific_date_info = self.find_price_by_date(q_clean)

        # 3. KLASIFIKASI INTENT
        intent = self._classify_intent(raw_q)

        # Override: Jika ada tahun tapi bukan prediksi, masuk Analysis
        if re.search(r'\b(19|20)\d{2}\b', q_clean) and "prediksi" not in q_clean:
            intent = "ANALYSIS"

        context     = ""
        instruction = "Jawab dengan ramah sebagai Sobat INL."

        # 4. LOGIC EXECUTION
        if intent == "FORECAST":
            if not self.pipeline or not self.pipeline.is_trained:
                return "Maaf, modul prediksi sedang dipersiapkan. Silakan coba beberapa saat lagi."

            num_days = 7
            if "besok" in q_clean:
                num_days = 1
            elif "lusa" in q_clean:
                num_days = 2
            else:
                d = re.search(r'(\d+)', q_clean)
                if d:
                    num_days = min(int(d.group(1)), 30)

            try:
                # Prioritas: gunakan cache dari api_stream (SAMA dengan grafik dashboard)
                # Fallback: panggil pipeline.forecast() jika tidak ada cache
                if cached_forecast:
                    result = cached_forecast
                    # Jika user minta < 7 hari, potong hasilnya
                    if num_days < len(result["forecasts"]):
                        result = dict(result)
                        result["forecasts"] = result["forecasts"][:num_days]
                else:
                    result = self.pipeline.forecast(horizon=num_days, n_mc_samples=100)

                forecasts  = result["forecasts"]
                last_price = result["last_known_price"]

                context = f"PREDIKSI HARGA CPO (USD/MT) — Last known: USD {last_price:.2f}/MT:\n"
                for f in forecasts:
                    context += (
                        f"- {f['date']} ({f['day_of_week']}): "
                        f"USD {f['predicted_price']:.2f}/MT "
                        f"[{f['lower_90']:.2f}–{f['upper_90']:.2f}] "
                        f"({f['change_pct']:+.2f}%)\n"
                    )
                instruction = (
                    "Sampaikan hasil prediksi harga CPO dalam satuan USD per Metric Ton. "
                    "Sertakan rentang kepercayaan 90% dan perubahan persentase."
                )
            except Exception as e:
                return f"Maaf, prediksi gagal: {e}"


        elif intent == "INFO_COMPANY":
            faq     = self.faq_handler.search(q_clean, top_k=3)
            context = f"KNOWLEDGE BASE:\n{faq}"
            instruction = "Jawab berdasarkan profil perusahaan INL."

        elif intent in ("ANALYSIS", "GENERAL"):
            analysis_result = self.analyzer.analyze(q_clean)

            if specific_date_info and "KONFIRMASI" in specific_date_info:
                context     = (f"DATA HARGA HARIAN:\n{specific_date_info}\n\n"
                               f"STATISTIK PERIODE:\n{analysis_result}")
                instruction = ("Sebutkan harga harian yang ditemukan, "
                               "lalu berikan analisis statistik periode tersebut. "
                               "Satuan harga adalah USD per Metric Ton (USD/MT).")
            else:
                context     = analysis_result
                instruction = ("Berikan analisis mendalam berdasarkan data statistik "
                               "harga CPO yang disediakan. Satuan harga adalah USD per Metric Ton (USD/MT).")

        full_prompt = (f"{SYSTEM_PERSONA}\n\nDATA:\n{context}\n\n"
                       f"INSTRUKSI:\n{instruction}\n{STRICT_PROMPT}\n\n"
                       f"USER: {raw_q}\nJAWABAN:")
        return self._call_ollama(full_prompt)


if __name__ == "__main__":
    bot = SmartChatbot()
    while True:
        txt = input("\nUser: ")
        if txt.lower() == 'exit':
            break
        print("Bot:", bot.get_response(txt))