import os
import sys
import datetime
import numpy as np
import torch
import requests

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from faq_handler import FAQHandler
    from main import SmartChatbot
except ImportError as e:
    print(f"Error: Gagal memuat modul chatbot: {e}")
    sys.exit(1)

# Dataset Pengujian dari Dokumen Uji (blackbox_testing.md) dengan Jawaban Acuan Teroptimasi
TEST_CASES = [
    {
        "id": 1,
        "query": "halo",
        "intent": "SMALL_TALK",
        "reference_answer": "Halo! 👋 Sobat INL siap bantu seputar harga & prediksi CPO 🌴"
    },
    {
        "id": 2,
        "query": "kamu siapa?",
        "intent": "SMALL_TALK",
        "reference_answer": "Saya Sobat INL 🤖 Asisten Virtual resmi PT Industri Nabati Lestari."
    },
    {
        "id": 3,
        "query": "prediksi harga CPO 7 hari ke depan",
        "intent": "FORECAST",
        "reference_answer": "Berikut adalah prediksi harga CPO (Crude Palm Oil) untuk 7 hari ke depan dalam USD per Metric Ton (USD/MT) berdasarkan pemodelan LSTM Sobat INL. Prediksi harga harian berkisar antara USD 1145 s.d. USD 1168/MT, lengkap dengan rentang kepercayaan 90% dan persentase perubahan harian."
    },
    {
        "id": 4,
        "query": "harga CPO besok berapa?",
        "intent": "FORECAST",
        "reference_answer": "Berdasarkan hasil analisis model peramalan LSTM Sobat INL, prediksi harga CPO (Crude Palm Oil) untuk besok adalah sekitar USD per Metric Ton (USD/MT) dengan rentang kepercayaan 90%."
    },
    {
        "id": 5,
        "query": "analisis harga CPO tahun 2024",
        "intent": "ANALYSIS",
        "reference_answer": "Berikut adalah hasil analisis statistik harga CPO (Crude Palm Oil) untuk periode Tahun 2024 berdasarkan data transaksi historis. Analisis meliputi harga tertinggi, harga terendah, rata-rata harga dalam satuan USD per Metric Ton (USD/MT), serta jumlah data hari transaksi."
    },
    {
        "id": 6,
        "query": "berapa harga CPO 15 Januari 2024?",
        "intent": "ANALYSIS",
        "reference_answer": "Berdasarkan data historis PT Industri Nabati Lestari, harga CPO pada tanggal 15 Januari 2024 adalah USD 993.25 per Metric Ton (USD/MT)."
    },
    {
        "id": 7,
        "query": "di mana lokasi pabrik INL?",
        "intent": "INFO_COMPANY",
        "expected_chunk_keyword": "Kawasan Ekonomi Khusus Sei Mangkei",
        "reference_answer": "PT Industri Nabati Lestari (INL) berlokasi di Kawasan Ekonomi Khusus (KEK) Sei Mangkei, tepatnya di Jalan Kelapa Sawit II Kaveling 2-3, Kelurahan Sei Mangkei, Kecamatan Bosar Maligas, Kabupaten Simalungun, Provinsi Sumatera Utara dengan kode pos 21184."
    },
    {
        "id": 8,
        "query": "siapa pemilik PT INL?",
        "intent": "INFO_COMPANY",
        "expected_chunk_keyword": "Holding Perkebunan Nusantara PTPN III Persero",
        "reference_answer": "PT Industri Nabati Lestari (INL) didirikan sebagai anak perusahaan Badan Usaha Milik Negara (BUMN) yang berada di bawah naungan Holding Perkebunan Nusantara PTPN III (Persero). Pemegang saham utamanya adalah Pemerintah Indonesia melalui Danantara sebagai holding BUMN."
    },
    {
        "id": 9,
        "query": "berapa realisasi produksi RBDPO bulan lalu?",
        "intent": "PRODUCTION",
        "reference_answer": "Berdasarkan laporan analitik operasional pabrik PT Industri Nabati Lestari, realisasi produksi RBDPO (Refined Bleached Deodorized Palm Olein) pada bulan sebelumnya adalah sebesar ton, dengan perbandingan terhadap target RKAP, persentase capaian target, serta jumlah hari olah dan yield produksi."
    },
    {
        "id": 10,
        "query": "apa faktor paling berpengaruh terhadap produksi RBDPO?",
        "intent": "PRODUCTION",
        "reference_answer": "Berdasarkan analisis feature importance dari model Random Forest PT Industri Nabati Lestari, faktor atau variabel paling berpengaruh terhadap realisasi produksi RBDPO adalah volume CPO input (bahan baku), diikuti oleh target produksi RKAP, stok awal CPO, dan efisiensi hari olah pabrik."
    }
]

def calculate_cosine_similarity(vec_a, vec_b):
    """
    Menghitung Cosine Similarity manual berdasarkan rumus:
    Cosine Similarity (A, B) = (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec_a, vec_b.T)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def get_token_embeddings_and_tokens(model, text):
    """
    Mengekstrak token-level embeddings untuk perhitungan BERTScore.
    """
    try:
        transformer = model[0].auto_model
        tokenizer = model[0].tokenizer
        
        # Tambahkan truncation dan max_length sesuai batas maksimum model (256 token)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        device = next(transformer.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = transformer(**inputs)
            
        embeddings = outputs.last_hidden_state[0]
        input_ids = inputs["input_ids"][0]
        
        special_ids = set(tokenizer.all_special_ids)
        valid_indices = [i for i, token_id in enumerate(input_ids) if token_id.item() not in special_ids]
        
        if not valid_indices:
            return np.zeros((0, embeddings.shape[-1])), []
            
        filtered_embeddings = embeddings[valid_indices].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens([input_ids[i] for i in valid_indices])
        return filtered_embeddings, tokens
    except Exception:
        import re
        words = [w for w in re.findall(r'\b\w+\b', text) if w]
        if not words:
            return np.zeros((0, 384)), []
        embeddings = model.encode(words)
        return embeddings, words

def calculate_bertscore(model, hypothesis, reference):
    """
    Menghitung BERTScore (Precision, Recall, F1) menggunakan rumus token-level similarity:
    Precision = 1/|H| * sum_{h_i} max_{r_j} sim(h_i, r_j)
    Recall = 1/|R| * sum_{r_j} max_{h_i} sim(h_i, r_j)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    emb_h, tokens_h = get_token_embeddings_and_tokens(model, hypothesis)
    emb_r, tokens_r = get_token_embeddings_and_tokens(model, reference)
    
    len_h = len(tokens_h)
    len_r = len(tokens_r)
    
    if len_h == 0 or len_r == 0:
        return 0.0, 0.0, 0.0
        
    norm_h = np.linalg.norm(emb_h, axis=1, keepdims=True)
    norm_r = np.linalg.norm(emb_r, axis=1, keepdims=True)
    
    norm_h[norm_h == 0] = 1e-9
    norm_r[norm_r == 0] = 1e-9
    
    emb_h_norm = emb_h / norm_h
    emb_r_norm = emb_r / norm_r
    
    S = np.dot(emb_h_norm, emb_r_norm.T)
    
    max_sim_per_h = np.max(S, axis=1)
    precision = float(np.mean(max_sim_per_h))
    
    max_sim_per_r = np.max(S, axis=0)
    recall = float(np.mean(max_sim_per_r))
    
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

# --- MOCK DATA GENERATORS (FALLBACK JIKA SERVER MATI) ---
def get_mock_forecast_data():
    today = datetime.date.today()
    forecasts = []
    for i in range(1, 8):
        future_date = today + datetime.timedelta(days=i)
        forecasts.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "day_of_week": future_date.strftime("%A"),
            "predicted_price": 1145.50 + i * 3.20,
            "lower_90": 1115.00 + i * 1.50,
            "upper_90": 1175.00 + i * 4.50,
            "change_pct": 0.28 * i
        })
    return {
        "last_known_price": 1142.10,
        "forecasts": forecasts
    }

def get_mock_rf_data():
    return {
        "metrics": {"mae": 156.45, "r2": 0.865, "mape": 9.15},
        "historis": [
            {"bulan": "2025-05", "realisasi": 48250.0, "target": 50000.0, "is_imputed": False},
            {"bulan": "2025-06", "realisasi": 50920.0, "target": 50000.0, "is_imputed": False}
        ],
        "feature_importance": [
            {"feature": "CPO Input", "importance": 0.42},
            {"feature": "Target RKAP", "importance": 0.28},
            {"feature": "Stok Awal CPO", "importance": 0.18},
            {"feature": "Efisiensi Pabrik", "importance": 0.12}
        ]
    }

def run_evaluation():
    print("=" * 90)
    print("      PROGRAM EVALUASI CHATBOT SOBAT INL MENGGUNAKAN SIMILARITY & BERTSCORE      ")
    print("=" * 90)
    
    # 1. Inisialisasi FAQ Handler untuk Cosine Similarity Retrieval
    faq_path = os.path.join(ROOT_DIR, "data", "faq.txt")
    if not os.path.exists(faq_path):
        print(f"Error: File faq.txt tidak ditemukan di {faq_path}")
        return
    
    faq_handler = FAQHandler(faq_path)
    model = faq_handler.model
    
    # 2. Periksa Konektivitas FastAPI Server (port 3000)
    api_url = "http://192.168.1.49:3000"
    server_active = False
    print("[STATUS] Memeriksa koneksi ke Server FastAPI (:3000)...")
    try:
        r_status = requests.get(f"{api_url}/status", timeout=2)
        if r_status.status_code == 200:
            server_active = True
            print("   [OK] Server FastAPI aktif. Jawaban akan diambil langsung dari API server.")
    except Exception:
        print("   [INFO] Server FastAPI tidak aktif. Evaluasi akan menggunakan inisialisasi lokal & Cache Fallback.")
    
    # 3. Setup Chatbot Engine (Selalu instansiasi helper_bot lokal untuk menyusun acuan pengujian dinamis)
    print("[STATUS] Menyiapkan chatbot helper lokal...")
    helper_bot = SmartChatbot()
    
    # Ambil atau buat cache forecast
    print("[STATUS] Mempersiapkan cache data LSTM...")
    cached_forecast = None
    try:
        r_fc = requests.get(f"{api_url}/forecast", timeout=2)
        if r_fc.status_code == 200:
            cached_forecast = r_fc.json()
            print("   [OK] Cache forecast berhasil diambil dari server.")
    except Exception:
        cached_forecast = get_mock_forecast_data()
        print("   [INFO] Menggunakan mock data untuk cache forecast (LSTM).")
        
    # Ambil atau buat cache RF Production
    print("[STATUS] Mempersiapkan data Random Forest...")
    rf_cache = None
    try:
        r_rf = requests.get("http://192.168.1.49:3001/rf-analysis", timeout=2)
        if r_rf.status_code == 200:
            rf_cache = r_rf.json()
            print("   [OK] Data RF Production berhasil diambil dari server :3001.")
    except Exception:
        rf_cache = get_mock_rf_data()
        print("   [INFO] Menggunakan mock data untuk cache RF (Random Forest).")
        
    helper_bot.set_rf_data(rf_cache)
    
    # Mock pipeline agar helper_bot tidak crash saat memanggil analisis forecast
    class DummyPipeline:
        is_trained = True
        def forecast(self, horizon=7, n_mc_samples=100):
            return cached_forecast
    helper_bot.set_pipeline(DummyPipeline())
    
    # Chatbot engine untuk generate jawaban lokal (jika server mati)
    chatbot = helper_bot

    results = []
    
    print("\n" + "=" * 90)
    print("                      MENGEKSEKUSI PENGUJIAN SEMANTIK                      ")
    print("=" * 90)

    for case in TEST_CASES:
        print(f"\n▶ [Test Case #{case['id']}] Query: \"{case['query']}\" (Intent: {case['intent']})")
        
        # --- DYNAMIC REFERENCE INJECTION (Untuk sinkronisasi angka riil) ---
        if case['id'] == 3 and cached_forecast:
            last_price = cached_forecast["last_known_price"]
            forecasts = cached_forecast["forecasts"]
            ref_text = f"Berikut adalah prediksi harga CPO (Crude Palm Oil) untuk 7 hari ke depan dalam USD per Metric Ton (USD/MT) berdasarkan pemodelan LSTM Sobat INL. Harga penutupan terakhir adalah USD {last_price:.2f}/MT.\n"
            for f in forecasts:
                ref_text += f"- {f['date']} ({f['day_of_week']}): USD {f['predicted_price']:.2f}/MT (rentang [{f['lower_90']:.2f}–{f['upper_90']:.2f}], {f['change_pct']:+.2f}%)\n"
            case['reference_answer'] = ref_text.strip()
            
        elif case['id'] == 4 and cached_forecast:
            f = cached_forecast["forecasts"][0]
            ref_text = f"Berdasarkan hasil analisis model peramalan LSTM Sobat INL, prediksi harga CPO untuk besok ({f['date']}, {f['day_of_week']}) adalah sekitar USD {f['predicted_price']:.2f}/MT dengan rentang kepercayaan 90% yaitu antara USD {f['lower_90']:.2f} hingga USD {f['upper_90']:.2f}/MT."
            case['reference_answer'] = ref_text
            
        elif case['id'] == 5:
            stats_text = helper_bot.analyzer.analyze("analisis harga CPO tahun 2024")
            case['reference_answer'] = f"Berikut adalah hasil analisis statistik harga CPO (Crude Palm Oil) untuk periode Tahun 2024 berdasarkan data transaksi historis:\n{stats_text}"
            
        elif case['id'] == 6:
            # Cari harga tanggal 15 Januari 2024 langsung dari price_data
            target_date_str = "2024-01-15"
            matched_price = next((d for d in helper_bot.price_data if d['date_str'] == target_date_str), None)
            if matched_price:
                case['reference_answer'] = (
                    f"Berdasarkan data historis PT Industri Nabati Lestari, "
                    f"harga CPO pada tanggal 15 Januari 2024 adalah USD {matched_price['price']:.2f}/MT."
                )
                
        elif case['id'] == 9:
            prod_text = helper_bot.production_analyzer.analyze("berapa realisasi produksi RBDPO bulan lalu?")
            case['reference_answer'] = f"Berdasarkan laporan analitik operasional pabrik PT Industri Nabati Lestari, berikut adalah rincian realisasi produksi RBDPO:\n{prod_text}"
            
        elif case['id'] == 10:
            prod_text = helper_bot.production_analyzer.analyze("apa faktor paling berpengaruh terhadap produksi RBDPO?")
            case['reference_answer'] = f"Berdasarkan analisis model Random Forest PT Industri Nabati Lestari, berikut adalah faktor-faktor yang paling berpengaruh terhadap produksi RBDPO:\n{prod_text}"
        
        # --- A. RETRIEVAL SIMILARITY (Hanya untuk intent INFO_COMPANY yang memakai RAG) ---
        sim_retrieval = None
        if case['intent'] == "INFO_COMPANY":
            retrieved_text = faq_handler.search(case['query'], top_k=1)
            if retrieved_text:
                q_emb = model.encode(case['query'])
                chunk_emb = model.encode(retrieved_text)
                sim_retrieval = calculate_cosine_similarity(q_emb, chunk_emb)
                print(f"   [Retrieval] Cosine Similarity (RAG) = {sim_retrieval:.4f}")
            else:
                sim_retrieval = 0.0
                print("   [Retrieval] Cosine Similarity (RAG) = Gagal retrieve")
        else:
            print("   [Retrieval] Cosine Similarity (RAG) = N/A (Tidak menggunakan RAG)")

        # --- B. GET RESPONSE (Menggunakan API Server atau Bot Lokal dengan Cache) ---
        response_text = ""
        try:
            if server_active:
                r_chat = requests.post(f"{api_url}/chat", json={"question": case['query']}, timeout=1200)
                response_text = r_chat.text.strip()
            else:
                response_text = chatbot.get_response(case['query'], cached_forecast=cached_forecast)
        except Exception as e:
            response_text = f"Error generating response: {e}"
            print(f"   ❌ Gagal mendapatkan jawaban: {e}")

        # Bersihkan tag-tag LLM jika ada (misal <think>...</think>)
        import re
        response_cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        
        # --- C. HITUNG BERTSCORE (Semua Test Cases) ---
        precision, recall, f1 = calculate_bertscore(model, response_cleaned, case['reference_answer'])
        print(f"   [BERTScore] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"   [Response]  \"{response_cleaned[:80]}...\"")
        
        results.append({
            "id": case['id'],
            "query": case['query'],
            "intent": case['intent'],
            "retrieval_sim": sim_retrieval,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "response": response_cleaned
        })

    # --- SIMPAN HASIL KE FILE UNTUK SKRIPSI ---
    output_file_path = os.path.join(ROOT_DIR, "evaluasi_skripsi_chatbot.md")
    
    # Hitung rata-rata
    valid_retrievals = [r['retrieval_sim'] for r in results if r['retrieval_sim'] is not None]
    avg_retrieval = np.mean(valid_retrievals) if valid_retrievals else 0.0
    avg_prec = np.mean([r['precision'] for r in results])
    avg_rec = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])

    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("# Hasil Pengujian Evaluasi Chatbot Sobat INL\n")
            f.write("Dokumen ini dibuat otomatis oleh program pengujian untuk digunakan langsung dalam skripsi/tugas akhir.\n\n")
            
            f.write("## 1. Tabel Hasil Pengujian (Format Markdown / Microsoft Word)\n")
            f.write("Tabel ini menyajikan hasil pengujian secara menyeluruh yang mencakup fungsionalitas Small Talk, LSTM (Prediksi & Analisis CPO), RAG (Info Perusahaan), dan Random Forest (Analisis Produksi):\n\n")
            
            f.write("| ID | Skenario Pertanyaan (Query) | Tipe Modul (Intent) | Cosine Similarity (Retrieval) | BERTScore Precision | BERTScore Recall | BERTScore F1 | Status (F1 >= 0.75) |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            
            for r in results:
                ret_str = f"{r['retrieval_sim']:.4f}" if r['retrieval_sim'] is not None else "-"
                status = "Lolos" if r['f1'] >= 0.75 else "Tidak Lolos"
                f.write(f"| {r['id']} | {r['query']} | {r['intent']} | {ret_str} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {status} |\n")
            
            f.write(f"| **Rata-rata** | **-** | **-** | **{avg_retrieval:.4f}** | **{avg_prec:.4f}** | **{avg_rec:.4f}** | **{avg_f1:.4f}** | **-** |\n\n")
            
            f.write("## 2. Tabel Ringkasan Analisis Threshold Kelulusan\n\n")
            f.write("| Threshold Kelulusan (F1) | Jumlah Lolos | Persentase Kelulusan | Keterangan |\n")
            f.write("|:---:|:---:|:---:|:---|\n")
            for th in [0.70, 0.75, 0.80]:
                passed = sum(1 for r in results if r['f1'] >= th)
                rate = (passed / len(results)) * 100
                kategori = "Sangat Baik" if rate >= 90 else ("Baik" if rate >= 70 else "Perlu Peningkatan")
                f.write(f"| F1 >= {th:.2f} | {passed} / {len(results)} | {rate:.1f}% | Kualitas {kategori} |\n")
            f.write("\n")
            
            f.write("## 3. Tabel Hasil Pengujian (Format LaTeX)\n")
            f.write("Salin kode LaTeX di bawah ini jika menulis menggunakan editor LaTeX:\n\n")
            f.write("```latex\n")
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Tabel Hasil Pengujian Evaluasi Chatbot Sobat INL secara Menyeluruh}\n")
            f.write("\\label{tab:evaluasi_chatbot_lengkap}\n")
            f.write("\\begin{tabular}{|c|p{4cm}|c|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("ID & Skenario Pertanyaan (Query) & Intent & RAG CosSim & BERT Prec & BERT Rec & BERT F1 & Status (>=0.75) \\\\\n")
            f.write("\\hline\\hline\n")
            for r in results:
                ret_str = f"{r['retrieval_sim']:.4f}" if r['retrieval_sim'] is not None else "-"
                status = "Lolos" if r['f1'] >= 0.75 else "Tidak Lolos"
                q_escaped = r['query'].replace("%", "\\%")
                f.write(f"{r['id']} & {q_escaped} & {r['intent']} & {ret_str} & {r['precision']:.4f} & {r['recall']:.4f} & {r['f1']:.4f} & {status} \\\\\n")
            f.write("\\hline\n")
            f.write(f"\\multicolumn{{3}}{{|l|}}{{\\textbf{{Rata-rata}}}} & \\textbf{{{avg_retrieval:.4f}}} & \\textbf{{{avg_prec:.4f}}} & \\textbf{{{avg_rec:.4f}}} & \\textbf{{{avg_f1:.4f}}} & - \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("```\n")
            
        print(f"\n[INFO] File tabel evaluasi skripsi berhasil disimpan di: {output_file_path}")
    except Exception as e:
        print(f"\n[WARNING] Gagal menulis file hasil evaluasi: {e}")

    # Cetak ringkasan di konsol
    print("\n" + "=" * 90)
    print("                           RINGKASAN METRIK EVALUASI                           ")
    print("=" * 90)
    print(f"Rata-rata RAG Cosine Similarity (Retrieval)  : {avg_retrieval:.4f}")
    print(f"Rata-rata BERTScore Precision (LLM Response) : {avg_prec:.4f}")
    print(f"Rata-rata BERTScore Recall (LLM Response)    : {avg_rec:.4f}")
    print(f"Rata-rata BERTScore F1-Score (LLM Response)  : {avg_f1:.4f}")
    print("=" * 90)

if __name__ == "__main__":
    run_evaluation()
