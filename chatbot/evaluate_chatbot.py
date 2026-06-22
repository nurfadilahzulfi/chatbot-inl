import os
import sys
import numpy as np
import torch

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

# Dataset Pengujian (Test Cases)
TEST_CASES = [
    {
        "id": 1,
        "query": "Di mana lokasi kantor dan pabrik PT INL?",
        "expected_chunk_keyword": "Kawasan Ekonomi Khusus Sei Mangkei",
        "reference_answer": "PT INL berlokasi di Kawasan Ekonomi Khusus Sei Mangkei, Jalan Kelapa Sawit II Kaveling 2-3, Kelurahan Sei Mangkei, Kecamatan Bosar Maligas, Kabupaten Simalungun, Provinsi Sumatera Utara."
    },
    {
        "id": 2,
        "query": "Kapan PT Industri Nabati Lestari didirikan?",
        "expected_chunk_keyword": "didirikan pada tanggal 23 Desember 2015",
        "reference_answer": "PT Industri Nabati Lestari didirikan pada tanggal 23 Desember 2015 sebagai langkah strategis hilirisasi industri kelapa sawit."
    },
    {
        "id": 3,
        "query": "Siapa nama Direktur PT INL yang menjabat saat ini?",
        "expected_chunk_keyword": "Rahmanto Amin Jatmiko",
        "reference_answer": "PT INL dipimpin oleh Direktur Rahmanto Amin Jatmiko yang resmi dilantik pada tanggal 9 Januari 2024."
    },
    {
        "id": 4,
        "query": "Berapa kapasitas produksi pabrik minyak kelapa sawit INL?",
        "expected_chunk_keyword": "kapasitas produksi sebesar 600.000 ton per tahun",
        "reference_answer": "PT INL memiliki kapasitas produksi sebesar 600.000 ton per tahun dengan mengoperasikan Palm Oil Refining and Fractionation Plant."
    },
    {
        "id": 5,
        "query": "Apa saja merek minyak goreng kemasan yang diproduksi oleh PT INL?",
        "expected_chunk_keyword": "Salvaco, Minyakita, Nusakita, dan INL",
        "reference_answer": "PT INL memroduksi minyak goreng kemasan dengan merek Salvaco, Minyakita, Nusakita, dan INL."
    },
    {
        "id": 6,
        "query": "Siapa pemilik atau induk perusahaan dari PT INL?",
        "expected_chunk_keyword": "Holding Perkebunan Nusantara PTPN III Persero",
        "reference_answer": "PT INL merupakan anak perusahaan dari Holding Perkebunan Nusantara PTPN III Persero."
    },
    {
        "id": 7,
        "query": "Apa visi utama dari pembentukan PT INL?",
        "expected_chunk_keyword": "mendukung program hilirisasi industri kelapa sawit",
        "reference_answer": "Visi utama INL adalah mendukung program hilirisasi industri kelapa sawit Indonesia dengan mengubah CPO menjadi produk bernilai tambah tinggi."
    },
    {
        "id": 8,
        "query": "Mengapa KEK Sei Mangkei dipilih sebagai lokasi pabrik?",
        "expected_chunk_keyword": "akses mudah ke bahan baku CPO dari kebun-kebun PTPN",
        "reference_answer": "Lokasi di KEK Sei Mangkei memberikan akses mudah ke bahan baku CPO dari kebun-kebun PTPN di Sumatera Utara serta efisiensi biaya logistik."
    },
    {
        "id": 9,
        "query": "Apa saja tantangan utama yang dihadapi oleh PT INL?",
        "expected_chunk_keyword": "fluktuasi harga CPO",
        "reference_answer": "Tantangan utama PT INL meliputi fluktuasi harga CPO global, persaingan industri minyak goreng, dan tuntutan standar keberlanjutan."
    },
    {
        "id": 10,
        "query": "Siapa pemegang saham utama dari PT INL?",
        "expected_chunk_keyword": "Pemerintah Indonesia melalui Danantara",
        "reference_answer": "Pemegang saham utama PT INL adalah Pemerintah Indonesia melalui Danantara sebagai holding BUMN, serta di bawah Holding PTPN III."
    }
]

def calculate_cosine_similarity(vec_a, vec_b):
    """
    Menghitung Cosine Similarity manual menggunakan rumus:
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
    Mengekstrak token embeddings menggunakan model Transformer dari SentenceTransformer.
    Jika gagal, akan fallback ke word-level embeddings menggunakan tokenisasi regex.
    """
    try:
        transformer = model[0].auto_model
        tokenizer = model[0].tokenizer
        
        inputs = tokenizer(text, return_tensors="pt")
        # Cocokkan device dengan model
        device = next(transformer.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = transformer(**inputs)
            
        embeddings = outputs.last_hidden_state[0] # shape: (seq_len, hidden_dim)
        input_ids = inputs["input_ids"][0]
        
        special_ids = set(tokenizer.all_special_ids)
        valid_indices = [i for i, token_id in enumerate(input_ids) if token_id.item() not in special_ids]
        
        if not valid_indices:
            return np.zeros((0, embeddings.shape[-1])), []
            
        filtered_embeddings = embeddings[valid_indices].cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens([input_ids[i] for i in valid_indices])
        return filtered_embeddings, tokens
    except Exception:
        # Fallback ke kata individual jika struktur HuggingFace di dalam model berbeda
        import re
        words = [w for w in re.findall(r'\b\w+\b', text) if w]
        if not words:
            return np.zeros((0, 384)), []
        embeddings = model.encode(words)
        return embeddings, words

def calculate_bertscore(model, hypothesis, reference):
    """
    Menghitung BERTScore (Precision, Recall, F1) berdasarkan rumus:
    Precision = 1/|H| * sum_{h_i} max_{r_j} sim(h_i, r_j)
    Recall = 1/|R| * sum_{r_j} max_{h_i} sim(h_i, r_j)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    # 1. Dapatkan token-level embeddings
    emb_h, tokens_h = get_token_embeddings_and_tokens(model, hypothesis)
    emb_r, tokens_r = get_token_embeddings_and_tokens(model, reference)
    
    len_h = len(tokens_h)
    len_r = len(tokens_r)
    
    if len_h == 0 or len_r == 0:
        return 0.0, 0.0, 0.0
        
    # 2. Hitung Matriks Cosine Similarity untuk setiap pasang token (h_i, r_j)
    # Normalisasi vektor agar perkalian dot product langsung menghasilkan cosine similarity
    norm_h = np.linalg.norm(emb_h, axis=1, keepdims=True)
    norm_r = np.linalg.norm(emb_r, axis=1, keepdims=True)
    
    # Hindari pembagian dengan nol
    norm_h[norm_h == 0] = 1e-9
    norm_r[norm_r == 0] = 1e-9
    
    emb_h_norm = emb_h / norm_h
    emb_r_norm = emb_r / norm_r
    
    # Matriks similarity S berukuran (|H|, |R|)
    S = np.dot(emb_h_norm, emb_r_norm.T)
    
    # 3. Hitung Precision
    # Untuk setiap token di hypothesis, cari kemiripan maksimum dengan token di reference
    max_sim_per_h = np.max(S, axis=1) # shape: (|H|,)
    precision = float(np.mean(max_sim_per_h))
    
    # 4. Hitung Recall
    # Untuk setiap token di reference, cari kemiripan maksimum dengan token di hypothesis
    max_sim_per_r = np.max(S, axis=0) # shape: (|R|,)
    recall = float(np.mean(max_sim_per_r))
    
    # 5. Hitung F1 Score
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def run_evaluation():
    print("=" * 90)
    print("      PROGRAM EVALUASI RAG & CHATBOT SEMANTIC SIMILARITY (BERTScore)      ")
    print("=" * 90)
    
    # 1. Inisialisasi FAQ Handler
    faq_path = os.path.join(ROOT_DIR, "data", "faq.txt")
    print(f"[STATUS] Memuat FAQ Handler dari: {faq_path}")
    if not os.path.exists(faq_path):
        print(f"Error: File faq.txt tidak ditemukan di {faq_path}")
        return
    
    faq_handler = FAQHandler(faq_path)
    model = faq_handler.model
    
    # 2. Cek ketersediaan Server Ollama / Chatbot
    run_llm_test = False
    chatbot = None
    print("\n[STATUS] Memeriksa ketersediaan Server Ollama (Qwen)...")
    try:
        chatbot = SmartChatbot()
        test_response = chatbot.get_response("halo")
        if "Koneksi AI terputus" not in test_response and "Error" not in test_response:
            run_llm_test = True
            print("   [OK] Server Ollama aktif. Pengujian BERTScore akan dijalankan.")
        else:
            print("   [INFO] Server Ollama tidak aktif. Evaluasi dibatasi pada Retrieval Cosine Similarity.")
    except Exception as e:
        print(f"   [INFO] Tidak dapat menginisialisasi Chatbot ({e}). Evaluasi dibatasi pada Retrieval Cosine Similarity.")

    # List untuk menyimpan semua metrik
    results = []
    
    print("\n" + "=" * 90)
    print("                      MULAI PENGHITUNGAN PERSAMAAN METRIK                      ")
    print("=" * 90)

    for case in TEST_CASES:
        print(f"\n▶ [Test Case #{case['id']}] Query: \"{case['query']}\"")
        
        # --- 1. EVALUASI RETRIEVAL (RAG PHASE) ---
        # Ambil chunk terbaik
        retrieved_text = faq_handler.search(case['query'], top_k=1)
        
        if not retrieved_text:
            print("   ❌ Gagal melakukan retrieval data.")
            continue
            
        # Hitung embedding
        query_emb = model.encode(case['query'])
        chunk_emb = model.encode(retrieved_text)
        
        # Hitung similarity menggunakan rumus Cosine Similarity (A, B) = A.B / (||A||*||B||)
        sim_retrieval = calculate_cosine_similarity(query_emb, chunk_emb)
        print(f"   [RAG Retrieval] Cosine Similarity (A, B) = {sim_retrieval:.4f}")
        
        # --- 2. EVALUASI GENERATION (LLM PHASE - BERTScore) ---
        precision, recall, f1 = 0.0, 0.0, 0.0
        llm_response = ""
        
        if run_llm_test and chatbot:
            print("   [LLM Generation] Menghasilkan jawaban dari model Qwen...")
            llm_response = chatbot.get_response(case['query'])
            
            # Hitung BERTScore menggunakan rumus token-level similarity
            precision, recall, f1 = calculate_bertscore(model, llm_response, case['reference_answer'])
            
            print(f"   [BERTScore] Precision : {precision:.4f}")
            print(f"   [BERTScore] Recall    : {recall:.4f}")
            print(f"   [BERTScore] F1-Score  : {f1:.4f}")
        else:
            print("   [LLM Generation] SKIPPED (Ollama tidak aktif)")
            
        results.append({
            "id": case['id'],
            "query": case['query'],
            "retrieval_sim": sim_retrieval,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "response": llm_response
        })

    # --- TABEL HASIL LENGKAP ---
    print("\n" + "=" * 125)
    print(f"{'ID':<3} | {'Query':<45} | {'RAG CosSim':<10} | {'Prec (H,R)':<11} | {'Rec (H,R)':<10} | {'BERT F1':<8}")
    print("-" * 125)
    for r in results:
        p_str = f"{r['precision']:.4f}" if run_llm_test else "N/A"
        rec_str = f"{r['recall']:.4f}" if run_llm_test else "N/A"
        f1_str = f"{r['f1']:.4f}" if run_llm_test else "N/A"
        print(f"{r['id']:<3} | {r['query'][:45]:<45} | {r['retrieval_sim']:<10.4f} | {p_str:<11} | {rec_str:<10} | {f1_str:<8}")
    print("=" * 125)

    # Rata-rata metrik
    avg_retrieval = np.mean([r['retrieval_sim'] for r in results])
    print(f"Rata-rata RAG Cosine Similarity Retrieval : {avg_retrieval:.4f}")
    
    avg_prec = np.mean([r['precision'] for r in results]) if run_llm_test else 0.0
    avg_rec = np.mean([r['recall'] for r in results]) if run_llm_test else 0.0
    avg_f1 = np.mean([r['f1'] for r in results]) if run_llm_test else 0.0

    if run_llm_test:
        print(f"Rata-rata BERTScore Precision             : {avg_prec:.4f}")
        print(f"Rata-rata BERTScore Recall                : {avg_rec:.4f}")
        print(f"Rata-rata BERTScore F1-Score              : {avg_f1:.4f}")
        
        # Tingkat Kelulusan berdasarkan Threshold
        thresholds = [0.70, 0.75, 0.80]
        for th in thresholds:
            passed = sum(1 for r in results if r['f1'] >= th)
            rate = (passed / len(results)) * 100
            print(f"Tingkat Kelulusan Jawaban (F1 >= {th:.2f})  : {rate:.1f}% ({passed}/{len(results)} case)")
    print("=" * 125)

    # --- SIMPAN HASIL KE FILE UNTUK SKRIPSI ---
    output_file_path = os.path.join(ROOT_DIR, "evaluasi_skripsi_chatbot.md")
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("# Hasil Pengujian Evaluasi Chatbot Sobat INL\n")
            f.write("Dokumen ini dibuat otomatis oleh program pengujian untuk digunakan langsung dalam skripsi/tugas akhir.\n\n")
            
            f.write("## 1. Tabel Hasil Pengujian (Format Markdown / Microsoft Word)\n")
            f.write("Berikut adalah tabel lengkap hasil pengujian retrieval RAG dan semantic similarity (BERTScore) untuk setiap skenario:\n\n")
            
            f.write("| ID | Skenario Pertanyaan (Query) | Cosine Similarity (Retrieval) | BERTScore Precision | BERTScore Recall | BERTScore F1 | Status (F1 >= 0.75) |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|:---:|\n")
            
            for r in results:
                p_str = f"{r['precision']:.4f}" if run_llm_test else "N/A"
                rec_str = f"{r['recall']:.4f}" if run_llm_test else "N/A"
                f1_str = f"{r['f1']:.4f}" if run_llm_test else "N/A"
                status = "Lolos" if (run_llm_test and r['f1'] >= 0.75) else ("N/A" if not run_llm_test else "Tidak Lolos")
                f.write(f"| {r['id']} | {r['query']} | {r['retrieval_sim']:.4f} | {p_str} | {rec_str} | {f1_str} | {status} |\n")
            
            p_avg = f"{avg_prec:.4f}" if run_llm_test else "N/A"
            rec_avg = f"{avg_rec:.4f}" if run_llm_test else "N/A"
            f1_avg = f"{avg_f1:.4f}" if run_llm_test else "N/A"
            f.write(f"| **Rata-rata** | **-** | **{avg_retrieval:.4f}** | **{p_avg}** | **{rec_avg}** | **{f1_avg}** | **-** |\n\n")
            
            f.write("## 2. Tabel Ringkasan Analisis Threshold Kelulusan\n\n")
            f.write("| Threshold (F1) | Jumlah Lolos | Persentase Kelulusan |\n")
            f.write("|:---:|:---:|:---:|\n")
            if run_llm_test:
                for th in [0.70, 0.75, 0.80]:
                    passed = sum(1 for r in results if r['f1'] >= th)
                    rate = (passed / len(results)) * 100
                    f.write(f"| F1 >= {th:.2f} | {passed} / {len(results)} | {rate:.1f}% |\n")
            else:
                f.write("| F1 >= 0.70 | N/A | N/A |\n| F1 >= 0.75 | N/A | N/A |\n| F1 >= 0.80 | N/A | N/A |\n")
            f.write("\n")
            
            f.write("## 3. Tabel Hasil Pengujian (Format LaTeX)\n")
            f.write("Jika Anda menggunakan LaTeX, Anda dapat menyalin kode tabel di bawah ini:\n\n")
            f.write("```latex\n")
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\caption{Tabel Hasil Pengujian Semantic Similarity Chatbot Sobat INL}\n")
            f.write("\\label{tab:evaluasi_chatbot}\n")
            f.write("\\begin{tabular}{|c|p{6cm}|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("ID & Skenario Pertanyaan (Query) & RAG CosSim & Prec (H,R) & Rec (H,R) & BERT F1 & Status (>=0.75) \\\\\n")
            f.write("\\hline\\hline\n")
            for r in results:
                p_str = f"{r['precision']:.4f}" if run_llm_test else "N/A"
                rec_str = f"{r['recall']:.4f}" if run_llm_test else "N/A"
                f1_str = f"{r['f1']:.4f}" if run_llm_test else "N/A"
                status = "Lolos" if (run_llm_test and r['f1'] >= 0.75) else ("N/A" if not run_llm_test else "Tidak Lolos")
                # Escape characters like %
                q_escaped = r['query'].replace("%", "\\%")
                f.write(f"{r['id']} & {q_escaped} & {r['retrieval_sim']:.4f} & {p_str} & {rec_str} & {f1_str} & {status} \\\\\n")
            f.write("\\hline\n")
            f.write(f"\\multicolumn{{2}}{{|l|}}{{\\textbf{{Rata-rata}}}} & \\textbf{{{avg_retrieval:.4f}}} & \\textbf{{{p_avg}}} & \\textbf{{{rec_avg}}} & \\textbf{{{f1_avg}}} & - \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
            f.write("```\n")
            
        print(f"\n[INFO] File tabel evaluasi skripsi berhasil disimpan di: {output_file_path}")
    except Exception as e:
        print(f"\n[WARNING] Gagal menulis file hasil evaluasi: {e}")

if __name__ == "__main__":
    run_evaluation()
