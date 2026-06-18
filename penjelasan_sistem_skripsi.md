# Penjelasan Metodologi & Cara Kerja Sistem Sobat INL

Dokumen ini menjelaskan secara rinci mekanisme kerja dari setiap modul cerdas pada sistem "Sobat INL". Uraian di bawah ini disusun menggunakan sudut pandang akademis formal agar dapat disesuaikan untuk kebutuhan penulisan **Bab III (Metodologi Penelitian)** atau **Bab IV (Hasil dan Pembahasan)** pada skripsi.

---

## 1. Mekanisme Kerja Chatbot "Sobat INL"
Asisten virtual "Sobat INL" dirancang dengan konsep hybrid yang menggabungkan tiga pilar pemrosesan bahasa: **Intent Routing (Penyaringan Niat)**, **Semantic Search / RAG (Pencarian Semantik)**, dan **Generative AI (Model Bahasa)**.

```
[ Input Query User ]
         │
         ▼
[ Intent Routing (Regex & Keyword Match) ]
         │
         ├─► PRODUCTION   ──► Ambil data dari RF Cache (Port 3001) ─┐
         ├─► FORECAST     ──► Ambil data dari LSTM Cache (Port 3000) ├─► [ Penyusunan Prompt ] ──► [ LLM qwen3-coder ] ──► [ Output Response ]
         ├─► INFO_COMPANY ──► Semantic Search (FAQ Embeddings) ─────┤
         └─► ANALYSIS/GEN ──► Price Statistics / Daily Price Match ─┘
```

### Detail Langkah Pemrosesan:
1. **Pendeteksian Niat (Intent Routing)**:
   Setiap kali pengguna memasukkan pertanyaan, sistem tidak langsung mengirimkannya ke model bahasa. Query dibersihkan terlebih dahulu (case normalization) dan dicocokkan menggunakan pola ekspresi reguler (*regex*) dan kamus kata kunci yang didefinisikan pada `config_keywords.py`.
2. **Pengambilan Konteks (Retrieval-Augmented Generation / RAG)**:
   * **Profil Perusahaan (`INFO_COMPANY`)**: Query diubah menjadi vektor numerik menggunakan model representasi kalimat **`all-MiniLM-L6-v2`** (SentenceTransformers). Sistem kemudian mengukur kedekatan sudut antar-vektor (*Cosine Similarity*) dengan kumpulan data FAQ pada `faq.txt`. Dokumen dengan nilai kesamaan tertinggi diambil sebagai konteks.
   * **Prediksi Harga CPO (`FORECAST`)**: Sistem langsung menarik data prediksi dari cache memori (`_forecast_cache`) yang dihitung saat startup.
   * **Analisis Produksi (`PRODUCTION`)**: Sistem memanggil modul `ProductionAnalyzer` yang menyimpan rangkuman data operasional pabrik dari model Random Forest.
3. **Generasi Teks (Inference)**:
   Konteks data yang berhasil diambil digabungkan bersama aturan identitas (*System Persona*) dan aturan batasan ketat (*Strict Prompt*). Gabungan ini dikirim ke server **Ollama** yang menjalankan model **`qwen3-coder:480b-cloud`** lokal. Parameter temperatur diatur sangat rendah (**0.1**) untuk memastikan model hanya menjawab berdasarkan fakta yang disediakan pada konteks (mencegah *hallucination*).

---

## 2. Pemodelan Prediksi Harga CPO (LSTM dengan Attention)
Untuk menganalisis harga CPO berjangka USD/MT, sistem menggunakan arsitektur jaringan saraf berulang (*Recurrent Neural Network*) jenis LSTM yang dimodifikasi.

### Jaringan LSTM + Bahdanau Attention:
* **Alasan Penggunaan**: Jaringan LSTM standar memiliki kelemahan dalam mengingat informasi jangka panjang ketika runtutan waktu (*sequence*) terlalu panjang. Dengan menambahkan lapisan **Bahdanau Attention**, model dapat secara dinamis mengevaluasi hari-hari transaksi mana dalam jendela observasi 20 hari sebelumnya yang memiliki pengaruh paling signifikan terhadap pergerakan harga hari berikutnya.
* **Feature Engineering**: Model dilatih tidak hanya menggunakan harga penutupan (*Close Price*), tetapi juga 25+ fitur teknikal seperti Moving Averages (MA), MACD, RSI, Bollinger Bands, volatilitas, momentum, dan lag harga untuk mewakili tren pasar secara kuantitatif.
* **Mekanisme Pelatihan runtime**:
  Untuk menghindari masalah penurunan performa model akibat pergeseran tren pasar (*data drift*), model dilatih secara dinamis (*train-in-runtime*) langsung di memori server setiap kali aplikasi dijalankan. Model menggunakan fungsi optimasi **AdamW** dengan scheduler **Cosine Annealing** dan meminimalkan **Huber Loss** untuk menjaga ketahanan model dari fluktuasi harga ekstrem (*outliers*).

---

## 3. Analisis Kinerja Produksi RBDPO (Random Forest & LOOCV)
Modul ini bertugas memperkirakan jumlah realisasi produksi produk RBDPO bulanan berdasarkan riwayat operasional pabrik.

### Metodologi Model:
* **Random Forest Regressor**:
  Dipilih karena data operasional pabrik berupa data tabular terstruktur. Model berbasis ensambel keputusan (500 decision trees) ini sangat kuat dalam mengenali hubungan non-linear antar variabel operasional tanpa mengalami *overfitting*.
* **Validasi Leave-One-Out Cross-Validation (LOOCV)**:
  * **Kendala**: Dataset laporan produksi bulanan memiliki jumlah sampel yang sangat terbatas (skala kecil). Pembagian data latih/uji konvensional (misalnya split 80/20) akan membatasi data pelatihan secara signifikan.
  * **Solusi**: LOOCV melatih model sebanyak $N$ kali (di mana $N$ adalah jumlah baris data). Pada setiap iterasi, tepat $1$ sampel bulan digunakan sebagai data uji dan sisa $N-1$ sampel bulan digunakan sebagai data latih. Langkah ini menjamin evaluasi performa model ($R^2$ Score, MAPE, MAE) tetap objektif dan menggunakan data latih semaksimal mungkin.
* **Feature Importance**:
  Model menghitung bobot kontribusi setiap variabel input (seperti jumlah CPO yang dikonsumsi, hari kerja aktif, yield persentase, dan rata-rata stok CPO harian) menggunakan metode *Mean Decrease Impurity* (MDI) dan *Permutation Importance* guna mengidentifikasi faktor paling berpengaruh terhadap realisasi produksi.

---

## 4. Aliran Integrasi Frontend dan Backend
Proses sinkronisasi data dan pengiriman informasi antara dasbor Vue 3 dan Backend AI dirancang untuk menjaga konsistensi data dan kenyamanan interaksi pengguna.

* **Konsistensi Visual (Caching)**:
  Hasil kalkulasi prediksi LSTM dan analisis LOOCV Random Forest dihitung sekali saat startup backend dan disimpan di memori RAM (*caching*). Dasbor visualisasi grafik (ApexCharts) dan modul Chatbot memanggil objek cache yang sama. Hal ini memastikan angka prediksi yang diucapkan chatbot senada dengan garis tren grafik dasbor.
* **Responsivitas Interaksi (SSE Streaming)**:
  Ketika pengguna mengirim pesan chat, server backend mengirimkan respons bertipe aliran data (*stream*). Sisi klien (Vue 3) menggunakan objek pembaca stream (`response.body.getReader()`) untuk mendekode data teks mentah kata demi kata secara asinkron. Dengan cara ini, pengguna dapat membaca jawaban chatbot secara bertahap saat teks sedang dibuat, tanpa harus menunggu seluruh paragraf selesai diproses di backend (mengurangi *perceived latency*).
