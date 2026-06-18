# Detail Arsitektur & Spesifikasi Sistem Sobat INL

Dokumen ini berisi detail teknis arsitektur sistem "Sobat INL" yang menggabungkan antarmuka dashboard monitoring dengan mesin kecerdasan buatan (LSTM Forecasting & Random Forest). Dokumen ini disusun dengan format akademis formal yang dapat digunakan sebagai bahan penulisan skripsi.

---

## 1. Spesifikasi Teknologi & Stack Sistem

Sistem ini dirancang menggunakan arsitektur terdekopling (*decoupled*) yang memisahkan bagian antarmuka pengguna (*frontend*) dengan pemrosesan data (*backend*).

| Komponen | Peran | Teknologi |
| :--- | :--- | :--- |
| **Frontend** | Antarmuka Pengguna & Visualisasi Data | Vue 3 (Composition API), Vite, TailwindCSS, PrimeVue 4, ApexCharts |
| **Backend AI 1** | Chatbot Asisten & Prediksi Harga CPO | Python 3, FastAPI, PyTorch, SentenceTransformers, Ollama (`qwen3-coder:480b-cloud`) |
| **Backend AI 2** | Analisis & Estimasi Produksi RBDPO | Python 3, FastAPI, scikit-learn (Random Forest, LOOCV), Pandas |
| **Penyimpanan Data** | Sumber Data Historis & Knowledge Base | CSV (Harga CPO), JSON (Data Produksi & Stok), TXT (FAQ Profil INL) |

---

## 2. Arsitektur Backend (Sisi Server AI)

Backend sistem dibagi menjadi dua layanan microservice berbasis FastAPI untuk membagi beban komputasi model *Deep Learning* (LSTM) dan *Machine Learning* (Random Forest).

### A. Server Chatbot & LSTM Forecasting (Port 3000 — `api_stream.py`)

Server ini bertanggung jawab atas dua fitur utama: prediksi harga CPO (Crude Palm Oil) berjangka dan logika interaksi agen percakapan (*chatbot*).

#### 1. Model Prediksi Harga CPO (LSTM + Attention)
*   **Arsitektur Model**: Menggunakan kombinasi dua layer *Long Short-Term Memory* (LSTM) dengan 128 *hidden units* dan *dropout rate* sebesar 0.3. Untuk menangkap pengaruh jangka panjang, model dilengkapi dengan **Bahdanau-style Additive Attention Layer**. Keluaran dari attention dihubungkan ke *Fully Connected Regression Head* dengan struktur: `LayerNorm` → `Linear(128 → 64)` → `GELU` → `Dropout` → `Linear(64 → 32)` → `GELU` → `Linear(32 → 1)`.
*   **Panjang Runtutan (Sequence Length)**: 20 hari transaksi aktif historis untuk memprediksi harga 1 hari ke depan secara autoregresif.
*   **Konfigurasi Pelatihan**:
    *   **Metode**: *Train-in-runtime* (model dilatih langsung di memori server menggunakan data terbaru setiap kali aplikasi dijalankan guna mencegah *data drift*).
    *   **Epoch Maksimal**: 200 epoch dengan mekanisme *Early Stopping* (patience = 15 epoch, min delta = $10^{-6}$).
    *   **Loss Function**: *Huber Loss* ($\delta = 0.5$) untuk meminimalkan dampak data pencilan (*outliers*).
    *   **Optimizer**: AdamW dengan *Learning Rate* awal 0.0005 dan penjadwalan *Cosine Annealing LR*.

#### 2. Agen Percakapan "Sobat INL"
*   **Intent Router**: Mengklasifikasikan pertanyaan pengguna ke dalam 5 kategori intent utama: `PRODUCTION`, `FORECAST`, `INFO_COMPANY`, `ANALYSIS`, dan `GENERAL`.
*   **Large Language Model (LLM)**: Menggunakan model `qwen3-coder:480b-cloud` yang di-hosting lokal melalui Ollama. Parameter temperatur diatur ke 0.1 guna menjamin jawaban yang konsisten dan faktual.
*   **Semantic Search FAQ**: Menggunakan model embedding `all-MiniLM-L6-v2` (SentenceTransformers) untuk mencocokkan kemiripan kosinus (*cosine similarity*) pertanyaan pengguna dengan basis pengetahuan internal (`faq.txt`).

### B. Server Random Forest Production Analysis (Port 3001 — `rf_api.py`)

Layanan ini berfokus pada analisis kinerja produksi RBDPO (*Refined Bleached Deodorized Palm Oil*) bulanan milik PT INL.

#### 1. Model Regresi Random Forest
*   **Struktur Model**: Menggunakan algoritma `RandomForestRegressor` dengan konfigurasi 500 decision trees dan pembatasan pemilihan fitur acak menggunakan kriteria `sqrt(n_features)`.
*   **Metode Validasi LOOCV (*Leave-One-Out Cross Validation*)**: Mengingat ukuran dataset produksi bulanan yang terbatas, LOOCV diterapkan untuk mengevaluasi kinerja generalisasi model tanpa kehilangan data latih yang signifikan. Model dilatih sebanyak $N$ kali (di mana $N$ adalah jumlah total bulan) menggunakan $N-1$ data sebagai set pelatihan dan 1 data sisa sebagai set pengujian.
*   **Variabel Input (14 Fitur)**:
    1.  `stok_rata2` (Rata-rata stok CPO harian)
    2.  `stok_max` (Stok CPO maksimum dalam bulan berjalan)
    3.  `stok_hari_aktif` (Jumlah hari dengan stok aktif)
    4.  `target_rkap` (Target produksi berdasarkan rencana kerja perusahaan)
    5.  `cpo_consume` (Total konsumsi bahan baku CPO)
    6.  `hari_olah` (Hari kerja aktif pabrik)
    7.  `yield_rbdpo` (Rasio efisiensi konversi bahan baku ke produk jadi)
    8.  `pfad_total` (Jumlah produksi produk sampingan PFAD)
    9.  `cpo_per_hari` (Rata-rata konsumsi CPO harian)
    10. `bulan_ke` (Indeks bulan kalender, 1-12)
    11. `kuartal` (Indeks kuartal tahunan, 1-4)
    12. `realisasi_prev` (Realisasi produksi pada bulan sebelumnya)
    13. `cpo_prev` (Konsumsi CPO pada bulan sebelumnya)
    14. `hari_olah_prev` (Hari kerja aktif pabrik pada bulan sebelumnya)

---

## 3. Detail Integrasi Frontend ↔ Backend

Proses pertukaran data antara aplikasi web Frontend (Vue 3) dan Backend API berjalan melalui dua metode utama:

### A. Aliran Data Streaming (Chatbot Interface)
*   **Metode**: HTTP POST Request ke endpoint `/chat` dengan muatan (*payload*) berupa JSON `{ "question": "..." }`.
*   **Protokol**: Menggunakan **Fetch API** untuk membaca aliran data bertipe `text/plain`. 
*   **Penanganan Sisi Klien**: Menggunakan pembaca stream (`response.body.getReader()`) yang secara iteratif membaca dan mendekode potongan teks (*chunk*) UTF-8. Teks kemudian disaring untuk menghapus tag `<think>...</think>` dan diuraikan ke elemen HTML menggunakan pustaka `marked.js` agar mendukung format Markdown.

### B. Visualisasi Grafik (Monitoring Dashboard)
*   **Metode**: HTTP GET Request ke endpoint `/forecast-data` (Server 3000) dan `/rf-analysis` (Server 3001).
*   **Protokol**: Menggunakan pustaka **Axios** dengan konfigurasi waktu tunggu (*timeout*) hingga 5.000.000 ms khusus untuk inisialisasi awal komputasi LOOCV.
*   **Penanganan Sisi Klien**: Data yang diterima kemudian ditransformasikan ke struktur data *series* dan *options* yang kompatibel dengan pustaka grafik **ApexCharts** untuk dirender secara reaktif.

---

## 4. Pipeline Pengolahan Data & Preprocessing

Sebelum data historis digunakan untuk melatih model LSTM dan Random Forest, data melalui beberapa tahapan preprocessing:

1.  **Pembersihan & Parsing Angka**: Mengeliminasi tanda pemisah ribuan berupa titik (`.`) dan mengganti tanda desimal koma (`,`) menjadi format titik (`.`) standar pemrograman Python.
2.  **Winsorization**: Menangani nilai ekstrim (*outliers*) pada data harga historis menggunakan teknik IQR (*Interquartile Range*) dengan batas faktor 3.0.
3.  **Feature Engineering**: Menghitung indikator teknikal untuk melengkapi dataset harga penutupan (*Close*):
    *   *Moving Average* (MA_7, MA_21, MA_50)
    *   *Exponential Moving Average* (EMA_12, EMA_26)
    *   *MACD* (MACD, Signal, Histogram)
    *   *Relative Strength Index* (RSI 14 hari)
    *   *Bollinger Bands* (Upper, Lower, Width)
    *   *Volatility* (Vol_7, Vol_30)
    *   *Momentum* (Mom_5, Mom_14)
    *   *Lag Features* (Lag_1, Lag_3, Lag_7)
4.  **Scaling**: Normalisasi skala nilai fitur menggunakan `RobustScaler` untuk meminimalkan deviasi pencilan pada data runtun waktu (*time-series*).
