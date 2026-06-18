# BAB III / BAB IV: PERANCANGAN DAN IMPLEMENTASI SISTEM

Dokumen ini memaparkan secara komprehensif arsitektur, spesifikasi teknis, dan alur integrasi dari sistem "Sobat INL". Sistem ini merupakan ekosistem *backend* dan *frontend* berbasis Kecerdasan Buatan (AI) yang dikembangkan untuk PT Industri Nabati Lestari (INL), menggabungkan model *Deep Learning*, *Machine Learning*, dan *Large Language Models* (LLM).

---

## 1. Arsitektur Sistem Keseluruhan

Sistem dirancang dengan pendekatan *decoupled architecture* berbasis *microservices*, di mana antarmuka pengguna (*Frontend*) berinteraksi secara independen dengan dua layanan *Backend AI* terpisah.

Sistem terbagi menjadi tiga komponen utama:
1. **Frontend Web Application (Dashboard INL)**: Bertugas menangani antarmuka pemantauan data (*monitoring*) dan interaksi pengguna dengan asisten virtual.
2. **Backend AI 1 (Port 3000)**: Melayani *Forecasting* Harga CPO menggunakan model LSTM dan melayani agen percakapan interaktif (*Chatbot*).
3. **Backend AI 2 (Port 3001)**: Melayani analisis kinerja produksi RBDPO (*Refined Bleached Deodorized Palm Oil*) menggunakan model *Random Forest Regressor*.

---

## 2. Spesifikasi Antarmuka Pengguna (Frontend)

Antarmuka dibangun dengan tujuan memvisualisasikan hasil komputasi model AI ke dalam bentuk dasbor analitik dan *widget* interaktif.

*   **Teknologi Utama**: Sistem dikembangkan menggunakan *framework* **Vue.js 3** dengan *Composition API*, dibangun melalui *bundler* **Vite**.
*   **Visualisasi Data**: Memanfaatkan pustaka **ApexCharts** (`vue3-apexcharts`) untuk memproyeksikan data runtun waktu (*time-series*) prediksi LSTM dan evaluasi *Leave-One-Out Cross-Validation* (LOOCV) dari Random Forest.
*   **Desain Pola Integrasi**: Menerapkan pola abstraksi **Controller-API**.
    *   *API Layer* (`src/api/thisAPI/`): Menggunakan **Fetch API** untuk menangani *Server-Sent Events* (SSE) guna mendukung *streaming response* pada chatbot (teks dirender kata per kata), dan **Axios** untuk panggilan data *RESTful* biasa.
    *   *Controller Layer* (`src/controller/`): Memproses (*parsing* & *transforming*) respons JSON mentah sebelum diinjeksikan sebagai *props* (*properties*) ke komponen Vue.

---

## 3. Spesifikasi Backend AI 1 (Chatbot & LSTM Forecasting)

Layanan ini dikembangkan menggunakan *framework* **FastAPI** (Python) dan beroperasi pada *port* 3000. Layanan ini merepresentasikan kapabilitas prediksi harga pasar dan interaksi bahasa alami.

### A. Model Prediksi Harga CPO (CPO_LSTM)
Model dirancang untuk memprediksi harga CPO (dalam USD/MT) 7 hari ke depan secara *autoregressive*.
1. **Arsitektur Model**: 
   * Dibangun menggunakan **PyTorch**.
   * Memanfaatkan 2 lapisan *Long Short-Term Memory* (LSTM) dengan 128 *hidden units* dan *dropout rate* 0.3.
   * Lapisan *Attention*: Mengadopsi mekanisme **Bahdanau-style Additive Attention** untuk memberikan bobot (*attention weights*) spesifik pada bagian data runtutan historis (panjang *sequence* = 20 hari) yang paling berpengaruh terhadap harga masa depan.
   * *Fully Connected Head*: Keluaran *attention* dilewatkan ke fungsi regresi: `LayerNorm` → `Linear(128 ke 64)` → `GELU` → `Dropout` → `Linear(64 ke 32)` → `GELU` → `Linear(32 ke 1)`.
2. **Prapemrosesan Data (Preprocessing)**:
   * **Pembersihan Data**: Parsing format angka Indonesia (mis. `1.140,25` → `1140.25`).
   * **Outlier Handling**: Metode *Interquartile Range (IQR) Winsorization* dengan parameter faktor 3.0.
   * **Feature Engineering**: Ekstraksi 25+ indikator teknikal (MA, EMA, MACD, RSI 14, *Bollinger Bands*, Volatilitas, Momentum, dan *Lag Features*).
   * **Normalisasi**: Transformasi dengan `RobustScaler` (tahan terhadap nilai pencilan).
3. **Skenario Pelatihan (*Training*)**:
   * **Metode**: *Train-in-runtime* (model melatih dirinya sendiri dari iterasi ke-0 pada setiap inisialisasi server untuk menyerap data pasar paling aktual secara penuh).
   * **Fungsi Objektif (*Loss Function*)**: Menggunakan **Huber Loss** ($\delta = 0.5$).
   * **Optimisasi**: AdamW (*Learning Rate*: $0.0005$, *Weight Decay*: $10^{-5}$) yang dipadukan dengan *Cosine Annealing LR Scheduler*. Dilengkapi dengan *Early Stopping* (*patience* = 15).

### B. Otak Chatbot ("Sobat INL")
Sistem pemrosesan bahasa alami dirancang dengan pendekatan semi-deterministik dan *Retrieval-Augmented Generation* (RAG) sederhana.
1. **Model Bahasa (LLM)**: Menggunakan model **`qwen3-coder:480b-cloud`** (diakses melalui antarmuka Ollama). Parameter temperatur ditetapkan secara ketat di angka **0.1** untuk menekan halusinasi (*hallucination*) dan memaksa penyampaian respons analitis.
2. **Intent Routing**: Setiap instruksi (*query*) melalui proses klasifikasi regex menjadi 5 kategori:
   * `PRODUCTION`: Memicu kelas `ProductionAnalyzer`, memuat data pabrik aktual dari memori.
   * `FORECAST`: Mengambil hasil *cache* prediksi LSTM 7 hari ke depan.
   * `INFO_COMPANY`: Melakukan *semantic search* pada teks dokumentasi profil INL.
   * `ANALYSIS`: Memicu analitik statistik harga (maksimum, minimum, rata-rata) tahun/bulan tertentu.
   * `GENERAL`: Melakukan pencarian harga CPO spesifik berdasarkan tanggal.
3. **Pencarian Semantik (Semantic Search)**: Memanfaatkan pustaka `SentenceTransformers` dengan *pre-trained embedding* **`all-MiniLM-L6-v2`** untuk melakukan pencocokan kesamaan kosinus (*cosine similarity*) terhadap basis pengetahuan FAQ perusahaan.

---

## 4. Spesifikasi Backend AI 2 (Random Forest Production Analysis)

Layanan ini berfokus pada sisi manufaktur/operasional, diakses melalui *port* 3001, bertujuan untuk memperkirakan dan mengevaluasi jumlah realisasi produksi RBDPO harian dan bulanan.

1. **Pengumpulan Data**:
   * Hirarki prioritas sumber data (Fall-back mechanism): Mengambil data REST-API operasional secara *real-time* (Server Utama `103.176.66.42:9009`) → Berkas cadangan JSON lokal → Berkas agregat Excel Historis 2021-2023.
2. **Arsitektur Pemodelan (*Random Forest*)**:
   * Menggunakan algoritma **Random Forest Regressor** dari pustaka `scikit-learn`.
   * **Hyperparameter**: Membentuk struktur ensambel (*ensemble learning*) dengan 500 pohon keputusan (`n_estimators = 500`) dan pembagian pencabangan (`max_features = 'sqrt'`).
   * **Fitur Input**: Melibatkan 14 dimensi data operasional (antara lain: target RKAP, rata-rata stok CPO harian, hari kerja aktif, *yield* konversi RBDPO, produksi produk turunan/PFAD, hingga matriks produksi bulan sebelumnya/lag data).
3. **Skenario Evaluasi Model (LOOCV)**:
   * Kendala analitik operasional adalah sempitnya jumlah observasi historis bulanan pabrik (kumpulan data skala menengah-kecil).
   * Solusi validasi: Menggunakan **Leave-One-Out Cross-Validation (LOOCV)**. Model dilatih dari awal berulang-ulang sebanyak $N$ kali iterasi (di mana $N$ adalah jumlah data bulan), dengan memisahkan persis $1$ data sampel pengujian pada tiap *fold*-nya guna mencegah kebocoran data (*data leakage*) namun tetap melatih dengan matriks yang maksimal.
4. **Analisis Kepentingan Fitur (*Feature Importance*)**:
   * Kalkulasi atribut pengaruh tiap metrik pabrik menggunakan algoritma penimbang silang: **Mean Decrease Impurity (MDI)** dan **Permutation Importance** secara iteratif.
5. **Caching Memori**: 
   * Hasil kalkulasi LOOCV (*metrik evaluasi R², MAPE, MAE, RMSE*) dibekukan sementara di dalam RAM dengan *Time-To-Live* (TTL) selama 7.200 detik (2 jam). Metode sinkronisasi antar-utas dilakukan melalui protokol `threading.Lock` guna melayani API (*endpoint*) secara instan.
