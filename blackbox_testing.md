# Tabel Pengujian Black Box — Sistem Sobat INL
**Proyek:** Dashboard INL (Frontend Vue.js + Backend FastAPI)
**Tanggal Pengujian:** ___________
**Penguji:** ___________

---

## Kategori 1 — Chatbot Sobat INL (`POST /chat`)

| NO | TEST SCENARIO | TEST CASE | PRECONDITION | TEST STEPS | TEST DATA | EXPECTED RESULT | ACTUAL RESULT | STATUS |
|----|---------------|-----------|--------------|------------|-----------|-----------------|---------------|--------|
| 1 | Small Talk / Sapaan | User mengirim pesan sapaan | Backend :3000 berjalan | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"halo"` | Chatbot merespons dengan pesan sambutan tanpa memanggil LLM | | |
| 2 | Small Talk / Identitas | User bertanya identitas bot | Backend :3000 berjalan | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"kamu siapa?"` | Chatbot menjawab "Saya Sobat INL 🤖 Asisten Virtual resmi PT Industri Nabati Lestari" | | |
| 3 | Intent FORECAST | User meminta prediksi 7 hari | Backend :3000 berjalan, model LSTM terlatih | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"prediksi harga CPO 7 hari ke depan"` | Chatbot menampilkan prediksi harga dalam USD/MT untuk 7 hari ke depan dengan confidence interval | | |
| 4 | Intent FORECAST — besok | User meminta prediksi 1 hari | Backend :3000 berjalan | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"harga CPO besok berapa?"` | Chatbot menampilkan prediksi harga CPO untuk 1 hari ke depan | | |
| 5 | Intent ANALYSIS — periode tahun | User meminta analisis historis | Backend :3000 berjalan, data CSV tersedia | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"analisis harga CPO tahun 2024"` | Chatbot menampilkan statistik (tertinggi, terendah, rata-rata) harga CPO tahun 2024 | | |
| 6 | Intent ANALYSIS — tanggal spesifik | User menanyakan harga pada tanggal tertentu | Backend :3000 berjalan, data CSV tersedia | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"berapa harga CPO 15 Januari 2024?"` | Chatbot menampilkan harga CPO pada tanggal tersebut dalam USD/MT | | |
| 7 | Intent INFO_COMPANY | User bertanya profil perusahaan | Backend :3000 berjalan, faq.txt tersedia | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"di mana lokasi pabrik INL?"` | Chatbot menjawab "KEK Sei Mangkei, Sumatera Utara" | | |
| 8 | Intent INFO_COMPANY — kepemilikan | User bertanya induk perusahaan | Backend :3000 berjalan | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"siapa pemilik PT INL?"` | Chatbot menjawab INL adalah anak perusahaan PTPN III (Persero) | | |
| 9 | Intent PRODUCTION — realisasi | User menanyakan data produksi RBDPO | Backend :3000 & :3001 keduanya berjalan | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"berapa realisasi produksi RBDPO bulan lalu?"` | Chatbot menampilkan data realisasi vs target RKAP bulan sebelumnya dalam satuan ton | | |
| 10 | Intent PRODUCTION — feature importance | User menanyakan faktor produksi | Backend :3000 & :3001 keduanya berjalan | 1. Buka halaman `/ai/chatbot` 2. Ketik pesan 3. Tekan Enter | `"apa faktor paling berpengaruh terhadap produksi RBDPO?"` | Chatbot menampilkan daftar feature importance dari model Random Forest | | |
| 11 | Streaming response | Respons bot ditampilkan per kata | Backend :3000 berjalan | 1. Kirim pertanyaan apapun 2. Perhatikan area respons bot | `"harga CPO hari ini berapa?"` | Teks muncul secara bertahap (streaming per kata), bukan sekaligus | | |
| 12 | Typing indicator | Indikator loading muncul saat menunggu | Backend :3000 berjalan | 1. Kirim pesan 2. Amati sebelum respons muncul | `"prediksi harga"` | Tiga titik animasi (typing indicator) muncul sebelum teks respons mulai mengalir | | |
| 13 | Suggestion Chip | Klik chip mengirim pesan langsung | Halaman baru dibuka, belum ada chat | 1. Buka halaman `/ai/chatbot` 2. Klik salah satu chip pertanyaan | Klik chip `"Prediksi harga CPO 7 hari ke depan"` | Pertanyaan terkirim otomatis dan chatbot merespons | | |
| 14 | Hapus percakapan | Tombol trash mengosongkan chat | Sudah ada percakapan | 1. Kirim beberapa pesan 2. Klik ikon 🗑️ di header | — | Semua pesan terhapus, halaman kembali ke tampilan awal (welcome banner + suggestion chips) | | |
| 15 | Filter tag `<think>` | Tag LLM tidak ikut ditampilkan | Backend mengembalikan respons dengan tag `<think>` | 1. Kirim pertanyaan yang memicu LLM berpikir panjang | `"analisis mendalam tren harga CPO"` | Teks antara `<think>...</think>` tidak terlihat di UI, hanya jawaban akhir yang muncul | | |

---

## Kategori 2 — Grafik Forecast LSTM (`GET /forecast-data`)

| NO | TEST SCENARIO | TEST CASE | PRECONDITION | TEST STEPS | TEST DATA | EXPECTED RESULT | ACTUAL RESULT | STATUS |
|----|---------------|-----------|--------------|------------|-----------|-----------------|---------------|--------|
| 16 | Toggle panel forecast | Tombol grafik membuka panel | Backend :3000 berjalan, model terlatih | 1. Buka halaman `/ai/chatbot` 2. Klik ikon 📈 di header | — | Panel forecast muncul di sisi kanan dengan grafik line chart | | |
| 17 | Tampilan data aktual | Garis aktual muncul pada grafik | Backend :3000 berjalan | 1. Buka panel forecast | — | Garis hijau menampilkan data harga CPO aktual mulai 1 Januari 2025 | | |
| 18 | Tampilan data prediksi | Garis prediksi 7 hari muncul | Backend :3000 berjalan | 1. Buka panel forecast | — | Garis oranye (dashed) menampilkan prediksi 7 hari ke depan setelah tanggal terakhir data aktual | | |
| 19 | Satuan label Y-axis | Label sumbu Y menggunakan USD/MT | Backend :3000 berjalan | 1. Buka panel forecast 2. Amati label sumbu Y | — | Label sumbu Y menampilkan format `USD 1,140` (bukan Rupiah) | | |
| 20 | Tooltip grafik | Hover pada titik data menampilkan nilai | Backend :3000 berjalan | 1. Buka panel forecast 2. Arahkan kursor ke titik pada grafik | — | Tooltip menampilkan nilai dalam format `USD 1,140.00 /MT` | | |
| 21 | Loading state forecast | Spinner muncul saat memuat data | Backend :3000 berjalan, koneksi lambat | 1. Klik ikon 📈 pertama kali | — | Teks "Memuat prediksi..." dengan spinner muncul sebelum grafik tampil | | |
| 22 | Tutup panel forecast | Tombol × menutup panel | Panel forecast terbuka | 1. Klik tombol × pada panel forecast | — | Panel forecast tertutup, area chat kembali penuh | | |
| 23 | Cache forecast | Data grafik konsisten dengan jawaban chatbot | Backend :3000 berjalan | 1. Buka panel forecast 2. Tanya chatbot prediksi 7 hari | — | Angka prediksi di grafik dan jawaban chatbot identik | | |

---

## Kategori 3 — RF Production Analysis (`GET /rf-analysis` & `/rf-production-history`)

| NO | TEST SCENARIO | TEST CASE | PRECONDITION | TEST STEPS | TEST DATA | EXPECTED RESULT | ACTUAL RESULT | STATUS |
|----|---------------|-----------|--------------|------------|-----------|-----------------|---------------|--------|
| 24 | Load halaman RF Analysis | Data dimuat otomatis saat halaman dibuka | Backend :3001 berjalan | 1. Navigasi ke `/ai/rf-analysis` | — | Loading overlay muncul lalu diganti konten (KPI cards, grafik, tabel) | | |
| 25 | KPI Card — MAE | Nilai MAE ditampilkan | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Amati KPI card MAE | — | KPI card "MAE" menampilkan nilai dalam satuan ton | | |
| 26 | KPI Card — R² Score | Nilai R² ditampilkan dengan warna dinamis | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Amati KPI card R² | — | Jika R² ≥ 0.8 → warna hijau; jika < 0.8 → warna oranye | | |
| 27 | KPI Card — MAPE | Nilai MAPE ditampilkan dengan warna dinamis | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Amati KPI card MAPE | — | Jika MAPE ≤ 10% → warna hijau; jika > 10% → warna oranye | | |
| 28 | Grafik produksi — 3 seri | Realisasi, Target RKAP, dan Prediksi LOOCV tampil | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Amati grafik utama | — | Line chart menampilkan 3 garis: hijau (Realisasi), biru dashed (Target RKAP), oranye (Prediksi LOOCV) | | |
| 29 | Grafik Feature Importance | Top-5 fitur ditampilkan horizontal bar chart | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Amati grafik feature importance | — | Bar chart horizontal menampilkan 5 fitur teratas diurutkan berdasarkan MDI Importance (%) dari besar ke kecil | | |
| 30 | Tabel dataset — jumlah baris | Tabel menampilkan semua bulan | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Scroll ke tabel | — | Jumlah baris tabel sesuai dengan jumlah bulan data historis yang tersedia | | |
| 31 | Tabel dataset — kolom Realisasi | Nilai realisasi produksi tampil benar | Backend :3001 berjalan | 1. Buka tabel dataset | — | Kolom "Realisasi (ton)" menampilkan nilai dalam format angka ribuan (mis. `12.500,00`) | | |
| 32 | Tabel dataset — Achievement bar | Progress bar capaian target | Backend :3001 berjalan | 1. Buka tabel dataset 2. Amati kolom Achievement | — | Progress bar berwarna hijau jika ≥ 95%, kuning jika 80-94%, merah jika < 80% | | |
| 33 | Tabel dataset — badge status | Badge Aktual / Imputasi tampil | Backend :3001 berjalan | 1. Buka tabel dataset 2. Amati kolom Status | — | Baris dengan `is_imputed = true` menampilkan badge oranye "Imputasi"; baris lain badge hijau "Aktual" | | |
| 34 | Tabel dataset — highlight imputasi | Baris imputasi memiliki background berbeda | Backend :3001 berjalan | 1. Buka tabel dataset | — | Baris dengan data imputasi memiliki background oranye transparan | | |
| 35 | Tombol Refresh | Data dimuat ulang dari backend | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Klik tombol "Refresh" | — | Loading spinner muncul lalu data terbaru dimuat ulang (last_updated berubah) | | |
| 36 | Informasi Last Updated | Waktu update terakhir ditampilkan | Backend :3001 berjalan | 1. Buka halaman `/ai/rf-analysis` 2. Amati header halaman | — | Timestamp "last updated" dari backend ditampilkan di header halaman | | |

---

## Kategori 4 — Penanganan Error & Kondisi Batas

| NO | TEST SCENARIO | TEST CASE | PRECONDITION | TEST STEPS | TEST DATA | EXPECTED RESULT | ACTUAL RESULT | STATUS |
|----|---------------|-----------|--------------|------------|-----------|-----------------|---------------|--------|
| 37 | Backend chatbot mati | Error ditampilkan saat backend tidak aktif | Backend :3000 **tidak** berjalan | 1. Buka halaman `/ai/chatbot` 2. Kirim pesan | `"halo"` | Pesan error ditampilkan di bubble bot: "⚠️ Koneksi ke Sobat INL gagal. Pastikan backend berjalan." | | |
| 38 | Backend RF mati | Error card ditampilkan | Backend :3001 **tidak** berjalan | 1. Navigasi ke `/ai/rf-analysis` | — | Error card merah muncul dengan pesan gagal koneksi dan tombol "Coba Lagi" | | |
| 39 | Tombol Coba Lagi | Retry setelah error | Backend :3001 **tidak** berjalan lalu **dinyalakan** | 1. Buka `/ai/rf-analysis` saat backend mati 2. Nyalakan backend 3. Klik "Coba Lagi" | — | Data berhasil dimuat setelah retry | | |
| 40 | Input kosong chatbot | Tombol kirim nonaktif | Backend :3000 berjalan | 1. Buka halaman `/ai/chatbot` 2. Biarkan textarea kosong | — | Tombol "Send" berstatus disabled, tidak bisa diklik | | |
| 41 | Intent PRODUCTION tanpa RF server | Pesan informatif saat RF tidak aktif | Backend :3000 berjalan, :3001 **tidak** berjalan | 1. Kirim pertanyaan produksi | `"berapa produksi RBDPO bulan ini?"` | Chatbot merespons "Data produksi RBDPO belum tersedia saat ini. Pastikan server RF Production (port 3001) sudah berjalan…" | | |
| 42 | Tanggal tidak ada di database | Pesan informatif saat data tidak ditemukan | Backend :3000 berjalan | 1. Tanyakan harga pada tanggal hari libur/weekend | `"harga CPO 25 Desember 2023?"` | Chatbot menjawab data untuk tanggal tersebut tidak ada di database (hari libur tidak memiliki data trading) | | |

---

## Kategori 5 — Autentikasi & Navigasi

| NO | TEST SCENARIO | TEST CASE | PRECONDITION | TEST STEPS | TEST DATA | EXPECTED RESULT | ACTUAL RESULT | STATUS |
|----|---------------|-----------|--------------|------------|-----------|-----------------|---------------|--------|
| 43 | Akses tanpa login | Redirect ke halaman login | User belum login | 1. Buka langsung URL `/ai/chatbot` | — | User di-redirect ke halaman `/` (login) | | |
| 44 | Akses dengan login valid | Halaman AI Insight dapat diakses | User sudah login dengan token valid | 1. Login 2. Navigasi ke `/ai/chatbot` | Username & password valid | Halaman chatbot berhasil dimuat | | |
| 45 | Role-based access | Semua role dapat akses AI Insight | User login dengan berbagai role | 1. Login dengan role `finance` 2. Akses `/ai/chatbot` | Role: finance, operation, sales, scm, sdm, sourcing | Halaman berhasil diakses (semua role memiliki akses ke AI Insight) | | |
| 46 | Navigasi menu AI Insight | Link menu mengarah ke halaman yang benar | User sudah login | 1. Klik menu "AI Insight" di sidebar 2. Klik "Sobat INL" | — | Halaman `/ai/chatbot` terbuka | | |
| 47 | Navigasi RF Analysis | Link menu RF Analysis benar | User sudah login | 1. Klik menu "AI Insight" 2. Klik "RF Production Analysis" | — | Halaman `/ai/rf-analysis` terbuka | | |

---

## Kategori 6 — Responsivitas UI

| NO | TEST SCENARIO | TEST CASE | PRECONDITION | TEST STEPS | TEST DATA | EXPECTED RESULT | ACTUAL RESULT | STATUS |
|----|---------------|-----------|--------------|------------|-----------|-----------------|---------------|--------|
| 48 | Chatbot — tampilan mobile | Layout menyesuaikan layar kecil | Browser dengan viewport ≤ 768px | 1. Buka `/ai/chatbot` di viewport 768px | — | Panel forecast tersembunyi, bubble chat melebar hingga 88% lebar layar | | |
| 49 | RF Analysis — KPI mobile | Grid KPI menyesuaikan | Viewport ≤ 1024px | 1. Buka `/ai/rf-analysis` di viewport 1024px | — | KPI grid berubah dari 4 kolom menjadi 2 kolom | | |
| 50 | RF Analysis — chart mobile | Grafik menyesuaikan layout | Viewport ≤ 1024px | 1. Buka `/ai/rf-analysis` di viewport 1024px | — | Chart feature importance berubah dari panel samping menjadi di bawah chart produksi (flex-direction: column) | | |
| 51 | Auto-resize textarea | Input chat membesar sesuai teks | Browser desktop | 1. Buka `/ai/chatbot` 2. Ketik teks panjang multi-baris di textarea | Teks 3–5 baris | Textarea otomatis membesar hingga maksimum 120px, lalu scroll muncul | | |
| 52 | Enter untuk kirim | Shortcut keyboard berfungsi | Backend :3000 berjalan | 1. Ketik pesan di textarea 2. Tekan `Enter` | `"halo"` | Pesan terkirim (sama seperti klik tombol Send) | | |
| 53 | Shift+Enter baris baru | Shortcut tidak mengirim pesan | Backend :3000 berjalan | 1. Ketik pesan di textarea 2. Tekan `Shift+Enter` | `"halo"` | Baris baru dibuat di textarea, pesan **tidak** terkirim | | |
