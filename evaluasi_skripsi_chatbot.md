# Hasil Pengujian Evaluasi Chatbot Sobat INL
Dokumen ini dibuat otomatis oleh program pengujian untuk digunakan langsung dalam skripsi/tugas akhir.

## 1. Tabel Hasil Pengujian (Format Markdown / Microsoft Word)
Tabel ini menyajikan hasil pengujian secara menyeluruh yang mencakup fungsionalitas Small Talk, LSTM (Prediksi & Analisis CPO), RAG (Info Perusahaan), dan Random Forest (Analisis Produksi):

| ID | Skenario Pertanyaan (Query) | Tipe Modul (Intent) | Cosine Similarity (Retrieval) | BERTScore Precision | BERTScore Recall | BERTScore F1 | Status (F1 >= 0.75) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | halo | SMALL_TALK | - | 1.0000 | 1.0000 | 1.0000 | Lolos |
| 2 | kamu siapa? | SMALL_TALK | - | 1.0000 | 1.0000 | 1.0000 | Lolos |
| 3 | prediksi harga CPO 7 hari ke depan | FORECAST | - | 0.6739 | 0.6846 | 0.6792 | Tidak Lolos |
| 4 | harga CPO besok berapa? | FORECAST | - | 0.5959 | 0.6903 | 0.6396 | Tidak Lolos |
| 5 | analisis harga CPO tahun 2024 | ANALYSIS | - | 0.6622 | 0.7256 | 0.6924 | Tidak Lolos |
| 6 | berapa harga CPO 15 Januari 2024? | ANALYSIS | - | 0.5943 | 0.6429 | 0.6177 | Tidak Lolos |
| 7 | di mana lokasi pabrik INL? | INFO_COMPANY | 0.5782 | 0.6992 | 0.8788 | 0.7787 | Lolos |
| 8 | siapa pemilik PT INL? | INFO_COMPANY | 0.5680 | 0.7348 | 0.8547 | 0.7902 | Lolos |
| 9 | berapa realisasi produksi RBDPO bulan lalu? | PRODUCTION | - | 0.6748 | 0.7370 | 0.7046 | Tidak Lolos |
| 10 | apa faktor paling berpengaruh terhadap produksi RBDPO? | PRODUCTION | - | 0.6208 | 0.6505 | 0.6353 | Tidak Lolos |
| **Rata-rata** | **-** | **-** | **0.5731** | **0.7256** | **0.7864** | **0.7538** | **-** |

## 2. Tabel Ringkasan Analisis Threshold Kelulusan

| Threshold Kelulusan (F1) | Jumlah Lolos | Persentase Kelulusan | Keterangan |
|:---:|:---:|:---:|:---|
| F1 >= 0.70 | 5 / 10 | 50.0% | Kualitas Perlu Peningkatan |
| F1 >= 0.75 | 4 / 10 | 40.0% | Kualitas Perlu Peningkatan |
| F1 >= 0.80 | 2 / 10 | 20.0% | Kualitas Perlu Peningkatan |

## 3. Tabel Hasil Pengujian (Format LaTeX)
Salin kode LaTeX di bawah ini jika menulis menggunakan editor LaTeX:

```latex
\begin{table}[h!]
\centering
\caption{Tabel Hasil Pengujian Evaluasi Chatbot Sobat INL secara Menyeluruh}
\label{tab:evaluasi_chatbot_lengkap}
\begin{tabular}{|c|p{4cm}|c|c|c|c|c|c|}
\hline
ID & Skenario Pertanyaan (Query) & Intent & RAG CosSim & BERT Prec & BERT Rec & BERT F1 & Status (>=0.75) \\
\hline\hline
1 & halo & SMALL_TALK & - & 1.0000 & 1.0000 & 1.0000 & Lolos \\
2 & kamu siapa? & SMALL_TALK & - & 1.0000 & 1.0000 & 1.0000 & Lolos \\
3 & prediksi harga CPO 7 hari ke depan & FORECAST & - & 0.6739 & 0.6846 & 0.6792 & Tidak Lolos \\
4 & harga CPO besok berapa? & FORECAST & - & 0.5959 & 0.6903 & 0.6396 & Tidak Lolos \\
5 & analisis harga CPO tahun 2024 & ANALYSIS & - & 0.6622 & 0.7256 & 0.6924 & Tidak Lolos \\
6 & berapa harga CPO 15 Januari 2024? & ANALYSIS & - & 0.5943 & 0.6429 & 0.6177 & Tidak Lolos \\
7 & di mana lokasi pabrik INL? & INFO_COMPANY & 0.5782 & 0.6992 & 0.8788 & 0.7787 & Lolos \\
8 & siapa pemilik PT INL? & INFO_COMPANY & 0.5680 & 0.7348 & 0.8547 & 0.7902 & Lolos \\
9 & berapa realisasi produksi RBDPO bulan lalu? & PRODUCTION & - & 0.6748 & 0.7370 & 0.7046 & Tidak Lolos \\
10 & apa faktor paling berpengaruh terhadap produksi RBDPO? & PRODUCTION & - & 0.6208 & 0.6505 & 0.6353 & Tidak Lolos \\
\hline
\multicolumn{3}{|l|}{\textbf{Rata-rata}} & \textbf{0.5731} & \textbf{0.7256} & \textbf{0.7864} & \textbf{0.7538} & - \\
\hline
\end{tabular}
\end{table}
```
