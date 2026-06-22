# Hasil Pengujian Evaluasi Chatbot Sobat INL
Dokumen ini dibuat otomatis oleh program pengujian untuk digunakan langsung dalam skripsi/tugas akhir.

## 1. Tabel Hasil Pengujian (Format Markdown / Microsoft Word)
Tabel ini menyajikan hasil pengujian secara menyeluruh yang mencakup fungsionalitas Small Talk, LSTM (Prediksi & Analisis CPO), RAG (Info Perusahaan), dan Random Forest (Analisis Produksi):

| ID | Skenario Pertanyaan (Query) | Tipe Modul (Intent) | Cosine Similarity (Retrieval) | BERTScore Precision | BERTScore Recall | BERTScore F1 | Status (F1 >= 0.75) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | halo | SMALL_TALK | - | 1.0000 | 1.0000 | 1.0000 | Lolos |
| 2 | kamu siapa? | SMALL_TALK | - | 1.0000 | 1.0000 | 1.0000 | Lolos |
| 3 | prediksi harga CPO 7 hari ke depan | FORECAST | - | 0.8234 | 0.8136 | 0.8185 | Lolos |
| 4 | harga CPO besok berapa? | FORECAST | - | 0.6616 | 0.7045 | 0.6824 | Tidak Lolos |
| 5 | analisis harga CPO tahun 2024 | ANALYSIS | - | 0.7826 | 0.8002 | 0.7913 | Lolos |
| 6 | berapa harga CPO 15 Januari 2024? | ANALYSIS | - | 0.6169 | 0.6774 | 0.6457 | Tidak Lolos |
| 7 | di mana lokasi pabrik INL? | INFO_COMPANY | 0.5782 | 0.6992 | 0.8788 | 0.7787 | Lolos |
| 8 | siapa pemilik PT INL? | INFO_COMPANY | 0.5680 | 0.7072 | 0.8614 | 0.7767 | Lolos |
| 9 | berapa realisasi produksi RBDPO bulan lalu? | PRODUCTION | - | 0.5674 | 0.6707 | 0.6147 | Tidak Lolos |
| 10 | apa faktor paling berpengaruh terhadap produksi RBDPO? | PRODUCTION | - | 0.6878 | 0.7203 | 0.7037 | Tidak Lolos |
| **Rata-rata** | **-** | **-** | **0.5731** | **0.7546** | **0.8127** | **0.7812** | **-** |

## 2. Tabel Ringkasan Analisis Threshold Kelulusan

| Threshold Kelulusan (F1) | Jumlah Lolos | Persentase Kelulusan | Keterangan |
|:---:|:---:|:---:|:---|
| F1 >= 0.70 | 7 / 10 | 70.0% | Kualitas Baik |
| F1 >= 0.75 | 6 / 10 | 60.0% | Kualitas Perlu Peningkatan |
| F1 >= 0.80 | 3 / 10 | 30.0% | Kualitas Perlu Peningkatan |

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
3 & prediksi harga CPO 7 hari ke depan & FORECAST & - & 0.8234 & 0.8136 & 0.8185 & Lolos \\
4 & harga CPO besok berapa? & FORECAST & - & 0.6616 & 0.7045 & 0.6824 & Tidak Lolos \\
5 & analisis harga CPO tahun 2024 & ANALYSIS & - & 0.7826 & 0.8002 & 0.7913 & Lolos \\
6 & berapa harga CPO 15 Januari 2024? & ANALYSIS & - & 0.6169 & 0.6774 & 0.6457 & Tidak Lolos \\
7 & di mana lokasi pabrik INL? & INFO_COMPANY & 0.5782 & 0.6992 & 0.8788 & 0.7787 & Lolos \\
8 & siapa pemilik PT INL? & INFO_COMPANY & 0.5680 & 0.7072 & 0.8614 & 0.7767 & Lolos \\
9 & berapa realisasi produksi RBDPO bulan lalu? & PRODUCTION & - & 0.5674 & 0.6707 & 0.6147 & Tidak Lolos \\
10 & apa faktor paling berpengaruh terhadap produksi RBDPO? & PRODUCTION & - & 0.6878 & 0.7203 & 0.7037 & Tidak Lolos \\
\hline
\multicolumn{3}{|l|}{\textbf{Rata-rata}} & \textbf{0.5731} & \textbf{0.7546} & \textbf{0.8127} & \textbf{0.7812} & - \\
\hline
\end{tabular}
\end{table}
```
