# Hasil Pengujian Evaluasi Chatbot Sobat INL
Dokumen ini dibuat otomatis oleh program pengujian untuk digunakan langsung dalam skripsi/tugas akhir.

## 1. Tabel Hasil Pengujian (Format Markdown / Microsoft Word)
Tabel ini menyajikan hasil pengujian secara menyeluruh yang mencakup fungsionalitas Small Talk, LSTM (Prediksi & Analisis CPO), RAG (Info Perusahaan), dan Random Forest (Analisis Produksi):

| ID | Skenario Pertanyaan (Query) | Tipe Modul (Intent) | Cosine Similarity (Retrieval) | BERTScore Precision | BERTScore Recall | BERTScore F1 | Status (F1 >= 0.75) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | halo | SMALL_TALK | - | 0.5556 | 0.5574 | 0.5565 | Tidak Lolos |
| 2 | kamu siapa? | SMALL_TALK | - | 1.0000 | 1.0000 | 1.0000 | Lolos |
| 3 | prediksi harga CPO 7 hari ke depan | FORECAST | - | 0.4654 | 0.7154 | 0.5639 | Tidak Lolos |
| 4 | harga CPO besok berapa? | FORECAST | - | 0.5227 | 0.6903 | 0.5949 | Tidak Lolos |
| 5 | analisis harga CPO tahun 2024 | ANALYSIS | - | 0.4734 | 0.7018 | 0.5654 | Tidak Lolos |
| 6 | berapa harga CPO 15 Januari 2024? | ANALYSIS | - | 0.6178 | 0.8256 | 0.7067 | Tidak Lolos |
| 7 | di mana lokasi pabrik INL? | INFO_COMPANY | 0.5782 | 0.5520 | 0.8422 | 0.6669 | Tidak Lolos |
| 8 | siapa pemilik PT INL? | INFO_COMPANY | 0.5680 | 0.6237 | 0.8954 | 0.7352 | Tidak Lolos |
| 9 | berapa realisasi produksi RBDPO bulan lalu? | PRODUCTION | - | 0.4802 | 0.7221 | 0.5768 | Tidak Lolos |
| 10 | apa faktor paling berpengaruh terhadap produksi RBDPO? | PRODUCTION | - | 0.4784 | 0.6573 | 0.5537 | Tidak Lolos |
| **Rata-rata** | **-** | **-** | **0.5731** | **0.5769** | **0.7607** | **0.6520** | **-** |

## 2. Tabel Ringkasan Analisis Threshold Kelulusan

| Threshold Kelulusan (F1) | Jumlah Lolos | Persentase Kelulusan | Keterangan |
|:---:|:---:|:---:|:---|
| F1 >= 0.70 | 3 / 10 | 30.0% | Kualitas Perlu Peningkatan |
| F1 >= 0.75 | 1 / 10 | 10.0% | Kualitas Perlu Peningkatan |
| F1 >= 0.80 | 1 / 10 | 10.0% | Kualitas Perlu Peningkatan |

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
1 & halo & SMALL_TALK & - & 0.5556 & 0.5574 & 0.5565 & Tidak Lolos \\
2 & kamu siapa? & SMALL_TALK & - & 1.0000 & 1.0000 & 1.0000 & Lolos \\
3 & prediksi harga CPO 7 hari ke depan & FORECAST & - & 0.4654 & 0.7154 & 0.5639 & Tidak Lolos \\
4 & harga CPO besok berapa? & FORECAST & - & 0.5227 & 0.6903 & 0.5949 & Tidak Lolos \\
5 & analisis harga CPO tahun 2024 & ANALYSIS & - & 0.4734 & 0.7018 & 0.5654 & Tidak Lolos \\
6 & berapa harga CPO 15 Januari 2024? & ANALYSIS & - & 0.6178 & 0.8256 & 0.7067 & Tidak Lolos \\
7 & di mana lokasi pabrik INL? & INFO_COMPANY & 0.5782 & 0.5520 & 0.8422 & 0.6669 & Tidak Lolos \\
8 & siapa pemilik PT INL? & INFO_COMPANY & 0.5680 & 0.6237 & 0.8954 & 0.7352 & Tidak Lolos \\
9 & berapa realisasi produksi RBDPO bulan lalu? & PRODUCTION & - & 0.4802 & 0.7221 & 0.5768 & Tidak Lolos \\
10 & apa faktor paling berpengaruh terhadap produksi RBDPO? & PRODUCTION & - & 0.4784 & 0.6573 & 0.5537 & Tidak Lolos \\
\hline
\multicolumn{3}{|l|}{\textbf{Rata-rata}} & \textbf{0.5731} & \textbf{0.5769} & \textbf{0.7607} & \textbf{0.6520} & - \\
\hline
\end{tabular}
\end{table}
```
