# Hasil Pengujian Evaluasi Chatbot Sobat INL
Dokumen ini dibuat otomatis oleh program pengujian untuk digunakan langsung dalam skripsi/tugas akhir.

## 1. Tabel Hasil Pengujian (Format Markdown / Microsoft Word)
Berikut adalah tabel lengkap hasil pengujian retrieval RAG dan semantic similarity (BERTScore) untuk setiap skenario:

| ID | Skenario Pertanyaan (Query) | Cosine Similarity (Retrieval) | BERTScore Precision | BERTScore Recall | BERTScore F1 | Status (F1 >= 0.75) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | Di mana lokasi kantor dan pabrik PT INL? | 0.5301 | 0.6759 | 0.9026 | 0.7730 | Lolos |
| 2 | Kapan PT Industri Nabati Lestari didirikan? | 0.6026 | 0.5563 | 0.6938 | 0.6175 | Tidak Lolos |
| 3 | Siapa nama Direktur PT INL yang menjabat saat ini? | 0.6411 | 0.5375 | 0.7843 | 0.6379 | Tidak Lolos |
| 4 | Berapa kapasitas produksi pabrik minyak kelapa sawit INL? | 0.7492 | 0.4592 | 0.4628 | 0.4610 | Tidak Lolos |
| 5 | Apa saja merek minyak goreng kemasan yang diproduksi oleh PT INL? | 0.6918 | 0.5101 | 0.7567 | 0.6094 | Tidak Lolos |
| 6 | Siapa pemilik atau induk perusahaan dari PT INL? | 0.5854 | 0.7344 | 0.9128 | 0.8139 | Lolos |
| 7 | Apa visi utama dari pembentukan PT INL? | 0.6284 | 0.6040 | 0.8228 | 0.6966 | Tidak Lolos |
| 8 | Mengapa KEK Sei Mangkei dipilih sebagai lokasi pabrik? | 0.5652 | 0.4950 | 0.7686 | 0.6022 | Tidak Lolos |
| 9 | Apa saja tantangan utama yang dihadapi oleh PT INL? | 0.6587 | 0.2940 | 0.4357 | 0.3511 | Tidak Lolos |
| 10 | Siapa pemegang saham utama dari PT INL? | 0.5940 | 0.5331 | 0.7141 | 0.6105 | Tidak Lolos |
| **Rata-rata** | **-** | **0.6247** | **0.5400** | **0.7254** | **0.6173** | **-** |

## 2. Tabel Ringkasan Analisis Threshold Kelulusan

| Threshold (F1) | Jumlah Lolos | Persentase Kelulusan |
|:---:|:---:|:---:|
| F1 >= 0.70 | 2 / 10 | 20.0% |
| F1 >= 0.75 | 2 / 10 | 20.0% |
| F1 >= 0.80 | 1 / 10 | 10.0% |

## 3. Tabel Hasil Pengujian (Format LaTeX)
Jika Anda menggunakan LaTeX, Anda dapat menyalin kode tabel di bawah ini:

```latex
\begin{table}[h!]
\centering
\caption{Tabel Hasil Pengujian Semantic Similarity Chatbot Sobat INL}
\label{tab:evaluasi_chatbot}
\begin{tabular}{|c|p{6cm}|c|c|c|c|c|}
\hline
ID & Skenario Pertanyaan (Query) & RAG CosSim & Prec (H,R) & Rec (H,R) & BERT F1 & Status (>=0.75) \\
\hline\hline
1 & Di mana lokasi kantor dan pabrik PT INL? & 0.5301 & 0.6759 & 0.9026 & 0.7730 & Lolos \\
2 & Kapan PT Industri Nabati Lestari didirikan? & 0.6026 & 0.5563 & 0.6938 & 0.6175 & Tidak Lolos \\
3 & Siapa nama Direktur PT INL yang menjabat saat ini? & 0.6411 & 0.5375 & 0.7843 & 0.6379 & Tidak Lolos \\
4 & Berapa kapasitas produksi pabrik minyak kelapa sawit INL? & 0.7492 & 0.4592 & 0.4628 & 0.4610 & Tidak Lolos \\
5 & Apa saja merek minyak goreng kemasan yang diproduksi oleh PT INL? & 0.6918 & 0.5101 & 0.7567 & 0.6094 & Tidak Lolos \\
6 & Siapa pemilik atau induk perusahaan dari PT INL? & 0.5854 & 0.7344 & 0.9128 & 0.8139 & Lolos \\
7 & Apa visi utama dari pembentukan PT INL? & 0.6284 & 0.6040 & 0.8228 & 0.6966 & Tidak Lolos \\
8 & Mengapa KEK Sei Mangkei dipilih sebagai lokasi pabrik? & 0.5652 & 0.4950 & 0.7686 & 0.6022 & Tidak Lolos \\
9 & Apa saja tantangan utama yang dihadapi oleh PT INL? & 0.6587 & 0.2940 & 0.4357 & 0.3511 & Tidak Lolos \\
10 & Siapa pemegang saham utama dari PT INL? & 0.5940 & 0.5331 & 0.7141 & 0.6105 & Tidak Lolos \\
\hline
\multicolumn{2}{|l|}{\textbf{Rata-rata}} & \textbf{0.6247} & \textbf{0.5400} & \textbf{0.7254} & \textbf{0.6173} & - \\
\hline
\end{tabular}
\end{table}
```
