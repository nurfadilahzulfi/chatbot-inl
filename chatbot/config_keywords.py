# config_keywords.py (UPDATED)

# ==========================================
# 1. PERSONA & INSTRUKSI SISTEM
# ==========================================
SYSTEM_PERSONA = """
Kamu adalah 'Sobat INL', Asisten Virtual resmi dari PT Industri Nabati Lestari (INL).

IDENTITAS PERUSAHAAN (WAJIB DIINGAT):
1. PT Industri Nabati Lestari (INL) adalah ANAK PERUSAHAAN dari PTPN III (Persero) atau Holding Perkebunan Nusantara.
2. PT INL BUKAN bagian dari Salim Group, Sinar Mas, atau swasta lainnya.
3. Lokasi pabrik: KEK Sei Mangkei, Sumatera Utara.
4. Produk utama: Minyak Goreng (Salvaco, INL), Margarin, dan turunan CPO lainnya.

Gaya bicara: Ramah, Profesional, Informatif, Emoji secukupnya (🌴📈🤖).
Tugas utama:
- Menjelaskan profil perusahaan dengan akurat (Anak usaha PTPN III).
- Menjawab harga CPO aktual & historis.
- Menyajikan prediksi berbasis data.
"""

# === PROMPT KHUSUS ROBOT (STRICT) ===
STRICT_PROMPT = """
[ATURAN PENTING]
1. Jika user bertanya "Siapa INL?" atau "Apa itu PT INL?", WAJIB sebutkan bahwa INL adalah anak perusahaan PTPN III (Persero).
2. Gunakan data yang tersedia di KONTEKS untuk menjawab pertanyaan harga/data.
3. Jika data tidak ditemukan, gunakan knowledge kamu sendiri untuk membantu menjawab pertanyaannya (utamakan data terlebih dahulu).
4. JANGAN MENGARANG informasi tentang struktur perusahaan selain fakta di atas.
5. PENTING: Satuan harga CPO dalam dataset ini adalah USD per Metric Ton (USD/MT). Jangan menyebutkan satuan Rupiah atau per Kg untuk data harga berjangka ini.
"""

# ==========================================
# 2. KONVERSI KATA KE ANGKA & WAKTU
# ==========================================
WORD_TO_NUM = {
    "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    "besok": 1, "lusa": 2,
    "minggu": 7, "pekan": 7,
    "bulan": 30
}

# ==========================================
# 3. NORMALISASI ISTILAH DOMAIN
# ==========================================
DOMAIN_TERMS = {
    # Perusahaan
    "inl": "PT Industri Nabati Lestari",
    "pt inl": "PT Industri Nabati Lestari",
    "ptpn": "PTPN III (Persero)",
    "holding": "Holding Perkebunan Nusantara",
    "kek": "Kawasan Ekonomi Khusus",
    "sei mangkei": "Sei Mangkei",
    "cpo": "Crude Palm Oil",
    "minyak sawit": "Crude Palm Oil",

    # Aksi & sinonim
    "predict": "prediksi",
    "forecast": "prediksi",
    "estimasi": "prediksi",
    "price": "harga",
    "hrg": "harga",
    "brp": "berapa",
    "thanks": "terima kasih",
    "thx": "terima kasih",
    "pls": "tolong"
}

# ==========================================
# 4. INTENT KEYWORDS (DIHALUSKAN & DIPERLUAS)
# ==========================================
INTENT_KEYWORDS = {
    # ── Intent produksi RBDPO (Random Forest) ──────────────────────────────
    "PRODUCTION": [
        # Produk & realisasi
        "produksi", "realisasi", "rbdpo", "produk sawit",
        "output pabrik", "hasil produksi", "volume produksi",
        "throughput", "kapasitas produksi",

        # Target & capaian
        "target", "rkap", "capaian", "pencapaian",
        "tercapai", "tidak tercapai", "melebihi target", "kurang dari target",
        "sesuai target",

        # Operasional pabrik
        "hari olah", "running days", "cpo consume", "cpo dikonsumsi",
        "yield", "yield rbdpo", "pfad", "stok cpo", "stok bahan",

        # Model & analitik
        "akurasi produksi", "performa model", "prediksi produksi",
        "random forest", "feature importance", "faktor produksi",
        "paling berpengaruh", "variabel penting", "determinan produksi",
        "loocv", "mape produksi", "error prediksi",

        # Temporal produksi
        "produksi bulan", "produksi januari", "produksi februari",
        "produksi maret", "produksi april", "produksi mei", "produksi juni",
        "produksi juli", "produksi agustus", "produksi september",
        "produksi oktober", "produksi november", "produksi desember",

        # Frasa alami
        "berapa produksi", "produksi berapa", "berapa rbdpo",
        "gimana produksinya", "update produksi", "info produksi",
        "laporan produksi", "data produksi", "rekap produksi",
        "produksi kita", "produksi pabrik", "produksi inl",
    ],

    "FORECAST": [
        # Formal
        "prediksi", "forecast", "ramalan", "estimasi", "proyeksi",
        "kedepan", "ke depan", "mendatang", "berikutnya",
        "future", "tren", "arah", "outlook", "potensi",
        "naik", "turun", "stabil",

        # Bahasa sehari-hari
        "ke depannya", "bakal", "akan", "nanti",
        "besok", "lusa", "minggu depan", "pekan depan", "bulan depan",
        "gimana besok", "gimana kedepan", "arahnya gimana",
        "naik gak", "turun gak", "bakal naik", "bakal turun",
        "prospek ke depan"
    ],

    "RECOMMENDATION": [
        # Formal
        "rekomendasi", "saran", "anjuran",
        "beli", "jual", "tahan", "hold", "buy", "sell",
        "layak", "prospek", "peluang",
        "untung", "rugi", "resiko", "risiko",
        "posisi", "sebaiknya",

        # Bahasa sehari-hari
        "mending", "enaknya", "bagusan",
        "perlu beli gak", "perlu jual gak",
        "aman gak", "worth it", "masih bagus gak",
        "cocok gak", "ambil gak", "lepas gak"
    ],

    "ANALYSIS": [
        # Formal
        "analisis", "analisa",
        "tertinggi", "terendah",
        "rata-rata", "average", "mean",
        "historis", "history", "riwayat",
        "grafik", "chart", "tren historis",
        "rekap", "ringkasan","hari terakhir"

        # Bahasa sehari-hari
        "riwayatnya", "dari dulu", "sebelumnya",
        "kemarin", "bulan lalu", "tahun lalu",
        "pernah berapa", "kisaran", "range",
        "paling mahal", "paling murah",
        "naik turun", "pergerakannya"
    ],

    "PRICE_QUERY": [
        # Formal
        "harga", "berapa harga", "price",
        "nilai", "rate", "tanggal",

        # Semi formal
        "harga cpo", "harga hari ini", "harga sekarang",
        "harga per hari", "harga per tanggal",

        # Bahasa sehari-hari
        "berapa", "berapa sih", "sekarang berapa",
        "hari ini", "sekarang", "saat ini",
        "harga terbaru", "harga terkini",
        "cek harga", "info harga",

        # Pola tanya alami (natural language)
        "cpo hari ini berapa",
        "harga cpo sekarang",
        "harga cpo hari ini",
        "berapa harga cpo",
        "cpo sekarang berapa",

        # Variasi santai / lisan
        "cpo berapa",
        "lagi di harga berapa",
        "posisi harga",
        "harga lagi di berapa",
        "update harga",

        # Pola berbasis waktu
        "tanggal ini berapa",
        "di tanggal ini",
        "pada tanggal",
        "harga tanggal",
        "harga di tanggal"
    ],

    "INFO_COMPANY": [
        # Formal
        "profil", "tentang inl", "siapa inl",
        "alamat", "lokasi", "kantor", "pabrik",
        "kontak", "hubungi", "email", "telepon",
        "siapa inl", "profil", "lokasi", "dimana", 
        "alamat", "kontak", "sejarah", "kapan didirikan", 
        "pendirian", "tanggal berdiri","tahun berapa",
        "visi", "misi", "direksi", "pemilik", "holding",
        "tentang inl", "perusahaan apa",
        
        # JABATAN (Ini yang kemarin kurang lengkap)
        "direktur", "direksi", "pimpinan", "bos", "ceo", "manajemen",
        "kepala", "manager", "komisaris",
        
        # Lainnya
        "sejarah", "kapan berdiri",
        "visi", "misi", "produk", "bisnis",
        "struktur", "organisasi"

        # Bahasa sehari-hari
        "inl itu apa", "inl perusahaan apa",
        "kantornya dimana", "alamatnya dimana",
        "bisa kontak dimana", "nomor kantor",
        "usahanya apa", "bergerak di bidang apa"
    ]
}

# ==========================================
# 5. SMALL TALK & SOCIAL INTENT
# ==========================================
SMALL_TALK = {
    "sapaan": {
        "keywords": ["halo", "hi", "hai", "pagi", "siang", "sore", "malam", "assalamualaikum", "tes", "p"],
        "response": "Halo! 👋 Sobat INL siap bantu seputar harga & prediksi CPO 🌴"
    },
    "kabar": {
        "keywords": ["apa kabar", "gimana kabar", "sehat"],
        "response": "Sistem berjalan optimal 🚀 Data siap disajikan kapan saja."
    },
    "identitas": {
        "keywords": ["kamu siapa", "siapa kamu", "bot apa"],
        "response": "Saya Sobat INL 🤖 Asisten Virtual resmi PT Industri Nabati Lestari."
    },
    "pamit": {
        "keywords": ["dadah", "bye", "exit", "quit", "keluar"],
        "response": "Terima kasih sudah mampir 👋 Sampai jumpa kembali!"
    }
}
