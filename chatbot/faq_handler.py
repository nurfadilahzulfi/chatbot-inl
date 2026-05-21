import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FAQHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_chunks = []
        self.doc_embeddings = None
        self.model = None
        self._load_data()

    def _load_data(self):
        print("Memuat Knowledge Base (Keyword Boost Enabled)...")
        if os.path.exists(self.file_path):
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                with open(self.file_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
                    
                    # Split berdasarkan enter 2x
                    chunks = full_text.split('\n\n')
                    self.text_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
                
                if self.text_chunks:
                    self.doc_embeddings = self.model.encode(self.text_chunks)
                    print(f"Loaded {len(self.text_chunks)} blocks.")
                else:
                    print("File FAQ kosong.")
            except Exception as e:
                print(f"Gagal memuat FAQ: {e}")

    def search(self, query, top_k=5): # Default top_k kita naikkan jadi 5
        if self.doc_embeddings is None or not self.text_chunks:
            return ""

        # 1. Hitung Skor Semantik (Vector)
        query_embedding = self.model.encode([query])
        # flatten ke 1D array
        scores = cosine_similarity(query_embedding, self.doc_embeddings)[0] 

        # === 2. LOGIC HYBRID: KEYWORD BOOSTING ===
        # Kita manipulasi skor berdasarkan kata kunci penting
        
        query_lower = query.lower()
        keywords_critical = [
            # Jabatan / Struktur Organisasi
            "direktur", "direksi", "pimpinan", "ketua", "ceo", "commissioner",
            "komisaris", "manager", "manajer", "wakil direktur", "vp", 
            "kepala divisi", "head", "chief", "bos", "atasan",

            # Lokasi / Alamat / Identitas Tempat
            "lokasi", "alamat", "dimana", "letak", "posisi", "lokasinya dimana",
            "kodepos", "kode pos", "wilayah", "provinsi", "kabupaten", "kecamatan",
            "kantor pusat", "head office", "pabrik", "gedung", "site", "area",

            # Kapasitas / Produksi / Angka sensitivitas
            "kapasitas", "ton", "tonase", "produksi", "output", 
            "kapasitas produksi", "jumlah produksi", "volume produksi",
            "berapa ton", "hasil produksi", "kapasitas pabrik", "yield",

            # Struktur Perusahaan / Kepemilikan
            "saham", "pemilik", "induk", "holding", "parent company",
            "anak perusahaan", "subsidiary", "group", "grup", 
            "struktur perusahaan", "pemegang saham",

            # Sejarah / Waktu / Legalitas
            "sejarah", "pendirian", "didirikan","tahun", "tahun berdiri", "kapan berdiri",
            "akta", "legalitas", "nib", "izin usaha", "nomor izin", 
            "registrasi", "peresmian",

            # Informasi sensitif lain
            "omset", "pendapatan", "revenue", "keuangan", "kapital", 
            "aset", "nilai perusahaan", "market share",

            # Unit / Divisi / Departemen
            "departemen", "divisi", "unit bisnis", "operasional", "plant",
            "warehouse", "gudang", "line produksi", "unit kerja"
        ]


        boosted_scores = scores.copy()
        
        for idx, text in enumerate(self.text_chunks):
            text_lower = text.lower()
            
            # Cek setiap kata kunci krusial
            for kw in keywords_critical:
                # Jika kata kunci muncul di QUERY dan juga muncul di TEKS CHUNK
                if kw in query_lower and kw in text_lower:
                    # BERIKAN BOOST MASSIF (+0.3 itu sangat besar di dunia vector)
                    boosted_scores[idx] += 0.35 
                    # Print debug biar tau ada boosting
                    # print(f"   🚀 Boost '{kw}' pada chunk: {text[:20]}...")

        # 3. Ambil Top-K berdasarkan skor yang SUDAH DIBOOST
        top_indices = np.argsort(boosted_scores)[-top_k:][::-1]
        
        relevant_texts = []
        print(f"\n🔍 [DEBUG SEARCH HYBRID]")
        
        for idx in top_indices:
            score = boosted_scores[idx]
            original_score = scores[idx]
            text = self.text_chunks[idx]
            
            # Threshold aman (0.25). 
            # Note: Score bisa > 1.0 karena boosting, itu tidak masalah.
            if score > 0.25:
                # Tampilkan skor. Jika boosted, kasih tanda 🚀
                mark = "🚀" if score > original_score else ""
                preview = text.split('\n')[0][:50]
                print(f"   - Match ({score:.4f}) {mark}: {preview}...")
                
                relevant_texts.append(text)

        if not relevant_texts:
            return None

        return "\n\n".join(relevant_texts)