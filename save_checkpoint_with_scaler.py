"""
save_checkpoint_with_scaler.py
================================
Jalankan script ini di Colab / Jupyter SETELAH training selesai,
MENGGANTIKAN perintah torch.save yang lama di notebook.

Tujuan: menyertakan scaler & close_scaler di dalam .pth
agar chatbot production TIDAK perlu re-fit scaler
(konsisten dengan data training).

Cara pakai:
    1. Jalankan training notebook sampai selesai (semua step)
    2. Jalankan script ini (copy-paste ke cell baru di Colab)
    3. Download file cpo_lstm_model_v2.pth yang dihasilkan
    4. Ganti file cpo_lstm_model.pth di folder data/ chatbot
"""

if CONFIG['save_model']:
    torch.save(
        {
            'model_state' : model.state_dict(),
            'config'      : CONFIG,
            'feature_cols': feature_cols,
            'target_idx'  : target_idx,
            # ── TAMBAHAN: simpan scaler agar chatbot tidak re-fit ──
            'scaler'      : scaler,        # fit pada train data
            'close_scaler': close_scaler,  # fit pada train Close column
        },
        'cpo_lstm_model_v2.pth'
    )
    print('💾 Model + Scaler saved → cpo_lstm_model_v2.pth')
    print('   Rename menjadi cpo_lstm_model.pth lalu upload ke folder data/')
