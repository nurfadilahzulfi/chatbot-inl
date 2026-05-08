"""
forecast_handler.py  — Train-in-Runtime Edition
================================================
Model LSTM dilatih langsung dalam runtime yang sama dengan chatbot,
mengikuti step-by-step notebook CPO_LSTM_Prediction (1).ipynb.

Tidak ada load file .pth — scaler, model, dan last_window semuanya
hidup dalam memori Python yang sama → hasil prediksi 100% konsisten
dengan proses training.

Pipeline (identik dengan notebook):
  Step 2 : Outlier Handling  (IQR Winsorization, factor=3.0)
  Step 3 : Feature Engineering (25+ fitur teknikal)
  Step 4 : Preprocessing  (MinMaxScaler fit on TRAIN only + build sequences)
  Step 5 : Model Architecture (Stacked LSTM + Attention)
  Step 6 : Training (Adam + CosineAnnealing + EarlyStopping)
  Step 8 : Forecast (rolling autoregressive)
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# ARSITEKTUR MODEL (identik dengan notebook Step 5)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor):
        scores  = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights


class CPO_LSTM(nn.Module):
    """Stacked LSTM + optional Attention + FC regression head."""
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout, bidirectional, use_attention):
        super().__init__()
        self.use_attention  = use_attention
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            batch_first   = True,
            bidirectional = bidirectional,
        )

        lstm_out_size  = hidden_size * self.num_directions
        self.attention = AttentionLayer(lstm_out_size) if use_attention else None

        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        if self.use_attention and self.attention is not None:
            context, attn_w = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
            attn_w  = None
        return self.fc(context), attn_w


# ─────────────────────────────────────────────────────────────────────────────
# FORECASTER UTAMA
# ─────────────────────────────────────────────────────────────────────────────

class CPOForecaster:
    """
    Train LSTM CPO dalam runtime yang sama, lalu siap prediksi.

    Cara pakai:
        forecaster = CPOForecaster(ohlc_df)   # train sekali saat startup
        preds = forecaster.predict_next_days(days_to_predict=7)
    """

    # === CONFIG (identik dengan notebook) ===
    CONFIG = {
        'target_col'      : 'Close',
        'train_split'     : 0.80,
        'val_split'       : 0.10,
        'sequence_length' : 60,
        'forecast_horizon': 7,
        'lstm_hidden_size': 128,
        'lstm_num_layers' : 3,
        'dropout_rate'    : 0.3,
        'bidirectional'   : False,
        'use_attention'   : True,
        'epochs'          : 100,
        'batch_size'      : 32,
        'learning_rate'   : 0.001,
        'weight_decay'    : 1e-5,
        'gradient_clip'   : 1.0,
        'patience'        : 15,
        'min_delta'       : 1e-6,
        'iqr_factor'      : 3.0,
        'seed'            : 42,
    }

    def __init__(self, df_ohlc: pd.DataFrame):
        """
        df_ohlc : DataFrame dengan kolom Date, Close, Open, High, Low, Pct_Change
                  (output dari SmartChatbot.build_ohlc_dataframe())
        """
        torch.manual_seed(self.CONFIG['seed'])
        np.random.seed(self.CONFIG['seed'])

        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model        = None
        self.scaler       = None
        self.close_scaler = None
        self.feature_cols = []
        self.target_idx   = 0
        self.seq_length   = self.CONFIG['sequence_length']
        self.last_window  = None   # (seq_len, n_features) — disimpan setelah training
        self.is_trained   = False

        print(f"\n{'='*60}")
        print(f"🏋️  CPO LSTM — TRAIN-IN-RUNTIME (Mengikuti Notebook)")
        print(f"   Device  : {self.device}")
        print(f"   Epochs  : {self.CONFIG['epochs']} (max, early-stop aktif)")
        print(f"{'='*60}")

        self._run_full_pipeline(df_ohlc)

    # =========================================================================
    # STEP 2 — Outlier Handling (IQR Winsorization)
    # =========================================================================
    def _step2_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        d      = df.copy()
        factor = self.CONFIG['iqr_factor']
        for col in ['Open', 'High', 'Low', 'Close']:
            if col not in d.columns:
                continue
            Q1, Q3 = d[col].quantile([0.25, 0.75])
            IQR    = Q3 - Q1
            lb, ub = Q1 - factor * IQR, Q3 + factor * IQR
            n_clip = ((d[col] < lb) | (d[col] > ub)).sum()
            d[col] = d[col].clip(lower=lb, upper=ub)
            if n_clip:
                print(f"   {col}: {n_clip} outlier di-clip ke [{lb:.1f}, {ub:.1f}]")
        return d

    # =========================================================================
    # STEP 3 — Feature Engineering
    # =========================================================================
    def _step3_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy().sort_values('Date').reset_index(drop=True)

        d['MA_7']  = d['Close'].rolling(7).mean()
        d['MA_21'] = d['Close'].rolling(21).mean()
        d['MA_50'] = d['Close'].rolling(50).mean()

        d['EMA_12'] = d['Close'].ewm(span=12, adjust=False).mean()
        d['EMA_26'] = d['Close'].ewm(span=26, adjust=False).mean()

        d['MACD']        = d['EMA_12'] - d['EMA_26']
        d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
        d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']

        delta = d['Close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        d['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        rm  = d['Close'].rolling(20).mean()
        rs  = d['Close'].rolling(20).std()
        d['Bollinger_Upper'] = rm + 2 * rs
        d['Bollinger_Lower'] = rm - 2 * rs
        d['BB_Width']        = d['Bollinger_Upper'] - d['Bollinger_Lower']

        d['Volatility_7']  = d['Close'].rolling(7).std()
        d['Volatility_30'] = d['Close'].rolling(30).std()
        d['Momentum_5']    = d['Close'].diff(5)
        d['Momentum_14']   = d['Close'].diff(14)

        for lag in [1, 3, 7]:
            d[f'Close_Lag_{lag}'] = d['Close'].shift(lag)

        d['HL_Ratio'] = (d['High'] - d['Low']) / (d['Close'] + 1e-9)
        d['OC_Ratio'] = (d['Close'] - d['Open']) / (d['Open'] + 1e-9)

        d['DayOfWeek'] = d['Date'].dt.dayofweek
        d['Month']     = d['Date'].dt.month
        d['Quarter']   = d['Date'].dt.quarter

        d.dropna(inplace=True)
        d.reset_index(drop=True, inplace=True)
        return d

    # =========================================================================
    # STEP 4 — Preprocessing: Select Features, Scale, Build Sequences
    # =========================================================================
    def _step4_preprocess(self, df: pd.DataFrame):
        # Pilih SEMUA kolom numerik — identik dengan notebook Cell 16
        feature_cols = [c for c in df.select_dtypes(include='number').columns
                        if c != 'Close'] + ['Close']
        feature_cols = [c for c in feature_cols if c in df.columns]
        target_idx   = feature_cols.index('Close')

        self.feature_cols = feature_cols
        self.target_idx   = target_idx

        data_array = df[feature_cols].values
        n          = len(data_array)
        n_train    = int(n * self.CONFIG['train_split'])
        n_val      = int(n * (self.CONFIG['train_split'] + self.CONFIG['val_split']))

        train_data = data_array[:n_train]
        val_data   = data_array[n_train:n_val]
        test_data  = data_array[n_val:]

        # Fit scaler HANYA pada train — identik dengan notebook Step 4
        self.scaler = MinMaxScaler()
        train_scaled = self.scaler.fit_transform(train_data)
        val_scaled   = self.scaler.transform(val_data)
        test_scaled  = self.scaler.transform(test_data)

        self.close_scaler = MinMaxScaler()
        self.close_scaler.fit(train_data[:, target_idx].reshape(-1, 1))

        # Build sequences
        SEQ = self.seq_length

        def build_seq(scaled, seq_len, t_idx):
            X, y = [], []
            for i in range(len(scaled) - seq_len):
                X.append(scaled[i:i + seq_len])
                y.append(scaled[i + seq_len, t_idx])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        X_train, y_train = build_seq(train_scaled, SEQ, target_idx)
        X_val,   y_val   = build_seq(
            np.concatenate([train_scaled[-SEQ:], val_scaled]), SEQ, target_idx)
        X_test,  y_test  = build_seq(
            np.concatenate([val_scaled[-SEQ:],   test_scaled]), SEQ, target_idx)

        print(f"   Features  : {len(feature_cols)} kolom | target_idx={target_idx} ('Close')")
        print(f"   Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)} sequences")
        print(f"   Scaler fit pada {n_train}/{n} baris (train 80%)")

        BS = self.CONFIG['batch_size']
        def to_loader(X, y, shuffle=False):
            ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
            return DataLoader(ds, batch_size=BS, shuffle=shuffle,
                              pin_memory=(self.device.type == 'cuda'))

        train_loader = to_loader(X_train, y_train, shuffle=True)
        val_loader   = to_loader(X_val,   y_val)

        # Simpan last_window dari full scaled data
        full_scaled      = np.concatenate([train_scaled, val_scaled, test_scaled])
        self.last_window = full_scaled[-SEQ:].copy()

        return train_loader, val_loader, len(feature_cols)

    # =========================================================================
    # STEP 5+6 — Model Architecture + Training
    # =========================================================================
    def _step56_train(self, train_loader, val_loader, input_size) -> CPO_LSTM:
        C = self.CONFIG

        model = CPO_LSTM(
            input_size    = input_size,
            hidden_size   = C['lstm_hidden_size'],
            num_layers    = C['lstm_num_layers'],
            dropout       = C['dropout_rate'],
            bidirectional = C['bidirectional'],
            use_attention = C['use_attention'],
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Model params: {total_params:,}")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = C['learning_rate'],
            weight_decay = C['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50)

        best_val  = float('inf')
        best_state = None
        patience  = 0

        print(f"\n   {'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12}")
        print(f"   {'─'*42}")

        for epoch in range(1, C['epochs'] + 1):
            # — Train —
            model.train()
            t_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred, _ = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), C['gradient_clip'])
                optimizer.step()
                t_loss += loss.item()
            t_loss /= len(train_loader)

            # — Validate —
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    pred, _ = model(Xb)
                    v_loss += criterion(pred, yb).item()
            v_loss /= len(val_loader)

            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"   {epoch:>6} | {t_loss:>12.6f} | {v_loss:>12.6f}")

            # Early stopping
            if v_loss < best_val - C['min_delta']:
                best_val   = v_loss
                patience   = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            if patience >= C['patience']:
                print(f"\n   ⚡ Early stop @ epoch {epoch} (best val: {best_val:.6f})")
                break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        print(f"   {'─'*42}")
        print(f"   ✅ Training selesai! Best Val Loss: {best_val:.6f}")
        return model

    # =========================================================================
    # Full Pipeline Runner
    # =========================================================================
    def _run_full_pipeline(self, df_ohlc: pd.DataFrame):
        try:
            print(f"\n📥 Step 2 — Outlier Handling (IQR factor={self.CONFIG['iqr_factor']})...")
            df = self._step2_outlier(df_ohlc)

            print(f"📊 Step 3 — Feature Engineering...")
            df = self._step3_features(df)
            print(f"   Shape: {df.shape} | "
                  f"{df['Date'].min().date()} → {df['Date'].max().date()}")

            print(f"🧹 Step 4 — Preprocessing (Scale + Sequences)...")
            train_loader, val_loader, input_size = self._step4_preprocess(df)

            print(f"🧠 Step 5+6 — Build Model + Training...")
            self.model      = self._step56_train(train_loader, val_loader, input_size)
            self.is_trained = True
            print(f"\n🚀 CPOForecaster SIAP — prediksi in-memory aktif!\n")

        except Exception as e:
            print(f"❌ Pipeline gagal: {e}")
            import traceback; traceback.print_exc()
            self.is_trained = False

    # =========================================================================
    # STEP 8 — Rolling Autoregressive Forecast (MC Dropout — identik Colab)
    # =========================================================================
    def predict_next_days(self, historical_df: pd.DataFrame = None,
                          days_to_predict: int = 7,
                          n_mc_samples: int = 100) -> list:
        """
        Prediksi harga Close N hari ke depan menggunakan Monte Carlo Dropout,
        IDENTIK dengan notebook Step 8 (forecast_n_days di Colab).

        Colab pakai model.train() + n_mc_samples → mean prediction.
        Ini menghasilkan fluktuasi realistis, bukan garis monoton lurus.

        Parameters
        ----------
        historical_df   : Opsional. Jika None, pakai last_window dari training.
        days_to_predict : Jumlah hari ke depan.
        n_mc_samples    : Jumlah MC Dropout samples (default 100, sama dgn Colab).
        """
        if not self.is_trained or self.model is None:
            print("❌ Model belum dilatih.")
            return []

        try:
            # Tentukan starting window
            if historical_df is not None and len(historical_df) > 0:
                df_c  = self._step2_outlier(historical_df)
                df_f  = self._step3_features(df_c)
                miss  = [c for c in self.feature_cols if c not in df_f.columns]
                if miss:
                    print(f"⚠️  Kolom hilang: {miss} — pakai last_window training.")
                    window_base = self.last_window.copy()
                else:
                    arr = df_f[self.feature_cols].values.astype(np.float32)
                    if len(arr) < self.seq_length:
                        print(f"⚠️  Data kurang, pakai last_window training.")
                        window_base = self.last_window.copy()
                    else:
                        scaled      = self.scaler.transform(arr)
                        window_base = scaled[-self.seq_length:].copy()
                        print(f"📊 Window dari data terbaru "
                              f"(Close terakhir: {arr[-1, self.target_idx]:.2f})")
            else:
                window_base = self.last_window.copy()
                print(f"📊 Menggunakan last_window dari training.")

            # ── MC Dropout: identik dengan Colab Step 8 ──────────────────────
            # model.train() → dropout AKTIF → setiap sample berbeda
            self.model.train()
            all_samples = []

            for s in range(n_mc_samples):
                window       = window_base.copy()
                preds_scaled = []

                for _ in range(days_to_predict):
                    x_t = torch.tensor(
                        window[np.newaxis], dtype=torch.float32
                    ).to(self.device)

                    with torch.no_grad():
                        pred_s, _ = self.model(x_t)

                    pred_val = pred_s.item()
                    preds_scaled.append(pred_val)

                    # Roll window
                    new_row = window[-1].copy()
                    new_row[self.target_idx] = pred_val
                    window = np.vstack([window[1:], new_row])

                # Inverse transform → USD/MT
                preds_arr   = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
                preds_price = self.close_scaler.inverse_transform(preds_arr).flatten()
                all_samples.append(preds_price)

            # Kembali ke eval mode setelah sampling
            self.model.eval()

            # Ambil MEAN dari semua sampel (identik Colab)
            all_samples = np.array(all_samples)       # (n_mc, n_days)
            mean_pred   = all_samples.mean(axis=0)
            std_pred    = all_samples.std(axis=0)

            predictions = [round(float(v), 2) for v in mean_pred]

            print(f"📈 Prediksi {days_to_predict} hari (MC n={n_mc_samples}): {predictions}")
            print(f"   Std dev per hari: {[round(float(s), 4) for s in std_pred]}")
            return predictions

        except Exception as e:
            self.model.eval()   # pastikan kembali ke eval jika error
            print(f"❌ Error prediksi: {e}")
            import traceback; traceback.print_exc()
            return []