"""
pipeline.py — End-to-end CPO pipeline:
    Data loading → Feature engineering → Preprocessing →
    Training → Evaluation → Forecasting

CATATAN: Model hanya hidup dalam memori (in-memory only).
Tidak ada ekspor .pth — menjamin konsistensi prediksi.
"""

import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, Dict, Any

from model import CPO_LSTM

logger = logging.getLogger("cpo_pipeline")

# ── Default config — 100% identik dengan notebook CONFIG ─────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # Data
    "target_col"        : "Close",
    "train_split"       : 0.80,
    "val_split"         : 0.10,
    "scaler_type"       : "Robust",       # ← notebook: RobustScaler
    # Sequence
    "sequence_length"   : 20,
    "forecast_horizon"  : 7,
    # Model — identik notebook
    "lstm_hidden_size"  : 128,
    "lstm_num_layers"   : 2,
    "dropout_rate"      : 0.3,
    "bidirectional"     : False,
    "use_attention"     : True,
    # Training — identik notebook
    "epochs"            : 200,
    "batch_size"        : 64,
    "learning_rate"     : 0.0005,
    "optimizer"         : "AdamW",
    "loss_function"     : "Huber",        # HuberLoss(delta=0.5)
    "weight_decay"      : 1e-5,
    "gradient_clip"     : 1.0,
    # Scheduler — identik notebook: CosineAnnealing T_max=50
    "scheduler_type"    : "CosineAnnealing",
    "lr_step_size"      : 20,
    "lr_gamma"          : 0.5,
    "lr_patience"       : 10,
    "lr_t_max"          : 50,
    # Early stopping — identik notebook
    "early_stopping"    : True,
    "patience"          : 15,
    "min_delta"         : 1e-6,
    # Outlier — identik notebook: IQR factor=3.0
    "outlier_method"    : "IQR",
    "iqr_factor"        : 3.0,
    "zscore_threshold"  : 3.5,
    # Misc
    "seed"              : 42,
    # save_model dihapus — model in-memory only, no .pth export
}


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Simpan best state dalam CPU memory — tidak ke disk
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


class CPOPipeline:
    """
    Full ML pipeline untuk prediksi harga CPO.
    Pipeline identik step-by-step dengan notebook CPO_LSTM_Prediction.ipynb:
      Step 1 : _load_and_clean()       — parse format Indonesia, sort by date
      Step 2 : _handle_outliers()      — IQR Winsorization (factor=3.0)
      Step 3 : _engineer_features()    — 25+ technical indicators
      Step 4 : train() scaling & seq   — RobustScaler fit on train only
      Step 5 : CPO_LSTM model          — dari model.py
      Step 6 : _train_loop()           — AdamW + CosineAnnealing + EarlyStopping
      Step 7 : _predict_loader()       — metrics on val & test
      Step 8 : forecast()              — autoregressive MC Dropout (100 samples)

    Model sepenuhnya in-memory — tidak ada ekspor/impor file .pth.
    Hasil prediksi 100% konsisten antara grafik dashboard dan chatbot.
    """

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config_override or {})}
        self._set_seed(self.config["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Device: {self.device}")

        # State — diisi setelah train()
        self.model        : Optional[CPO_LSTM]      = None
        self.scaler                                  = None   # full-feature scaler
        self.close_scaler                            = None   # Close-only scaler
        self.feature_cols : list                     = []
        self.target_idx   : int                      = -1
        self.n_features   : int                      = 0
        self.total_params : int                      = 0
        self.is_trained   : bool                     = False
        self.metrics      : Optional[dict]           = None
        self._last_window : Optional[np.ndarray]     = None   # (SEQ, n_features) scaled
        self._last_close  : Optional[float]          = None
        self._last_date   : Optional[str]            = None
        self._df          : Optional[pd.DataFrame]   = None
        # Backtest: prediksi vs aktual pada test set (untuk overlay chart)
        self.backtest_dates     : list                = []
        self.backtest_actual    : list                = []
        self.backtest_predicted : list                = []

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, df_raw: pd.DataFrame) -> dict:
        """Full training pipeline. Returns summary dict."""
        logger.info("🔄 Starting training pipeline...")

        df = self._load_and_clean(df_raw)
        df = self._handle_outliers(df)
        df = self._engineer_features(df)
        self._df = df.copy()

        # ── Feature matrix — identik notebook Step 4 ─────────────────────────
        # Semua kolom numerik kecuali Close, lalu Close di-append terakhir
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != "Close"
        ] + ["Close"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        target_idx   = feature_cols.index("Close")

        self.feature_cols = feature_cols
        self.target_idx   = target_idx
        data_array        = df[feature_cols].values

        # ── Temporal split — no shuffle, no data leakage ──────────────────────
        n         = len(data_array)
        train_end = int(n * self.config["train_split"])
        val_end   = int(n * (self.config["train_split"] + self.config["val_split"]))
        train_data = data_array[:train_end]
        val_data   = data_array[train_end:val_end]
        test_data  = data_array[val_end:]
        logger.info(
            f"📊 Split: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )

        # ── Scaling — fit HANYA pada train (identik notebook Step 4) ─────────
        scaler_cls = (
            MinMaxScaler if self.config["scaler_type"] == "MinMax" else RobustScaler
        )
        self.scaler   = scaler_cls()
        train_scaled  = self.scaler.fit_transform(train_data)
        val_scaled    = self.scaler.transform(val_data)
        test_scaled   = self.scaler.transform(test_data)

        # Close-only scaler untuk inverse transform prediksi (identik notebook)
        self.close_scaler = scaler_cls()
        self.close_scaler.fit(train_data[:, target_idx].reshape(-1, 1))

        # ── Sequences — identik notebook Step 4 ──────────────────────────────
        # val/test diberi konteks dari split sebelumnya agar tidak ada gap
        SEQ      = self.config["sequence_length"]
        X_train, y_train = self._build_sequences(train_scaled, SEQ, target_idx)
        X_val,   y_val   = self._build_sequences(
            np.concatenate([train_scaled[-SEQ:], val_scaled], axis=0), SEQ, target_idx
        )
        X_test,  y_test  = self._build_sequences(
            np.concatenate([val_scaled[-SEQ:], test_scaled], axis=0), SEQ, target_idx
        )

        BS           = self.config["batch_size"]
        train_loader = self._to_loader(X_train, y_train, BS, shuffle=True)
        val_loader   = self._to_loader(X_val,   y_val,   BS)
        test_loader  = self._to_loader(X_test,  y_test,  BS)

        # ── Model — identik notebook Step 5 ──────────────────────────────────
        self.n_features = X_train.shape[2]
        self.model = CPO_LSTM(
            input_size   = self.n_features,
            hidden_size  = self.config["lstm_hidden_size"],
            num_layers   = self.config["lstm_num_layers"],
            dropout      = self.config["dropout_rate"],
            bidirectional= self.config["bidirectional"],
            use_attention= self.config["use_attention"],
        ).to(self.device)
        self.total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"🧠 Model params: {self.total_params:,} | Features: {self.n_features}")

        # ── Training — identik notebook Step 6 ───────────────────────────────
        result = self._train_loop(train_loader, val_loader)

        # ── Evaluation — identik notebook Step 7 ─────────────────────────────
        val_preds,  val_trues  = self._predict_loader(val_loader)
        test_preds, test_trues = self._predict_loader(test_loader)
        self.metrics = {
            "validation": self._compute_metrics(val_trues,  val_preds),
            "test"       : self._compute_metrics(test_trues, test_preds),
        }
        logger.info(
            f"📈 Test — MAE={self.metrics['test']['mae']:.4f} "
            f"RMSE={self.metrics['test']['rmse']:.4f} "
            f"MAPE={self.metrics['test']['mape']:.2f}% "
            f"R²={self.metrics['test']['r2']:.4f} "
            f"DA={self.metrics['test']['directional_accuracy']:.2f}%"
        )

        # ── Backtest: simpan prediksi vs aktual pada test set ─────────────────
        # Test sequences dibangun dari concat([val[-SEQ:], test_scaled]).
        # _build_sequences: y[i] = concat[i+SEQ] = test_scaled[i]
        # Jadi y[0] → data_array[val_end], y[i] → data_array[val_end + i]
        # Tanggal: df["Date"].iloc[val_end + i]
        test_date_start = val_end  # posisi pertama test di df asli
        n_test = len(test_preds)
        self.backtest_dates = [
            str(df["Date"].iloc[test_date_start + i].date())
            for i in range(n_test)
            if (test_date_start + i) < len(df)
        ]
        self.backtest_actual    = [round(float(v), 2) for v in test_trues[:len(self.backtest_dates)]]
        self.backtest_predicted = [round(float(v), 2) for v in test_preds[:len(self.backtest_dates)]]
        logger.info(
            f"📊 Backtest saved: {len(self.backtest_dates)} data points "
            f"({self.backtest_dates[0] if self.backtest_dates else '?'} → "
            f"{self.backtest_dates[-1] if self.backtest_dates else '?'})"
        )

        # ── Last window — identik notebook Step 8: full_scaled[-SEQ:] ────────
        full_scaled       = np.concatenate([train_scaled, val_scaled, test_scaled], axis=0)
        self._last_window = full_scaled[-SEQ:].copy()
        self._last_close  = float(df["Close"].iloc[-1])
        self._last_date   = str(df["Date"].iloc[-1].date())

        self.is_trained = True
        logger.info("✅ Training complete!")

        return {
            "epochs_trained": result["epochs_trained"],
            "best_val_loss" : result["best_val_loss"],
            "train_samples" : len(X_train),
            "val_samples"   : len(X_val),
            "test_samples"  : len(X_test),
            "input_features": self.n_features,
        }

    def forecast(self, horizon: int = 7, n_mc_samples: int = 100) -> dict:
        """
        Autoregressive forecast dengan MC Dropout — identik notebook Step 8.

        Alur:
          1. model.train() → dropout aktif (MC Dropout)
          2. Loop n_mc_samples: tiap sample gunakan window berbeda karena dropout
          3. Per sample: loop horizon hari, tiap hari Close prediksi di-roll ke window
          4. Inverse transform → harga asli
          5. Agregasi: mean, std, percentile 5/25/75/95
        """
        if not self.is_trained or self._last_window is None:
            raise RuntimeError("Pipeline belum dilatih. Panggil train() terlebih dahulu.")

        target_idx  = self.target_idx
        window_base = self._last_window.copy()  # (SEQ, n_features), sudah scaled

        # MC Dropout: model.train() → dropout aktif → tiap forward pass berbeda
        self.model.train()
        all_samples: list = []

        for _ in range(n_mc_samples):
            window       = window_base.copy()
            preds_scaled = []

            for _ in range(horizon):
                x_t = torch.tensor(
                    window[np.newaxis, :, :], dtype=torch.float32
                ).to(self.device)
                with torch.no_grad():
                    pred_s, _ = self.model(x_t)
                val = pred_s.item()
                preds_scaled.append(val)

                # Roll window: hapus baris terlama, append baris baru dgn Close diupdate
                new_row             = window[-1].copy()
                new_row[target_idx] = val
                window              = np.vstack([window[1:], new_row])

            prices = self.close_scaler.inverse_transform(
                np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
            ).flatten()
            all_samples.append(prices)

        self.model.eval()
        samples = np.array(all_samples)  # (n_mc_samples, horizon)

        mean     = samples.mean(axis=0)
        std      = samples.std(axis=0)
        lower_90 = np.percentile(samples,  5, axis=0)
        upper_90 = np.percentile(samples, 95, axis=0)
        lower_50 = np.percentile(samples, 25, axis=0)
        upper_50 = np.percentile(samples, 75, axis=0)

        last_close   = self._last_close
        future_dates = pd.bdate_range(
            start=pd.Timestamp(self._last_date) + pd.Timedelta(days=1),
            periods=horizon,
        )

        forecasts = []
        for i, dt in enumerate(future_dates):
            forecasts.append({
                "date"           : str(dt.date()),
                "day_of_week"    : dt.strftime("%A"),
                "predicted_price": round(float(mean[i]),     2),
                "lower_50"       : round(float(lower_50[i]), 2),
                "upper_50"       : round(float(upper_50[i]), 2),
                "lower_90"       : round(float(lower_90[i]), 2),
                "upper_90"       : round(float(upper_90[i]), 2),
                "uncertainty_std": round(float(std[i]),      4),
                "change_vs_last" : round(float(mean[i] - last_close),                    2),
                "change_pct"     : round(float((mean[i] - last_close) / last_close * 100), 2),
            })

        return {
            "last_known_date"  : self._last_date,
            "last_known_price" : round(last_close, 2),
            "forecast_horizon" : horizon,
            "mc_samples_used"  : n_mc_samples,
            "forecasts"        : forecasts,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_and_clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Identik notebook Step 1:
          - Rename kolom Indonesia → English
          - Parse tanggal DD/MM/YYYY
          - Parse angka format Indonesia "1.135,75" → 1135.75
          - Drop Volume (semua null)
          - Sort chronologically
        """
        df = df_raw.copy()

        # Auto-detect kolom Indonesia
        col_map_id = {
            "tanggal"    : "Date",   "terakhir"   : "Close",
            "pembukaan"  : "Open",   "tertinggi"  : "High",
            "terendah"   : "Low",    "vol."       : "Volume",
            "perubahan%" : "Pct_Change", "perubahan": "Pct_Change",
        }
        df.columns = [
            col_map_id.get(c.lower().strip(), c.strip()) for c in df.columns
        ]

        required = ["Date", "Close", "Open", "High", "Low"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {missing}. "
                f"Kolom tersedia: {list(df.columns)}"
            )

        # Parse tanggal — coba beberapa format
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                df["Date"] = pd.to_datetime(df["Date"], format=fmt, dayfirst=True)
                break
            except Exception:
                continue
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        # Parse angka format Indonesia "1.135,75" → 1135.75 (identik notebook)
        for col in ["Close", "Open", "High", "Low", "Pct_Change"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(".", "", regex=False)   # hapus thousands separator
                    .str.replace(",", ".", regex=False)  # ganti desimal koma → titik
                    .str.replace("%", "", regex=False)   # hapus simbol persen
                    .pipe(pd.to_numeric, errors="coerce")
                )

        if "Volume" in df.columns:
            df.drop(columns=["Volume"], inplace=True, errors="ignore")

        df = df.sort_values("Date").reset_index(drop=True)
        df.dropna(subset=required, inplace=True)
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identik notebook Step 2:
          - IQR Winsorization (clip, bukan drop) — preserves timestamp continuity
          - factor=3.0 (toleran, sesuai CONFIG notebook)
        """
        method = self.config["outlier_method"]
        if method == "None":
            return df

        d = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col not in d.columns:
                continue
            if method == "IQR":
                Q1, Q3 = d[col].quantile([0.25, 0.75])
                IQR    = Q3 - Q1
                factor = self.config["iqr_factor"]
                lb, ub = Q1 - factor * IQR, Q3 + factor * IQR
            elif method == "ZScore":
                threshold  = self.config["zscore_threshold"]
                mean_v     = d[col].mean()
                std_v      = d[col].std()
                lb, ub     = mean_v - threshold * std_v, mean_v + threshold * std_v
            else:
                continue
            d[col] = d[col].clip(lower=lb, upper=ub)  # Winsorize (bukan drop)
        return d

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identik notebook Step 3 — semua 25+ technical indicator:
          MA_7, MA_21, MA_50
          EMA_12, EMA_26
          MACD, MACD_Signal, MACD_Hist
          RSI (14-period)
          Bollinger_Upper, Bollinger_Lower, BB_Width (20-period, 2σ)
          Volatility_7, Volatility_30
          Momentum_5, Momentum_14
          Close_Lag_1, Close_Lag_3, Close_Lag_7
          HL_Ratio, OC_Ratio
          DayOfWeek, Month, Quarter
        """
        d = df.copy()

        # Moving Averages
        d["MA_7"]  = d["Close"].rolling(7).mean()
        d["MA_21"] = d["Close"].rolling(21).mean()
        d["MA_50"] = d["Close"].rolling(50).mean()

        # EMA
        d["EMA_12"] = d["Close"].ewm(span=12, adjust=False).mean()
        d["EMA_26"] = d["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        d["MACD"]        = d["EMA_12"] - d["EMA_26"]
        d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
        d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]

        # RSI (14-period) — identik notebook: gain/loss rolling mean + epsilon
        delta = d["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        d["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-period, 2σ)
        roll_mean = d["Close"].rolling(20).mean()
        roll_std  = d["Close"].rolling(20).std()
        d["Bollinger_Upper"] = roll_mean + 2 * roll_std
        d["Bollinger_Lower"] = roll_mean - 2 * roll_std
        d["BB_Width"]        = d["Bollinger_Upper"] - d["Bollinger_Lower"]

        # Volatility & Momentum
        d["Volatility_7"]  = d["Close"].rolling(7).std()
        d["Volatility_30"] = d["Close"].rolling(30).std()
        d["Momentum_5"]    = d["Close"].diff(5)
        d["Momentum_14"]   = d["Close"].diff(14)

        # Lag Features
        for lag in [1, 3, 7]:
            d[f"Close_Lag_{lag}"] = d["Close"].shift(lag)

        # Price Ratios
        d["HL_Ratio"] = (d["High"] - d["Low"])    / (d["Close"] + 1e-9)
        d["OC_Ratio"] = (d["Close"] - d["Open"])  / (d["Open"]  + 1e-9)

        # Calendar Features
        d["DayOfWeek"] = d["Date"].dt.dayofweek   # 0=Mon … 4=Fri
        d["Month"]     = d["Date"].dt.month
        d["Quarter"]   = d["Date"].dt.quarter

        # Pct_Change — hitung jika belum ada dari raw CSV
        if "Pct_Change" not in d.columns:
            d["Pct_Change"] = d["Close"].pct_change() * 100

        # Drop NaN dari rolling windows (identik notebook: df.dropna())
        d.dropna(inplace=True)
        d.reset_index(drop=True, inplace=True)
        return d

    @staticmethod
    def _build_sequences(
        scaled_data: np.ndarray, seq_len: int, target_col_idx: int
    ):
        """
        Identik notebook Step 4 build_sequences():
          X[i] = scaled_data[i : i+seq_len, :]          (semua fitur)
          y[i] = scaled_data[i+seq_len, target_col_idx] (Close berikutnya)
        """
        X, y = [], []
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i : i + seq_len, :])
            y.append(scaled_data[i + seq_len, target_col_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _to_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def _train_loop(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict:
        """
        Identik notebook Step 6:
          - HuberLoss(delta=0.5)
          - AdamW(lr=0.0005, weight_decay=1e-5)
          - CosineAnnealingLR(T_max=50)
          - gradient clipping max_norm=1.0
          - EarlyStopping(patience=15, min_delta=1e-6)
          - best_state di-restore setelah stopping
        """
        cfg   = self.config
        model = self.model

        # Loss — identik notebook
        loss_map = {
            "MSE"  : nn.MSELoss(),
            "MAE"  : nn.L1Loss(),
            "Huber": nn.HuberLoss(delta=0.5),   # delta=0.5 identik notebook
        }
        criterion = loss_map[cfg["loss_function"]]

        # Optimizer — identik notebook
        opt_map = {
            "Adam" : torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "SGD"  : torch.optim.SGD,
        }
        optimizer = opt_map[cfg["optimizer"]](
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

        # Scheduler — identik notebook
        if cfg["scheduler_type"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"]
            )
        elif cfg["scheduler_type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg["lr_gamma"], patience=cfg["lr_patience"]
            )
        else:  # CosineAnnealing (default)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["lr_t_max"]
            )

        early_stop = EarlyStopping(cfg["patience"], cfg["min_delta"])
        epochs_ran = 0

        logger.info(
            f"🏋️ Training: epochs={cfg['epochs']} batch={cfg['batch_size']} "
            f"optimizer={cfg['optimizer']} loss={cfg['loss_function']} "
            f"scheduler={cfg['scheduler_type']} device={self.device}"
        )

        for epoch in range(1, cfg["epochs"] + 1):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            batch_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred, _  = model(X_b)
                loss     = criterion(pred, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
                batch_loss += loss.item()
            train_loss = batch_loss / len(train_loader)

            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    val_loss += criterion(model(X_b)[0], y_b).item()
            val_loss /= len(val_loader)

            # ── Scheduler step ────────────────────────────────────────────────
            if cfg["scheduler_type"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            epochs_ran = epoch
            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Epoch [{epoch:>4}/{cfg['epochs']}] "
                    f"train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.2e}"
                )

            # ── Early stopping — restore best state, tidak simpan ke disk ─────
            if cfg["early_stopping"] and early_stop(val_loss, model):
                logger.info(
                    f"⚡ Early stopping at epoch {epoch} "
                    f"(best val: {early_stop.best_loss:.6f})"
                )
                model.load_state_dict(early_stop.best_state)  # restore in-memory
                break

        model.eval()
        return {
            "epochs_trained": epochs_ran,
            "best_val_loss" : early_stop.best_loss,
        }

    def _predict_loader(self, loader: DataLoader):
        """
        Identik notebook Step 7 predict_all():
          - Inference pada DataLoader
          - Inverse transform dengan close_scaler
        """
        self.model.eval()
        preds_s, trues_s = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                out, _ = self.model(X_b.to(self.device))
                preds_s.append(out.cpu().numpy())
                trues_s.append(y_b.numpy())
        preds_s = np.concatenate(preds_s)
        trues_s = np.concatenate(trues_s)
        preds   = self.close_scaler.inverse_transform(preds_s).flatten()
        trues   = self.close_scaler.inverse_transform(trues_s).flatten()
        return preds, trues

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Identik notebook Step 7 compute_metrics(): MAE, RMSE, MAPE, R², DA."""
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
        r2   = float(r2_score(y_true, y_pred))
        da   = float(
            np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        )
        return {
            "mae"                 : round(mae,  4),
            "rmse"                : round(rmse, 4),
            "mape"                : round(mape, 4),
            "r2"                  : round(r2,   4),
            "directional_accuracy": round(da,   4),
        }