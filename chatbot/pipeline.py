"""
pipeline.py — End-to-end CPO pipeline:
    Data loading → Feature engineering → Preprocessing →
    Training → Evaluation → Forecasting
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Optional, Dict, Any
from datetime import datetime

from model import CPO_LSTM

logger = logging.getLogger("cpo_pipeline")

# ── Default hyperparameter config ────────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # Data
    "target_col": "Close",
    "train_split": 0.80,
    "val_split": 0.10,
    "scaler_type": "MinMax",
    # Sequence
    "sequence_length": 60,
    "forecast_horizon": 7,
    # Model
    "lstm_hidden_size": 128,
    "lstm_num_layers": 3,
    "dropout_rate": 0.3,
    "bidirectional": False,
    "use_attention": True,
    # Training
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "MSE",
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    # Scheduler
    "scheduler_type": "CosineAnnealing",
    "lr_step_size": 20,
    "lr_gamma": 0.5,
    "lr_patience": 10,
    "lr_t_max": 50,
    # Early stopping
    "early_stopping": True,
    "patience": 15,
    "min_delta": 1e-6,
    # Outlier
    "outlier_method": "IQR",
    "iqr_factor": 3.0,
    "zscore_threshold": 3.5,
    # Misc
    "seed": 42,
    "save_model": True,
}


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience


class CPOPipeline:
    """Full ML pipeline for CPO price forecasting."""

    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config_override or {})}
        self._set_seed(self.config["seed"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️  Device: {self.device}")

        # State filled after train()
        self.model: Optional[CPO_LSTM] = None
        self.scaler = None
        self.close_scaler = None
        self.feature_cols: list = []
        self.target_idx: int = -1
        self.n_features: int = 0
        self.total_params: int = 0
        self.is_trained: bool = False
        self.metrics: Optional[dict] = None
        self._last_window: Optional[np.ndarray] = None  # for forecast
        self._last_close: Optional[float] = None
        self._last_date: Optional[str] = None
        self._df: Optional[pd.DataFrame] = None

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

        # Feature matrix
        feature_cols = [c for c in df.select_dtypes(include="number").columns if c != "Close"] + ["Close"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        target_idx = feature_cols.index("Close")

        self.feature_cols = feature_cols
        self.target_idx = target_idx
        data_array = df[feature_cols].values

        # Split
        n = len(data_array)
        train_end = int(n * self.config["train_split"])
        val_end = int(n * (self.config["train_split"] + self.config["val_split"]))
        train_data, val_data, test_data = (
            data_array[:train_end],
            data_array[train_end:val_end],
            data_array[val_end:],
        )
        logger.info(f"📊 Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

        # Scale
        scaler_cls = MinMaxScaler if self.config["scaler_type"] == "MinMax" else RobustScaler
        self.scaler = scaler_cls()
        train_scaled = self.scaler.fit_transform(train_data)
        val_scaled = self.scaler.transform(val_data)
        test_scaled = self.scaler.transform(test_data)

        self.close_scaler = scaler_cls()
        self.close_scaler.fit(train_data[:, target_idx].reshape(-1, 1))

        # Sequences
        SEQ = self.config["sequence_length"]
        X_train, y_train = self._build_sequences(train_scaled, SEQ, target_idx)
        X_val, y_val = self._build_sequences(
            np.concatenate([train_scaled[-SEQ:], val_scaled]), SEQ, target_idx
        )
        X_test, y_test = self._build_sequences(
            np.concatenate([val_scaled[-SEQ:], test_scaled]), SEQ, target_idx
        )

        BS = self.config["batch_size"]
        train_loader = self._to_loader(X_train, y_train, BS, shuffle=True)
        val_loader = self._to_loader(X_val, y_val, BS)
        test_loader = self._to_loader(X_test, y_test, BS)

        # Model
        self.n_features = X_train.shape[2]
        self.model = CPO_LSTM(
            input_size=self.n_features,
            hidden_size=self.config["lstm_hidden_size"],
            num_layers=self.config["lstm_num_layers"],
            dropout=self.config["dropout_rate"],
            bidirectional=self.config["bidirectional"],
            use_attention=self.config["use_attention"],
        ).to(self.device)
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"🧠 Model params: {self.total_params:,}")

        # Train
        result = self._train_loop(train_loader, val_loader)

        # Metrics
        val_preds, val_trues = self._predict_loader(val_loader)
        test_preds, test_trues = self._predict_loader(test_loader)
        val_m = self._compute_metrics(val_trues, val_preds)
        test_m = self._compute_metrics(test_trues, test_preds)
        self.metrics = {"validation": val_m, "test": test_m}

        # Save last window for forecast
        full_scaled = np.concatenate([train_scaled, val_scaled, test_scaled])
        self._last_window = full_scaled[-SEQ:]
        self._last_close = float(df["Close"].iloc[-1])
        self._last_date = str(df["Date"].iloc[-1].date())

        self.is_trained = True
        logger.info("✅ Training complete!")

        return {
            "epochs_trained": result["epochs_trained"],
            "best_val_loss": result["best_val_loss"],
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "input_features": self.n_features,
        }

    def forecast(self, horizon: int = 7, n_mc_samples: int = 100) -> dict:
        """Autoregressive forecast with MC Dropout uncertainty estimation."""
        if self._last_window is None:
            raise RuntimeError("Pipeline not trained or window not available.")

        target_idx = self.target_idx
        window_base = self._last_window.copy()

        self.model.train()  # activate dropout for MC sampling
        all_samples = []

        for _ in range(n_mc_samples):
            window = window_base.copy()
            preds_scaled = []
            for _ in range(horizon):
                x_t = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    pred_s, _ = self.model(x_t)
                val = pred_s.item()
                preds_scaled.append(val)
                new_row = window[-1].copy()
                new_row[target_idx] = val
                window = np.vstack([window[1:], new_row])

            prices = self.close_scaler.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).flatten()
            all_samples.append(prices)

        self.model.eval()
        all_samples = np.array(all_samples)  # (n_mc, horizon)

        mean = all_samples.mean(axis=0)
        std = all_samples.std(axis=0)
        lower_90 = np.percentile(all_samples, 5, axis=0)
        upper_90 = np.percentile(all_samples, 95, axis=0)
        lower_50 = np.percentile(all_samples, 25, axis=0)
        upper_50 = np.percentile(all_samples, 75, axis=0)

        last_close = self._last_close
        last_date = self._last_date
        future_dates = pd.bdate_range(
            start=pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon
        )

        forecasts = []
        for i, dt in enumerate(future_dates):
            forecasts.append({
                "date": str(dt.date()),
                "day_of_week": dt.strftime("%A"),
                "predicted_price": round(float(mean[i]), 2),
                "lower_50": round(float(lower_50[i]), 2),
                "upper_50": round(float(upper_50[i]), 2),
                "lower_90": round(float(lower_90[i]), 2),
                "upper_90": round(float(upper_90[i]), 2),
                "uncertainty_std": round(float(std[i]), 4),
                "change_vs_last": round(float(mean[i] - last_close), 2),
                "change_pct": round(float((mean[i] - last_close) / last_close * 100), 2),
            })

        return {
            "last_known_date": last_date,
            "last_known_price": round(last_close, 2),
            "forecast_horizon": horizon,
            "mc_samples_used": n_mc_samples,
            "forecasts": forecasts,
        }

    def predict_next(self, df_raw: pd.DataFrame) -> dict:
        """Single-step prediction from a new CSV (must have >= sequence_length rows)."""
        SEQ = self.config["sequence_length"]
        df = self._load_and_clean(df_raw)
        df = self._handle_outliers(df)
        df = self._engineer_features(df)

        # Align features
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Kolom tidak ditemukan di CSV: {missing}")

        data = df[self.feature_cols].values
        if len(data) < SEQ:
            raise ValueError(
                f"CSV harus berisi minimal {SEQ} baris data. "
                f"Diterima: {len(data)} baris."
            )

        # Scale using fitted scaler
        scaled = self.scaler.transform(data[-SEQ:])
        window = scaled  # (SEQ, n_features)

        self.model.eval()
        x_t = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_s, _ = self.model(x_t)

        pred_price = float(
            self.close_scaler.inverse_transform([[pred_s.item()]])[0][0]
        )
        last_close = float(df["Close"].iloc[-1])
        last_date = str(df["Date"].iloc[-1].date())

        return {
            "last_known_date": last_date,
            "last_known_price": round(last_close, 2),
            "predicted_next_price": round(pred_price, 2),
            "change_vs_last": round(pred_price - last_close, 2),
            "change_pct": round((pred_price - last_close) / last_close * 100, 2),
        }

    def save_model(self, path: str):
        """Simpan model + metadata ke file .pth"""
        import json
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config,
                "feature_cols": self.feature_cols,
                "target_idx": self.target_idx,
                "n_features": self.n_features,
                "scaler": self.scaler,
                "close_scaler": self.close_scaler,
                "last_window": self._last_window,
                "last_close": self._last_close,
                "last_date": self._last_date,
                "metrics": self.metrics,
            },
            path,
        )
        logger.info(f"💾 Model saved → {path}")

    def load_model(self, path: str):
        """Load model + metadata dari file .pth"""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint["config"]
        self.feature_cols = checkpoint["feature_cols"]
        self.target_idx = checkpoint["target_idx"]
        self.n_features = checkpoint["n_features"]
        self.scaler = checkpoint["scaler"]
        self.close_scaler = checkpoint["close_scaler"]
        self._last_window = checkpoint["last_window"]
        self._last_close = checkpoint["last_close"]
        self._last_date = checkpoint["last_date"]
        self.metrics = checkpoint.get("metrics")

        self.model = CPO_LSTM(
            input_size=self.n_features,
            hidden_size=self.config["lstm_hidden_size"],
            num_layers=self.config["lstm_num_layers"],
            dropout=self.config["dropout_rate"],
            bidirectional=self.config["bidirectional"],
            use_attention=self.config["use_attention"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.is_trained = True
        logger.info(f"✅ Model loaded from {path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_and_clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names, parse Indonesian number format, sort by date."""
        df = df_raw.copy()

        # Auto-detect column names (Indonesian or English)
        col_map_id = {
            "tanggal": "Date", "terakhir": "Close", "pembukaan": "Open",
            "tertinggi": "High", "terendah": "Low", "vol.": "Volume",
            "perubahan%": "Pct_Change", "perubahan": "Pct_Change",
        }
        df.columns = [
            col_map_id.get(c.lower().strip(), c.strip())
            for c in df.columns
        ]

        # Ensure required columns exist
        required = ["Date", "Close", "Open", "High", "Low"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {missing}. "
                f"Kolom yang ada: {list(df.columns)}"
            )

        # Parse dates
        for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]:
            try:
                df["Date"] = pd.to_datetime(df["Date"], format=fmt, dayfirst=True)
                break
            except Exception:
                continue
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, dayfirst=True)

        # Parse numeric (handle Indonesian "1.135,75" format)
        for col in ["Close", "Open", "High", "Low", "Pct_Change"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace("%", "", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )

        # Drop Volume (usually all null in CPO data)
        if "Volume" in df.columns:
            df.drop(columns=["Volume"], inplace=True, errors="ignore")

        df = df.sort_values("Date").reset_index(drop=True)
        df.dropna(subset=required, inplace=True)
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize outliers based on configured method."""
        method = self.config["outlier_method"]
        if method == "None":
            return df

        d = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            if col not in d.columns:
                continue
            if method == "IQR":
                Q1, Q3 = d[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                factor = self.config["iqr_factor"]
                lb, ub = Q1 - factor * IQR, Q3 + factor * IQR
            elif method == "ZScore":
                threshold = self.config["zscore_threshold"]
                mean_v, std_v = d[col].mean(), d[col].std()
                lb, ub = mean_v - threshold * std_v, mean_v + threshold * std_v
            else:
                continue
            d[col] = d[col].clip(lower=lb, upper=ub)
        return d

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators as model features."""
        d = df.copy()

        # Moving Averages
        d["MA_7"] = d["Close"].rolling(7).mean()
        d["MA_21"] = d["Close"].rolling(21).mean()
        d["MA_50"] = d["Close"].rolling(50).mean()

        # EMA
        d["EMA_12"] = d["Close"].ewm(span=12, adjust=False).mean()
        d["EMA_26"] = d["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        d["MACD"] = d["EMA_12"] - d["EMA_26"]
        d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
        d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

        # RSI (14-period)
        delta = d["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        d["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-period, 2σ)
        roll_mean = d["Close"].rolling(20).mean()
        roll_std = d["Close"].rolling(20).std()
        d["Bollinger_Upper"] = roll_mean + 2 * roll_std
        d["Bollinger_Lower"] = roll_mean - 2 * roll_std
        d["BB_Width"] = d["Bollinger_Upper"] - d["Bollinger_Lower"]

        # Volatility & Momentum
        d["Volatility_7"] = d["Close"].rolling(7).std()
        d["Volatility_30"] = d["Close"].rolling(30).std()
        d["Momentum_5"] = d["Close"].diff(5)
        d["Momentum_14"] = d["Close"].diff(14)

        # Lag Features
        for lag in [1, 3, 7]:
            d[f"Close_Lag_{lag}"] = d["Close"].shift(lag)

        # Price Ratios
        d["HL_Ratio"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-9)
        d["OC_Ratio"] = (d["Close"] - d["Open"]) / (d["Open"] + 1e-9)

        # Calendar Features
        d["DayOfWeek"] = d["Date"].dt.dayofweek
        d["Month"] = d["Date"].dt.month
        d["Quarter"] = d["Date"].dt.quarter

        if "Pct_Change" not in d.columns:
            d["Pct_Change"] = d["Close"].pct_change() * 100

        d.dropna(inplace=True)
        d.reset_index(drop=True, inplace=True)
        return d

    @staticmethod
    def _build_sequences(scaled_data: np.ndarray, seq_len: int, target_col_idx: int):
        X, y = [], []
        for i in range(len(scaled_data) - seq_len):
            X.append(scaled_data[i: i + seq_len, :])
            y.append(scaled_data[i + seq_len, target_col_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _to_loader(self, X, y, batch_size, shuffle=False) -> DataLoader:
        ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            pin_memory=self.device.type == "cuda",
        )

    def _train_loop(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        cfg = self.config
        model = self.model

        # Loss
        loss_map = {"MSE": nn.MSELoss(), "MAE": nn.L1Loss(), "Huber": nn.HuberLoss(delta=0.5)}
        criterion = loss_map[cfg["loss_function"]]

        # Optimizer
        opt_map = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}
        optimizer = opt_map[cfg["optimizer"]](
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )

        # Scheduler
        if cfg["scheduler_type"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"]
            )
        elif cfg["scheduler_type"] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg["lr_gamma"], patience=cfg["lr_patience"]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["lr_t_max"]
            )

        early_stop = EarlyStopping(cfg["patience"], cfg["min_delta"])
        best_val = float("inf")
        train_losses = []
        epochs_ran = 0

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            batch_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred, _ = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
                batch_loss += loss.item()

            train_loss = batch_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    val_loss += criterion(model(X_b)[0], y_b).item()
            val_loss /= len(val_loader)

            if cfg["scheduler_type"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            train_losses.append(train_loss)
            best_val = min(best_val, val_loss)
            epochs_ran = epoch

            if epoch % 10 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch:>4}/{cfg['epochs']}] "
                    f"train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.2e}"
                )

            if cfg["early_stopping"] and early_stop(val_loss, model):
                logger.info(f"⚡ Early stopping at epoch {epoch}")
                model.load_state_dict(early_stop.best_state)
                best_val = early_stop.best_loss
                break

        model.eval()
        return {"epochs_trained": epochs_ran, "best_val_loss": best_val}

    def _predict_loader(self, loader: DataLoader):
        self.model.eval()
        preds_s, trues_s = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                out, _ = self.model(X_b.to(self.device))
                preds_s.append(out.cpu().numpy())
                trues_s.append(y_b.numpy())
        preds_s = np.concatenate(preds_s)
        trues_s = np.concatenate(trues_s)
        preds = self.close_scaler.inverse_transform(preds_s).flatten()
        trues = self.close_scaler.inverse_transform(trues_s).flatten()
        return preds, trues

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
        r2 = float(r2_score(y_true, y_pred))
        da = float(
            np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        )
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "r2": round(r2, 4),
            "directional_accuracy": round(da, 4),
        }
