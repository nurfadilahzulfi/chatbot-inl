"""
schemas.py — Pydantic models untuk request/response FastAPI
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  System
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    architecture: str
    hidden_size: int
    num_layers: int
    bidirectional: bool
    use_attention: bool
    sequence_length: int
    forecast_horizon: int
    input_features: int
    total_parameters: int
    config: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    """Opsional — override CONFIG default via JSON body."""
    epochs: Optional[int] = Field(None, ge=1, le=1000)
    lstm_hidden_size: Optional[int] = Field(None, ge=16, le=512)
    lstm_num_layers: Optional[int] = Field(None, ge=1, le=6)
    dropout_rate: Optional[float] = Field(None, ge=0.0, le=0.8)
    sequence_length: Optional[int] = Field(None, ge=10, le=200)
    learning_rate: Optional[float] = Field(None, gt=0)
    batch_size: Optional[int] = Field(None, ge=8, le=256)


class TrainResponse(BaseModel):
    message: str
    epochs_trained: int
    best_val_loss: float
    train_samples: int
    val_samples: int
    test_samples: int
    input_features: int
    config_used: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction & Forecast
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Tidak digunakan langsung (file upload), tapi bisa diperluas."""
    pass


class DayForecast(BaseModel):
    date: str
    day_of_week: str
    predicted_price: float
    lower_50: float
    upper_50: float
    lower_90: float
    upper_90: float
    uncertainty_std: float
    change_vs_last: float
    change_pct: float


class ForecastResponse(BaseModel):
    last_known_date: str
    last_known_price: float
    forecast_horizon: int
    mc_samples_used: int
    forecasts: List[DayForecast]


class PredictResponse(BaseModel):
    last_known_date: str
    last_known_price: float
    predicted_next_price: float
    change_vs_last: float
    change_pct: float


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

class SetMetrics(BaseModel):
    mae: float = Field(description="Mean Absolute Error")
    rmse: float = Field(description="Root Mean Squared Error")
    mape: float = Field(description="Mean Absolute Percentage Error (%)")
    r2: float = Field(description="R² — Coefficient of Determination")
    directional_accuracy: float = Field(description="Directional Accuracy (%)")


class MetricsResponse(BaseModel):
    validation: SetMetrics
    test: SetMetrics
