# rf_schemas.py
from pydantic import BaseModel
from typing import List, Optional


class FeatureImportanceItem(BaseModel):
    peringkat: int
    fitur: str
    mdi_pct: float          # MDI importance dalam persen
    permutation: float      # Permutation importance
    permutation_std: float


class BulanHistoris(BaseModel):
    bulan: str              # format: "2024-01"
    bulan_label: str        # format: "Jan 2024"
    realisasi: float        # kg
    target_rkap: float      # kg
    cpo_consume: float      # kg
    hari_olah: float
    yield_rbdpo: float      # %
    stok_rata2: float       # kg
    prediksi_loocv: float   # kg
    error_pct: float        # %
    is_imputed: bool


class EvaluasiModel(BaseModel):
    mae: float
    rmse: float
    r2: float
    mape: float
    jumlah_data: int
    jumlah_imputed: int
    interpretasi_r2: str
    interpretasi_mape: str


class RFAnalysisResponse(BaseModel):
    status: str
    last_updated: str
    evaluasi: EvaluasiModel
    feature_importance: List[FeatureImportanceItem]
    historis: List[BulanHistoris]


class RFFeatureImportanceResponse(BaseModel):
    status: str
    last_updated: str
    jumlah_data: int
    feature_importance: List[FeatureImportanceItem]


class RFHistorisResponse(BaseModel):
    status: str
    last_updated: str
    # Format siap pakai untuk ApexCharts di Vue.js
    categories: List[str]           # label sumbu X
    realisasi: List[float]
    target_rkap: List[float]
    cpo_consume: List[float]
    hari_olah: List[float]
    prediksi_loocv: List[float]
    is_imputed: List[bool]
    evaluasi: EvaluasiModel