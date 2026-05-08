# rf_schemas.py
from pydantic import BaseModel
from typing import List, Optional

class BulanData(BaseModel):
    bulan: str
    realisasi_produksi: float
    target_produksi: float
    stok_cpo: float
    is_imputed: bool

class FeatureItem(BaseModel):
    fitur: str
    kontribusi_pct: float
    permutation: float
    permutation_std: float

class EvaluasiMetrik(BaseModel):
    mae: float
    rmse: float
    r2: float
    mape: float
    jumlah_data: int
    jumlah_imputasi: int

class RFAnalysisResponse(BaseModel):
    status: str
    pesan: Optional[str] = None
    dataset: List[BulanData]
    feature_importance: List[FeatureItem]
    evaluasi: EvaluasiMetrik
    last_updated: str

class ProductionHistoryResponse(BaseModel):
    status: str
    categories: List[str]           # label sumbu X (bulan)
    realisasi: List[Optional[float]] # nilai aktual
    target: List[Optional[float]]    # nilai target RKAP
    prediksi_loocv: List[Optional[float]]  # nilai prediksi LOOCV
    stok_cpo: List[Optional[float]]  # nilai stok CPO
    is_imputed: List[bool]           # flag mana yang diimputasi
    last_updated: str