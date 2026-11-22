from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    user_code: str = Field(min_length=3, max_length=40)
    full_name: str = Field(min_length=1, max_length=120)
    date_of_birth: date


class ScanCreate(BaseModel):
    scan_code: str
    user_code: str
    quality: str
    quality_metrics: dict
    landmarks: list
    features: dict


class PhenotypicRequest(BaseModel):
    """Nine biomarkers. unit_system 'conventional': albumin g/dL, creatinine mg/dL, glucose mg/dL,
    CRP mg/dL.  'si': albumin g/L, creatinine umol/L, glucose mmol/L, CRP mg/L.
    Lymphocytes %, MCV fL, RDW %, ALP U/L, WBC 10^3/uL (= 10^9/L) in both."""
    user_code: str
    unit_system: Literal["conventional", "si"] = "conventional"
    chronological_age: Optional[float] = Field(default=None, gt=0, le=120)  # optional; server uses DOB
    albumin: float = Field(gt=0)
    creatinine: float = Field(gt=0)
    glucose: float = Field(gt=0)
    crp: float = Field(gt=0)
    lymphocyte_percent: float = Field(gt=0, le=100)
    mcv_fl: float = Field(gt=0)
    rdw_percent: float = Field(gt=0)
    alp_u_l: float = Field(gt=0)
    wbc_10e3_ul: float = Field(gt=0)
    scan_code: Optional[str] = None
