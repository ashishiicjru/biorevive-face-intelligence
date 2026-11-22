from datetime import datetime, timezone
from pydantic import NaiveDatetime
from typing import Optional
from sqlmodel import SQLModel, Field, Column, JSON
def _utcnow(): return datetime.now(timezone.utc).replace(tzinfo=None)   # naive UTC, stored as such
class User(SQLModel,table=True):
    id: Optional[int]=Field(default=None,primary_key=True)
    user_code: str=Field(index=True,unique=True)
    full_name: str
    date_of_birth: str
    created_at: NaiveDatetime=Field(default_factory=_utcnow)
class Scan(SQLModel,table=True):
    id: Optional[int]=Field(default=None,primary_key=True)
    scan_code: str=Field(index=True,unique=True)
    user_code: str=Field(index=True)
    captured_at: NaiveDatetime=Field(default_factory=_utcnow)
    quality: str
    quality_metrics: dict=Field(default={},sa_column=Column(JSON))
    landmark_count: int=0
    landmarks: list=Field(default=[],sa_column=Column(JSON))
    features: dict=Field(default={},sa_column=Column(JSON))
    apparent_age: Optional[float]=None
    age_model: Optional[str]=None
class PhenotypicAssessment(SQLModel,table=True):
    id: Optional[int]=Field(default=None,primary_key=True)
    assessment_code: str=Field(index=True,unique=True)
    user_code: str=Field(index=True)
    scan_code: Optional[str]=None
    assessed_at: NaiveDatetime=Field(default_factory=_utcnow)
    chronological_age: float
    biological_age: float
    age_difference: float
    biomarkers: dict=Field(default={},sa_column=Column(JSON))
    algorithm: str="PhenotypicAge"
class StudyRecord(SQLModel,table=True):
    id: Optional[int]=Field(default=None,primary_key=True)
    session_code: str=Field(index=True,unique=True)
    user_code: Optional[str]=Field(default=None,index=True)
    started_at: NaiveDatetime=Field(default_factory=_utcnow)
    duration_s: float=0.0
    average_score: float=0.0
    summary: dict=Field(default={},sa_column=Column(JSON))
class Snapshot(SQLModel,table=True):
    id: Optional[int]=Field(default=None,primary_key=True)
    snapshot_code: str=Field(index=True,unique=True)
    user_code: Optional[str]=Field(default=None,index=True)
    created_at: NaiveDatetime=Field(default_factory=_utcnow)
    caption: str=""
    source: str="live"
    filename: str
    analysis: dict=Field(default={},sa_column=Column(JSON))
