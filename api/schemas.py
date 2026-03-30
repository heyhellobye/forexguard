from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class ScoreRequest(BaseModel):
    user_id: str = Field(..., description="User ID to score")

    class Config:
        json_schema_extra = {"example": {"user_id": "user_0042"}}


class PredictRequest(BaseModel):
    user_id : str
    features: dict[str, float] = Field(
        ..., description="Feature name -> value (z-score normalised)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "new_user_999",
                "features": {
                    "n_unique_ips": 3.2,
                    "volume_spike_ratio": 4.1,
                    "impossible_travel_count": 3.0,
                    "micro_deposit_count": 2.5,
                    "off_hours_trade_ratio": 3.0,
                },
            }
        }


class ModelScore(BaseModel):
    isolation_forest: float
    lof             : float
    lstm_ae         : float


class AlertResponse(BaseModel):
    user_id              : str
    timestamp            : str
    ensemble_score       : float
    severity             : str
    model_scores         : ModelScore
    top_features         : dict[str, float]
    summary              : str
    action_required      : str
    flags                : list[str]
    ensemble_disagreement: float = 0.0


class HealthResponse(BaseModel):
    status  : str
    models  : dict[str, bool]
    n_users : int
    version : str = "2.0.0"


class StreamStatusResponse(BaseModel):
    is_running  : bool
    n_processed : int
    n_alerts    : int
    top_alerts  : list[dict]


class LLMAlertResponse(AlertResponse):
    llm_summary: str = ""
