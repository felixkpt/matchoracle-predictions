from pydantic import BaseModel
from typing import Optional


class TrainRequest(BaseModel):
    competition: Optional[int] = None
    season_id: Optional[int] = None
    season_start_date: Optional[str] = None
    target: Optional[str] = None
    model_type: Optional[str] = None
    prefer_saved_matches: Optional[bool] = False
    is_grid_search: Optional[bool] = False
    is_random_search: Optional[bool] = False
    retrain_if_last_train_is_before: Optional[str] = None
    ignore_trained: Optional[bool] = False
    per_page: Optional[int] = None
    job_id: Optional[str] = None


class PredictRequest(BaseModel):
    competition: Optional[int] = None
    season_id: Optional[int] = None
    season_start_date: Optional[str] = None
    target: Optional[str] = None
    last_predict_date: Optional[str] = None
    prediction_type: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    target_match: Optional[int] = None
    job_id: Optional[str] = None
