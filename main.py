from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from train import train
from predict import predict
from metrics import metrics
from app.auth.get_user_token import get_user_token
from configs.settings import EMAIL, PASSWORD
from configs.logger import Logger

app = FastAPI()

# Models for request bodies


class TrainRequest(BaseModel):
    competition: Optional[int] = None
    target: Optional[str] = None
    ignore_saved: bool = False
    is_grid_search: bool = False
    ignore_trained: bool = False
    last_train_date: Optional[str] = None
    prediction_type: Optional[str] = None
    per_page: int = 380


class PredictParams(BaseModel):
    competition: Optional[int] = None
    target: Optional[str] = None
    last_predict_date: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    target_match: Optional[int] = None

# Get the user token


def get_user_token_or_404():
    user_token = get_user_token(EMAIL, PASSWORD)
    if not user_token:
        raise HTTPException(
            status_code=401, detail="Failed to obtain user token.")
    return user_token


@app.post("/train")
async def train_model(request: TrainRequest):
    user_token = get_user_token_or_404()
    Logger.info("User token obtained successfully.")

    prediction_type = request.prediction_type or f"regular_prediction_12_6_4_{request.per_page}"
    train(user_token, prediction_type, request.model_dump())
    return {"message": "Training started"}


@app.post("/predict")
async def predict_model(request: PredictParams):
    user_token = get_user_token_or_404()
    Logger.info("User token obtained successfully.")

    rediction_type = request.prediction_type or f"regular_prediction_12_6_4_{request.per_page}"
    predict(user_token, rediction_type, request.model_dump())
    return {"message": "Prediction started"}


@app.get("/metrics")
async def get_metrics():
    user_token = get_user_token_or_404()
    Logger.info("User token obtained successfully.")

    metrics(user_token)
    return {"message": "Metrics generated"}

# Optional: Run using Uvicorn for testing
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
