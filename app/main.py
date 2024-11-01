from fastapi import FastAPI, HTTPException
from app.train import train
from app.predict import predict
from app.metrics import metrics
from app.auth.get_user_token import get_user_token
from app.configs.settings import EMAIL, PASSWORD
from app.configs.logger import Logger
from app.requests.prediction_request import TrainRequest, PredictRequest
import asyncio

app = FastAPI()

from typing import Optional
from pydantic import BaseModel, Field

class TestSchema(BaseModel):
    user_id: Optional[int] = Field(None, max_length=None)

    class Config:
        from_attributes = True
        protected_namespaces = ()

# Add root endpoint

@app.get("/")
def read_root():
    return {"message": "Welcome to Matchoracle Prediction App!"}

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

    prediction_type = f"regular_prediction_12_6_4_{request.per_page}"

    # Run train function in the background
    asyncio.create_task(train(user_token, prediction_type, request.model_dump()))

    return {"message": "Training started in the background"}

@app.post("/predict")
async def predict_model(request: PredictRequest):
    user_token = get_user_token_or_404()
    Logger.info("User token obtained successfully.")

    prediction_type = request.prediction_type or f"regular_prediction_12_6_4_1000"
    
    # Run train function in the background
    asyncio.create_task(predict(user_token, prediction_type, request.model_dump()))

    return {"message": "Prediction started in the background"}


@app.get("/metrics")
async def get_metrics():
    user_token = get_user_token_or_404()
    Logger.info("User token obtained successfully.")

    metrics(user_token)
    return {"message": "Metrics generated"}

# Optional: Run using Uvicorn for testing
if __name__ == "__main__":
    pass
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)