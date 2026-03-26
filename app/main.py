from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.predictor import predict_quality

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API for predicting wine quality using ML model",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "Wine Quality Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = predict_quality(request)
    return result