from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.inference import predict

class TextIn(BaseModel):
    text: str

class Prediction(BaseModel):
    label: str
    score: float

app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple FastAPI service using Hugging Face DistilBERT for sentiment analysis.",
    version="0.1.0"
)

@app.post("/predict", response_model=Prediction)
async def predict_endpoint(item: TextIn):
    try:
        result = predict(item.text)
        return Prediction(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local development: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
