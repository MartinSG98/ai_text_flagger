from fastapi import FastAPI
from pydantic import BaseModel
from model import predict


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

app = FastAPI()    

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest):
    result = predict(request.text)
    return result
