from fastapi import FastAPI
from pydantic import BaseModel
from model import predict


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

app = FastAPI()    