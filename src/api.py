import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from pydantic import BaseModel
from src.model import predict
from io import BytesIO
from docx import Document
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file into the system environment
load_dotenv()

API_KEY = os.getenv("API_KEY")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    ai_probability: float

app = FastAPI()    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key verification dependency
async def verify_api_key(x_api_key: str = Header(...)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured on server")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# Health check endpoint - NO AUTH (needed for monitoring/uptime checks)
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Depends(verify_api_key) runs the verification before this endpoint executes
@app.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest, _: str = Depends(verify_api_key)):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please provide a text in order for the prediction to be made.")
    
    try:
        result = predict(request.text)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Same auth pattern - Depends(verify_api_key) protects this endpoint
@app.post("/predict/file", response_model=PredictResponse)
async def predict_from_file(file: UploadFile = File(...), _: str = Depends(verify_api_key)):
    filename = file.filename.lower()
    if not filename.endswith(('.txt', '.docx', '.pdf')):
        raise HTTPException(status_code=400, detail="Please upload a .txt, .docx, or .pdf file")
    
    try:
        content = await file.read()
        
        if filename.endswith('.txt'):
            text = content.decode('utf-8')
        elif filename.endswith('.docx'):
            doc = Document(BytesIO(content))
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif filename.endswith('.pdf'):
            reader = PdfReader(BytesIO(content))
            text = '\n'.join([page.extract_text() or '' for page in reader.pages])

        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file is empty or cannot be read.")
            
        result = predict(text)
        return result
        
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")