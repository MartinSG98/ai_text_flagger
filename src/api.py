from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from src.model import predict
from io import BytesIO
from docx import Document
from pypdf import PdfReader
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please provide a text in order for the prediction to be made.")
    
    try:
        result = predict(request.text)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.post("/predict/file", response_model=PredictResponse)
async def predict_from_file(file: UploadFile = File(...)):
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
