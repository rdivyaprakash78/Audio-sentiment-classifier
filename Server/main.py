from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from prediction import preprocess_audio, get_predictions

app = FastAPI()

UPLOAD_FOLDER = "inputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    waveform, sample_rate = preprocess_audio(file_path)
    probabilities, predicted_class, confidence = get_predictions(waveform)
    
    response = {
        "class" : predicted_class,
        "confidence" : confidence
    }

    return JSONResponse(content=response)
