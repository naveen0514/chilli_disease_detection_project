import io
import json
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

MODEL_PATH = "models/chilli_disease.keras"
CLASS_PATH = "models/class_names.json"

app = FastAPI(title="Chilli Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        class_names = json.load(f)

@app.get("/")
def root():
    return {"message": "Chilli disease detection API"}

@app.get("/health")
def health():
    return {"model_loaded": model is not None, "classes": class_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, class_names

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train the model first with: python ml/train.py"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")

    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    probabilities = model.predict(arr, verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    return {
        "disease": class_names[index],
        "confidence": round(confidence * 100, 2),
        "message": f"Predicted class: {class_names[index]}"
    }
