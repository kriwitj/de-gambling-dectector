from io import BytesIO

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.predict import MODEL_PATH, predict_image

app = FastAPI()


def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        raise RuntimeError(f"Error loading model from {MODEL_PATH}") from exc


MODEL = load_model()


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload.")

    image_stream = BytesIO(image_bytes)
    try:
        label, confidence, _ = predict_image(image_stream, MODEL)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {exc}") from exc

    return {"label": label, "confidence": float(confidence)}
