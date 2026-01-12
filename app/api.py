from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf

from src.predict import predict_label

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "weight" / "gambling_classifier_mobilenetv2_gemini_150_ep_Augment.h5"

app = FastAPI(title="DE-Gambling-Detector API")


def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as exc:
        raise RuntimeError(f"Error loading model: {exc}") from exc


MODEL = load_model()


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image uploaded")

    suffix = Path(image.filename).suffix or ".jpg"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await image.read())
            tmp_path = tmp.name
        label, confidence = predict_label(tmp_path, MODEL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"label": label, "confidence": confidence}
