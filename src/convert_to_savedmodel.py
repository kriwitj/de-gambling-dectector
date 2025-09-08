import tensorflow as tf
from pathlib import Path

# Path ของโมเดล .h5
MODEL_H5 = Path("weight/gambling_classifier_mobilenetv2_gemini_150_ep_Augment.h5")

# Path สำหรับบันทึก SavedModel
SAVED_MODEL_DIR = Path("weight/saved_model_gambling")

# โหลดโมเดล
print(f"Loading .h5 model from {MODEL_H5} ...")
model = tf.keras.models.load_model(MODEL_H5)
print("Model loaded successfully!")

# Save เป็น SavedModel format
print(f"Saving as SavedModel to {SAVED_MODEL_DIR} ...")
model.save(SAVED_MODEL_DIR, save_format="tf")
print("SavedModel created successfully!")
