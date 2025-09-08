import tensorflow as tf
from pathlib import Path

# Path ของโมเดล .h5
MODEL_H5 = Path("weight/gambling_classifier_mobilenetv2_gemini_150_ep_Augment.h5")
MODEL_KERAS = Path("weight/gambling_classifier_mobilenetv2_gemini_150_ep_Augment.keras")

# โหลดโมเดล H5
model = tf.keras.models.load_model(MODEL_H5)

# เซฟเป็น .keras format (Keras V3)
model.save(MODEL_KERAS, save_format="keras")

print("SavedModel (.keras) created successfully!")