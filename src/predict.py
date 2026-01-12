from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "weight" / "gambling_classifier_mobilenetv2_gemini_150_ep_Augment.h5"

IMG_HEIGHT = 224
IMG_WIDTH = 224
# The class names must be in the same order as they were during training.
# Check the output of the training script to be sure.
# Usually, it's alphabetical: ['gambling', 'not_gambling']
CLASS_NAMES = ["gambling", "not_gambling"]


def preprocess_image(image_source):
    """Loads an image and returns the PIL image and prepared batch array."""
    img = tf.keras.utils.load_img(image_source, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img, img_array


def _interpret_score(score):
    confidence = 1 - score if score < 0.5 else score
    predicted_class = CLASS_NAMES[0] if score < 0.5 else CLASS_NAMES[1]
    return predicted_class, confidence


def predict_label(image_source, model):
    """Returns predicted label and confidence for the provided image."""
    _, img_array = preprocess_image(image_source)
    predictions = model.predict(img_array)
    score = float(predictions[0][0])
    return _interpret_score(score)


def predict_image(image_source, model):
    """Loads an image, preprocesses it, and returns the prediction and preview image."""
    img, img_array = preprocess_image(image_source)
    predictions = model.predict(img_array)
    score = float(predictions[0][0])
    predicted_class, confidence = _interpret_score(score)
    return predicted_class, confidence, img


def predict_image_with_preview(image_source, model):
    """Backward-compatible wrapper for streamlit app usage."""
    return predict_image(image_source, model)
