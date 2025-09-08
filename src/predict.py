
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "weight" / "gambling_classifier_mobilenetv2_gemini_150_ep_Augment.h5"

IMG_HEIGHT = 224
IMG_WIDTH = 224
# The class names must be in the same order as they were during training.
# Check the output of the training script to be sure.
# Usually, it's alphabetical: ['gambling', 'not_gambling']
CLASS_NAMES = ['gambling', 'not_gambling']

# --- 2. LOAD THE SAVED MODEL ---
# print(f"Loading model from {MODEL_PATH}...")
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# print("Model loaded successfully.")

# --- 3. CREATE A PREDICTION FUNCTION ---
# This function prepares a single image and makes a prediction.
def predict_image(image_path, model):
    """Loads an image, preprocesses it, and returns the prediction."""
    print(f"\nPredicting image: {image_path}")
    
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    
    # Convert the image to a NumPy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # The model expects a "batch" of images, so we add an extra dimension
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make the prediction
    predictions = model.predict(img_array)
    score = float(predictions[0][0]) # Get the single prediction value from the batch

    # Interpret the prediction
    # The sigmoid function outputs a value between 0 and 1.
    # We'll use 0.5 as the threshold to decide the class.
    # IMPORTANT: The index (0 or 1) depends on the class order.
    # If CLASS_NAMES = ['gambling', 'not_gambling']:
    #   - A low score (close to 0) means it's the first class ('gambling').
    #   - A high score (close to 1) means it's the second class ('not_gambling').
    
    confidence = 1 - score if score < 0.5 else score
    predicted_class = CLASS_NAMES[0] if score < 0.5 else CLASS_NAMES[1]
    # Show image with label
    # plt.imshow(img)  # use PIL Image (already resized)
    # plt.axis("off")
    # plt.title(f"{predicted_class} ({confidence:.1%})", fontsize=12, color="blue")
    # plt.show()
    
    return predicted_class, confidence, img
