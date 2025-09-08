# # app/streamlit_app.py
# import streamlit as st
# from src.predict import predict_image
# import tempfile

# st.title("AI ตรวจจับรูปเว็บพนัน 🕵️‍♂️")

# uploaded_file = st.file_uploader("อัพโหลดรูปภาพ", type=["jpg","jpeg","png"])

# if uploaded_file:
#     # save temp file
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(uploaded_file.read())
#         img_path = tmp.name

#     st.image(img_path, caption="รูปที่อัพโหลด", use_column_width=True)

#     if st.button("🔍 ตรวจสอบ"):
#         score = predict_image(img_path)
#         label = "⚠️ พนัน" if score > 0.5 else "✅ ไม่ใช่พนัน"
#         st.subheader(f"ผลลัพธ์: {label}")
#         st.progress(float(score) if score > 0.5 else 1-float(score))


# import streamlit as st

# st.title("AI ตรวจจับรูปเว็บพนัน 🕵️‍♂️")
# st.write("อัพโหลดภาพเพื่อทดสอบการตรวจจับ")

# uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])
# if uploaded_file:
#     st.image(uploaded_file, caption="รูปที่อัพโหลด", use_column_width=True)
#     st.success("ไฟล์ถูกอัพโหลดเรียบร้อยแล้ว!")

# app/streamlit_app.py
import sys
from pathlib import Path
import streamlit as st
import tempfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.predict import predict_image

# path ไปยัง weight และ test images
MODEL_PATH = BASE_DIR / "weight" / "gambling_classifier_mobilenetv2_gemini_150_ep_Augment.h5"
# MODEL_PATH = BASE_DIR / "weight" / "gambling_classifier_mobilenetv2_gemini_150_ep_Augment.keras"
# MODEL_PATH = BASE_DIR / "weight" / "saved_model_gambling"

# ===== LOAD MODEL ONCE =====
@st.cache_resource  # โหลดครั้งเดียว cache ไว้
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

MODEL = load_model()


st.title("AI ตรวจจับรูปเว็บพนัน 🕵️‍♂️")

# uploaded_file = st.file_uploader("อัพโหลดรูปภาพ", type=["jpg","jpeg","png"])

# if uploaded_file:
#     # save temp file
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(uploaded_file.read())
#         img_path = tmp.name

#     st.image(img_path, caption="รูปที่อัพโหลด", use_container_width=True)

#     if st.button("🔍 ตรวจสอบ"):
#         # score = predict_image(img_path, MODEL)
#         label, confidence, img = predict_image(img_path, MODEL)  # ✅ รับ 2 ค่า
#         # label = "⚠️ พนัน" if score > 0.5 else "✅ ไม่ใช่พนัน"
#         st.image(img, caption=f"{label} ({confidence:.1%})", use_container_width=True)
#         st.subheader(f"ผลลัพธ์: {label} : ({confidence})")
#         st.metric("ความมั่นใจ (Confidence)", f"{confidence:.1%}")
#         st.progress(confidence)
#         # st.progress(float(confidence) if confidence > 0.5 else 1-float(confidence))

# multi-file uploader
uploaded_files = st.file_uploader(
    "เลือกรูปภาพ", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        st.image(img_path, caption="รูปที่อัพโหลด", use_container_width=True)
        
        #predict
        label, confidence, img = predict_image(img_path, MODEL)  # ✅ รับ 2 ค่า
            # label = "⚠️ พนัน" if score > 0.5 else "✅ ไม่ใช่พนัน"
        # st.image(img, caption=f"{label} ({confidence:.1%})", use_container_width=True)
        st.subheader(f"ผลลัพธ์: {label} ({confidence:.1%})")
        # st.metric("ความมั่นใจ (Confidence)", f"{confidence:.1%}")
        st.progress(confidence)