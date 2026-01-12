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

from src.predict import predict_image_with_preview

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


st.title("DE-Gamling-Detector : AI ตรวจจับรูปเว็บพนัน 🕵️‍♂️")
st.write("อัพโหลดภาพหลายภาพเพื่อทดสอบ และดูสรุปผลรวม")

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

results = []

if uploaded_files:
    st.subheader("📂 ผลลัพธ์รายภาพ")
    # cols = st.columns(2)  # 2 คอลัมน์แสดงภาพ
    
    cols = st.columns(3)  # grid 3 คอลัมน์

    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        ######################### 1 #################################
        # st.image(img_path, caption="รูปที่อัพโหลด", use_container_width=True)
        
        #predict
        label, confidence, img = predict_image_with_preview(img_path, MODEL)  # ✅ รับ 2 ค่า
        # st.subheader(f"ผลลัพธ์: {label} ({confidence:.1%})")
        # st.progress(confidence)
        ######################### 1 #################################

        ##########################
        # เก็บผลลัพธ์
        results.append({"file": uploaded_file.name, "label": label, "confidence": float(confidence)})

        # แสดงภาพ + ผล
        col = cols[i % 3]
        with col:
            st.image(img_path, caption=uploaded_file.name, use_container_width=True)
            st.markdown(f"**ผลลัพธ์:** {label} ({confidence:.1%})")
            st.progress(float(confidence))

    # ===== Summary Dashboard =====
    st.markdown("---")
    st.header("📊 Summary Dashboard")

    total = len(results)
    avg_conf = sum(r["confidence"] for r in results) / total
    gambling_count = sum(1 for r in results if r["label"] == "gambling")
    not_gambling_count = total - gambling_count

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("จำนวนรูป", total)
    m2.metric("ค่าเฉลี่ยความมั่นใจ", f"{avg_conf:.1%}")
    m3.metric("พนัน", gambling_count)
    m4.metric("ไม่ใช่พนัน", not_gambling_count)

    st.dataframe(results)
