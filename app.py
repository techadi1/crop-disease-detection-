# app.py — Smart Apple Leaf Disease Detector 🍎

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import tempfile
import os

# ==============================
# PAGE CONFIG & STYLING
# ==============================
st.set_page_config(
    page_title="🍎 Apple Leaf Disease Detector",
    page_icon="🍏",
    layout="centered",
)

st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main-title {
            text-align: center;
            font-size: 40px !important;
            color: #1e5631;
            font-weight: bold;
        }
        .sub {
            text-align: center;
            color: #555;
            font-size: 18px;
        }
        .footer {
            text-align:center;
            margin-top: 60px;
            color: #888;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🍎 Apple Leaf Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Upload an Apple leaf image to get instant AI-based diagnosis.</p>", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
MODEL_PATH = "apple_leaf_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Full PlantVillage 38-class list
class_names_38 = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy",
]

apple_classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

# ==============================
# FILE UPLOAD
# ==============================
uploaded = st.file_uploader("📂 Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((128, 128))
    arr = tf.keras.utils.img_to_array(img_resized)
    arr = arr / 255.0   # ✅ normalization fix
    arr = np.expand_dims(arr, 0)

    # Prediction
    preds = model.predict(arr, verbose=0)[0]
    num_classes = preds.shape[0]

    if num_classes == 4:
        class_names = apple_classes
    elif num_classes == 38:
        class_names = class_names_38
    else:
        st.error(f"Unexpected number of classes: {num_classes}")
        st.stop()

    max_prob = float(np.max(preds))
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    entropy = -float(np.sum(preds * np.log(preds + 1e-12)))

    # Relaxed thresholds
    MAX_PROB_THRESHOLD = 0.50
    ENTROPY_THRESHOLD = 1.5

    st.markdown("---")

    # ==============================
    # CONFIDENCE CHECK
    # ==============================
    if (max_prob < MAX_PROB_THRESHOLD) or (entropy > ENTROPY_THRESHOLD):
        st.warning("⚠️ Low confidence prediction (image may not match training data)")

    # ==============================
    # ALWAYS SHOW OUTPUT
    # ==============================
    st.subheader("🔍 Prediction Result:")
    st.success(f"**{pred_class}**  \nConfidence: `{max_prob*100:.2f}%`")

    # ==============================
    # CHART
    # ==============================
    st.markdown("### 📊 Class Probabilities")

    chart_data = {}
    for app_class in apple_classes:
        if app_class in class_names:
            idx = class_names.index(app_class)
            value = float(preds[idx])
        else:
            value = 0.0

        pretty = app_class.replace("Apple___", "").replace("_", " ")
        chart_data[pretty] = value

    st.bar_chart(chart_data, use_container_width=True)

    # ==============================
    # PDF
    # ==============================
    def generate_pdf(image, pred_class, conf):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 20)
        pdf.cell(0, 10, "Apple Leaf Disease Detection Report", ln=True, align='C')

        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 10, f"Predicted Disease: {pred_class}", ln=True)
        pdf.cell(0, 10, f"Confidence: {conf:.2f}%", ln=True)
        pdf.ln(10)

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_img.name)
        pdf.image(temp_img.name, x=30, y=None, w=150)
        temp_img.close()

        pdf_path = os.path.join(tempfile.gettempdir(), "apple_leaf_report.pdf")
        pdf.output(pdf_path)
        return pdf_path

    if st.button("📄 Download PDF Report"):
        pdf_path = generate_pdf(img, pred_class, max_prob*100)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Click to Download Report",
                data=f,
                file_name="AppleLeaf_Report.pdf",
                mime="application/pdf"
            )

    st.markdown("---")
    st.markdown(
        "<div class='footer'>Model trained on PlantVillage Apple Leaf dataset • Smart version 🍏</div>",
        unsafe_allow_html=True,
    )

else:
    st.info("👆 Upload a leaf image to begin prediction.")