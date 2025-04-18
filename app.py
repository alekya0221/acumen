import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os
import shap
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import pdfplumber
import whisper
from paddleocr import PaddleOCR

# --- Page Config ---
st.set_page_config(page_title="Acumen ML Predictor", layout="wide")
st.sidebar.title("📎 Multimodal Upload")

# --- Multimodal Upload Sidebar ---
uploaded_file = st.sidebar.file_uploader(
    "Drop a PDF, MP3, or PNG file here",
    type=["pdf", "mp3", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    file_suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if file_suffix == ".pdf":
        st.sidebar.success("📄 PDF uploaded successfully!")
        with pdfplumber.open(tmp_path) as pdf:
            all_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        st.markdown("### 📄 Extracted PDF Text")
        st.text_area("Text from PDF:", all_text, height=300)

    elif file_suffix == ".mp3":
        st.sidebar.success("🎧 MP3 uploaded — transcribing...")
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        st.markdown("### 🎧 Transcribed Audio")
        st.text_area("Transcription:", result["text"], height=300)

    elif file_suffix == ".png":
        st.sidebar.success("🖼 PNG uploaded — running OCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(tmp_path, cls=True)

        ocr_text = ""
        for line in result[0]:
            ocr_text += f"{line[1][0]}\n"

        st.markdown("### 🖼 OCR Text from Image")
        st.text_area("OCR Result:", ocr_text, height=300)

    else:
        st.sidebar.warning("⚠️ Unsupported file type.")

# --- Load Model ---
st.title("🧠 Acumen: Model Inference Dashboard")
feature_names = ["feature_0", "feature_1", "feature_2", "feature_3"]

@st.cache_resource
def load_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    return mlflow.sklearn.load_model("models:/AcumenModel/Production")

model = load_model()

# --- Single Prediction ---
st.markdown("### 🎛️ Enter Features for Prediction")
inputs = [st.slider(f, 0.0, 10.0, 5.0, 0.1) for f in feature_names]
input_array = np.array(inputs).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=feature_names)

if st.button("🚀 Predict"):
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)

    st.success(f"✅ Predicted Class: **{prediction[0]}**")
    st.dataframe(input_df.style.highlight_max(axis=1))

    st.markdown("### 🔢 Class Probabilities")
    prob_df = pd.DataFrame(probabilities, columns=[f"Class {i}" for i in range(probabilities.shape[1])])
    st.bar_chart(prob_df.T)

    st.markdown("### 🧠 SHAP Feature Importance")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots()

    if isinstance(shap_values, list) and len(shap_values) > 1:
        predicted_class = int(prediction[0])
        st.markdown(f"🔎 SHAP for class: {predicted_class}")
        shap.summary_plot(shap_values[predicted_class], input_df, plot_type="bar", show=False)
    else:
        st.markdown("🔎 SHAP (binary classifier)")
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)

    st.pyplot(fig)

# --- Batch CSV Prediction ---
st.markdown("## 📎 Batch Prediction from CSV")
uploaded_csv = st.file_uploader("Upload CSV file with the same feature columns", type=["csv"])

if uploaded_csv is not None:
    try:
        batch_df = pd.read_csv(uploaded_csv)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(batch_df.head())

        missing_cols = [col for col in feature_names if col not in batch_df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
        else:
            if st.button("📊 Predict for Batch"):
                batch_predictions = model.predict(batch_df)
                batch_probabilities = model.predict_proba(batch_df)

                result_df = batch_df.copy()
                result_df["Predicted Class"] = batch_predictions

                for i in range(batch_probabilities.shape[1]):
                    result_df[f"Class_{i}_Prob"] = batch_probabilities[:, i]

                st.success("✅ Batch prediction completed!")
                st.dataframe(result_df.head())

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name='acumen_predictions.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

