import streamlit as st 
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import pdfplumber
import whisper
from paddleocr import PaddleOCR
import openai
from sklearn.base import is_classifier

# --- Page Config ---
st.set_page_config(page_title="Acumen ML Predictor", layout="wide")
st.title("🧠 Acumen: Multimodal ML Dashboard")
st.markdown("Upload a file or enter features to get model predictions.")

# --- Sidebar Upload ---
st.sidebar.title("📌 Multimodal Upload")
uploaded_file = st.sidebar.file_uploader(
    "Drop a PDF, MP3, PNG, or CSV file",
    type=["pdf", "mp3", "png", "csv"],
    accept_multiple_files=False
)

# --- Define Features (Assuming 4 features for the model) ---
feature_names = ["feature_0", "feature_1", "feature_2", "feature_3"]
inputs = []

# --- Load Model from MLflow Registry ---
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    return mlflow.sklearn.load_model("models:/AcumenModel/Production")

model = load_model()
is_classification = is_classifier(model)

# --- Summarization Utilities ---
def summarize_text_openai(text, model="gpt-3.5-turbo", max_tokens=1000):
    prompt = f"Summarize the following:\n\n{text[:3000]}"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

# --- PDF / MP3 / PNG / CSV Handling ---
if uploaded_file is not None:
    file_suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    if file_suffix == ".pdf":
        st.sidebar.success("📄 PDF uploaded successfully!")
        with pdfplumber.open(tmp_path) as pdf:
            all_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        st.subheader("📄 Extracted Text from PDF")
        st.text_area("PDF Content:", all_text, height=250)
        summary = summarize_text_openai(all_text)
        st.subheader("🖍️ Summary")
        st.text_area("Summary:", summary, height=200)

    elif file_suffix == ".mp3":
        st.sidebar.success("🎷 MP3 uploaded — transcribing...")
        model_audio = whisper.load_model("base")
        result = model_audio.transcribe(tmp_path)
        st.subheader("🎷 Transcribed Audio")
        st.text_area("Transcription:", result["text"], height=250)
        summary = summarize_text_openai(result["text"])
        st.subheader("🖍️ Summary")
        st.text_area("Summary:", summary, height=200)

    elif file_suffix == ".png":
        st.sidebar.success("🖼️ PNG uploaded — extracting text...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(tmp_path, cls=True)
        extracted_text = "\n".join([line[1][0] for line in result[0]])
        st.subheader("🖼️ Extracted Text from Image")
        st.text_area("OCR Output:", extracted_text, height=250)
        summary = summarize_text_openai(extracted_text)
        st.subheader("🖍️ Summary")
        st.text_area("Summary:", summary, height=200)

    elif file_suffix == ".csv":
        st.sidebar.success("📊 CSV uploaded — predicting batch...")
        df = pd.read_csv(tmp_path)
        predictions = model.predict(df)
        st.subheader("📊 Batch Predictions")
        df["prediction"] = predictions
        st.dataframe(df)
        st.download_button("📅 Download Results", df.to_csv(index=False), "predictions.csv", "text/csv")

# --- Manual Feature Input ---
st.subheader("🛠️ Enter Features Manually")
cols = st.columns(len(feature_names))
for i, feature in enumerate(feature_names):
    val = cols[i].slider(f"{feature}", 0.0, 10.0, 5.0, step=0.1)
    inputs.append(val)

input_df = pd.DataFrame([inputs], columns=feature_names)

# --- Predict Button ---
if st.button("🚀 Predict"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: **{prediction[0]}**")
    st.dataframe(input_df.style.highlight_max(axis=1))

    if is_classification:
        proba = model.predict_proba(input_df)
        st.markdown("### 🔮 Prediction Probabilities")
        prob_df = pd.DataFrame(proba, columns=[f"Class {i}" for i in range(proba.shape[1])])
        st.bar_chart(prob_df.T)

    # --- SHAP Explainability ---
    st.markdown("### 🔍 SHAP Feature Importance")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    try:
        shap.initjs()
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ SHAP couldn't generate plot: {e}")
