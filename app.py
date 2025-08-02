
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from imblearn.pipeline import Pipeline as ImbPipeline
import numpy as np
from PIL import Image

@st.cache_resource
def load_components():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("selector.pkl", "rb") as f:
        selector = pickle.load(f)
    with open("model_pipeline.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
    return vectorizer, selector, model_pipeline

vectorizer, selector, model_pipeline = load_components()
label_map = {1: "Positif", 0: "Negatif"}

def predict_sentiment(texts):
    results = []
    for text in texts:
        if not text.strip():
            results.append(None)
            continue
        vect = vectorizer.transform([text])
        selected = selector.transform(vect)
        pred = model_pipeline.predict(selected)[0]
        prob = model_pipeline.predict_proba(selected)[0]
        label = pred.capitalize()
        pos = prob[1] * 100
        neg = prob[0] * 100
        results.append({
            "label": label,
            "pos": pos,
            "neg": neg,
            "text": text
        })
    return results

image_path = "/content/drive/MyDrive/data skripsi/panduan-daftar-aplikasi-signal-samsat-2025-cara-mudah-bayar-pajak-stnk-secara-online-tanpa-ribet.jpg"
image = Image.open(image_path)
st.image(image, width=200)

st.title("📊 Analisis Sentimen Review Aplikasi SIGNAL ")
st.markdown("Masukkan hingga 5 teks. Sistem akan memprediksi apakah sentimennya **positif** atau **negatif**, beserta persentasenya.")

text_inputs = []
for i in range(5):
    default = "Contoh: film ini sangat bagus dan menyentuh hati" if i == 0 else ""
    text = st.text_input(f"Teks {i+1}", default)
    text_inputs.append(text)

if st.button("🔍 Prediksi"):
    outputs = predict_sentiment(text_inputs)
    for i, res in enumerate(outputs):
        if res is None:
            st.warning(f"Teks {i+1} kosong, silakan isi.")
            continue
        color = "green" if res["label"] == "Positif" else "red"
        st.markdown(f"**Teks {i+1}:** {res['text']}")
        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>➡️ Prediksi: {res['label']}</span><br>"
            f"👍 Positif: {res['pos']:.2f}%<br>"
            f"👎 Negatif: {res['neg']:.2f}%",
            unsafe_allow_html=True
        )
        st.markdown("---")
