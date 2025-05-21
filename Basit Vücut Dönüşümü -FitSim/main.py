import streamlit as st
from PIL import Image
import os
import numpy as np

from app.segmentor import load_model, segment
from app.deformator import tps_deform

st.set_page_config(page_title="FitSim - Vücut Dönüşüm", layout="centered")
st.title("FitSim - Zayıf/Kilolu Görsel Dönüştürücü")

uploaded_file = st.file_uploader("Bir görsel seçin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    with st.spinner("Segmentasyon modeli uygulanıyor..."):
        model = load_model()
        mask = segment(image, model)

    st.image(mask, caption="Segmentasyon Maskesi", use_column_width=True)

    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(mask).save("outputs/mask.png")

    col1, col2 = st.columns(2)

    if col1.button("Zayıf Göster"):
        with st.spinner("Zayıf versiyon hazırlanıyor..."):
            result = tps_deform(image, Image.fromarray(mask), mode="slim")
            st.image(result, caption="Zayıflamış Versiyon", use_column_width=True)
            Image.fromarray(result).save("outputs/slimmed.png")

    if col2.button("Kilolu Göster"):
        with st.spinner("Kilolu versiyon hazırlanıyor..."):
            result = tps_deform(image, Image.fromarray(mask), mode="fat")
            st.image(result, caption="Kilo Almış Versiyon", use_column_width=True)
            Image.fromarray(result).save("outputs/fattened.png")
