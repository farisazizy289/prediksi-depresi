# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('rf_prediksidepresi.pkl')

# Judul aplikasi
st.title('Prediksi Risiko Depresi Remaja')

st.write('Masukkan data berikut untuk melakukan prediksi risiko depresi.')

# =========================
# INPUT USER
# =========================

age = st.number_input('Usia', min_value=10, max_value=25, value=18)

daily_social_media_hours = st.number_input(
    'Durasi Penggunaan Media Sosial (jam/hari)',
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

sleep_hours = st.number_input(
    'Durasi Tidur (jam/hari)',
    min_value=0.0,
    max_value=12.0,
    value=7.0
)

stress_level = st.slider(
    'Tingkat Stress',
    min_value=1,
    max_value=10,
    value=5
)

anxiety_level = st.slider(
    'Tingkat Anxiety',
    min_value=1,
    max_value=10,
    value=5
)


# =========================
# DATAFRAME INPUT
# =========================

input_data = pd.DataFrame({
    'Age': [age],
    'Daily_Social_Media_Hours': [social_media_hours],
    'Sleep_Hours': [sleep_hours],
    'Stress_Level': [stress_level],
    'Anxiety_Level': [anxiety_level],
})

# =========================
# PREDIKSI
# =========================

if st.button('Prediksi'):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('Terindikasi Berisiko Depresi')
    else:
        st.success('Tidak Terindikasi Depresi')

