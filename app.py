# app.py

import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load('rf_prediksidepresi.pkl')

# =========================
# TITLE
# =========================

st.title('Prediksi Risiko Depresi Remaja')

st.write(
    'Aplikasi ini digunakan untuk memprediksi risiko depresi '
    'berdasarkan aktivitas media sosial dan gaya hidup remaja.'
)

# =========================
# INPUT USER
# =========================

age = st.number_input(
    'Usia',
    min_value=10,
    max_value=25,
    value=18
)

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

screen_time_before_sleep = st.number_input(
    'Screen Time Sebelum Tidur (jam)',
    min_value=0.0,
    max_value=10.0,
    value=2.0
)

academic_performance = st.slider(
    'Performa Akademik',
    min_value=1,
    max_value=10,
    value=5
)

physical_activity = st.number_input(
    'Aktivitas Fisik (jam/minggu)',
    min_value=0.0,
    max_value=20.0,
    value=5.0
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

addiction_level = st.slider(
    'Tingkat Kecanduan Media Sosial',
    min_value=1,
    max_value=10,
    value=5
)

gender = st.selectbox(
    'Gender',
    ['Male', 'Female']
)

platform = st.selectbox(
    'Platform yang Paling Sering Digunakan',
    ['Instagram', 'TikTok']
)

social_interaction = st.selectbox(
    'Tingkat Interaksi Sosial',
    ['Low', 'Medium']
)

# =========================
# DATAFRAME INPUT
# =========================

input_data = pd.DataFrame({
    'age': [age],
    'daily_social_media_hours': [daily_social_media_hours],
    'sleep_hours': [sleep_hours],
    'screen_time_before_sleep': [screen_time_before_sleep],
    'academic_performance': [academic_performance],
    'physical_activity': [physical_activity],
    'stress_level': [stress_level],
    'anxiety_level': [anxiety_level],
    'addiction_level': [addiction_level],

    'gender_male': [
        1 if gender == 'Male' else 0
    ],

    'platform_usage_Instagram': [
        1 if platform == 'Instagram' else 0
    ],

    'platform_usage_TikTok': [
        1 if platform == 'TikTok' else 0
    ],

    'social_interaction_level_low': [
        1 if social_interaction == 'Low' else 0
    ],

    'social_interaction_level_medium': [
        1 if social_interaction == 'Medium' else 0
    ]
})

# =========================
# PREDIKSI
# =========================

if st.button('Prediksi'):

    prediction = model.predict(input_data)

    prediction_proba = model.predict_proba(input_data)

    probability = prediction_proba[0][1]

    st.subheader('Hasil Prediksi')

    if prediction[0] == 1:
        st.error('Terindikasi Berisiko Depresi')
    else:
        st.success('Tidak Terindikasi Depresi')

    st.write(f'Probabilitas Risiko Depresi: {probability:.2%}')
