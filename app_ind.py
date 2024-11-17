import streamlit as st
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Prediction function
def make_prediction(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return "Malignant" if prediction[0] == 1 else "Benign"

# App title and subtitle
st.title("🏥 IE Cancer Center 🏥")
st.write("### Mariano Lara García de Alba - Machine Learning: Individual Assignment")

st.write("Enter the cell characteristics below to predict whether the diagnosis is benign or malignant.")

# Input sliders for all 30 features
radius_mean = st.slider("Radius Mean", 6.0, 30.0, 14.0)
texture_mean = st.slider("Texture Mean", 10.0, 40.0, 20.0)
perimeter_mean = st.slider("Perimeter Mean", 40.0, 190.0, 80.0)
area_mean = st.slider("Area Mean", 100.0, 2500.0, 500.0)
smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.2, 0.1)
compactness_mean = st.slider("Compactness Mean", 0.02, 0.35, 0.1)
concavity_mean = st.slider("Concavity Mean", 0.0, 0.5, 0.1)
concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.2, 0.05)
symmetry_mean = st.slider("Symmetry Mean", 0.1, 0.3, 0.2)
fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.05, 0.15, 0.08)

radius_se = st.slider("Radius SE", 0.1, 3.0, 0.5)
texture_se = st.slider("Texture SE", 0.5, 5.0, 1.0)
perimeter_se = st.slider("Perimeter SE", 0.5, 25.0, 5.0)
area_se = st.slider("Area SE", 6.0, 550.0, 50.0)
smoothness_se = st.slider("Smoothness SE", 0.001, 0.03, 0.01)
compactness_se = st.slider("Compactness SE", 0.002, 0.15, 0.02)
concavity_se = st.slider("Concavity SE", 0.0, 0.4, 0.02)
concave_points_se = st.slider("Concave Points SE", 0.0, 0.05, 0.01)
symmetry_se = st.slider("Symmetry SE", 0.007, 0.08, 0.02)
fractal_dimension_se = st.slider("Fractal Dimension SE", 0.0005, 0.03, 0.01)

radius_worst = st.slider("Radius Worst", 7.0, 40.0, 16.0)
texture_worst = st.slider("Texture Worst", 12.0, 50.0, 25.0)
perimeter_worst = st.slider("Perimeter Worst", 50.0, 300.0, 100.0)
area_worst = st.slider("Area Worst", 150.0, 4000.0, 880.0)
smoothness_worst = st.slider("Smoothness Worst", 0.05, 0.25, 0.15)
compactness_worst = st.slider("Compactness Worst", 0.025, 1.2, 0.25)
concavity_worst = st.slider("Concavity Worst", 0.0, 1.5, 0.27)
concave_points_worst = st.slider("Concave Points Worst", 0.0, 0.3, 0.1)
symmetry_worst = st.slider("Symmetry Worst", 0.1, 0.6, 0.3)
fractal_dimension_worst = st.slider("Fractal Dimension Worst", 0.05, 0.4, 0.1)

# Collect input data in the correct order
input_data = [
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
    concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
    perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
    concave_points_worst, symmetry_worst, fractal_dimension_worst
]

# Predict button and display result
if st.button("Predict"):
    result = make_prediction(input_data)
    if result == "Benign":
        st.markdown("<h3 style='color: green;'>The diagnosis is: Benign ✅😊</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: red;'>The diagnosis is: Malignant ⚠️😞</h3>", unsafe_allow_html=True)
