import streamlit as st
import numpy as np
import joblib

# Cargar el modelo y el escalador
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Función para hacer la predicción
def make_prediction(input_data):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    return "Maligno" if prediction[0] == 1 else "Benigno"

# Título y descripción de la aplicación
st.title("Predicción de Cáncer de Mama")
st.write("Introduce las características de la célula para predecir si el diagnóstico es benigno o maligno.")

# Crear entradas para cada característica relevante
radius_mean = st.slider("Radio promedio", 6.0, 30.0, 14.0)
texture_mean = st.slider("Textura promedio", 10.0, 40.0, 20.0)
perimeter_mean = st.slider("Perímetro promedio", 40.0, 190.0, 80.0)
area_mean = st.slider("Área promedio", 100.0, 2500.0, 500.0)
smoothness_mean = st.slider("Suavidad promedio", 0.05, 0.2, 0.1)
compactness_mean = st.slider("Compacidad promedio", 0.02, 0.35, 0.1)
concavity_mean = st.slider("Concavidad promedio", 0.0, 0.5, 0.1)
concave_points_mean = st.slider("Puntos cóncavos promedio", 0.0, 0.2, 0.05)
symmetry_mean = st.slider("Simetría promedio", 0.1, 0.3, 0.2)
fractal_dimension_mean = st.slider("Dimensión fractal promedio", 0.05, 0.15, 0.08)

# Organizar los datos de entrada
input_data = [
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
]

# Botón para hacer la predicción
if st.button("Predecir"):
    result = make_prediction(input_data)
    st.success(f'El diagnóstico es: {result}')
