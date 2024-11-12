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

# Crear entradas para cada una de las 30 características
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

radius_se = st.slider("Radio SE", 0.1, 3.0, 0.5)
texture_se = st.slider("Textura SE", 0.5, 5.0, 1.0)
perimeter_se = st.slider("Perímetro SE", 0.5, 25.0, 5.0)
area_se = st.slider("Área SE", 6.0, 550.0, 50.0)
smoothness_se = st.slider("Suavidad SE", 0.001, 0.03, 0.01)
compactness_se = st.slider("Compacidad SE", 0.002, 0.15, 0.02)
concavity_se = st.slider("Concavidad SE", 0.0, 0.4, 0.02)
concave_points_se = st.slider("Puntos cóncavos SE", 0.0, 0.05, 0.01)
symmetry_se = st.slider("Simetría SE", 0.007, 0.08, 0.02)
fractal_dimension_se = st.slider("Dimensión fractal SE", 0.0005, 0.03, 0.01)

radius_worst = st.slider("Radio peor valor", 7.0, 40.0, 16.0)
texture_worst = st.slider("Textura peor valor", 12.0, 50.0, 25.0)
perimeter_worst = st.slider("Perímetro peor valor", 50.0, 300.0, 100.0)
area_worst = st.slider("Área peor valor", 150.0, 4000.0, 880.0)
smoothness_worst = st.slider("Suavidad peor valor", 0.05, 0.25, 0.15)
compactness_worst = st.slider("Compacidad peor valor", 0.025, 1.2, 0.25)
concavity_worst = st.slider("Concavidad peor valor", 0.0, 1.5, 0.27)
concave_points_worst = st.slider("Puntos cóncavos peor valor", 0.0, 0.3, 0.1)
symmetry_worst = st.slider("Simetría peor valor", 0.1, 0.6, 0.3)
fractal_dimension_worst = st.slider("Dimensión fractal peor valor", 0.05, 0.4, 0.1)

# Organizar los datos de entrada en el orden correcto
input_data = [
    radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
    concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
    perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
    concave_points_worst, symmetry_worst, fractal_dimension_worst
]

# Botón para hacer la predicción
if st.button("Predecir"):
    result = make_prediction(input_data)
    st.success(f'El diagnóstico es: {result}')
