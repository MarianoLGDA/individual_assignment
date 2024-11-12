import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo y scaler desde los archivos .pkl
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Función de predicción
def prediction(input_data):
    # Convertimos la predicción en términos de diagnóstico
    pred = model.predict(input_data)
    return ["Benigno" if p == 0 else "Maligno" for p in pred]

# Diseño principal de la app
def main():
    # Estilo CSS para fondo verde y texto blanco
    st.markdown(
        """
        <style>
        .main {
            background-color: #e6ffe6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Cabecera de la app con fondo verde y texto en blanco
    html_temp = """ 
    <div style="background-color:#66cc66;padding:15px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Cáncer de Mama: Predicción con KNN</h1> 
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Descripción de la app
    st.write("Sube tus datos de análisis de células y obtén una predicción sobre si el diagnóstico es benigno o maligno.")

    # Cargar modelo y scaler
    model, scaler = load_model_and_scaler()

    # Cargar archivo de datos
    uploaded_file = st.file_uploader("Sube un archivo CSV con las características de células", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        input_data = input_data.drop(columns=['id', 'diagnosis'], errors='ignore')

        # Escalado de datos
        input_data_scaled = scaler.transform(input_data)

        # Realizar predicción
        predictions = prediction(input_data_scaled)

        # Mostrar resultados
        st.write("Predicciones de diagnóstico:")
        st.write(predictions)

# Ejecutar la aplicación
if __name__ == '__main__':
    main()
