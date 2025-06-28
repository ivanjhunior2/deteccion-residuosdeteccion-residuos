import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import os
import csv
import pandas as pd
import altair as alt

# Configuración de página
st.set_page_config(page_title="Detector de Residuos", layout="wide")

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
        
        .title {
            text-align: center;
            color: #2e7d32;
        }
        .subtitle {
            text-align: center;
            color: #555;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            margin: 10px 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO ---
st.markdown("<h1 class='title'>♻️ Detector de Residuos</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>Equipo: Ivan Aiza, Alejandra Villarroel, Fernando Quinteros, Alan</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- CARGAR MODELO ---
@st.cache_resource
def load_model():
    return YOLO("modelo.pt")

model = load_model()

# --- CREAR ARCHIVOS NECESARIOS ---
os.makedirs("capturas", exist_ok=True)
csv_file = "detecciones.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["nombre_imagen", "clase", "confianza", "timestamp"])

# --- CONTROLES ---
col1, col2, col3 = st.columns(3)
start = col1.button("▶️ Iniciar cámara")
stop = col2.button("⛔ Detener cámara")
capture = col3.button("📸 Capturar")

FRAME_WINDOW = st.image([])

# --- ESTADO ---
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start:
    st.session_state.camera_active = True
if stop:
    st.session_state.camera_active = False

# --- FLUJO DE DETECCIÓN ---
if st.session_state.camera_active:
    cap = cv2.VideoCapture(0)
    st.info("Cámara activa. Presiona '⛔ Detener cámara' para finalizar.")
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara.")
            break

        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        detecciones_actuales = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"
                detecciones_actuales.append((model.names[cls], conf))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Captura
        if capture:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.jpg"
            path = os.path.join("capturas", filename)
            cv2.imwrite(path, annotated_frame)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for clase, conf in detecciones_actuales:
                    writer.writerow([filename, clase, f"{conf:.2f}", timestamp])

            st.success(f"📸 Imagen guardada como `{filename}` y datos en `detecciones.csv`")
            st.session_state.camera_active = False
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    st.warning("Cámara inactiva. Haz clic en '▶️ Iniciar cámara' para comenzar.")

# --- VISUALIZAR DATOS DETECTADOS ---
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    if not df.empty:
        st.markdown("### 📊 Resumen de Detecciones")
        resumen = df['clase'].value_counts().reset_index()
        resumen.columns = ['Clase', 'Cantidad']

        chart = alt.Chart(resumen).mark_bar().encode(
            x='Clase',
            y='Cantidad',
            color='Clase'
        ).properties(width=600)

        st.altair_chart(chart, use_container_width=True)
