import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import os
import csv
import pandas as pd
import altair as alt

# Configuración de página
st.set_page_config(page_title="Detector de Residuos en Vivo", layout="wide")

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

# --- Título ---
st.markdown("<h1 class='title'>♻️ Detector de Residuos</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>Equipo: Ivan Aiza, Alan Caceres, Alejandra Villarroel, Fernando Quinteros</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Cargar modelo (cacheado para eficiencia) ---
@st.cache_resource
def load_model():
    return YOLO("modelo.pt")  

model = load_model()

# Crear carpeta y CSV para guardado
os.makedirs("capturas", exist_ok=True)
csv_file = "detecciones.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["nombre_imagen", "clase", "confianza", "timestamp"])

# Estado para guardar capturas
if "capture_now" not in st.session_state:
    st.session_state.capture_now = False

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

if "last_detections" not in st.session_state:
    st.session_state.last_detections = []

# Botón para capturar frame
if st.button("📸 Capturar frame actual"):
    st.session_state.capture_now = True

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Detección con YOLO
    results = model(img, verbose=False)
    annotated = img.copy()
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"

            detections.append((model.names[cls], conf))

            # Dibujar caja y etiqueta
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Guardar último frame y detecciones en el estado de Streamlit
    st.session_state.last_frame = annotated
    st.session_state.last_detections = detections

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Mostrar el stream de video con callback
webrtc_ctx = webrtc_streamer(key="yolo_detector", video_frame_callback=video_frame_callback)

# Guardar imagen y datos si se solicitó captura
if st.session_state.capture_now and st.session_state.last_frame is not None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    path = os.path.join("capturas", filename)
    cv2.imwrite(path, st.session_state.last_frame)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for clase, conf in st.session_state.last_detections:
            writer.writerow([filename, clase, f"{conf:.2f}", timestamp])

    st.success(f"✅ Imagen guardada como `{filename}` y datos agregados a `detecciones.csv`")

    # Reset flag
    st.session_state.capture_now = False

# Mostrar resumen con Altair si hay datos
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
