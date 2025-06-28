import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Detector de Residuos", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>♻️ Detección de Residuos en Tiempo Real</h1>
    <h4 style='text-align: center;'>Proyecto del equipo: Ivan Aiza, Alan Caceres, Alejandra Villarroel, Fernando Quinteros</h4>
    <hr>
""", unsafe_allow_html=True)

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO("modelo.pt")

model = load_model()

# Interfaz de control
col1, col2 = st.columns(2)
start = col1.button("▶️ Iniciar cámara")
stop = col2.button("⛔ Detener cámara")

capture = st.button("📸 Capturar imagen")
FRAME_WINDOW = st.image([])

# Crear carpeta para capturas
CAPTURE_DIR = "capturas"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Inicializar estado de la cámara
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if start:
    st.session_state.camera_active = True

if stop:
    st.session_state.camera_active = False

# Lista para historial
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

# Proceso de cámara
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

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar en Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        # Guardar captura si se presiona botón
        if capture:
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            path = os.path.join(CAPTURE_DIR, filename)
            cv2.imwrite(path, annotated_frame)
            st.success(f"📸 Captura guardada como `{filename}`")

            # Agregar al historial
            st.session_state.captured_images.append(path)

            # Detener cámara
            st.session_state.camera_active = False
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    st.warning("Cámara detenida. Haz clic en '▶️ Iniciar cámara' para comenzar.")

# Mostrar historial de capturas
if st.session_state.captured_images:
    st.markdown("## 🖼️ Historial de capturas")
    cols = st.columns(4)
    for i, img_path in enumerate(reversed(st.session_state.captured_images)):
        with cols[i % 4]:
            st.image(Image.open(img_path), caption=os.path.basename(img_path), use_container_width=True)

