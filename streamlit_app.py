import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocess import preprocess_image
from config import EMBEDDINGS_PATH, LABELS_PATH, KNN_MODEL_PATH

# -----------------------------------------
# Configuración visual
# -----------------------------------------
st.set_page_config(
    page_title="Clasificación de Regiones",
    page_icon="🌎",
    layout="centered"
)

# CSS personalizado (premium)
st.markdown("""
<style>
    body {
        background-color: #0E1117;
    }
    .title {
        color: #4CAF50;
        text-align: center;
        font-size: 50px;
        font-weight: 900;
        margin-bottom: -10px;
    }
    .subtitle {
        color: #CCCCCC;
        text-align: center;
        font-size: 20px;
        margin-bottom: 25px;
    }
    .card {
        background-color: #1A1D23;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 0px 25px rgba(0,0,0,0.45);
        text-align: center;
        margin-top: 20px;
    }
    .result-region {
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
    }
    .confidence {
        color: #FFD700;
        font-size: 30px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Cargar modelo y labels
# -----------------------------------------
@st.cache_resource
def load_knn():
    with open(KNN_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    return model, labels

knn_model, labels = load_knn()

# -----------------------------------------
# Títulos
# -----------------------------------------
st.markdown("<h1 class='title'>Clasificación de Regiones</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sube o captura una imagen para clasificarla como Costa, Sierra o Selva</p>", unsafe_allow_html=True)

# -----------------------------------------
# Layout: cámara + subida de imagen
# -----------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Cámara")
    img_camera = st.camera_input("Toma una foto")

with col2:
    st.subheader("📂 Subir imagen")
    img_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

# Elegir imagen final
image = img_camera if img_camera else img_file

# -----------------------------------------
# Procesamiento
# -----------------------------------------
if image:

    st.markdown("<p class='subtitle'>Procesando imagen...</p>", unsafe_allow_html=True)

    # Convertimos a numpy
    img_pil = Image.open(image)
    img_np = np.array(img_pil)

    # Preprocesar
    embedding = preprocess_image(img_np)

    if embedding is None:
        st.error("No se pudo procesar la imagen.")
    else:
        embedding = np.array(embedding).astype("float32").reshape(1, -1)

        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm

        # Predicción KNN
        dist, idx = knn_model.kneighbors(embedding, n_neighbors=3)

        pred_label = labels[idx[0][0]]

        # Distancia promedio
        mean_dist = float(np.mean(dist[0]))

        # Escala de confianza ajustada
        MAX_DIST = 2.0
        confidence = max(0, 1 - (mean_dist / MAX_DIST)) * 100

        # -----------------------------------------
        # Tarjeta de resultado
        # -----------------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.image(img_np, caption="Imagen procesada", use_container_width=True)

        st.markdown(f"<p class='result-region'>{pred_label}</p>", unsafe_allow_html=True)
        st.markdown("<p style='color:#AAAAAA;'>Confianza del modelo:</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='confidence'>{confidence:.2f}%</p>", unsafe_allow_html=True)

        st.progress(min(confidence / 100, 1.0))

        st.markdown("</div>", unsafe_allow_html=True)

        # =============================================================
        # DASHBOARD DE ANÁLISIS DETALLADO
        # =============================================================
        st.markdown("---")
        st.subheader("📊 Análisis detallado de esta predicción")

        # Tabla vecinos
        neighbors_labels = labels[idx[0]]
        neighbors_dist = dist[0]

        df_neighbors = pd.DataFrame({
            "Vecino": [1, 2, 3],
            "Clase": neighbors_labels,
            "Distancia": neighbors_dist
        })

        st.write("### 🧩 Vecinos más cercanos")
        st.dataframe(df_neighbors)

        # Gráfico de distancias
        fig1, ax1 = plt.subplots()
        ax1.bar(["Vecino 1", "Vecino 2", "Vecino 3"], neighbors_dist)
        ax1.set_ylabel("Distancia")
        ax1.set_title("Distancia a cada vecino")
        st.pyplot(fig1)

        # Gráfico de confianza
        fig2, ax2 = plt.subplots()
        ax2.bar(["Confianza"], [confidence / 100])
        ax2.set_ylim(0, 1)
        ax2.set_title("Nivel de confianza")
        st.pyplot(fig2)

        # Similitud (invertimos distancia)
        similarities = 1 - (neighbors_dist / MAX_DIST)

        fig3, ax3 = plt.subplots()
        ax3.bar(["Vecino 1", "Vecino 2", "Vecino 3"], similarities)
        ax3.set_ylim(0, 1)
        ax3.set_title("Similitud con cada vecino (1 = más similar)")
        st.pyplot(fig3)

        # Resumen
        st.info(f"""
**Resultado final:** {pred_label}  
**Confianza estimada:** {confidence:.2f}%  
**Distancia promedio:** {mean_dist:.4f}  
""")
