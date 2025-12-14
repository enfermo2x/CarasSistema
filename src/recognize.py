import numpy as np
import pickle
from src.preprocess import preprocess_image
from config import EMBEDDINGS_PATH, LABELS_PATH, KNN_MODEL_PATH

# Cargar KNN
print("Cargando modelo KNN...")
with open(KNN_MODEL_PATH, "rb") as f:
    knn_model = pickle.load(f)

def predict_region(img):
    """
    Devuelve 'costa', 'sierra' o 'selva' según el embedding.
    """
    try:
        embedding = preprocess_image(img)
        if embedding is None:
            return None

        # Normalizar embedding (si tu KNN fue entrenado así)
        embedding = embedding / np.linalg.norm(embedding)

        # Convertir a shape válido
        embedding = embedding.reshape(1, -1)

        # Predicción
        pred = knn_model.predict(embedding)[0]
        region = str(pred).lower()

        # Validación de región
        if region not in ["costa", "sierra", "selva"]:
            return None

        return region

    except Exception as e:
        print("Error en predict_region:", e)
        return None
