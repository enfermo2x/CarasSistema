import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas del dataset
DATASET_DIR = os.path.join(BASE_DIR, "data", "train")

# Rutas de la base de datos de embeddings
DATABASE_DIR = os.path.join(BASE_DIR, "database")

EMBEDDINGS_PATH = os.path.join(DATABASE_DIR, "embeddings.npy")
LABELS_PATH = os.path.join(DATABASE_DIR, "labels.npy")
KNN_MODEL_PATH = os.path.join(DATABASE_DIR, "knn_model.pkl")

# Tamaño de entrada de imagen para MobileNetV2
IMG_SIZE = (160, 160)

# Vecinos para KNN
KNN_NEIGHBORS = 3

# Clases
CLASS_NAMES = ["costa", "sierra", "selva"]
