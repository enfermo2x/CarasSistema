import os
import numpy as np
import pickle
import cv2
from sklearn.neighbors import KNeighborsClassifier

from src.preprocess import preprocess_image
from config import DATASET_DIR, EMBEDDINGS_PATH, LABELS_PATH, KNN_MODEL_PATH, KNN_NEIGHBORS


def load_dataset_images():
    embeddings = []
    labels = []

    print("Escaneando dataset:", DATASET_DIR)

    for label in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, label)

        if not os.path.isdir(class_dir):
            continue

        print(f"Procesando clase: {label}")

        for file in os.listdir(class_dir):
            path = os.path.join(class_dir, file)

            if not path.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img = cv2.imread(path)
            if img is None:
                print("Imagen no válida:", path)
                continue

            embedding = preprocess_image(img)

            if embedding is None:
                print("Error en embedding:", path)
                continue

            # 🔥 Normalización obligatoria (para KNN)
            embedding = embedding / np.linalg.norm(embedding)
            embedding = embedding.astype("float32")

            embeddings.append(embedding)
            labels.append(label)

    return np.array(embeddings), np.array(labels)


def train_knn(embeddings, labels):
    print("\nEntrenando modelo KNN...")
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
    knn.fit(embeddings, labels)
    return knn


if __name__ == "__main__":
    print("=== INICIANDO ENTRENAMIENTO ===")

    embeddings, labels = load_dataset_images()

    if len(embeddings) == 0:
        print("ERROR: No se generaron embeddings. Revisa tu dataset.")
        exit()

    # Guardar embeddings y labels
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)

    print("\nEmbeddings generados:", embeddings.shape)
    print("Labels generados:", labels.shape)

    # Entrenar y guardar modelo
    knn_model = train_knn(embeddings, labels)

    with open(KNN_MODEL_PATH, "wb") as f:
        pickle.dump(knn_model, f)

    print(f"\nModelo guardado en: {KNN_MODEL_PATH}")
    print("=== ENTRENAMIENTO COMPLETADO ===")
