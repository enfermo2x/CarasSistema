import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Inicializar MobileNetV2
model = MobileNetV2(include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Cargar detector de rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_image(img):
    """Procesa toda la imagen completa y obtiene embedding."""
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        arr = preprocess_input(img_rgb.astype("float32"))
        arr = np.expand_dims(arr, axis=0)

        embedding = model.predict(arr, verbose=0)[0]

        # Normalización recomendada
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        print("Error en preprocess_image:", e)
        return None


def preprocess_face(img):
    """Detecta rostro, recorta y obtiene embedding."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=7,
            minSize=(140, 140)
        )

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]

        face_resized = cv2.resize(face_img, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        arr = preprocess_input(face_rgb.astype("float32"))
        arr = np.expand_dims(arr, axis=0)

        embedding = model.predict(arr, verbose=0)[0]

        # Normalización recomendada
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        print("Error en preprocess_face:", e)
        return None
