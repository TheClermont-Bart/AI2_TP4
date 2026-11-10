import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf # type: ignore[import-untyped]

IMG_HEIGHT: int = 30
IMG_WIDTH: int = 30
NUM_CATEGORIES = 43

def load_data(dataset_path: Path) -> tuple[list[np.ndarray], list[int]]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for dossier in dataset_path.iterdir():
        label = int(dossier.name)
        for img_path in dossier.iterdir():
            img = cv2.imread(img_path, 1)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            destination = np.empty(img.shape, dtype=np.float32)
            img = cv2.normalize(img, destination, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(img)
            labels.append(label)

    return images,labels

def get_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(), #Applatie l'image en un gros vector
        tf.keras.layers.Dense(128, activation="relu"), # 128 neurones , relu met a zero les valeurs negatives
        tf.keras.layers.Dropout(0.25), #pendant l'entrainement ignore pourcentage des neuronnes
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"), #Une sortie par categorie et softmax transforme en probabilite total 1
    ])

    model.compile(
        optimizer="adam", # Methode qui ajuste les poirds apres chaque erreur
        loss="categorical_crossentropy", # Mesure d'erreur que l'optimiseur cherche a diminuer
        metrics=["accuracy"],
    )
    return model