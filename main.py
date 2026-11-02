import cv2
from pathlib import Path
import numpy
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Comment changer path pour fonctionner PC/Mac
dataset_path = Path("assets")
IMG_HEIGHT:int = 30
IMG_WIDTH:int = 30
TEST_SIZE = 1
model_save_filepath = "model.keras"
EPOCHS = 10
NUM_CATEGORIES = 43

#categories = dataset_path / "10"
#img_path = categories / "00000_00000.ppm"
#img = cv2.imread(str(img_path), -1)
#img = cv2.resize(img, (30, 30))
#img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#print(img.shape)
#print(type(img))
#print(img[0][0:30])
#cv2.imshow("Une Fenetre", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


def load_data(dataset_path: Path):
    images= []
    labels = []

    for dossier in dataset_path.iterdir():
        label = int(dossier.name)
        for img_path in dossier.iterdir():
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(img)
            labels.append(label)

    return images,labels

def get_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Flatten(), #Applatie l'image en un gros vector
        tf.keras.layers.Dense(128, activation="relu"), # 128 neurones , relu met a zero les valeurs negatives
        tf.keras.layers.Dropout(0.5), #pendant l'entrainement ignore la moitie des neuronnes
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"), #Une sortie par categorie et softmax transforme en probabilite total 1
    ])

    model.compile(
        optimizer="adam", # Methode qui ajuste les poirds apres chaque erreur
        loss="mean_squared_error", # Mesure d'erreur que l'optimiseur cherche a diminuer
        metrics=["mae", "accuracy"], # mae pour suivre progression et accuracy pour proportion de bonne reponse
    )
    return model

images,labels = load_data(dataset_path)
labels = tf.keras.utils.to_categorical(labels)
x_train , x_test , y_train , y_test = train_test_split(numpy.array(images), numpy.array(labels), test_size=TEST_SIZE)
model = get_model()
model.fit(x_train , y_train , epochs=EPOCHS) # Trains the model for a fixed number of epochs.
model.evaluate(x_test , y_test , verbose =2) # Returns the loss and metrics values for the model
model.save(model_save_filepath)







