import samva
import numpy
from pathlib import Path
import tensorflow as tf # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split # type: ignore[import-untyped]

dataset_path = Path("assets")
model_save_filepath = "model.keras"
TEST_SIZE = 20
EPOCHS = 10

images,labels = samva.load_data(dataset_path)
labels = tf.keras.utils.to_categorical(labels)
x_train , x_test , y_train , y_test = train_test_split(numpy.array(images), numpy.array(labels), test_size=TEST_SIZE)
model = samva.get_model()
model.fit(x_train , y_train , epochs=EPOCHS) # EPOCHS Nombre d'entrainement du model
model.evaluate(x_test , y_test , verbose =2) # Renvoi la perte et precision
model.save(model_save_filepath)







