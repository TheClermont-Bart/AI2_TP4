import cv2
from pathlib import Path
import numpy as np
from typing import List
import tensorflow as tf

#Comment changer path pour fonctionner PC/Mac
dataset_path = Path("assets")
categories = dataset_path / "10"
img_path = categories / "00000_00000.ppm"

img = cv2.imread(str(img_path), -1)
img = cv2.resize(img, (30, 30)) #Pourquoi le dernier 3(channel) rentre pas...
#img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

print(img.shape)
print(type(img))
print(img[0][0:30])

cv2.imshow("Une Fenetre", img)
cv2.waitKey(0) #Ferme avec n'importe quelle touche
cv2.destroyAllWindows()

'''
def load_data(dataset_path: Path):
    IMG_HEIGHT:int = 30
    IMG_WIDTH:int = 30
    images:np.ndarray = []
    labels:int = []

    img = cv2.imread(dataset_path, 1)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) #Pourquoi le dernier 3(channel) rentre pas...
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return images,labels
'''

'''
#def samva.get_model()-> model:

'''




