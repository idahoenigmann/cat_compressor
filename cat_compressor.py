import numpy as np
import pickle
from cat_pca import get_data
import random

TRAIN_DIR = '/home/ida/.keras/datasets/cat_faces/train/train/'
VAL_DIR = '/home/ida/.keras/datasets/cat_faces/validation/validation/'

IMG_WIDTH, IMG_HEIGHT = 64, 64
SHOW_IMG = False
BATCH_SIZE = 1


def compress(source="val"):
    CNT_IMG = 1

    if source == "val":
        images = get_data(VAL_DIR)
    else:
        images = get_data(TRAIN_DIR)

    idx = random.randint(0, 1000)

    images = images[idx:idx+1]
    images = np.reshape(images, [CNT_IMG, IMG_WIDTH * IMG_HEIGHT])

    with open("pca.txt", "rb") as f:
        pca = pickle.load(f)
    principle_components = pca.transform(images)

    np.savetxt('data.csv', principle_components, delimiter=',')


if __name__ == '__main__':
    compress()
