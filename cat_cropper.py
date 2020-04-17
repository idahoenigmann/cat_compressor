import tensorflow as tf
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from matplotlib.patches import Circle
import math
from scipy import ndimage, misc
import threading


origin = 'file:///home/ida/.keras/datasets/cat-dataset.zip'
fname = 'cat-dataset'

IMG_HEIGHT = 0
IMG_WIDTH = 0

lock = threading.Lock()


def load_data(start, end):
    global IMG_WIDTH
    global IMG_HEIGHT

    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    labels = []

    print("loading {} out of {} images".format(end - start, len(list(data_dir.glob('*/*/*.jpg')))))

    for file in list(data_dir.glob('*/*/*.jpg'))[start:end]:
        with lock:
            img = load_img(file.as_posix(), color_mode="rgb")
        img = np.array(img)

        IMG_WIDTH = img.shape[0]
        IMG_HEIGHT = img.shape[1]

        img = img / 255.0
        images.append(img)

        label_file = open(file.as_posix() + ".cat", "r")
        label = label_file.readline()
        label = label.split(" ")
        labels.append(label)

    return images, labels


def plot_image(image, name=""):
    with lock:
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        if name != "":
            plt.savefig("/home/ida/.keras/datasets/cat_faces/{}".format(name))
        else:
            plt.show()


def crop_face(image, label):
    mouth_x = int(label[6])
    mouth_y = int(label[5])

    left_ear_x = int(label[10])
    left_ear_y = int(label[9])

    right_ear_x = int(label[16])
    right_ear_y = int(label[15])

    top_x = int((left_ear_x + right_ear_x) / 2)
    top_y = int((left_ear_y + right_ear_y) / 2)

    if top_y != mouth_y:
        alpha = math.atan((top_x - mouth_x) / (top_y - mouth_y))
    else:
        alpha = -math.pi / 2
    dist = int(math.sqrt(((mouth_x - top_x) ** 2) + ((mouth_y - top_y) ** 2)))

    if alpha < 0:
        alpha += math.pi

    alpha -= math.pi / 2

    padX = [image.shape[1] - mouth_y, mouth_y]
    padY = [image.shape[0] - mouth_x, mouth_x]
    image = np.pad(image, [padY, padX, [0, 0]], 'constant')
    image = ndimage.rotate(image, math.degrees(alpha), reshape=False)
    image = image[padY[0]: -padY[1], padX[0]: -padX[1]]

    image = image[max(mouth_x - dist, 0): min(mouth_x + int(dist * 0.5), image.shape[0]),
                  max(mouth_y - dist, 0): min(mouth_y + dist, image.shape[1])]

    return image


def crop_and_save_images(start, end, step):
    i = start
    for y in range(start, end, step):
        images, labels = load_data(y, y + step)
        for x in range(len(images)):
            images[x] = crop_face(images[x], labels[x])
            plot_image(images[x], "{:05d}.jpg".format(i))
            i += 1


if __name__ == '__main__':
    threads = []

    k = 901
    for i in range(0, 4):
        t = threading.Thread(target=crop_and_save_images, args=(k, min(k + 500, 9990), 10))
        t.start()
        threads.append(t)
        k += 500

    for t in threads:
        t.join()
