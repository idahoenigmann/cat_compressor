import keras
from PIL import Image
import numpy as np
from main import IMG_WIDTH, IMG_HEIGHT
import time
from keras.preprocessing.image import load_img
import tensorflow as tf
import pathlib
import os.path
import matplotlib.pyplot as plt
import atexit
import random

LOOP = True
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 200

origin = 'file:///home/ida/.keras/datasets/cat-dataset.zip'
fname = 'cat-dataset'
model = keras.models.Sequential()


def exit_handler():
    model.save('model_cat_classifier.h5')


def load_images(len=50):
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    labels = []

    cats = 0
    non_cats = 0

    all_files = list(data_dir.glob('*.jpg'))
    random.shuffle(all_files)

    for file in all_files[0:len]:
        # print(file)
        img = load_img(file.as_posix(), color_mode="grayscale", target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img)
        img = img.reshape((IMG_WIDTH, IMG_HEIGHT, -1))
        img = img / 255.0
        images.append(img)

        if 'cat_' in file.as_posix().split("/")[-1]:
            labels.append([1])
            cats += 1
        else:
            labels.append([0])
            non_cats += 1

    print("cats: {}".format(cats))
    print("non cats: {}".format(non_cats))

    return (np.array(images), np.array(labels))


if __name__ == '__main__':
    atexit.register(exit_handler)

    if not os.path.isfile('model_cat_classifier.h5'):
        model = keras.Sequential([
            keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape=[IMG_WIDTH, IMG_HEIGHT, 1],
                                data_format='channels_last', activation=keras.activations.relu),
            keras.layers.MaxPool2D(pool_size=(2, 2)),

            keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.MaxPool2D(pool_size=(2, 2)),

            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ])
    else:
        print("Using trained model 'model_cat_classifier.h5'!")
        model = keras.models.load_model('model_cat_classifier.h5')

    print(model.summary())

    model.compile(metrics=[keras.metrics.accuracy],
                  loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.rmsprop())

    history_loss_all = []
    history_accuracy_all = []
    while True:
        data, label = load_images(BATCH_SIZE)

        history = model.fit(data, label, batch_size=BATCH_SIZE, epochs=50, shuffle=True)
        model.save('model_cat_classifier.h5')

        history_loss_all.append(np.average(history.history['loss']))

        history_accuracy_all.append(np.average(history.history['accuracy']))
        # history_accuracy_all = np.concatenate((history_accuracy_all, history.history['accuracy']))

        # Plot training & validation loss values
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(np.ravel(history_loss_all), color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(np.ravel(history_accuracy_all), color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

        if not LOOP:
            break
