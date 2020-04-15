import keras
from PIL import Image
import numpy as np
from main import IMG_WIDTH, IMG_HEIGHT
import time
from keras.preprocessing.image import load_img
import tensorflow as tf
import pathlib

origin = 'file:///home/ida/.keras/datasets/cat10-dataset.zip'
fname = 'cat10-dataset'


def load_images(len=9993):
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    labels = []
    for file in list(data_dir.glob('*.jpg'))[0:len]:
        img = load_img(file.as_posix(), color_mode="grayscale", target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img)
        img = img.reshape((IMG_WIDTH, IMG_HEIGHT, -1))
        img = img / 255.0
        images.append(img)

        if file.name.find('/cat_'):
            labels.append([1])
        else:
            labels.append([0])

    return np.array(images), np.array(labels)


if __name__ == '__main__':
    data, label = load_images(500)

    model = keras.Sequential([
        keras.layers.Conv2D(10, kernel_size=3, strides=1, input_shape=[IMG_WIDTH, IMG_HEIGHT, 1],
                            data_format='channels_last', padding='same', activation=keras.activations.relu),
        keras.layers.MaxPool2D(pool_size=2, padding='same'),

        keras.layers.Conv2D(50, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
        keras.layers.MaxPool2D(pool_size=2, padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(50),
        keras.layers.Dense(1, activation=keras.activations.sigmoid)
    ])

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))

    history = model.fit(data, label, batch_size=50, epochs=1, shuffle=True, validation_split=0.1)
