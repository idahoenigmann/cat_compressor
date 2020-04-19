import tensorflow as tf
import keras
import pathlib
import numpy as np
from keras.preprocessing.image import load_img
from PIL import Image
import atexit
import os.path
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random

NEW_MODEL = True
LOOP = True

origin = 'file:///home/ida/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 50


def exit_handler():
    model.save('model_cat_classifier.h5')


def load_images(start, end):
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    for file in list(data_dir.glob('*.jpg'))[start:end]:
        img = load_img(file.as_posix(), color_mode="rgb", target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img)
        img = img.reshape((IMG_WIDTH, IMG_HEIGHT, 3))
        img = img / 255.0
        images.append(img)

    return np.array(images)


def make_prediction(model, img):
    img = np.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 3])

    output = model.predict(img)

    output = np.reshape(output[0], [IMG_WIDTH, IMG_HEIGHT])

    output *= 255

    pil_img = Image.fromarray(output)
    # pil_img.show()


def main():
    global model

    if NEW_MODEL or (not os.path.isfile('model_cat_classifier.h5')):
        model = keras.Sequential([
            keras.layers.Conv2D(8, kernel_size=3, strides=1, input_shape=[IMG_WIDTH, IMG_HEIGHT, 3],
                                data_format='channels_last', padding='same', activation=keras.activations.relu),
            keras.layers.MaxPool2D(pool_size=2, padding='same'),

            keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.MaxPool2D(pool_size=2, padding='same'),

            keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.MaxPool2D(pool_size=2, padding='same'),

            keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),

            keras.layers.Reshape([80 * 60 * 64]),
            #keras.layers.Dense(4 * 4),
            #keras.layers.Dense(4 * 4 * 200),
            keras.layers.Reshape([80, 60, 64]),

            keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),

            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),

            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),

            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu)

        ])
    else:
        print("Using trained model 'model_cat_classifier.h5'!")
        model = keras.models.load_model('model_cat_classifier.h5')

    print(model.summary())

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))

    history_all_loss = []
    history_all_validation_loss = []
    img_idx = 0
    while True:
        data = load_images(img_idx, img_idx + BATCH_SIZE)

        history = model.fit(data, data, batch_size=BATCH_SIZE, epochs=1, shuffle=True, validation_split=0.1)
        model.save('model_cat_classifier.h5')

        history_all_loss = np.concatenate((history_all_loss, history.history['loss']))
        history_all_validation_loss = np.concatenate((history_all_validation_loss, history.history['val_loss']))

        # Plot training & validation loss values
        plt.plot(np.ravel(history_all_loss))
        plt.plot(np.ravel(history_all_validation_loss))
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        img_idx += BATCH_SIZE

        if not LOOP:
            break


if __name__ == '__main__':
    atexit.register(exit_handler)
    main()
