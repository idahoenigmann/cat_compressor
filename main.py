import tensorflow as tf
import keras
import pathlib
import numpy as np
from keras.preprocessing.image import load_img
from PIL import Image
import atexit
import os.path
import matplotlib.pyplot as plt

NEW_MODEL = False
LOOP = True

origin = 'file:///home/ida/.keras/datasets/cat10-dataset.zip'
fname = 'cat10-dataset'
model = keras.models.Sequential()

IMG_WIDTH = 64
IMG_HEIGHT = 64


def exit_handler():
    model.save('model.h5')


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    for file in list(data_dir.glob('*.jpg'))[0:100]:
        img = load_img(file.as_posix(), color_mode="grayscale", target_size=(IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img)
        img = img.reshape((IMG_WIDTH, IMG_HEIGHT, -1))
        img = img / 255.0
        images.append(img)
    return np.array(images)


def make_prediction(model, img):
    img = np.reshape(img, [1, IMG_WIDTH, IMG_HEIGHT, 1])

    output = model.predict(img)

    output = np.reshape(output[0], [IMG_WIDTH, IMG_HEIGHT])

    output *= 255

    pil_img = Image.fromarray(output)
    pil_img.convert('RGB').show()


def main():
    global model

    if NEW_MODEL or (not os.path.isfile('model.h5')):
        model = keras.Sequential([
            keras.layers.Conv2D(10, kernel_size=3, strides=1, input_shape=[IMG_WIDTH, IMG_HEIGHT, 1],
                                data_format='channels_last', padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),
            keras.layers.MaxPool2D(),

            keras.layers.Conv2D(50, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),
            keras.layers.MaxPool2D(),

            keras.layers.Conv2D(100, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),
            keras.layers.MaxPool2D(),

            keras.layers.Conv2D(200, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),

            keras.layers.Flatten(),

            keras.layers.Dense(12800),

            keras.layers.Reshape([8, 8, 200]),

            keras.layers.Conv2D(100, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),

            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(50, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),

            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(10, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),

            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),

        ])
    else:
        print("Using trained model 'model.h5'!")
        model = keras.models.load_model('model.h5')

    print(model.summary())

    data = load_images()

    model.compile(metrics=[keras.metrics.mean_absolute_error],
                  loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.5, nesterov=True))

    while True:
        history = model.fit(data, data, batch_size=50, epochs=100)
        model.save('model.h5')

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()

        make_prediction(model, data[0])

        if not LOOP:
            break


if __name__ == '__main__':
    atexit.register(exit_handler)
    main()
