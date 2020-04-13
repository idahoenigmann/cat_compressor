import tensorflow as tf
import keras
import pathlib
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import Image
import atexit
import os.path

NEW_MODEL = False
LOOP = True

origin = 'file:///home/ida/.keras/datasets/cat10-dataset.zip'
fname = 'cat10-dataset'
model = keras.models.Sequential()


def exit_handler():
    model.save('model.h5')


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    for file in list(data_dir.glob('*.jpg'))[0:100]:
        img = load_img(file.as_posix(), color_mode="grayscale", target_size=(500, 375))
        img = np.array(img)
        img = img.reshape((500, 375, -1))
        img = img / 255.0
        images.append(img)
    return np.array(images)


def make_prediction(model, img):
    img = np.reshape(img, [1, 500, 375, 1])

    output = model.predict(img)

    output = np.reshape(output[0], [500, 375])

    output *= 255

    pil_img = Image.fromarray(output)
    pil_img.convert('RGB').show()


def main():
    global model

    if NEW_MODEL or (not os.path.isfile('model.h5')):
        model = keras.Sequential([
            keras.layers.Conv2D(3, kernel_size=5, strides=5, input_shape=[500, 375, 1], data_format='channels_last', padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(3, kernel_size=5, strides=1, padding='same', activation=keras.activations.relu),
            keras.layers.Activation(keras.activations.sigmoid),
            keras.layers.MaxPool2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(100, activation=keras.activations.relu),
            keras.layers.Dense(187500, activation=keras.activations.sigmoid),
            keras.layers.Reshape([500, 375, 1]),
        ])
    else:
        print("Using trained model 'model.h5'!")
        model = keras.models.load_model('model.h5')

    print(model.summary())

    data = load_images()

    model.compile(metrics=[keras.metrics.mean_absolute_error],
                  loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))

    while True:
        model.fit(data, data, batch_size=100, epochs=1000)
        model.save('model.h5')
        make_prediction(model, data[0])

        if not LOOP:
            break


if __name__ == '__main__':
    atexit.register(exit_handler)
    main()
