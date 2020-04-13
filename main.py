import tensorflow as tf
import keras
import pathlib
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import Image

origin = 'file:///home/ida/.keras/datasets/cat10-dataset.zip'
fname = 'cat10-dataset'


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    for file in list(data_dir.glob('*.jpg'))[0:5]:
        img = load_img(file.as_posix(), color_mode="grayscale", target_size=(500, 375))
        img = np.array(img)
        img = img.reshape((500, 375, -1))
        images.append(img)
    return np.array(images)


def main():
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

    print(model.summary())

    data = load_images()

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error, keras.metrics.mean_absolute_error],
                  loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.SGD())

    model.fit(data, data, batch_size=5, epochs=1000)

    img = data[0]
    img = np.reshape(img, [1, 500, 375, 1])

    output = model.predict(img)

    output = np.reshape(output[0], [500, 375])

    output *= 255

    pil_img = Image.fromarray(output)
    pil_img.show()

    img = np.reshape(img, [500, 375])

    img *= 255

    pil_img = Image.fromarray(img)
    pil_img.show()


if __name__ == '__main__':
    main()
