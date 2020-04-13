import tensorflow as tf
import keras
import pathlib
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2

origin = 'file:///home/ida/.keras/datasets/cat10-dataset.zip'
fname = 'cat10-dataset'


def load_images():
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    images = []
    for file in list(data_dir.glob('*.jpg')):
        print(file.as_posix())
        img = load_img(file.as_posix(), color_mode="grayscale")
        img = np.array(img)
        img = img.reshape((500, 375, -1))
        images.append(img)
    return np.array(images)


def main():
    model = keras.Sequential([
        keras.layers.Conv2D(3, kernel_size=5, strides=5, input_shape=[500, 375, 1], data_format='channels_last', padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(3, kernel_size=5, strides=1, padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(12),
        keras.layers.Dense(187500),
        keras.layers.Reshape([500, 375, 1]),
    ])

    print(model.summary())

    data = load_images()
    print(data.shape)

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.SGD())

    model.fit(data, data, batch_size=1)


if __name__ == '__main__':
    main()