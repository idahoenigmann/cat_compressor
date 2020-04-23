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

NEW_MODEL = False
LOOP = True

origin = 'file:///home/sascha/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 30
EPOCHS = 3

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def exit_handler():
    model.save('model_cat_classifier.h5')


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
            keras.layers.Dense(300),
            keras.layers.Dense(80 * 60 * 64),
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

    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    print(data_dir)

    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')

    print(train_dir)
    print(validation_dir)

    train_dir_path = pathlib.Path(os.path.join(train_dir, 'train'))
    cnt_files = len(list(train_dir_path.glob('*.jpg')))

    img_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = img_generator.flow_from_directory(directory=train_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE, class_mode="input", shuffle=True)

    val_data_gen = img_generator.flow_from_directory(directory=validation_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=BATCH_SIZE, class_mode="input", shuffle=True)

    history_all_loss = []
    history_all_validation_loss = []

    callback = keras.callbacks.ModelCheckpoint("model_cat_classifier.h5", period=1)

    while True:
        history = model.fit(train_data_gen, steps_per_epoch=cnt_files // BATCH_SIZE, epochs=EPOCHS,
                            validation_data=val_data_gen, validation_steps=10, callbacks=[callback])
        model.save('model_cat_classifier.h5')

        history_all_loss = np.concatenate((history_all_loss, history.history['loss']))
        history_all_validation_loss = np.concatenate((history_all_validation_loss, history.history['val_loss']))
        """
        # Plot training & validation loss values
        plt.plot(np.ravel(history_all_loss))
        plt.plot(np.ravel(history_all_validation_loss))
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()"""

        if not LOOP:
            break


if __name__ == '__main__':
    atexit.register(exit_handler)
    main()
