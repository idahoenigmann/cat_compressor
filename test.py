import tensorflow as tf
import keras
import pathlib
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import cv2
from PIL import Image
from main import load_images, make_prediction

if __name__ == '__main__':
    model = keras.models.load_model('model.h5')

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.SGD())

    data = load_images()

    make_prediction(model, data[0])
