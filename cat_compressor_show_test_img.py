import keras
from PIL import Image
import numpy as np
from cat_nn import IMG_WIDTH, IMG_HEIGHT
import tensorflow as tf
import pathlib
from keras.preprocessing.image import load_img

origin = 'file:///home/sascha/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'


def load_images(start, end):
    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    validation_dir = data_dir.joinpath('validation/validation')

    files = list(validation_dir.glob('*.jpg'))
    np.random.shuffle(files)
    images = []

    for file in files[start:end]:
        img = load_img(file, color_mode="rgb", target_size=(IMG_WIDTH, IMG_HEIGHT))
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

    return output


if __name__ == '__main__':
        model = keras.models.load_model('model_cat_classifier.h5')
        print(model.summary())

        model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                      loss=keras.losses.mean_absolute_percentage_error,
                      optimizer=keras.optimizers.SGD())
        data = load_images(0, 1)
        for img_idx in range(0, len(data)):
            input_img = np.reshape(data[img_idx], [1, IMG_WIDTH, IMG_HEIGHT, 3])
            output_img = model.predict(input_img)

            output_img = np.reshape(output_img[0], [IMG_WIDTH, IMG_HEIGHT, 3])
            output_img *= 255

            pil_output_img = Image.fromarray(np.uint8(output_img))
            pil_output_img.show()

            input_img = data[img_idx] * 255
            input_img = np.reshape(input_img, [IMG_WIDTH, IMG_HEIGHT, 3])
            pil_input_img = Image.fromarray(np.uint8(input_img))
            pil_input_img.show()

            diff_img = output_img - input_img
            diff_img = np.abs(diff_img)
            pil_diff_img = Image.fromarray(np.uint8(diff_img))
            pil_diff_img.show(title="difference")
