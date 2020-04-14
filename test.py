import keras
from main import load_images, make_prediction
from PIL import Image
import numpy as np
from main import IMG_WIDTH, IMG_HEIGHT

if __name__ == '__main__':
    img_idx = 0

    model = keras.models.load_model('model.h5')

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_percentage_error,
                  optimizer=keras.optimizers.SGD())

    data = load_images()

    make_prediction(model, data[img_idx])

    some_img = data[img_idx] * 255

    some_img = np.reshape(some_img, [IMG_WIDTH, IMG_HEIGHT])

    pil_img = Image.fromarray(some_img)
    pil_img.show()
