import keras
from main import load_images, make_prediction
from PIL import Image
import numpy as np
from main import IMG_WIDTH, IMG_HEIGHT
import time

if __name__ == '__main__':
        model = keras.models.load_model('model.h5')
        print(model.summary())

        model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                      loss=keras.losses.mean_absolute_percentage_error,
                      optimizer=keras.optimizers.SGD())
        data = load_images()
        for img_idx in range(0, 100):
            input_img = np.reshape(data[img_idx], [1, IMG_WIDTH, IMG_HEIGHT, 1])
            output_img = model.predict(input_img)
            output_img = np.reshape(output_img[0], [IMG_WIDTH, IMG_HEIGHT])
            output_img *= 255
            pil_output_img = Image.fromarray(output_img)
            # pil_output_img.show()

            input_img = data[img_idx] * 255
            input_img = np.reshape(input_img, [IMG_WIDTH, IMG_HEIGHT])
            pil_input_img = Image.fromarray(input_img)
            # pil_input_img.show()

            diff_img = output_img - input_img
            diff_img = np.abs(diff_img)
            pil_diff_img = Image.fromarray(diff_img)
            pil_diff_img = pil_diff_img.resize((IMG_WIDTH * 10, IMG_HEIGHT * 10))
            pil_diff_img.show(title="difference")
            time.sleep(0.1)
            pil_diff_img.close()
