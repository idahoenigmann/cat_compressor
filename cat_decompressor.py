import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import pickle

origin = 'file:///home/sascha/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 1

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def main():
    entire_model = keras.models.load_model("model_cat_classifier.h5")

    model = keras.Sequential([
        keras.layers.Dense(40 * 30 * 64),
        keras.layers.Reshape([40, 30, 64]),

        keras.layers.Conv2DTranspose(32, kernel_size=3, strides=(2, 2), padding='same', output_padding=(1, 1),
                                     data_format='channels_last', activation=keras.activations.relu),
        keras.layers.Conv2DTranspose(16, kernel_size=3, strides=(2, 2), padding='same', output_padding=(1, 1),
                                     data_format='channels_last', activation=keras.activations.relu),
        keras.layers.Conv2DTranspose(8, kernel_size=3, strides=(2, 2), padding='same', output_padding=(1, 1),
                                     data_format='channels_last', activation=keras.activations.relu),
        keras.layers.Conv2DTranspose(3, kernel_size=3, strides=(2, 2), padding='same', output_padding=(1, 1),
                                     data_format='channels_last', activation=keras.activations.relu)
    ])

    for layer_idx in range(-1, -1 - len(model.layers), -1):
        model.layers[layer_idx].set_weights(entire_model.layers[layer_idx].get_weights())

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))

    data = np.loadtxt('data.csv', delimiter=',')

    with open("pca.txt", "rb") as f:
        pca = pickle.load(f)

    all_components = pca.inverse_transform(data)
    all_components = all_components.reshape([BATCH_SIZE, 300])

    results = model.predict(all_components, steps=1)

    for image in results:
        output_img = np.reshape(image, [IMG_WIDTH, IMG_HEIGHT, 3])
        output_img *= 255

        pil_output_img = Image.fromarray(np.uint8(output_img))
        pil_output_img.show()


if __name__ == '__main__':
    main()
