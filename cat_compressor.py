import tensorflow as tf
import keras
import pathlib
import os.path
import numpy as np
from PIL import Image
import pickle

origin = 'file:///home/sascha/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 1
SHOW_IMG = True

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def compress(source="val"):
    entire_model = keras.models.load_model("model_cat_nn.h5")

    model = keras.Sequential([
        keras.layers.Conv2D(8, kernel_size=4, strides=(1, 1), input_shape=[IMG_WIDTH, IMG_HEIGHT, 3],
                            data_format='channels_last', padding='same', activation=keras.activations.relu),
        keras.layers.Conv2D(16, kernel_size=4, strides=(2, 2), padding='same', activation=keras.activations.relu,
                            data_format='channels_last'),
        keras.layers.Conv2D(32, kernel_size=4, strides=(2, 2), padding='same', activation=keras.activations.relu,
                            data_format='channels_last'),
        keras.layers.Conv2D(64, kernel_size=4, strides=(4, 4), padding='same', activation=keras.activations.relu,
                            data_format='channels_last'),

        keras.layers.Reshape([40 * 30 * 64]),
        keras.layers.Dense(300, activation=keras.activations.sigmoid),
    ])

    for layer_idx in range(len(model.layers)):
        model.layers[layer_idx].set_weights(entire_model.layers[layer_idx].get_weights())

    model.compile(metrics=[keras.metrics.mean_absolute_error],
                  loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))

    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    val_dir = None
    if source == "val":
        val_dir = os.path.join(data_dir, 'validation')
    elif source == "train":
        val_dir = os.path.join(data_dir, 'train')
    img_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_data_gen = img_generator.flow_from_directory(directory=val_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                     batch_size=BATCH_SIZE, class_mode="input", shuffle=True)

    images = val_data_gen.next()
    images = images[0]

    if SHOW_IMG:
        for image in images:
            output_img = np.reshape(np.copy(image), [IMG_WIDTH, IMG_HEIGHT, 3])
            output_img *= 255

            pil_output_img = Image.fromarray(np.uint8(output_img))
            pil_output_img.show()

    results = model.predict(images, steps=1)

    with open("pca.txt", "rb") as f:
        pca = pickle.load(f)
    principle_components = pca.transform(results)

    np.savetxt('data.csv', principle_components, delimiter=',')


if __name__ == '__main__':
    compress()
