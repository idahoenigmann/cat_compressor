import tensorflow as tf
import keras
import pathlib
import os.path
from sklearn.decomposition import PCA
import pickle

origin = 'file:///home/ida/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH, IMG_HEIGHT = 160, 120
BATCH_SIZE = 1
REDUCED_SIZE = 128
PCA_SIZE = 10

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def calc_pca():
    entire_model = keras.models.load_model("cat_faces.h5")

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=[IMG_WIDTH, IMG_HEIGHT, 3], name="input_1"),
        keras.layers.Conv2D(8, kernel_size=3, strides=(1, 1),
                            data_format='channels_last', padding='same', activation=keras.activations.relu,
                            name="compress_1"),
        keras.layers.Conv2D(16, kernel_size=5, strides=(2, 2), padding='same', activation=keras.activations.relu,
                            data_format='channels_last', name="compress_2"),
        keras.layers.Conv2D(32, kernel_size=10, strides=(5, 5), padding='same', activation=keras.activations.relu,
                            data_format='channels_last', name="compress_3d"),

        keras.layers.Reshape([16 * 12 * 32], name="compress_5"),
        keras.layers.Dense(128, activation=keras.activations.sigmoid, name="compress_6"),
    ])

    for layer_idx in range(len(model.layers)):
        model.layers[layer_idx].set_weights(entire_model.layers[layer_idx].get_weights())

    model.compile(metrics=[keras.metrics.mean_absolute_error],
                  loss=keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))

    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_dir = os.path.join(data_dir, 'train')
    img_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = img_generator.flow_from_directory(directory=train_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE, class_mode="input", shuffle=True)

    train_dir_path = pathlib.Path(os.path.join(train_dir, 'train'))
    cnt_files = len(list(train_dir_path.glob('*.jpg')))

    results = model.predict(train_data_gen, steps=cnt_files // BATCH_SIZE)

    print("all image data calculated")

    pca = PCA(n_components=PCA_SIZE)
    pca.fit(results)

    print("pca finished")

    with open("pca.txt", "wb") as f:
        pickle.dump(pca, f)

    return pca


if __name__ == '__main__':
    calc_pca()
