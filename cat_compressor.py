import tensorflow as tf
import keras
import pathlib
import atexit
import os.path

origin = 'file:///home/sascha/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 5
EPOCHS = 1

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def main():
    entire_model = keras.models.load_model("model_cat_classifier.h5")

    for layer in entire_model.layers:
        print(layer.get_config(), layer.get_weights())

    return
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
    ])

    print(model.summary())

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_error,
                  optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))

    data_dir = tf.keras.utils.get_file(
        origin=origin,
        fname=fname, untar=True)
    data_dir = pathlib.Path(data_dir)

    train_dir = os.path.join(data_dir, 'train')

    train_dir_path = pathlib.Path(os.path.join(train_dir, 'train'))
    cnt_files = len(list(train_dir_path.glob('*.jpg')))

    img_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = img_generator.flow_from_directory(directory=train_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE, class_mode="input", shuffle=True)


if __name__ == '__main__':
    main()
