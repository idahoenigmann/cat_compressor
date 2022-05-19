import tensorflow as tf
import keras
import pathlib
import atexit
import os.path

NEW_MODEL = False
LOOP = True

origin = 'file:///home/ida/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH, IMG_HEIGHT = 160, 120
BATCH_SIZE = 1
EPOCHS = 1
REDUCED_SIZE = 128

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def exit_handler():
    model.save('cat_faces.h5')


def main():
    global model

    if NEW_MODEL or (not os.path.isfile('cat_faces.h5')):
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
            keras.layers.Dense(16 * 12 * 32, name="decompress_1"),
            keras.layers.Reshape([16, 12, 32], name="decompress_2"),

            keras.layers.UpSampling2D(size=(5, 5), data_format='channels_last', name="decompress_3"),
            keras.layers.Conv2D(16, kernel_size=5, strides=(2, 2), padding='same', activation=keras.activations.relu,
                                name="decompress_4"),
            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last', name="decompress_5"),
            keras.layers.Conv2D(8, kernel_size=3, strides=(1, 1), padding='same', activation=keras.activations.relu,
                                name="decompress_6"),
            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last', name="decompress_7"),
            keras.layers.Conv2D(3, kernel_size=3, strides=(1, 1), padding='same', activation=keras.activations.relu,
                                name="decompress_8"),
        ])
    else:
        print("Using trained model 'cat_faces.h5'!")
        model = keras.models.load_model('cat_faces.h5')

    print(model.summary())

    model.compile(metrics=[keras.metrics.mean_absolute_error],
                  loss=keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True),
                  run_eagerly=True)

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

    while True:
        model.fit(train_data_gen, steps_per_epoch=cnt_files // BATCH_SIZE, epochs=EPOCHS)
        model.save('cat_faces.h5')

        if not LOOP:
            break


if __name__ == '__main__':
    atexit.register(exit_handler)
    main()
