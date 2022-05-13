import tensorflow as tf
import keras
import pathlib
import atexit
import os.path

NEW_MODEL = False
LOOP = True

origin = 'file:///home/ida/.keras/datasets/simple_shapes.zip'
fname = 'simple_shapes'
model = keras.models.Sequential()

IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 1
EPOCHS = 1

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def exit_handler():
    model.save('simple_shapes.h5')


def main():
    global model

    if NEW_MODEL or (not os.path.isfile('simple_shapes.h5')):
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
            keras.layers.Dense(40 * 30 * 64),
            keras.layers.Reshape([40, 30, 64]),

            keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last'),
            keras.layers.Conv2D(32, kernel_size=4, strides=(1, 1), padding='same', activation=keras.activations.relu),
            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(16, kernel_size=4, strides=(1, 1), padding='same', activation=keras.activations.relu),
            keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last'),
            keras.layers.Conv2D(8, kernel_size=4, strides=(1, 1), padding='same', activation=keras.activations.relu),

            keras.layers.Conv2D(3, kernel_size=4, strides=(1, 1), padding='same', activation=keras.activations.relu),

        ])
    else:
        print("Using trained model 'simple_shapes.h5'!")
        model = keras.models.load_model('simple_shapes.h5')

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
    cnt_files = len(list(train_dir_path.glob('*.png')))

    img_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = img_generator.flow_from_directory(directory=train_dir, target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                       batch_size=BATCH_SIZE, class_mode="input", shuffle=True)

    while True:
        model.fit(train_data_gen, steps_per_epoch=cnt_files // BATCH_SIZE, epochs=EPOCHS)
        model.save('simple_shapes.h5')

        if not LOOP:
            break


if __name__ == '__main__':
    atexit.register(exit_handler)
    main()
