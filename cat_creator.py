import tensorflow as tf
import keras
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from http.server import HTTPServer, SimpleHTTPRequestHandler
import pickle
from cat_compressor import compress

hostName = "localhost"
serverPort = 3000

origin = 'file:///home/ida/.keras/datasets/cat_faces.zip'
fname = 'cat_faces'
model = keras.models.Sequential()

IMG_WIDTH = 320
IMG_HEIGHT = 240

REDUCED_SIZE = 128

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class MyServer(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        if "parseCat=" in self.path:
            if "parseCat=val" in self.path:
                compress("val")
            elif "parseCat=train" in self.path:
                compress("train")
            data = np.loadtxt('data.csv', delimiter=',')
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes(','.join("{:.2f}".format(x) for x in data), "utf-8"))
        else:
            path = self.path.split("=")[-1].replace("%5B", "").replace("%5D", "")
            parameters = []
            for num in path.split(","):
                parameters.append(float(num))

            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes(get_image(parameters), "utf-8"))


def main():
    global model

    entire_model = keras.models.load_model("cat_faces.h5")

    model = keras.Sequential([
        keras.layers.Dense(32 * 24 * 32, input_shape=(128,), name="decompress_1"),
        keras.layers.Reshape([32, 24, 32], name="decompress_2"),

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

    #for layer_idx in range(len(entire_model.layers) - len(model.layers), len(entire_model.layers)):
    for layer_idx in range(-1, -1 - len(model.layers), -1):
        model.layers[layer_idx].set_weights(entire_model.layers[layer_idx].get_weights())

    model.compile(metrics=[keras.metrics.mean_absolute_percentage_error],
                  loss=keras.losses.mean_absolute_error,
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True))


def get_image(parameters):
    data = np.loadtxt('data.csv', delimiter=',')

    for i in range(len(parameters)):
        data[i] = parameters[i]

    with open("pca.txt", "rb") as f:
        pca = pickle.load(f)

    all_components = pca.inverse_transform(data)
    all_components = all_components.reshape([1, REDUCED_SIZE])

    img = model.predict(all_components, steps=1)
    img = np.reshape(img, [IMG_WIDTH, IMG_HEIGHT, 3])
    img *= 255.0
    pil_img = Image.fromarray(np.uint8(img))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return new_image_string


if __name__ == '__main__':
    main()

    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

