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

IMG_WIDTH, IMG_HEIGHT = 64, 64


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


def get_image(parameters):
    data = np.loadtxt('data.csv', delimiter=',')

    for i in range(len(parameters)):
        data[i] = parameters[i]

    with open("pca.txt", "rb") as f:
        pca = pickle.load(f)

    img = pca.inverse_transform(data)
    img = img*255
    img = np.reshape(img, [IMG_WIDTH, IMG_HEIGHT])
    pil_img = Image.fromarray(np.uint8(img))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return new_image_string


if __name__ == '__main__':
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")

