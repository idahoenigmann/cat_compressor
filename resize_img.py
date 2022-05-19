import os
from PIL import Image


def main(path, new_size):

    for file in os.listdir(path):
        if os.path.isfile(path+file) and file.split(".")[1].lower() in ('jpg', 'jpeg'):
            f, e = os.path.splitext(path + file)
            print(path + file)
            im = Image.open(path+file)
            new_im = im.resize(new_size, Image.ANTIALIAS)
            new_im = new_im.convert('L')
            new_im.save(path+file, 'JPEG', quality=90)


if __name__ == "__main__":
    main("/home/ida/.keras/datasets/cat_faces/train/train/", (320, 240))
