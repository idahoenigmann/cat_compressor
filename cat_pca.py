import pathlib
import os.path
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA
import time
import cv2

TRAIN_DIR = '/home/ida/.keras/datasets/cat_faces/train/train/'
VAL_DIR = '/home/ida/.keras/datasets/cat_faces/validation/validation/'
fname = 'cat_faces'
PCA_SIZE = 478
CNT_TRAIN_IMG = len([name for name in os.listdir(TRAIN_DIR) if name.endswith('.jpg')])
CNT_VAL_IMG = len([name for name in os.listdir(VAL_DIR) if name.endswith('.jpg')])
IMG_WIDTH = 64
IMG_HEIGHT = 64


def get_data(dir):
    data = []
    for file_name in os.listdir(dir):
        if not (os.path.isfile(os.path.join(dir, file_name)) and file_name.endswith('.jpg')):
            continue
        path = os.path.join(dir, file_name)

        img = mpimg.imread(path)
        img = cv2.resize(img, dsize=(IMG_WIDTH, IMG_HEIGHT))
        data.append(img)
    return data


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    print(f"found {CNT_TRAIN_IMG} training images")
    print(f"found {CNT_VAL_IMG} validation images")

    # initialize arrays
    reconstructed_data = []
    train_data = get_data(TRAIN_DIR)
    val_data = get_data(VAL_DIR)

    # flatten matrix
    train_data = np.reshape(train_data, [CNT_TRAIN_IMG, IMG_WIDTH * IMG_HEIGHT])
    val_data = np.reshape(val_data, [CNT_VAL_IMG, IMG_WIDTH * IMG_HEIGHT])

    pca = PCA(n_components=PCA_SIZE)
    pca.fit(train_data)

    compressed_data = pca.transform(val_data)

    """min = np.min(compressed_data)        TODO
    max = np.max(compressed_data)
    compressed_data = np.random.random(compressed_data.shape)
    compressed_data = (max - min) * compressed_data + min"""

    reconstructed_data = pca.inverse_transform(compressed_data).astype(int)
    reconstructed_data = np.round(reconstructed_data)

    # visualize
    empty_fig = np.ones([IMG_WIDTH, IMG_HEIGHT])
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=1)

    org_img_sp = axes[0].imshow(empty_fig, cmap="grey", vmin=0, vmax=255)
    new_img_sp = axes[1].imshow(empty_fig, cmap="grey", vmin=0, vmax=255)

    val_data = np.reshape(val_data, [CNT_VAL_IMG, IMG_WIDTH, IMG_HEIGHT])
    reconstructed_data = np.reshape(reconstructed_data, [CNT_VAL_IMG, IMG_WIDTH, IMG_HEIGHT])

    for i in range(CNT_VAL_IMG):
        original_img = val_data[i, :, :]
        reconstructed_img = reconstructed_data[i, :, :]

        org_img_sp.set_data(original_img)
        new_img_sp.set_data(reconstructed_img)

        plt.draw()
        plt.pause(1)
