import os

import numpy as np
from skimage import io


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype("float32")
    x = x / 255.0
    print("MNIST:", x.shape)
    return x, y


def load_usps(data_path="./data/usps"):
    import os

    if not os.path.exists(data_path + "/usps_train.jf"):
        if not os.path.exists(data_path + "/usps_train.jf.gz"):
            os.system("wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_train.jf.gz -P %s" % data_path)
            os.system("wget http://www-i6.informatik.rwth-aachen.de/~keysers/usps_test.jf.gz -P %s" % data_path)
        os.system("gunzip %s/usps_train.jf.gz" % data_path)
        os.system("gunzip %s/usps_test.jf.gz" % data_path)

    with open(data_path + "/usps_train.jf") as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + "/usps_test.jf") as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype("float32")
    x /= 2.0
    x = x.reshape([-1, 16, 16, 1])
    y = np.concatenate((labels_train, labels_test))
    print("USPS samples", x.shape)
    return x, y


def load_photos_and_prints(data_path="./data/photos_and_prints_split/train"):
    # load images
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".jpg")][:7500]
    images_raw = []
    for fp in image_paths:
        images_raw.append(io.imread(fp))
    images = np.array(images_raw)

    # Scale pixel values
    images = images / 255.0
    print("Christies Photos and Prints:", images.shape)

    return images, None


def load_christies(data_path="./data/final/"):
    # load images
    image_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".jpg")]
    images_raw = []
    for fp in image_paths:
        images_raw.append(io.imread(fp))
    images = np.array(images_raw)

    # Scale pixel values
    images = images / 255.0
    print("Christies final dataset:", images.shape)

    return images, None
