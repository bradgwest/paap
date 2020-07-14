from typing import Iterable, Tuple

import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential
# from keras.utils.vis_utils import plot_model

RELU = "relu"
ELU = "elu"

# TODO Document these
STRIDE = 2
PAD_SAME = "same"
PAD_VALID = "valid"


def CAE(input_shape: Tuple[int, int, int] = (128, 128, 3),
        filters: Iterable[int] = [32, 64, 128, 32],
        activation: str = ELU):
    """Convolutional Autoencoder

    :param input_shape: Shape of the input layer in the model
    :param filters: Number of filters in the convolutional layers, plus the size of the clustering layer. Hence the
        length should equal len(convolutional layers) + 1.
    :param activation: Activation function to use
    """
    assert len(filters) == 4, "Expected 4 filters, got, {}".format(filters)

    # TODO, I'm not sure why we're mod'ing by 8. https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t  # noqa: E501
    if input_shape[0] % 8 == 0:
        pad3 = PAD_SAME
    else:
        pad3 = PAD_VALID

    # TODO This could be much prettier and cleaner
    layers = [
        Conv2D(filters[0], kernel_size=5, strides=STRIDE, padding=PAD_SAME, activation=ELU, name="conv1", input_shape=input_shape),
        Conv2D(filters[1], kernel_size=5, strides=STRIDE, padding=PAD_SAME, activation=ELU, name="conv2"),
        Conv2D(filters[2], kernel_size=3, strides=STRIDE, padding=pad3, activation=ELU, name="conv3"),
        Flatten(),
        Dense(units=filters[3], name="embedding"),
        Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation=ELU),
        Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])),
        Conv2DTranspose(filters[1], kernel_size=3, strides=STRIDE, padding=pad3, activation=ELU, name="deconv3"),
        Conv2DTranspose(filters[0], kernel_size=5, strides=STRIDE, padding=PAD_SAME, activation=ELU, name="deconv2"),
        Conv2DTranspose(input_shape[2], kernel_size=5, strides=STRIDE, padding=PAD_SAME, name="deconv1")
    ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    model.summary()

    return model


# if __name__ == "__main__":
#     from time import time
# 
#     # setting the hyper parameters
#     import argparse
# 
#     parser = argparse.ArgumentParser(description="train")
#     parser.add_argument("--dataset", default="usps", choices=["mnist", "usps"])
#     parser.add_argument("--n_clusters", default=10, type=int)
#     parser.add_argument("--batch_size", default=256, type=int)
#     parser.add_argument("--epochs", default=200, type=int)
#     parser.add_argument("--save_dir", default="results/temp", type=str)
#     args = parser.parse_args()
#     print(args)
# 
#     import os
# 
#     if not os.path.exists(args.save_dir):
#         os.makedirs(args.save_dir)
# 
#     # load dataset
#     from datasets import load_mnist, load_usps
# 
#     if args.dataset == "mnist":
#         x, y = load_mnist()
#     elif args.dataset == "usps":
#         x, y = load_usps("data/usps")
# 
#     # define the model
#     model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
#     plot_model(model, to_file=args.save_dir + "/%s-pretrain-model.png" % args.dataset, show_shapes=True)
#     model.summary()
# 
#     # compile the model and callbacks
#     optimizer = "adam"
#     model.compile(optimizer=optimizer, loss="mse")
#     from keras.callbacks import CSVLogger
# 
#     csv_logger = CSVLogger(args.save_dir + "/%s-pretrain-log.csv" % args.dataset)
# 
#     # begin training
#     t0 = time()
#     model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, callbacks=[csv_logger])
#     print("Training time: ", time() - t0)
#     model.save(args.save_dir + "/%s-pretrain-model-%d.h5" % (args.dataset, args.epochs))
# 
#     # extract features
#     feature_model = Model(inputs=model.input, outputs=model.get_layer(name="embedding").output)
#     features = feature_model.predict(x)
#     print("feature shape=", features.shape)
# 
#     # use features for clustering
#     from sklearn.cluster import KMeans
# 
#     km = KMeans(n_clusters=args.n_clusters)
# 
#     features = np.reshape(features, newshape=(features.shape[0], -1))
#     pred = km.fit_predict(features)
#     from . import metrics
# 
#     print("acc=", metrics.acc(y, pred), "nmi=", metrics.nmi(y, pred), "ari=", metrics.ari(y, pred))
