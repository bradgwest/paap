import os

from tensorflow import keras
from keras import backend as K
# import matplotlib.pyplot as plt

from DCEC import ClusteringLayer
from datasets import load_christies


# Plots to do:
# ----
# t-SNE plot for 3-10 clusters
# For top cluster count, plots of t-SNE by epoch
# One artist's work on best cluster
# Linear plot of performance by cluster size
# Loss by epoch
# For top cluster parameter, 3-5 images per cluster
# Plot 5.5 in Xie et all


# Constants
ROOT = "/home/dubs/dev/paap"
IMG_DIR = os.path.join(ROOT, "img")
DATA_DIR = os.path.join(ROOT, "data")
IMAGES = os.path.join(DATA_DIR, "img/christies/s128/final/")
MODELS_DIR = os.path.join(DATA_DIR, "models")

RESULT_DIR = os.path.join(MODELS_DIR, "aug_13", "temp")
MODEL = os.path.join(RESULT_DIR, "dcec_model.h5")
WEIGHTS = os.path.join(RESULT_DIR, "dcec_model_14145.h5")

EMBEDDED_LAYER_INDEX = 5


def layer_outputs(model, layer=EMBEDDED_LAYER_INDEX):
    # https://stackoverflow.com/a/41712013
    return K.function(
        model.layers[0].input,
        [model.layers[layer].output, model.layers[-2].output]
    )


def main():
    # load model and saved weights
    model = keras.models.load_model(
        MODEL,
        custom_objects={"ClusteringLayer": ClusteringLayer}
    )
    model.load_weights(WEIGHTS)

    # load data
    x, _ = load_christies(IMAGES)
    print("Loaded", len(x), "images")

    # get embedded layer output function
    f = layer_outputs(model)

    # Deal with memory issues. I need a new computer
    embedded = []
    cluster = []
    for i in range(len(x)):
        # if i % 100 == 0:
        #     print(i)
        e, c = f([x[(i - 1):i]])
        embedded.append(e)
        cluster.append(c)


if __name__ == "__main__":
    main()
