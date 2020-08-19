import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import backend as K
from sklearn.manifold import TSNE

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
    """Get k functions at the embedded and clustering layers"""
    # https://stackoverflow.com/a/41712013
    return K.function(
        model.layers[0].input,
        [model.layers[layer].output, model.layers[-2].output]
    )


# Looping because of memory issues
def predict(k_func, x):
    """Given a K function and some data, x, get the predictions at the layers.
    This is really a wrapper around k_func([x]) to conserve memory.
    """
    # preallocate some arrays
    predictions = [np.zeros((len(x), *p.shape)) for p in k_func([x[0:1]])]

    for i in range(len(x)):
        p = k_func([x[i:(i + 1)]])
        for j in range(len(p)):
            predictions[j][i] = p[j]

    return tuple(predictions)


def layers_to_df(cluster_prop, embedded):
    """Combine np arrays into dataframes"""
    df = pd.DataFrame(np.concatenate(embedded))
    cluster = np.apply_along_axis(lambda x: np.where(x == x.max()), 2, cluster_prop).flatten()
    df["cluster"] = cluster
    # TODO Should think about transforming
    # df = StandardScaler().fit_transform(df)
    return df


def tsne(df, dim=2):
    """Create tsne plot"""
    tsne = TSNE(random_state=0, n_components=dim)
    tsne_results = tsne.fit_transform(df)
    return pd.DataFrame(tsne_results, columns=["tsne" + str(i) for i in range(dim)])


def plot_tsne(tsne_results, df, fn=os.path.join(IMG_DIR, "tsne.png")):
    fig, ax = plt.subplots()
    ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=df.cluster, s=1)
    fig.savefig(fn)


def plot_3d_tsne(tsne_results, df, fn=os.path.join(IMG_DIR, "tsne_3d.png")):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], tsne_results["tsne2"], c=df.cluster, s=1)
    fig.savefig(fn)


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
    embedded, cluster = predict(f, x)

    print(embedded.shape)
    print(cluster.shape)

    df = layers_to_df(cluster, embedded)
    tsne_results = tsne(df)

    plot_tsne(tsne_results, df)
    plot_3d_tsne(tsne_results, df)


if __name__ == "__main__":
    main()
