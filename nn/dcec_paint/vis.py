import argparse
import logging
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.manifold import TSNE

from DCEC import ClusteringLayer
from datasets import load_christies

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


# Plots to do:
# ----
# t-SNE plot for 3-10 clusters - X
# For top cluster count, plots of t-SNE by epoch - X
# One artist's work on best cluster
# Linear plot of performance by cluster size
# Loss by epoch - X
# For top cluster parameter, 3-5 images per cluster
# Plot 5.5 in Xie et all

DESCRIPTION = "Visualize DCEC paint results"

# Constants
ROOT = "/home/dubs/dev/paap"
BASE_IMG_DIR = os.path.join(ROOT, "img")
DATA_DIR = os.path.join(ROOT, "data")
IMAGES = os.path.join(DATA_DIR, "img/christies/s128/final/")
MODELS_DIR = os.path.join(DATA_DIR, "models")
FINAL_DATASET = os.path.join(DATA_DIR, "output", "christies.ndjson")
ARTWORK_DIR = os.path.join(DATA_DIR, "img", "christies", "s128", "final")

EMBEDDED_LAYER_INDEX = 5


class Plotter(object):

    # PLOTS
    TSNE_ALL = "tsne_all"
    TSNE_FINAL = "tsne_final"
    TSNE_ARTIST = "tsne_artist"
    LOSS = "loss"
    METRICS = "metrics"
    PLOTS = [
        TSNE_ALL,
        TSNE_FINAL,
        TSNE_ARTIST,
        LOSS,
        METRICS
    ]

    def __init__(self, clusters, plots, artist=None):
        for plot in plots:
            if plot not in self.PLOTS:
                raise ValueError("{} not a valid plot: {}".format(plot, self.PLOTS))

        self.plots = plots
        self.artist = artist
        cluster_dir = "n" + str(clusters)
        model_dir = os.path.join(MODELS_DIR, cluster_dir)
        self.result_dir = os.path.join(model_dir, os.listdir(model_dir)[0], "temp")
        if not os.path.exists(self.result_dir):
            raise RuntimeError("{} does not exist".format(self.result_dir))

        self.model_file = os.path.join(self.result_dir, "dcec_model.h5")
        if not os.path.exists(self.model_file):
            raise RuntimeError("{} does not exist".format(self.model_file))

        self.img_dir = os.path.join(BASE_IMG_DIR, cluster_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.log_file = os.path.join(self.result_dir, "dcec_log.csv")

        self.x, _ = load_christies(IMAGES)

        self.model = keras.models.load_model(
            self.model_file,
            custom_objects={"ClusteringLayer": ClusteringLayer}
        )
        self.weight_files = self.get_model_paths()
        self.model.load_weights(self.weight_files[-1])
        # self.cluster_and_embedded_df = self.make_cluster_and_embedded_df()
        # self.christies_df = pd.read_json(FINAL_DATASET, orient="records", lines=True)
        self.image_ids = self.image_file_ids()

        # # Final dataset
        self.final_df = self.make_final_df()

        self.loss_image_filename = os.path.join(self.img_dir, "loss.png")
        self.tsne_image_filename = os.path.join(self.img_dir, "tsne.png")

        self.plot_map = {
            self.TSNE_ALL: None,
            self.TSNE_FINAL: None,
            self.TSNE_ARTIST: None,
            self.LOSS: None,
            self.METRICS: None
        }

    def get_model_paths(self):
        return sorted(
            [os.path.join(self.result_dir, fn) for fn in os.listdir(self.result_dir) if re.search(r"[0-9]\.h5", fn)],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

    def layer_outputs(self):
        """Get k functions at the embedded and clustering layers"""
        # https://stackoverflow.com/a/41712013
        return K.function(
            self.model.layers[0].input,
            [self.model.layers[EMBEDDED_LAYER_INDEX].output, self.model.layers[-2].output]
        )

    # Looping because of memory issues
    def predict(self, k_func):
        """Given a K function and some data, x, get the predictions at the layers.
        This is really a wrapper around k_func([x]) to conserve memory.
        """
        # preallocate some arrays
        predictions = [np.zeros((len(self.x), *p.shape)) for p in k_func([self.x[0:1]])]

        for i in range(len(self.x)):
            p = k_func([self.x[i:(i + 1)]])
            for j in range(len(p)):
                predictions[j][i] = p[j]

        return tuple(predictions)

    @staticmethod
    def layers_to_df(cluster_prop, embedded):
        """Combine np arrays into dataframes"""
        df = pd.DataFrame(np.concatenate(embedded))
        cluster = np.apply_along_axis(lambda x: np.where(x == x.max()), 2, cluster_prop).flatten()
        df["cluster"] = cluster
        # TODO Should think about transforming
        # df = StandardScaler().fit_transform(df)
        return df

    def make_cluster_and_embedded_df(self):
        self.model.load_weights(self.weight_files[-1])
        f = self.layer_outputs()
        embedded, cluster = self.predict(f)
        df = self.layers_to_df(cluster, embedded)
        df["lot_image_id"] = self.image_ids["lot_image_id"]
        return df

    def make_final_df(self):
        # TODO need to test this
        christies = pd.read_json(FINAL_DATASET, orient="records", lines=True)
        print("loaded christies")
        # img_df = pd.DataFrame(np.array(self.image_ids()), columns="lot_image_id")
        # print("loaded_images")
        df = self.image_ids.merge(christies, how="left", on="lot_image_id")
        print("merged images")
        del christies
        cluster_and_embedded = self.make_cluster_and_embedded_df()
        print("made cluster and embedded")
        # df.reset_index(drop=True)
        assert df.shape[0] == cluster_and_embedded.shape[0], "cluster and embedded df shape does not match christies df"
        df = df.merge(cluster_and_embedded, how="left", on="lot_image_id")
        print("merged cluster_and_embedded")
        del cluster_and_embedded
        return df

    @staticmethod
    def tsne(df, dim=2):
        """Create tsne plot"""
        tsne = TSNE(random_state=0, n_components=dim)
        tsne_results = tsne.fit_transform(df)
        return pd.DataFrame(tsne_results, columns=["tsne" + str(i) for i in range(dim)])

    def plot_tsne(self, tsne_results, df):
        fig, ax = plt.subplots()
        ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=df.cluster, s=1)
        fig.savefig(self.tsne_image_filename)
        plt.close(fig)

    # def plot_3d_tsne(self, tsne_results, df):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], tsne_results["tsne2"], c=df.cluster, s=1)
    #     fig.savefig(fn)
    #     plt.close(fig)

    def plot_tsne_model(self, model, weight_file, fn):
        model.load_weights(weight_file)
        f = self.layer_outputs(model)
        embedded, cluster = self.predict(f, self.x)
        df = self.layers_to_df(cluster, embedded)
        tsne_results = self.tsne(df)
        self.plot_tsne(tsne_results, df, fn)

    def plot_tsne_by_time(self, model, weight_files, x):
        for i, fn in enumerate(weight_files[::3]):
            epoch = int(fn.split("_")[-1].split(".")[0])
            img_name = os.path.join(self.img_dir, "tsne_{}.png".format(epoch))
            print("plotting {} of {} - {}".format(i, len(weight_files), img_name))
            self.plot_tsne_model(model, fn, x, img_name)

    def plot_loss(self):
        df = pd.read_csv(self.log_file)
        # df = pd.read_csv("/home/dubs/Downloads/l.csv")
        # df = pd.read_csv("/home/dubs/dev/DCEC/results/temp/dcec_log.csv")
        df = df.drop(["acc", "nmi", "ari"], axis=1)
        fig, ax = plt.subplots()
        ax.scatter(df["iter"], df["L"], c="red", s=1)
        ax.scatter(df["iter"], df["Lc"] * 0.9, c="green", s=1)
        ax.scatter(df["iter"], df["Lr"] * 0.1, c="blue", s=1)
        fig.savefig(self.loss_image_filename)
        plt.close(fig)

    @staticmethod
    def image_file_ids():
        a = np.array(
            [f.split(".")[0] for f in os.listdir(ARTWORK_DIR) if f.endswith(".jpg")]
        )
        return pd.DataFrame(np.array(a), columns=["lot_image_id"])

    def images_with_metadata(self, metadata):
        image_ids = self.image_file_ids()
        img_df = pd.DataFrame(np.array(image_ids), columns=["lot_image_id"])
        df = img_df.merge(metadata, how="left", on="lot_image_id")
        return df

    def plot_tsne_by_artist(self, model, weight_file, df, x, artist="andy warhol"):
        model.load_weights(weight_file)
        f = self.layer_outputs(model)
        embedded, cluster = self.predict(f, x)
        layer_df = self.layers_to_df(cluster, embedded)
        tsne_results = self.tsne(layer_df)

        d = self.images_with_metadata(df[["lot_image_id", "lot_description"]])
        color = d["lot_description"] == artist

        fn = os.path.join(self.img_dir, "tsne_{}.png".format(artist.replace(" ", "_")))
        fig, ax = plt.subplots()
        ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=color, s=1)
        fig.savefig(fn)

    def plot_metrics(self):
        """Plots the silhouette coefficient and Calinski-Harabasz index"""
        pass

    def plot(self):
        pass


# def get_model_paths(d=RESULT_DIR):
#     return sorted(
#         [os.path.join(RESULT_DIR, fn) for fn in os.listdir(RESULT_DIR) if re.search(r"[0-9]\.h5", fn)],
#         key=lambda x: int(x.split("_")[-1].split(".")[0])
#     )


# def layer_outputs(model, layer=EMBEDDED_LAYER_INDEX):
#     """Get k functions at the embedded and clustering layers"""
#     # https://stackoverflow.com/a/41712013
#     return K.function(
#         model.layers[0].input,
#         [model.layers[layer].output, model.layers[-2].output]
#     )


# Looping because of memory issues
# def predict(k_func, x):
#     """Given a K function and some data, x, get the predictions at the layers.
#     This is really a wrapper around k_func([x]) to conserve memory.
#     """
#     # preallocate some arrays
#     predictions = [np.zeros((len(x), *p.shape)) for p in k_func([x[0:1]])]
# 
#     for i in range(len(x)):
#         p = k_func([x[i:(i + 1)]])
#         for j in range(len(p)):
#             predictions[j][i] = p[j]
# 
#     return tuple(predictions)


# def load_final_dataset():
#     return pd.read_json(FINAL_DATASET, orient="records", lines=True)


# def layers_to_df(cluster_prop, embedded):
#     """Combine np arrays into dataframes"""
#     df = pd.DataFrame(np.concatenate(embedded))
#     cluster = np.apply_along_axis(lambda x: np.where(x == x.max()), 2, cluster_prop).flatten()
#     df["cluster"] = cluster
#     # TODO Should think about transforming
#     # df = StandardScaler().fit_transform(df)
#     return df


# def tsne(df, dim=2):
#     """Create tsne plot"""
#     tsne = TSNE(random_state=0, n_components=dim)
#     tsne_results = tsne.fit_transform(df)
#     return pd.DataFrame(tsne_results, columns=["tsne" + str(i) for i in range(dim)])
# 
# 
# def plot_tsne(tsne_results, df, fn=None):
#     if fn is None:
#         fn = os.path.join(IMG_DIR, "tsne.png")
#     fig, ax = plt.subplots()
#     ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=df.cluster, s=1)
#     fig.savefig(fn)
# 
# 
# def plot_3d_tsne(tsne_results, df, fn=None):
#     if fn is None:
#         fn = os.path.join(IMG_DIR, "tsne_3d.png")
# 
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], tsne_results["tsne2"], c=df.cluster, s=1)
#     fig.savefig(fn)
#     plt.close(fig)
# 
# 
# def plot_tsne_model(model, weight_file, x, fn):
#     model.load_weights(weight_file)
#     f = layer_outputs(model)
#     embedded, cluster = predict(f, x)
#     df = layers_to_df(cluster, embedded)
#     tsne_results = tsne(df)
#     plot_tsne(tsne_results, df, fn)
# 
# 
# def plot_tsne_by_time(model, weight_files, x):
#     for i, fn in enumerate(weight_files[::3]):
#         epoch = int(fn.split("_")[-1].split(".")[0])
#         img_name = os.path.join(IMG_DIR, "tsne_{}.png".format(epoch))
#         print("plotting {} of {} - {}".format(i, len(weight_files), img_name))
#         plot_tsne_model(model, fn, x, img_name)
# 
# 
# def plot_loss(fn=None):
#     if fn is None:
#         fn = os.path.join(IMG_DIR, "loss.png")
# 
#     df = pd.read_csv(LOG_FILE)
#     # df = pd.read_csv("/home/dubs/Downloads/l.csv")
#     # df = pd.read_csv("/home/dubs/dev/DCEC/results/temp/dcec_log.csv")
#     df = df.drop(["acc", "nmi", "ari"], axis=1)
#     fig, ax = plt.subplots()
#     ax.scatter(df["iter"], df["L"], c="red", s=1)
#     ax.scatter(df["iter"], df["Lc"] * 0.9, c="green", s=1)
#     ax.scatter(df["iter"], df["Lr"] * 0.1, c="blue", s=1)
#     fig.savefig(fn)
#     plt.close(fig)
# 
# 
# def image_file_ids():
#     return np.array(
#         [f.split(".")[0] for f in os.listdir(ARTWORK_DIR) if f.endswith(".jpg")]
#     )
# 
# 
# def images_with_metadata(metadata):
#     image_ids = image_file_ids()
#     img_df = pd.DataFrame(np.array(image_ids), columns=["lot_image_id"])
#     df = img_df.merge(metadata, how="left", on="lot_image_id")
#     return df
# 
# 
# def plot_tsne_by_artist(model, weight_file, df, x, artist="andy warhol"):
#     model.load_weights(weight_file)
#     f = layer_outputs(model)
#     embedded, cluster = predict(f, x)
#     layer_df = layers_to_df(cluster, embedded)
#     tsne_results = tsne(layer_df)
# 
#     d = images_with_metadata(df[["lot_image_id", "lot_description"]])
#     color = d["lot_description"] == artist
# 
#     fn = os.path.join(IMG_DIR, "tsne_{}.png".format(artist.replace(" ", "_")))
#     fig, ax = plt.subplots()
#     ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=color, s=1)
#     fig.savefig(fn)
# 
# 
# def plot_metrics():
#     """Plots the silhouette coefficient and Calinski-Harabasz index"""
#     pass

"""
def main(plots, artist):
    if LOSS in plots:
        plot_loss()

    if METRICS in plots:
        plot_metrics()

    # load model
    model = keras.models.load_model(
        FULL_MODEL,
        custom_objects={"ClusteringLayer": ClusteringLayer}
    )
    model.load_weights(WEIGHTS[-1])

    # load data
    x, _ = load_christies(IMAGES)
    print("Loaded", len(x), "images")

    df = load_final_dataset()

    # Get clusters on df
    full = images_with_metadata(df)
    f = layer_outputs(model)
    embedded, cluster = predict(f, x)
    results_df = layers_to_df(cluster, embedded)
    full["cluster"] = results_df["cluster"]

    import shutil
    for i in [0, 1, 2, 3, 4]:
        if not os.path.exists("/tmp/n" + str(i)):
            os.makedirs("/tmp/n" + str(i))
        short = full[full["cluster"] == i]
        for img in list(short["lot_image_id"]):
            fn = os.path.join(ARTWORK_DIR, img + ".jpg")
            shutil.copyfile(fn, "/tmp/n" + str(i) + "/" + img + ".jpg")

    exit(0)

    # TODO need to confirm that images are correct here
    # i.e. nth image in x is identical to the image id'd in the nth row of "full"

    if TSNE_ARTIST in plots:
        print("plotting artist")
        plot_tsne_by_artist(model, WEIGHTS[-1], df, x, artist)

    # plot_3d_tsne(tsne_results, df)
    # Plot a series of tsne models
    if TSNE_ALL in plots:
        plot_tsne_by_time(model, WEIGHTS, x)

    # Plot the final model
    if TSNE_FINAL in plots:
        plot_tsne_model(
            model,
            WEIGHTS[-1],
            x,
            os.path.join(IMG_DIR, "tsne_{}.png".format(WEIGHTS[-3].split("_")[-1].split(".")[0]))
        )
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("n", help="number of clusters")
    parser.add_argument("--plots", nargs="*", help="which plots to create", default=Plotter.PLOTS)
    parser.add_argument("--artist", help="artist for tsne plot", default="andy warhol")
    args = parser.parse_args()

    plotter = Plotter(args.n, args.plots, args.artist)
    print(plotter.final_df)
    # plotter.plot()

    # cluster_dir = "n" + args.n
    # model_dir = os.path.join(MODELS_DIR, cluster_dir)
    # RESULT_DIR = os.path.join(model_dir, os.listdir(model_dir)[0], "temp")
    # if not os.path.exists(RESULT_DIR):
    #     print("{} does not exist".format(RESULT_DIR))
    #     exit(2)

    # FULL_MODEL = os.path.join(RESULT_DIR, "dcec_model.h5")
    # if not os.path.exists(FULL_MODEL):
    #     print("{} does not exist".format(FULL_MODEL))
    #     exit(2)

    # IMG_DIR = os.path.join(BASE_IMG_DIR, cluster_dir)
    # if not os.path.exists(IMG_DIR):
    #     os.makedirs(IMG_DIR)

    # LOG_FILE = os.path.join(RESULT_DIR, "dcec_log.csv")

    # WEIGHTS = get_model_paths()

    # main(args.plots, args.artist)
