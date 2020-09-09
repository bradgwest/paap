import argparse
import logging
import math
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import gap
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
ARTIST_DATASET = os.path.join(DATA_DIR, "output", "artists.csv")
ARTWORK_DIR = os.path.join(DATA_DIR, "img", "christies", "s128", "final")

EMBEDDED_LAYER_INDEX = 5


artists = pd.read_csv(ARTIST_DATASET)


# def check_order(lot_image_ids, training_samples):
#     """Check if the order of the final dataset and the order of the training samples are equivalent"""
#     i = 0
#     for iid, x in zip(lot_image_ids, training_samples):
#         img_path = os.path.join(final_dir, iid + ".jpg")
#         img = io.imread(img_path) / 255.0
#         assert np.array_equal(x, img), "{} Don't equal eachother".format(i)
#         del img
#         i += 1
def centroid(points):
    """From a list of n, equal length, iterables (points), calculate the centroid"""
    return [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))]


def distance(x, y):
    """Get euclidean distance"""
    assert len(x) == len(y)
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


# ("joan miro", "ansel adams", "hiroshi sugimoto")
# ("fauvism", "pop art", "guohou")
def keywords_in_column(keywords, column):
    def _f(x):
        # pandas is an unbelivably horrible API. Truly horrific.
        # It bears almost zero resemblance to actual python. My dear god
        if type(x) != str:
            return "other"

        for keyword in keywords:
            if keyword in x:
                return keyword
        else:
            return "other"

    def f(df):
        return df[column].apply(_f)

    return f


class Plotter(object):

    # PLOTS
    TSNE_TIME = "tsne_time"
    TSNE_FINAL = "tsne_final"
    # TSNE_GROUP = "tsne_group"
    TSNE_ARTIST = "tsne_artist"
    TSNE_GENRE = "tsne_genre"
    TSNE_PHOTO = "tsne_photo"
    LOSS = "loss"
    METRICS = "metrics"
    GAP = "gap"
    KMEANS_METRICS = "kmeans_metrics"
    CENTER_IMAGES = "center_images"
    FINAL_DATASET = "final_dataset"
    PLOTS = [
        TSNE_TIME,
        TSNE_FINAL,
        TSNE_ARTIST,
        TSNE_GENRE,
        TSNE_PHOTO,
        LOSS,
        METRICS,
        GAP,
        KMEANS_METRICS,
        CENTER_IMAGES,
        FINAL_DATASET,
    ]

    def __init__(self, clusters, no_cache):
        self.clusters = int(clusters)
        cluster_dir = str(clusters)
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

        self.final_df_path = os.path.join(self.img_dir, "df.ndjson")
        if no_cache or not os.path.exists(self.final_df_path):
            # Final dataset
            self.final_df = self.make_final_df()
            centers = self.calculate_cluster_centers()
            self.add_distance_from_cluster_center(centers)
        else:
            self.final_df = pd.read_json(self.final_df_path, orient="records", lines=True)
        # print(self.final_df.columns)

        # Merge the artist dataset
        self.final_df = self.final_df.merge(artists, how="left", left_on="lot_description", right_on="artist")

        self.kmeans_df = self.make_cluster_and_embedded_df(self.weight_files[0])

        self.loss_image_filename = os.path.join(self.img_dir, "loss.png")
        self.tsne_image_filename = os.path.join(self.img_dir, "tsne.png")

        self.plot_map = {
            self.TSNE_TIME: self.plot_tsne_by_time,
            self.TSNE_FINAL: self.plot_tsne,
            # self.TSNE_GROUP: self.plot_tsne_by_group,
            self.TSNE_ARTIST: self.plot_by_artist,
            self.TSNE_GENRE: self.plot_by_genre,
            self.TSNE_PHOTO: self.plot_by_medium,
            self.LOSS: self.plot_loss,
            self.METRICS: self.plot_metrics,
            self.GAP: self.plot_gap,
            self.KMEANS_METRICS: self.plot_kmeans_metrics,
            self.CENTER_IMAGES: self.plot_closest_images,
            self.FINAL_DATASET: self.plot_final_dataset,
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

    def make_cluster_and_embedded_df(self, wf=None):
        if wf is None:
            wf = self.weight_files[-1]
        self.model.load_weights(wf)
        f = self.layer_outputs()
        embedded, cluster = self.predict(f)
        df = self.layers_to_df(cluster, embedded)
        df["lot_image_id"] = self.image_ids["lot_image_id"]
        return df

    def embedded_df(self):
        return self.final_df[[i for i in range(32)]]

    def embedded_and_cluster_df(self):
        return self.final_df[[i for i in range(32)] + ["cluster"]]

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

    # def plot_3d_tsne(self, tsne_results, df):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], tsne_results["tsne2"], c=df.cluster, s=1)
    #     fig.savefig(fn)
    #     plt.close(fig)

    def tsne_results(self, dim=2, wf=None):
        if wf is None:
            wf = self.weight_files[-1]

        self.model.load_weights(wf)
        f = self.layer_outputs()
        embedded, cluster = self.predict(f)
        df = self.layers_to_df(cluster, embedded)
        return self.tsne(df, dim)

    def plot_tsne(self, fn=None, wf=None, *args, **kwargs):
        if fn is None:
            fn = self.tsne_image_filename

        if wf is None:
            wf = self.weight_files[-1]

        is_3d = kwargs.get("dim", False)
        tsne_results = self.tsne_results(dim=3 if is_3d else 2, wf=wf)

        fig = plt.figure()
        if not is_3d:
            ax = fig.add_subplot()
            ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=self.final_df["cluster"], s=1)
        else:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], tsne_results["tsne2"], c=self.final_df["cluster"], s=1)

        fig.savefig(fn)
        plt.close(fig)

    def _plot_tsne_by_group(self, group_fnc, base_fn):
        """group_fn takes a dataset and returns a series"""
        tsne_results = self.tsne_results(dim=2)
        tsne_results["group"] = group_fnc(self.final_df)

        base_fn = base_fn if base_fn.endswith(".png") else base_fn + ".png"
        fn = os.path.join(self.img_dir, base_fn)

        groups = tsne_results.groupby("group")

        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.tsne0, group.tsne1, marker='o', linestyle='', ms=1, label=name)
        ax.legend()
        # ax.scatter(tsne_results['tsne0'], tsne_results['tsne1'], c=color, s=1)
        fig.savefig(fn)

    def plot_by_genre(self, *args, **kwargs):
        genres = ("fauvism", "pop art", "guohuo")
        column = "movement"
        f = keywords_in_column(genres, column)
        self._plot_tsne_by_group(f, "tsne_genre.png")

    def plot_by_artist(self, *args, **kwargs):
        artists = ("joan miro", "henri cartier-bresson", "ansel adams", "hiroshi sugimoto")
        column = "lot_description"
        f = keywords_in_column(artists, column)
        self._plot_tsne_by_group(f, "tsne_artists.png")

    def plot_by_medium(self, *args, **kwargs):
        def f(df):
            return df["is_photograph"]
        self._plot_tsne_by_group(f, "tsne_photograph.png")

    def plot_tsne_by_time(self, *args, **kwargs):
        # Plot every 3rd weight file
        for i, wf in enumerate(self.weight_files[::3]):
            print(wf)
            epoch = int(wf.split("_")[-1].split(".")[0])
            img_name = os.path.join(self.img_dir, "tsne_{}.png".format(epoch))
            print("plotting {} of {} - {}".format(i, len(self.weight_files), img_name))
            self.plot_tsne(fn=img_name, wf=wf)

    def plot_loss(self, *args, **kwargs):
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

    def silhouette_coefficient(self):
        return silhouette_score(self.embedded_df(), self.final_df["cluster"])

    def calinski_harabasz(self):
        """This will need to be normalized to the largest value"""
        return calinski_harabasz_score(self.embedded_df(), self.final_df["cluster"])

    def plot_metrics(self, *args, **kwargs):
        """Plots the silhouette coefficient and Calinski-Harabasz index"""
        ss = self.silhouette_coefficient()
        ch = self.calinski_harabasz()
        print("clusters={}; ss={}; ch={}".format(self.clusters, ss, ch))
        fn = os.path.join(self.img_dir, "metrics.csv")
        with open(fn, "w") as f:
            f.write("k,ss,ch\n")
            f.write("{},{},{}\n".format(self.clusters, ss, ch))

        return self.clusters, ss, ch

    def plot_kmeans_metrics(self, *args, **kwargs):
        embedded = self.kmeans_df[[i for i in range(32)]]
        ss = silhouette_score(embedded, self.kmeans_df["cluster"])
        ch = calinski_harabasz_score(embedded, self.kmeans_df["cluster"])
        print("clusters={}; ss={}; ch={}".format(self.clusters, ss, ch))
        fn = os.path.join(self.img_dir, "kmeans_metrics.csv")
        with open(fn, "w") as f:
            f.write("k,ss,ch\n")
            f.write("{},{},{}\n".format(self.clusters, ss, ch))

        return self.clusters, ss, ch

    def calculate_cluster_centers(self):
        centers = {}
        em = self.embedded_df()
        for i in range(self.clusters):
            points = list(em[self.final_df["cluster"] == i].to_records(index=False))
            centers[i] = centroid(points)
        return centers

    def add_distance_from_cluster_center(self, centers):
        self.final_df["distance_from_centroid"] = self.embedded_and_cluster_df().apply(lambda x: distance(x[:-1], centers[x["cluster"]]), axis=1)

    def sort_by_distance_from_cluster_center(self, n=10):
        return self.final_df.groupby(["cluster"]) \
            .apply(lambda x: x.sort_values(["distance_from_centroid"], ascending=True)) \
            .reset_index(drop=True) \
            .groupby(["cluster"]) \
            .head(n)

    def get_closest_images_from_center(self):
        df = self.sort_by_distance_from_cluster_center()
        df["images_path"] = df.apply(lambda x: os.path.join(ARTWORK_DIR, x["lot_image_id"] + ".jpg"), axis=1)
        return df[["lot_image_id", "images_path", "lot_description", "cluster", "distance_from_centroid"]]

    def plot_closest_images(self, *args, **kwargs):
        df = self.get_closest_images_from_center()
        df.to_csv(os.path.join(self.img_dir, "center.csv"), index=False)

    def plot_final_dataset(self, *args, **kwargs):
        self.final_df.to_json(self.final_dataset, orient="records", lines=True)

    def plot_gap(self):
        if self.clusters != 1:
            raise RuntimeError("GAP only defined for 1 cluster")

        f = self.layer_outputs()
        embedded, _ = self.predict(f)
        embedded2 = np.apply_along_axis(lambda x: x[0], 1, embedded)
        x = list(range(1, 21))
        g = gap.gap(embedded2, ks=x)
        fig, ax = plt.subplots()
        ax.plot(x, g)
        ax.set_title("GAP Statistic on non-clustered dataset")
        ax.set_xlabel("Clusters (k)")
        ax.set_ylabel("GAP statistic")
        ax.set_xticks(x)
        fig.savefig(os.path.join(self.img_dir, "gap.png"))
        plt.close(fig)

    def plot(self, plots, *args, **kwargs):
        for plot in plots:
            if plot not in self.PLOTS:
                raise ValueError("{} not a valid plot: {}".format(plot, self.PLOTS))

        for plot in plots:
            self.plot_map[plot](*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("n", type=int, help="number of clusters")
    parser.add_argument("--plots", nargs="*", help="which plots to create", default=Plotter.PLOTS)
    parser.add_argument("--artist", help="artist for tsne plot", default="andy warhol")
    parser.add_argument("--no-cache", help="don't use cache", action="store_true")
    args = parser.parse_args()

    plotter = Plotter(args.n, args.no_cache)
    plotter.plot(args.plots, artist=args.artist)
