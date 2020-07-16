import csv
import logging
import os
import subprocess
from time import time
from typing import Tuple, Iterable

import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.engine.topology import InputSpec, Layer
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans

import metrics
from ConvAE import CAE
from datasets import load_mnist, load_usps, load_photos_and_prints


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


DESCRIPTION = """
DCEC-Paint implementation adapted from DCEC by Guo et al., 2017. See Castellano and Vessio, 2020 for more information.
The network is a Deep Convolutional Auto Encoder for clustering artwork images with a loss function that is jointly
optimized to minimize reporduction error and clustering loss.
""".strip()


class Defaults(object):
    DATASET_PHOTOS_AND_PRINTS = "photos_and_prints"
    DATASET_MNIST = "mnist"
    NUM_CLUSTERS = 10
    BATCH_SIZE = 256
    MAXITER = 2e4
    GAMMA = 0.1
    UPDATE_INTERVAL = 140
    TOL = 0.001
    CAE_WEIGHTS = None
    SAVE_DIR = "results/temp"
    # 128x128x3 (raw input) -> 64x64x32 (conv1) -> 32x32x64 (conv2) -> 16x16x128 (conv3) -> 32768 (flatten) ->
    # 32 (fully connected) -> (mirrored decoder)
    CONVOLUTIONAL_FILTERS = [32, 64, 128, 32]  # Final is the fully connected clustering layer
    ALPHA = 1.0


def save_model_to_gcs(src="results/temp", dst="gs://paap/nn/dcec_paint/results/"):
    cmd = ["gsutil", "-m", "cp", "-r", src, dst]
    p = subprocess.run(cmd)
    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        logger.exception("Failed to write to gcs")


# TODO This will become the Prediction layer
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            name="clusters", shape=(self.n_clusters, input_dim), initializer="glorot_uniform"
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            # TODO Why delete?
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {"n_clusters": self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# TODO need to rescale in here
class DCEC(object):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 filters: Iterable[int] = Defaults.CONVOLUTIONAL_FILTERS,
                 n_clusters: int = Defaults.NUM_CLUSTERS,
                 alpha: int = Defaults.ALPHA):
        """DCEC Model

        :param input_shape: Shape of the input layer in the model
        :param filters: Number of filters in the convolutional layers, plus the size of the clustering layer. Hence the
            length should equal len(convolutional layers) + 1.
        :param n_clusters: k, the number of clusters to target
        # TODO Do we need this parameter?
        :param alpha: parameter in Student's t distribution
        """
        # TODO Add activation as a parameter to this model
        super(DCEC, self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name="embedding").output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(hidden)
        self.model = Model(inputs=self.cae.input, outputs=[clustering_layer, self.cae.output])

    # TODO we should really be training for 200 epochs
    def pretrain(self, x, batch_size=256, epochs=200, optimizer="adam", save_dir="results/temp"):
        logger.info("...Pretraining...")
        self.cae.compile(optimizer=optimizer, loss="mse")
        from keras.callbacks import CSVLogger

        csv_logger = CSVLogger(args.save_dir + "/pretrain_log.csv")

        # TODO SAVE TO GCS (intermediate)
        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        logger.info("Pretraining time: {}".format(time() - t0))
        self.cae.save(save_dir + "/pretrain_cae_model.h5")
        logger.info("Pretrained weights are saved to %s/pretrain_cae_model.h5" % save_dir)
        save_model_to_gcs(save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=["kld", "mse"], loss_weights=[1, 1], optimizer="adam"):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(
        self,
        x,
        y=None,
        batch_size=256,
        maxiter=2e4,
        tol=1e-3,
        update_interval=140,
        cae_weights=None,
        save_dir="./results/temp",
    ):

        logger.info("Update interval {}".format(update_interval))
        save_interval = x.shape[0] / batch_size * 5
        logger.info("Save interval {}".format(save_interval))

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            logger.info("...pretraining CAE using default hyper-parameters:")
            logger.info("   optimizer='adam';   epochs=200")
            self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            logger.info("cae_weights is loaded successfully.")

        # TODO Will I need some way to initialize the predictions?
        # Step 2: initialize cluster centers using k-means
        t1 = time()
        logger.info("Initializing cluster centers with k-means.")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + "/dcec_log.csv", "w")
        logwriter = csv.DictWriter(logfile, fieldnames=["iter", "acc", "nmi", "ari", "L", "Lc", "Lr"])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    logger.info("Iter {}: Acc {}, nmi {}, ari {}; loss={}".format(ite, acc, nmi, ari, loss))

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    logger.info("delta_label {} < tol {}".format(delta_label, tol))
                    logger.info("Reached tolerance threshold. Stopping training.")
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size : :], y=[p[index * batch_size : :], x[index * batch_size : :]]
                )
                index = 0
            else:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size : (index + 1) * batch_size],
                    y=[
                        p[index * batch_size : (index + 1) * batch_size],
                        x[index * batch_size : (index + 1) * batch_size],
                    ],
                )
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                logger.info("saving model to: {}".format(save_dir + "/dcec_model_" + str(ite) + ".h5"))
                self.model.save_weights(save_dir + "/dcec_model_" + str(ite) + ".h5")

            ite += 1

        # save the trained model
        logfile.close()
        logger.info("saving model to: {}".format(save_dir + "/dcec_model_final.h5"))
        self.model.save_weights(save_dir + "/dcec_model_final.h5")
        t3 = time()
        logger.info("Pretrain time:   {}".format(t1 - t0))
        logger.info("Clustering time: {}".format(t3 - t1))
        logger.info("Total time:      {}".format(t3 - t0))


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("dataset", default=Defaults.DATASET_PHOTOS_AND_PRINTS, help="Dataset to run on, defaults to {}".format(Defaults.DATASET_PHOTOS_AND_PRINTS))
    parser.add_argument("--dataset-path", default="./data/photos_and_prints")
    parser.add_argument("--n-clusters", default=Defaults.NUM_CLUSTERS, type=int, help="Final number of clusters, k, defaults to {}".format(Defaults.NUM_CLUSTERS))
    parser.add_argument("--batch-size", default=Defaults.BATCH_SIZE, type=int, help="Training batch size, defaults to {}".format(Defaults.BATCH_SIZE))
    parser.add_argument("--maxiter", default=Defaults.MAXITER, type=int, help="defaults to {}".format(Defaults.MAXITER))
    parser.add_argument("--gamma", default=Defaults.GAMMA, type=float, help="coefficient of clustering loss, defaults to {}".format(Defaults.GAMMA))
    parser.add_argument("--update-interval", default=Defaults.UPDATE_INTERVAL, type=int, help="defaults to {}".format(Defaults.UPDATE_INTERVAL))
    parser.add_argument("--tol", default=Defaults.TOL, type=float, help="defaults to {}".format(Defaults.TOL))
    parser.add_argument("--cae-weights", default=Defaults.CAE_WEIGHTS, help="This argument must be given, defaults to {}".format(Defaults.CAE_WEIGHTS))
    parser.add_argument("--save-dir", default=Defaults.SAVE_DIR, help="defaults to {}".format(Defaults.SAVE_DIR))
    parser.add_argument('--assert-gpu', action="store_true")
    args = parser.parse_args()
    logger.info(args)

    # Make sure we actually have a GPU if we want one
    logger.info("Num GPUs Available: {}".format(len(tf.config.experimental.list_physical_devices('GPU'))))
    devices = device_lib.list_local_devices()
    print(devices)
    if args.assert_gpu:
        device_types = {d.device_type for d in devices}
        assert "GPU" in device_types, "No GPU found in devices"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # TODO Load the christies dataset - figure out what this is returning
    # load dataset

    if args.dataset == "mnist":
        x, y = load_mnist()
    elif args.dataset == "usps":
        x, y = load_usps("data/usps")
    elif args.dataset == "mnist-test":
        x, y = load_mnist()
        x, y = x[60000:], y[60000:]
    elif args.dataset == "photos_and_prints":
        x, y = load_photos_and_prints(args.dataset_path)

    # TODO Update filters to match what DCEC-Paint has
    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=Defaults.CONVOLUTIONAL_FILTERS, n_clusters=args.n_clusters)
    plot_model(dcec.model, to_file=args.save_dir + "/dcec_model.png", show_shapes=True)
    dcec.model.summary()

    # TODO Parameterize these things
    # begin clustering.
    optimizer = "adam"
    dcec.compile(loss=["kld", "mse"], loss_weights=[args.gamma, 1], optimizer=optimizer)

    dcec.fit(
        x,
        y=y,
        tol=args.tol,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        save_dir=args.save_dir,
        cae_weights=args.cae_weights,
    )
    y_pred = dcec.y_pred
    logger.info(
        "acc = %.4f, nmi = %.4f, ari = %.4f" % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred))
    )
