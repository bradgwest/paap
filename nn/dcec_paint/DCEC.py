import argparse
import csv
import datetime
import logging
import os
import subprocess
from time import time
from typing import Tuple, Iterable

import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.cluster import KMeans

import metrics
from ConvAE import CAE
from datasets import load_christies

# Memory leak issues
# https://github.com/keras-team/keras/issues/13118
# https://github.com/tensorflow/tensorflow/issues/33030

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

# Clear session on the off chance it helps with memory issues
# K.clear_session()

DESCRIPTION = """
DCEC-Paint implementation adapted from DCEC by Guo et al., 2017. See Castellano and Vessio, 2020 for more information.
The network is a Deep Convolutional Auto Encoder for clustering artwork images with a loss function that is jointly
optimized to minimize reproduction error and clustering loss.
""".strip()

UTC_NOW = datetime.datetime.utcnow().strftime("D%Y%m%dT%H%M%S")
CAE_LOCAL_WEIGHTS = None


def gcs_path():
    return "gs://paap/nn/dcec_paint/results/{}/".format(UTC_NOW)


def save_results_to_gcs(src="results/temp", dst=gcs_path()):
    cmd = ["gsutil", "-m", "cp", "-r", src, dst]
    p = subprocess.run(cmd)
    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        logger.exception("Failed to write to gcs")


def gcs_copy(src, dst=os.path.join(gcs_path(), "temp/")):
    cmd = ["gsutil", "cp", src, dst]
    p = subprocess.run(cmd)
    try:
        p.check_returncode()
    except subprocess.CalledProcessError:
        logger.exception("Failed to write to gcs")


def gpu_info(assert_gpu) -> None:
    # Make sure we actually have a GPU if we want one
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Num GPUs Available: {}".format(len(gpus)))
    devices = device_lib.list_local_devices()
    logger.info(devices)
    if assert_gpu:
        device_types = {d.device_type for d in devices}
        assert "GPU" in device_types, "No GPU found in devices"

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info("{} Physical GPUs, Logical GPUs {}".format(len(gpus), len(logical_gpus)))
        except RuntimeError:
            # Memory growth must be set before GPUs have been initialized
            logger.exception("Failed to get GPUS")


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


class DCEC(object):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 filters: Iterable[int] = [32, 64, 128, 32],
                 n_clusters: int = 32,
                 alpha: int = 1):
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
    # TODO Can we do a bigger batch size here?
    # TODO Should we train for longer?
    def pretrain(self, x, batch_size=512, epochs=200, optimizer="adam", save_dir="results/temp"):
        logger.info("...Pretraining...")
        self.cae.compile(optimizer=optimizer, loss="mse")
        from tensorflow.keras.callbacks import CSVLogger

        csv_logger = CSVLogger(args.save_dir + "/pretrain_log.csv")

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        logger.info("Pretraining time: {}".format(time() - t0))
        self.cae.save(save_dir + "/pretrain_cae_model.h5")
        logger.info("Pretrained weights are saved to %s/pretrain_cae_model.h5" % save_dir)
        save_results_to_gcs(save_dir)
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
        batch_size=512,  # This was 256, Castellano used 128
        maxiter=2e4,
        tol=1e-3,
        update_interval=140,  # Was 140
        cae_weights=None,
        save_dir="./results/temp",
    ):

        logger.info("Update interval {}".format(update_interval))
        save_interval = int(x.shape[0] / batch_size * 5)
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
        logfile_path = save_dir + "/dcec_log.csv"
        logfile = open(logfile_path, "w")
        logwriter = csv.DictWriter(logfile, fieldnames=["iter", "acc", "nmi", "ari", "L", "Lc", "Lr"])
        logwriter.writeheader()

        overall_log_loss = save_dir + "/dcec_log_all.csv"
        l2 = open(overall_log_loss, "w")
        lw2 = csv.DictWriter(l2, fieldnames=["iter", "L", "Lc", "Lr"])
        lw2.writeheader()

        # Convert input to tensor so that we can use different predict function
        x_tf = tf.convert_to_tensor(x)

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                logger.info("Updating. Iter {}".format(ite))
                # q, _ = self.model.predict(x, verbose=0)
                # model.predict() causes a memory leak. So, use model(). See notes above
                q, _ = self.model(x_tf, training=False)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    logger.info("{} calculating acc".format(ite))
                    acc = np.round(metrics.acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                    ari = np.round(metrics.ari(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    logger.info("Iter {}: Acc {}, nmi {}, ari {}; loss={}".format(ite, acc, nmi, ari, loss))

                loss_dict = {"iter": ite, "L": loss[0], "Lc": loss[1], "Lr": loss[2]}
                logwriter.writerow(loss_dict)
                logger.info("iter {i}; L {L}; Lc {Lc}; Lr {Lr}".format(i=ite, **loss_dict))

                logger.info("Evaluating full loss")
                loss_all = self.model.evaluate(x, y=[p, x], batch_size=batch_size, verbose=0)
                ld = {"iter": ite, "L": loss_all[0], "Lc": loss_all[1], "Lr": loss_all[2]}
                logger.info("Overall loss. iter {iter}; L {L}; Lc {Lc}; Lr {Lr}".format(**ld))
                lw2.writerow(ld)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                logger.info("delta_label={}".format(delta_label))
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

            loss_dict = {"iter": ite, "L": loss[0], "Lc": loss[1], "Lr": loss[2]}
            logwriter.writerow(loss_dict)

            if ite % 10 == 0:
                logger.info("iter={};L={};L_c={};L_r={}".format(ite, *loss))

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                logger.info("saving model to: {}".format(save_dir + "/dcec_model_" + str(ite) + ".h5"))
                path = save_dir + "/dcec_model_" + str(ite) + ".h5"
                self.model.save_weights(path)
                gcs_copy(path)
                gcs_copy(logfile_path)
                gcs_copy(overall_log_loss)

            ite += 1

        # save the trained model
        logfile.close()
        lw2.close()
        logger.info("saving model to: {}".format(save_dir + "/dcec_model_final.h5"))
        self.model.save_weights(save_dir + "/dcec_model_final.h5")
        t3 = time()
        logger.info("Pretrain time:   {}".format(t1 - t0))
        logger.info("Clustering time: {}".format(t3 - t1))
        logger.info("Total time:      {}".format(t3 - t0))

        save_results_to_gcs(save_dir)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--dataset-path", default="./data/final", help="Path to data")
    parser.add_argument("--batch-size", default=os.getenv("DCEC_BATCH_SIZE", 512), type=int, help="Training batch size")
    parser.add_argument("--n-clusters", default=os.getenv("DCEC_N_CLUSTERS", 10), type=int, help="Final number of clusters, k")
    parser.add_argument("--learning-rate", default=os.getenv("DCEC_LEARNING_RATE", 0.001), type=float, help="Learning rate")
    parser.add_argument("--maxiter", default=os.getenv("DCEC_MAX_ITER", 20000), type=int, help="Maximum iterations to perform on final training")
    parser.add_argument("--gamma", default=os.getenv("DCEC_GAMMA", 0.9), type=float, help="coefficient of clustering loss")
    parser.add_argument("--update-interval", default=os.getenv("DCEC_UPDATE_INTERVAL", 140), type=int, help="How frequently to update weights")
    parser.add_argument("--tol", default=os.getenv("DCEC_TOLERANCE", 0.001), type=float, help="Threshold at which to stop training")
    parser.add_argument("--cae-weights", default=os.getenv("DCEC_CAE_WEIGHTS"), help="Path to remote weight file containing pretrained weights, or None")
    parser.add_argument("--save-dir", default=os.getenv("DCEC_SAVE_DIR", "./results/temp"), help="Where to save results/model to")
    parser.add_argument("--data-dir", default=os.getenv("DCEC_DATA_DIR", "./data"), help="Where the data reside")
    parser.add_argument('--assert-gpu', default=os.getenv("DCEC_ASSERT_GPU", "true").lower() == "true", action="store_true")
    parser.add_argument('--no-assert-gpu', action="store_false", dest="assert_gpu")
    parser.add_argument("--epochs", default=os.getenv("DCEC_EPOCHS", 200), type=int, help="Number of epochs to train CAE")
    args = parser.parse_args()

    logger.info("Running with config: {}".format(["{}={}".format(k, v) for k, v in vars(args).items()]))

    # Log GPU info, optionally asserting if there isn't one
    gpu_info(args.assert_gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    cfg = ["{}={}\n".format(k, v) for k, v in vars(args).items()]
    path = os.path.join(args.save_dir, "config.txt")
    with open(path, "w") as f:
        f.writelines(cfg)
    gcs_copy(path)

    if args.cae_weights is not None:
        CAE_LOCAL_WEIGHTS = os.path.join(args.save_dir, os.path.basename(args.cae_weights))
        print("copying to {}".format(CAE_LOCAL_WEIGHTS))
        gcs_copy(args.cae_weights, CAE_LOCAL_WEIGHTS)

    x, y = load_christies(args.dataset_path)

    # TODO Update filters to match what DCEC-Paint has
    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 32], n_clusters=args.n_clusters)
    plot_model(dcec.model, to_file=args.save_dir + "/dcec_model.png", show_shapes=True)
    dcec.model.summary()

    # begin clustering.
    optimizer = Adam(learning_rate=args.learning_rate)
    losses = ["kld", "mse"]
    # How Keras accounts for loss_weights - https://stackoverflow.com/a/49406231
    loss_weights = [args.gamma, 1 - args.gamma]

    dcec.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)

    model_path = os.path.join(args.save_dir, "dcec_model.h5")
    dcec.model.save(model_path)
    gcs_copy(model_path)

    # tf.compat.v1.get_default_graph().finalize()

    dcec.fit(
        x,
        y=y,
        batch_size=args.batch_size,
        tol=args.tol,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        save_dir=args.save_dir,
        cae_weights=CAE_LOCAL_WEIGHTS,
    )

    # TODO Y will always be None
    if y is not None:
        y_pred = dcec.y_pred
        logger.info(
            "acc = %.4f, nmi = %.4f, ari = %.4f" % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred))
        )
