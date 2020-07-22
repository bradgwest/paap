import argparse
import csv
import logging
import os
from time import time
from typing import Tuple, Iterable, Optional, NamedTuple

import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.engine.topology import InputSpec, Layer
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential
from keras.callbacks import CSVLogger
from sklearn.cluster import KMeans
from skimage import io

from storage import is_gcs_blob, GCStorage, path_from_uri, tar, untar


DESCRIPTION = """DCEC-Pred implementation adapted from DCEC by Guo et al., 2017 and DCEC-Paint by Castellano and
 Vessio, 2020. The network is a Deep Convolutional Auto Encoder for predicting artwork auction prices with a loss
 function that is jointly optimized to minimize image reproduction error and prediction loss.
""".strip()

# Model constants
# It seems like you might need to set your embedded layer to a much higher number
FILTERS = [32, 64, 128, 32]
OPTIMIZER = "adam"
ALPHA = 1.0
GCS_PREFIX = "nn/dcec_pred"
N_IMAGES = 10000


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG)


gcs_client = None


class ImageData(NamedTuple):
    image: np.ndarray
    price: float


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


def load_images(x_dir: str, y_path: Optional[str] = None, n: Optional[int] = None) -> Tuple[Iterable[np.ndarray], Optional[Iterable[float]]]:
    """Load image data.

    :param x_dir: path to directory of images to load
    :param y_path: path to csv file with 2 columns, no header, first column is image id, pertaining to the suffix-less
        basename of the image path, second is that image's price. If None, no prediction will be returned.
    :param n: number of images to return, a subset of the images in the directory. If None, all images returned
    """
    logger.info("Loading up to {} images from {}. Prices: {}".format(n, x_dir, y_path))
    ext = ".jpg"
    # load images
    image_paths = [os.path.join(x_dir, f) for f in os.listdir(x_dir) if f.endswith(ext)]
    if n is not None:
        image_paths = image_paths[:n]

    # load predictions
    if y_path is not None:
        prices = {}
        with open(y_path) as f:
            for line in f:
                img_id, price = line.strip().split(",")
                prices[img_id] = float(price)

    data = [ImageData(io.imread(fp), prices[os.path.basename(fp).strip(ext)] if y_path is not None else None) for fp in image_paths]

    images = np.array([img.image for img in data])
    # Scale pixel values
    images = images / 255.0

    if y_path is not None:
        y = np.array([img.price for img in data])
    else:
        y = None

    logger.info("Loaded images: {}; Prices: {}".format(images.shape, len(y) if y is not None else None))

    return images, y


def CAE(input_shape: Tuple[int, int, int] = (128, 128, 3),
        filters: Iterable[int] = [32, 64, 128, 32],
        activation: str = "elu",
        stride: int = 2):
    """Convolutional Autoencoder

    :param input_shape: Shape of the input layer in the model
    :param filters: Number of filters in the convolutional layers, plus the size of the clustering layer. Hence the
        length should equal len(convolutional layers) + 1.
    :param activation: Activation function to use
    :param strice: stride length to use
    """
    assert len(filters) == 4, "Expected 4 filters, got, {}".format(filters)

    pad_same = "same"

    # TODO, I'm not sure why we're mod'ing by 8. https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t  # noqa: E501
    if input_shape[0] % 8 == 0:
        pad3 = pad_same
    else:
        pad3 = "valid"

    # TODO This could be much prettier and cleaner
    layers = [
        Conv2D(filters[0], kernel_size=5, strides=stride, padding=pad_same, activation=activation, name="conv1", input_shape=input_shape),
        Conv2D(filters[1], kernel_size=5, strides=stride, padding=pad_same, activation=activation, name="conv2"),
        Conv2D(filters[2], kernel_size=3, strides=stride, padding=pad3, activation=activation, name="conv3"),
        Flatten(),
        Dense(units=filters[3], name="embedding"),
        Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation=activation),
        Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])),
        Conv2DTranspose(filters[1], kernel_size=3, strides=stride, padding=pad3, activation=activation, name="deconv3"),
        Conv2DTranspose(filters[0], kernel_size=5, strides=stride, padding=pad_same, activation=activation, name="deconv2"),
        Conv2DTranspose(input_shape[2], kernel_size=5, strides=stride, padding=pad_same, name="deconv1")
    ]

    model = Sequential()
    for layer in layers:
        model.add(layer)

    model.summary()

    return model


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
                 filters: Iterable[int] = FILTERS):
        """DCEC Model

        :param input_shape: Shape of the input layer in the model
        :param filters: Number of filters in the convolutional layers, plus the size of the clustering layer. Hence the
            length should equal len(convolutional layers) + 1.
        :param n_clusters: k, the number of clusters to target
        # TODO Do we need this parameter?
        """
        # TODO Add activation as a parameter to this model
        super(DCEC, self).__init__()

        self.input_shape = input_shape
        self.pretrained = False
        self.y_pred = []

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name="embedding").output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        # clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(hidden)
        prediction_layer = Dense(1, kernel_initializer='normal', activation='linear', name="prediction")(hidden)
        self.model = Model(inputs=self.cae.input, outputs=[prediction_layer, self.cae.output])

    # TODO we should really be training for at least 200 epochs
    def pretrain(self, x, batch_size=256, epochs=200, optimizer="adam", save_dir="results/temp"):
        logger.info("...Pretraining...")
        self.cae.compile(optimizer=optimizer, loss="mse")

        csv_logger = CSVLogger(args.save_dir + "/pretrain_log.csv")

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        logger.info("Pretraining time: {}".format(time() - t0))
        self.cae.save(save_dir + "/pretrain_cae_model.h5")
        logger.info("Pretrained weights are saved to %s/pretrain_cae_model.h5" % save_dir)
        gcs_client.upload(save_dir, os.path.join(GCS_PREFIX, "/results/temp/"))  # TODO Tar this and save it
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    # @staticmethod
    # def target_distribution(q):
    #     weight = q ** 2 / q.sum(0)
    #     return (weight.T / weight.sum(1)).T

    # TODO you need to update these losses and loss weights
    # TODO is "mse" the optimal loss function for the prediction layer?
    def compile(self, loss=["mse", "mse"], loss_weights=[1, 1], optimizer="adam"):
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

        # TODO Will I need some way to initialize the predictions?
        # Step 2: initialize cluster centers using k-means
        t1 = time()

        # TODO old clustering
        # logger.info("Initializing cluster centers with k-means.")
        # kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        # self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        # y_pred_last = np.copy(self.y_pred)
        # self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        # TODO Set prediction weights
        # Set weights to median of y. Could also use:
        # Mean of training set. Random from training set, 0? Need to compare models
        weights, _ = self.model.get_layer(name="prediction").get_weights()
        median = np.median(y)
        self.model.get_layer(name="prediction").set_weights(np.array([median for _ in weights]))
        y_pred_last = np.copy(self.y_pred)

        # Step 3: deep prediction
        # logging file

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + "/dcec_log.csv", "w")
        logwriter = csv.DictWriter(logfile, fieldnames=["iter", "acc", "nmi", "ari", "L", "Lc", "Lr"])
        logwriter.writeheader()

        # TODO Can we just call model.fit here?
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                logger.info("Updating. Iter {}".format(ite))
                q, _ = self.model.predict(x, verbose=0)
                # p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                # self.y_pred = q.argmax(1)
                if y is not None:
                    logger.info("{} calculating acc".format(ite))
                    # acc = np.round(acc(y, self.y_pred), 5)
                    # nmi = np.round(nmi(y, self.y_pred), 5)
                    # ari = np.round(ari(y, self.y_pred), 5)
                    # loss = np.round(loss, 5)
                    # logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2])
                    # logwriter.writerow(logdict)
                    # logger.info("Iter {}: Acc {}, nmi {}, ari {}; loss={}".format(ite, acc, nmi, ari, loss))
                    logdict = dict(L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    logger.info("loss={}".format(loss))

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
                    x=x[index * batch_size : :],
                    y=[
                        y[index * batch_size :],
                        # p[index * batch_size : :],
                        x[index * batch_size : :]
                    ]
                )
                index = 0
            else:
                loss = self.model.train_on_batch(
                    x=x[index * batch_size : (index + 1) * batch_size],
                    y=[
                        y[index * batch_size : (index + 1) * batch_size],
                        # p[index * batch_size : (index + 1) * batch_size],
                        x[index * batch_size : (index + 1) * batch_size],
                    ],
                )
                index += 1

            # save intermediate model
            if ite % save_interval == 0:
                # save DCEC model checkpoints
                model_name = "dced_model_{}.h5".format(ite)
                logger.info("saving model to: {}".format(os.path.join(save_dir, model_name)))
                path = os.path.join(save_dir, model_name)
                self.model.save_weights(path)
                gcs_client.upload(path, os.path.join(GCS_PREFIX, "results/temp", model_name))

            ite += 1

        # save the trained model
        logfile.close()

        model_name = "dcec_model_final.h5"
        logger.info("saving model to: {}".format(os.path.join(save_dir, model_name)))
        self.model.save_weights(os.path.join(save_dir, model_name))
        gcs_client.upload(path, os.path.join(GCS_PREFIX, "results/temp", model_name))

        t3 = time()
        logger.info("Pretrain time:   {}".format(t1 - t0))
        logger.info("Clustering time: {}".format(t3 - t1))
        logger.info("Total time:      {}".format(t3 - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--dataset-path", default=os.getenv("DCEC_DATASET_PATH"), help="Path to data")
    parser.add_argument("--batch-size", default=os.getenv("DCEC_BATCH_SIZE", 512), type=int, help="Training batch size")
    parser.add_argument("--max-iter", default=os.getenv("DCEC_MAX_ITER", 20000), type=int, help="Maximum iterations to perform on final training")
    parser.add_argument("--gamma", default=os.getenv("DCEC_GAMMA", 0.1), type=float, help="coefficient of clustering loss")
    parser.add_argument("--update-interval", default=os.getenv("DCEC_UPDATE_INTERVAL", 140), type=int, help="How frequently to update weights")
    parser.add_argument("--tolerance", default=os.getenv("DCEC_TOLERANCE", 0.001), type=float, help="Threshold at which to stop training")
    parser.add_argument("--cae-weights", default=os.getenv("DCEC_CAE_WEIGHTS", "false").lower() == "true", type=bool, help="Whether to use the default CAE weights")
    parser.add_argument("--save-dir", default=os.getenv("DCEC_SAVE_DIR", "./results/temp"), help="Where to save results/model to")
    parser.add_argument("--data-dir", default=os.getenv("DCEC_DATA_DIR", "./data"), help="Where the data reside")
    parser.add_argument('--assert-gpu', default=os.getenv("DCEC_ASSERT_GPU", "false").lower() == "true", action="store_true")
    parser.add_argument("--epochs", default=os.getenv("DCEC_EPOCHS", 200), type=int, help="Number of epochs to train CAE")
    args = parser.parse_args()

    logger.info("Running with config: {}".format(["{}={}".format(k, v) for k, v in vars(args).items()]))

    # Log GPU info, optionally asserting if there isn't one
    gpu_info(args.assert_gpu)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    gcs_client = GCStorage(bucket="paap")

    tarball_path = os.path.join(args.data_dir, os.path.basename(args.dataset_path))
    if is_gcs_blob(args.dataset_path):
        # Get data from GCS
        _, path = path_from_uri(args.dataset_path)
        gcs_client.download(path, tarball_path)
    elif args.dataset_path.endswith(".tar.gz") and args.dataset_path != tarball_path:
        raise ValueError("Expected local data to be at: {}".format(tarball_path))

    if args.dataset_path.endswith(".tar.gz"):
        untar(tarball_path, args.data_dir)

    # load data
    data = os.path.join(args.data_dir, os.path.basename(args.dataset_path).split(".")[0])
    x, y = load_images(
        x_dir=os.path.join(data, "train"),
        y_path=os.path.join(data, "y.txt"),
        n=N_IMAGES)

    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=FILTERS)
    plot_model(dcec.model, to_file=args.save_dir + "/dcec_model.png", show_shapes=True)
    dcec.model.summary()

    # TODO Update these
    dcec.compile(loss=["mse", "mse"], loss_weights=[args.gamma, 1], optimizer=OPTIMIZER)

    dcec.fit(
        x,
        y=y,
        tol=args.tolerance,
        maxiter=args.max_iter,
        update_interval=args.update_interval,
        save_dir=args.save_dir,
        cae_weights=args.cae_weights,
    )

    if y is not None:
        y_pred = dcec.y_pred
        # TODO better metrics
        # logger.info(
        #     "acc = %.4f, nmi = %.4f, ari = %.4f" % (metrics.acc(y, y_pred), metrics.nmi(y, y_pred), metrics.ari(y, y_pred))
        # )
