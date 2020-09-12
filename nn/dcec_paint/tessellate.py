import os
import random
import sys

import numpy as np
from skimage import io


ROOT = "/home/dubs/dev/paap"
IMAGES_DIR = os.path.join(ROOT, "img")
TMP_DIR = "/tmp"
FILENAME = "center.csv"

SEED = 22
random.seed(SEED)


def build_collage(images, border=0, border_color=0, orient="horizontal"):
    # Add blank image if we have an odd number
    if len(images) % 2 != 0:
        images.append(np.zeros(images[0].shape))

    if border > 0:
        images = [np.pad(img, ((border, border), (border, border), (0, 0)), constant_values=1) for img in images]

    if orient[0] == "h":
        x = 2
    else:
        x = int(len(images) / 2)

    tuples = [np.concatenate(images[i:i + x]) for i in range(0, len(images), x)]
    return np.concatenate(tuples, axis=1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tessellate.py cluster 0 140 ...")
        exit(1)

    cluster = int(sys.argv[1])
    d = fn = os.path.join(IMAGES_DIR, str(cluster))

    paths = [os.path.join(d, "tsne_" + str(n) + ".png") for n in sys.argv[2:]]
    images = [io.imread(p) for p in paths]
    collage = build_collage(images, border=2, border_color=1)

    fn = os.path.join(d, "cluster_evolution.png")
    io.imsave(fn, collage)
