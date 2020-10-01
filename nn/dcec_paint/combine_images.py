import csv
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


def read_csv(fn):
    with open(fn) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def out_row(row):
    return [row[k] for k in ["lot_description", "cluster", "distance_from_centroid", "lot_image_id"]]


def print_images(rows):
    out = [out_row(row) for row in rows]
    max_lengths = [max(len(str(out[j][i])) for j in range(len(out))) for i in range(len(out[0]))]
    for i in range(len(out)):
        s = [str(out[i][j]) for j in range(len(out[i]))]
        for k in range(len(max_lengths)):
            s[k] = s[k].ljust(max_lengths[k])
        print(" ".join(s))


# def build_collage(images, border=0):
#     s = math.ceil(min(images[0].shape[:2]) * border)
#     # padded = [np.pad(img, ((s, s), (s, s), (0, 0)), constant_values=1) for img in images]
#     padded = images
#     pairs = [np.concatenate(p) for p in zip(padded[::2], padded[1::2])]
#     return np.concatenate(pairs, axis=1)


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

    # x = f(math.sqrt(len(images)))
    # if x % 2 != 0:
    #     x -= 1

    tuples = [np.concatenate(images[i:i + x]) for i in range(0, len(images), x)]
    return np.concatenate(tuples, axis=1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: combine_images.py <clusters> <n_images>")
        exit(1)

    clusters = int(sys.argv[1])
    n_images = int(sys.argv[2])

    if n_images % 2 != 0:
        print("n_images must be a multiple of 2")
        exit(1)

    fn = os.path.join(IMAGES_DIR, str(clusters), FILENAME)
    rows = read_csv(fn)

    collages = {}
    for i in range(clusters):
        cluster_rows = [row for row in rows if int(row["cluster"]) == i]
        indexes = random.sample(range(len(cluster_rows)), k=n_images)
        data = [cluster_rows[i] for i in indexes]
        paths = [d["images_path"] for d in data]
        images = [io.imread(p) for p in paths]
        collages[i] = {}
        collages[i]["collage"] = build_collage(images, border=2, border_color=1)
        collages[i]["data"] = data

    final = build_collage([c["collage"] for c in collages.values()], border=4, border_color=0, orient="v")
    # border = 4
    # padded = [np.pad(c["collage"], ((border, border), (border, border), (0, 0)), constant_values=0) for _, c in collages.items()]
    # final = np.concatenate(padded)
    fn = os.path.join(IMAGES_DIR, str(clusters), "collage.png")
    io.imsave(fn, final)

    print_images(rows)
