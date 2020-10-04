"""Get statistics from a set of images and write them to a file
"""

import csv
import os
import sys

from skimage import io
from skimage.color import rgb2lab, rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk


ROOT = "/home/dubs/dev/paap"
DATA_DIR = os.path.join(ROOT, "data")
IMAGES = os.path.join(DATA_DIR, "img/christies/s128/final/")
OUTPUT = os.path.join(DATA_DIR, "output", "image_stats.csv")


def pixel_mean(img):
    assert len(img.shape) == 3
    return tuple([img[:, :, i].flatten().mean() for i in range(3)])


def rgb(img):
    return pixel_mean(img)


def lab(img):
    """CIE-LAB color space"""
    cie_lab = rgb2lab(img)
    return pixel_mean(cie_lab)


def entropy_stat(img):
    gray = rgb2gray(img)
    entr_img = entropy(gray, disk(10))
    return entr_img.flatten().mean()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: images_stats.py")
        exit(1)

    image_paths = [os.path.join(IMAGES, f) for f in os.listdir(IMAGES) if f.endswith(".jpg")]
    results = []
    for fp in image_paths:
        img = io.imread(fp)

        r, g, b = rgb(img)
        cie_l, cie_a, cie_b = lab(img)
        ent = entropy_stat(img)
        img_stats = {
            "lot_image_id": os.path.basename(fp)[:-len(".jpg")],
            "red": r,
            "green": g,
            "blue": b,
            "cie_l": cie_l,
            "cie_a": cie_a,
            "cie_b": cie_b,
            "entropy": ent
        }
        results.append(img_stats)

    header = list(results[0].keys())
    with open(OUTPUT, "w") as f:
        writer = csv.DictWriter(f, header)
        writer.writeheader()
        writer.writerows(results)
