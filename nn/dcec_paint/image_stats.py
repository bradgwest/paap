"""Get statistics from a set of images and write them to a file
"""

import csv
import os
import sys

from skimage import io


ROOT = "/home/dubs/dev/paap"
DATA_DIR = os.path.join(ROOT, "data")
IMAGES = os.path.join(DATA_DIR, "img/christies/s128/final/")
OUTPUT = os.path.join(DATA_DIR, "output", "image_stats.csv")


def rgb(img):
    assert len(img.shape) == 3
    return tuple([img[:, :, i].flatten().mean() for i in range(3)])


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: images_stats.py")
        exit(1)

    image_paths = [os.path.join(IMAGES, f) for f in os.listdir(IMAGES) if f.endswith(".jpg")]
    results = []
    for fp in image_paths:
        img = io.imread(fp)

        r, g, b = rgb(img)
        img_stats = {
            "lot_image_id": os.path.basename(fp)[:-len(".jpg")],
            "red": r,
            "green": g,
            "blue": b
        }
        results.append(img_stats)

    with open(OUTPUT, "w") as f:
        writer = csv.DictWriter(f, ("lot_image_id", "red", "green", "blue"))
        writer.writeheader()
        writer.writerows(results)
