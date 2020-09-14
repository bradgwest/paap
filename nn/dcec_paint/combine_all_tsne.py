import os

import numpy as np
from skimage import io


ROOT = "/home/dubs/dev/paap"
IMAGES_DIR = os.path.join(ROOT, "img")


def img_matcher(x):
    mapper = {
        "tsne": lambda x: x == "tsne.png"
    }
    return mapper[x]


def final_tsne(d, t="tsne"):
    matcher = img_matcher(t)
    return sorted(fn for fn in os.listdir(d) if matcher(fn))[-1]


def build_filenames():
    clusters = range(2, 11)
    for c in clusters:
        cluster_dir = os.path.join(IMAGES_DIR, str(c))
        base = final_tsne(cluster_dir)
        yield os.path.join(cluster_dir, base)


def build_collage(images, border=0, border_color=0, orient="horizontal"):
    # Add blank image if we have an odd number
    if len(images) % 2 != 0:
        images.append(np.zeros(images[0].shape))

    if border > 0:
        images = [np.pad(img, ((border, border), (border, border), (0, 0)), constant_values=1) for img in images]

    if orient[0] == "h":
        x = 2
        axis = (1, 0)
    else:
        x = int(len(images) / 2)
        axis = (1, 0)

    tuples = [np.concatenate(images[i:i + x], axis=axis[0]) for i in range(0, len(images), x)]
    print(len(tuples))
    return np.concatenate(tuples, axis=axis[1])


if __name__ == "__main__":
    images = [io.imread(p) for p in build_filenames()]
    collage = build_collage(images, border=2, border_color=1, orient="h")

    fn = os.path.join(IMAGES_DIR, "all_tsne.png")
    io.imsave(fn, collage)
