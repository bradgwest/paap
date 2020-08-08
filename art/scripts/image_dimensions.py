import argparse
import json
import os

import numpy as np
from skimage import io


def is_bw(image):
    if len(image.shape) == 2:
        return True

    if image.shape[2] == 1:
        return True

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    return np.array_equal(r, g) and np.array_equal(g, b)


def main(image_dir, output):
    cnt = 0
    images = os.listdir(image_dir)
    print("Will process {} images".format(len(images)))

    out = open(output, "w")
    try:
        for i, fp in enumerate(images):
            if cnt % 1000 == 0:
                print("{}/{}".format(cnt, len(images)), end="\r", flush=True)

            row = {}
            image = io.imread(os.path.join(image_dir, fp))
            row["image_channels"] = 1 if is_bw(image) else image.shape[2]

            row["image_shape"] = str(image.shape)
            row["lot_image_id"] = fp.split(".")[0]

            out.write(json.dumps(row))
            out.write("\n")

            cnt += 1
    finally:
        out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get dimensions of images")
    parser.add_argument("--image-dir", default="/home/dubs/dev/paap/data/img/christies/s128/all/")
    parser.add_argument("--output", default="/home/dubs/dev/paap/data/output/image_dimensions_output.ndjson")
    args = parser.parse_args()

    main(args.image_dir, args.output)
