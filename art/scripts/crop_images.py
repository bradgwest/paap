import argparse
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import ImageFile
from skimage import io

from ..utils import valid_directory, valid_path

DESCRIPTION = """Crop images so they are square"""
SIZE = 128
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Required for truncated images


def center_crop_image(image: np.ndarray) -> np.ndarray:
    y, x, _ = image.shape
    size = min(y, x)
    start_x = x // 2 - size // 2
    start_y = y // 2 - size // 2
    return image[start_y:(start_y + size), start_x:(start_x + size), :]


def bw_to_rgb(image: np.ndarray) -> np.ndarray:
    return np.stack((image,) * 3, axis=-1)


def load_image_paths(path: Path) -> Generator[None, None, Path]:
    with open(path) as f:
        for line in f:
            yield Path(line.strip())


def main(input_file: Path, output_dir: Path) -> None:
    image_paths = load_image_paths(input_file)
    count = 0
    for i, p in enumerate(image_paths):
        # print("{} - processing {}".format(i, p), end="\r")

        try:
            image = io.imread(p)
        except FileNotFoundError:
            print("WARNING: Couldn't find file: {}. Perhaps it was already processed".format(p))
            continue
        except ValueError as e:
            print("ERROR: Failed to load image: {}. Exception: {}".format(p, e))
            continue

        if len(image.shape) == 2:
            print("converting non-3 channel image. {}".format(p))
            image = bw_to_rgb(image)

        cropped_image = center_crop_image(image)

        y, x, c = cropped_image.shape
        assert y == x == SIZE, \
            "Expected image of size {}x{}, got {}x{}. {}".format(SIZE, SIZE, y, x, p)
        assert c == 3, "Expected 3 channels, got {}. {}".format(len(c), p)

        output_path = output_dir / p.name
        io.imsave(output_path, cropped_image)
        count += 1

    print("Wrote {} images to {}".format(count, output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "images",
        type=valid_path,
        help="Images to process a newline separated file of image paths. A command like the following "
             "should get you started: `find data/img/christies/raw/ -type f -name '*.jpg' > "
             "data/img/christies/raw_images.txt`",
    )
    parser.add_argument("output_dir", type=valid_directory, help="Directory to save cropped images to")
    args = parser.parse_args()

    print("Writing images to `{}`.".format(args.output_dir))
    main(args.images, args.output_dir)
