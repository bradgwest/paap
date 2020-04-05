import argparse
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import ImageFile
from skimage import io
from skimage.transform import rescale

DESCRIPTION = """Resize images to a common minimum dimension, retaining aspect ratio"""
DEFAULT_IMAGE_SIZE = 256
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Required for truncated images


def valid_directory(directory_str: str) -> Path:
    directory = Path(directory_str)
    if not (directory.exists() and directory.is_dir()):
        raise ValueError("{} is not a valid directory".format(directory))
    return directory


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def resize_image(image: np.ndarray, image_size: int) -> np.ndarray:
    smallest_side = int(image.shape[1] <= image.shape[0])
    scaling_factor = image_size / image.shape[smallest_side]
    # Some images have three channels, others have one
    scale = (scaling_factor, scaling_factor) if len(image.shape) == 2 else (scaling_factor, scaling_factor, 1)
    resized_image = rescale(image, scale)
    return (resized_image * 255).astype(np.uint8)  # Get it back to non-lossy format


def load_image_paths(path: Path) -> Generator[None, None, Path]:
    with open(path) as f:
        for line in f:
            yield Path(line.strip())


def main(input_file: Path, output_dir: Path, image_size: int, delete: bool) -> None:
    image_paths = load_image_paths(input_file)
    for i, p in enumerate(image_paths):
        print("{} - processing {}".format(i, p), end="\r")

        try:
            image = io.imread(p)
        except FileNotFoundError:
            print("WARNING: Couldn't find file: {}. Perhaps it was already processed".format(p))
            continue
        except ValueError as e:
            print("ERROR: Failed to load image: {}. Exception: {}".format(p, e))
            continue

        resized_image = resize_image(image, image_size)

        output_path = output_dir / p.name
        io.imsave(output_path, resized_image)

        if delete:
            assert p != output_path, "Would delete the file just created"
            p.unlink()


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
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Images will be scaled to be this large in their minimum dimension, in pixels",
    )
    parser.add_argument("--delete", action="store_true", help="Delete the input photo after processing")
    args = parser.parse_args()

    print("Writing images to `{}`. Will {}delete input images.".format(args.output_dir, "" if args.delete else "not "))
    main(args.images, args.output_dir, args.image_size, args.delete)
