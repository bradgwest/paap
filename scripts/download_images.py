import argparse
import csv
import os
import time
from functools import wraps
from pathlib import Path
from typing import Callable

import requests

DESCRIPTION = """Given an input csv with image urls, download images and save them to a location"""
CHUNK_SIZE = 1024
RETRYABLE_STATUS_CODES = frozenset([429])


def valid_directory(directory: str) -> str:
    if not (os.path.exists(directory) and os.path.isdir(directory)):
        raise ValueError("{} is not a valid directory".format(directory))
    return directory


def build_image_path(directory: str, image_id: str, image_url: str) -> Path:
    ext = image_url.split(".")[-1]
    basename = ".".join([image_id, ext])
    return Path(directory) / basename


def retry(retry_on_exception: Callable, delay: int = 1, attempts: int = 4, multiplier: int = 2) -> Callable:
    def decorator_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mattempts, mdelay = attempts, delay  # Make mutable
            while mattempts > 1:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if not should_retry(e):
                        raise e
                    mattempts -= 1
                    time.sleep(mdelay)
                    mdelay *= multiplier
                    continue
            return f(*args, **kwargs)
        return f_retry
    return decorator_retry


def should_retry(exception: Exception) -> bool:
    exception.status_code in RETRYABLE_STATUS_CODES


@retry(retry_on_exception=should_retry)
def download_image(image_url: str, image_path: str) -> None:
    r = requests.get(image_url, stream=True)
    r.raise_for_status()
    with open(image_path, "wb") as f:
        for chunk in r.iter_content(CHUNK_SIZE):
            f.write(chunk)


def main(input_path: str, output_dir: str) -> None:
    failed = []

    with open(input_path) as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            image_id, image_url = line
            image_path = build_image_path(output_dir, image_id, image_url)
            print("{} - {} -> {}".format(i, image_url, image_path), end="\r")

            try:
                download_image(image_url, image_path)
            except requests.HTTPError as e:
                print("{} - Failed to download {}. {}".format(i, image_url, e))
                failed.append(line)

    if failed:
        print("Failed to download:")
        for line in failed:
            print(line)

    print("{} images downloaded to {}".format(i - len(failed), output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input", help="Input csv. Two columns (<lot_id>,<image_url>), no header.")
    parser.add_argument("output_dir", help="Directory to save images to.", type=valid_directory)
    args = parser.parse_args()

    print("Writing images to {}".format(Path(args.output_dir).resolve()))
    main(args.input, args.output_dir)
