import argparse
import json
import subprocess
import os
import time
from pathlib import Path


DESCRIPTION = "Sample images from a dataset and remove sculptures, sketches, and images with text"
CMD = "feh {path}"


def open_image(path: Path) -> None:
    cmd = CMD.format(path=path)
    # run it and don't wait for it to complete
    p = subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    time.sleep(0.4)
    p.kill()


def ask() -> bool:
    while True:
        raw = input("2D? (y/n)")
        if raw == "" or raw == "n":
            return raw == ""
        print("Invalid input. Respond y or n")


def main(image_dir: Path, output: str) -> None:
    image_paths = [os.path.join(image_dir, fp) for fp in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, fp))]

    try:
        with open(output) as f:
            count = len(f.readlines())
    except FileNotFoundError:
        count = 0

    f = open(output, "a")
    try:
        for i, fp in enumerate(image_paths[count:]):
            print("\n--- {}".format(i + 1 + count))
            open_image(fp)
            response = ask()

            to_write = {"lot_image_id": fp.split("/")[-1].split(".")[0], "lot_is_2d": response}
            f.write(json.dumps(to_write))
            f.write("\n")

            f.flush()
    finally:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--images", help="Path to directory of images", default="/home/dubs/dev/paap/data/img/christies/s128/final/")
    parser.add_argument("--output", help="Where to write the output", default="/home/dubs/dev/paap/data/output/final_is_usable.json")
    args = parser.parse_args()

    main(args.images, args.output)
