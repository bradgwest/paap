import argparse
import os
import random
from shutil import copyfile


DESCRIPTION = "Split data into train, test, validate"
random.seed(22)


def main(image_dir: str, output_dir: str, split: list):
    assert sum(split) == 1

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    random.shuffle(images)

    n = len(images)
    config = {
        "train": images[:int(n * split[0])],
        "test": images[int(n * split[0]):int(n * (1 - split[2]))],
        "validate": images[int(n * (1 - split[2])):]
    }

    print("train: {}\ntest: {}\nvalidate: {}".format(*[len(config[t]) for t in ("train", "test", "validate")]))

    for fldr, images in config.items():
        fldr_dir = os.path.join(output_dir, fldr)
        print(fldr_dir)
        if not os.path.exists(fldr_dir):
            os.makedirs(fldr_dir)

        for img in images:
            src = os.path.join(image_dir, img)
            dst = os.path.join(output_dir, fldr, img)
            copyfile(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_dir", help="directory of images to split")
    parser.add_argument("output_dir", help="where to write to")
    parser.add_argument("split", nargs=3, type=float, help="train, test, validate proportions")
    args = parser.parse_args()
    print(args)

    main(args.input_dir, args.output_dir, args.split)
