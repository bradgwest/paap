import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import Mapping, Any

from ..defaults import Paths, Columns
from ..utils import valid_path, read_data


DESCRIPTION = "Randomly sample images from a dataset and determine if they are 2d or not"
CMD = "feh {path}"


def message(row: Mapping[str, Any]) -> str:
    keys = [Columns.SALE_DATE, Columns.SALE_IMPUT_URL, Columns.SALE_LOCATION, Columns.SALE_CATEGORY,
            Columns.LOT_ARTIST, Columns.LOT_MEDIUM, Columns.LOT_DESCRIPTION]
    vals = ["{}: {}".format(k, row[k]) for k in keys]
    return "\n".join(vals)


def open_image(path: Path) -> None:
    cmd = CMD.format(path=path)
    # run it and don't wait for it to complete
    subprocess.Popen(cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)


def build_image_path(image_id: str) -> Path:
    image_base = image_id + ".jpg"
    return Path(Paths.S256_IMG_DIR, image_base)


def ask() -> bool:
    while True:
        raw = input("2D? (y/n)")
        if raw == "y" or raw == "n":
            return raw == "y"
        print("Invalid input. Respond y or n")


def main(input_json: Path, output: str) -> None:
    df = read_data(input_json)

    seed = random.randint(1, 1000000)
    sample = df.sample(df.shape[0], random_state=seed)

    count = 1
    f = open(output, "w")
    try:
        for i, row in sample.iterrows():
            print("\n{} --- {}".format(count, i))
            print(message(row))
            path = build_image_path(row[Columns.LOT_IMAGE_ID])
            open_image(path)

            response = ask()

            to_write = {Columns.LOT_IMAGE_ID: row[Columns.LOT_IMAGE_ID], Columns.LOT_IS_2D: response}
            f.write(json.dumps(to_write))
            f.write("\n")

            count += 1
    finally:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--input", help="Path to input dataframe", type=valid_path, default=Paths.WITH_IMAGES_OUTPUT)
    parser.add_argument("--output", help="Where to write the output", default=Paths.SAMPLE_IS_2D)
    args = parser.parse_args()

    main(args.input, args.output)
