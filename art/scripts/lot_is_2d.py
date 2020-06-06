import argparse
import json
import subprocess
from pathlib import Path
from typing import Mapping, Any

from ..defaults import Paths, Columns
from ..utils import valid_path, read_data


DESCRIPTION = "Randomly sample images from a dataset and determine if they are 2d or not"
CMD = "feh {path}"


def message(row: Mapping[str, Any]) -> str:
    keys = [Columns.SALE_DATE, Columns.SALE_IMPUT_URL, Columns.SALE_LOCATION, Columns.LOT_ARTIST]
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
        raw = input("2D? (;=y/'=n)")
        if raw == ";" or raw == "'":
            return raw == ";"
        print("Invalid input. Respond y or n")


def main(input_json: Path, output: str) -> None:
    df = read_data(input_json)

    try:
        with open(output) as f:
            count = len(f.readlines())
    except FileNotFoundError:
        count = 0

    rows = df[(count):]

    f = open(output, "a")
    try:
        for i, row in rows.iterrows():
            print("\n--- {}".format(i + 1))
            print(message(row))
            path = build_image_path(row[Columns.LOT_IMAGE_ID])
            open_image(path)

            response = ask()

            to_write = {Columns.LOT_IMAGE_ID: row[Columns.ID], Columns.LOT_IS_2D: response}
            f.write(json.dumps(to_write))
            f.write("\n")

            f.flush()
    finally:
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--input", help="Path to input dataframe", type=valid_path, default=Paths.PHOTOS_PRINTS_OUTPUT)
    parser.add_argument("--output", help="Where to write the output", default=Paths.PHOTOS_PRINTS_IS_2D)
    args = parser.parse_args()

    main(args.input, args.output)
