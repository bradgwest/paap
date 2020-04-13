import argparse
import csv
from typing import Tuple

DESCRIPTION = """Create a two column CSV (<lot_id>,<image_url>) of image metadata from an input CSV of raw art data."""

# Input csv keys
ID = "id"
IMAGE_URL = "lot_image_url"


def dict_to_row(d: dict) -> Tuple[str, str]:
    return d[ID], d[IMAGE_URL]


def main(input_path: str, output_path: str) -> None:
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        reader = csv.DictReader(f_in)
        assert ID in reader.fieldnames and IMAGE_URL in reader.fieldnames, "input csv doesn't have expected colums"

        writer = csv.writer(f_out)
        for line in reader:
            writer.writerow(dict_to_row(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input", help="Input csv with raw art data")
    parser.add_argument("output", help="Output path")
    args = parser.parse_args()

    main(args.input, args.output)
