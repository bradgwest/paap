import argparse
import re
from pathlib import Path

from typing import List

import pandas as pd

DESCRIPTION = "Filter out artwork that is unsuitable for analysis according to a set of rules"

NON_2D_KEYWORDS = re.compile(r"ceramic|bronze|calligraphy|sculpture|wooden|model|furniture|chair|table|glass|jade|carving|rug|carpet|bust|jewelry|bracelet")


def _is_3d(s: str) -> bool:
    return bool(re.search(NON_2D_KEYWORDS, s))


def is_3d(row: List[str]) -> bool:
    return any(_is_3d(v) for v in row)


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def main(input_json: Path, is_2d_json: Path, output_json: Path) -> None:
    df = pd.read_json(input_json, orient="records")
    is_2d = pd.read_json(is_2d_json, orient="records")

    # filter out sales that weren't 2d
    is_2d = is_2d[is_2d.sale_is_2d]
    df = pd.merge(df, is_2d, on=("sale_number", "sale_url"), how="inner")

    # filter out sales that have some of the disallowed keywords
    df = df[~df[["lot_description", "lot_artist", "lot_medium"]].apply(is_3d, axis=1)]
    df = df.drop("sale_is_2d", axis=1)

    print("Writing {} rows".format(df.shape[0]))
    df.to_json(output_json, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_json", type=valid_path,
                        help="Path to input json, the output from clean_data.py")
    parser.add_argument("is_2d_json", type=valid_path,
                        help="Path to is_2d json, the output from is_2d.py")
    parser.add_argument("output_json", type=str, help="Path to save output json containing only filtered data")
    args = parser.parse_args()

    main(args.input_json, args.is_2d_json, args.output_json)
