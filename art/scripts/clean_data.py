import argparse
import csv
import json
import re
from collections import namedtuple
from pathlib import Path
from uuid import uuid4

import pandas as pd
import numpy as np

column = namedtuple("Column", ["name", "dtype", "na_allowed"])


def strip_newlines(s: str) -> str:
    return re.sub(re.compile(r"\n+"), " ", s)


DESCRIPTION = "Clean raw tabular data"
# Columns we wish to keep. Types will be coerced to dtype if not None
COLUMNS = [
    # (Column Name, DataType Coerscion Function, Can Be NA)
    column("sale_number", int, False),
    column("sale_url", None, False),
    column("sale_input_url", None, False),
    column("sale_year", int, False),
    column("sale_month", int, False),
    column("sale_date", None, False),
    column("sale_location", None, False),
    column("sale_location_int", int, False),
    column("sale_category", None, False),
    column("sale_currency", None, False),
    column("lot_image_url", None, False),
    column("lot_artist", None, True),
    column("lot_realized_price", float, False),
    column("lot_description", None, True),
    column("lot_medium", None, True)
]
NO_IMAGE_URL = "https://www.christies.com/img/lotimages//Alert/NoImage/non_NoImag.jpg"
IMAGE_ID = "lot_image_id"
LOT_IMAGE_URL = "lot_image_url"
IMAGE_COLS = [IMAGE_ID, LOT_IMAGE_URL]


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c.name for c in COLUMNS]]  # select columns

    for c in COLUMNS:
        # filter out NA
        if not c.na_allowed:
            df = df[~df[c.name].isna()]

    return df.astype({c.name: c.dtype for c in COLUMNS if c.dtype is not None})


def drop_image_duplicates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[~df.duplicated(col)]


def write_json(df: pd.DataFrame, path: str) -> None:
    rows = df.to_dict("records")
    print("Will write {} rows".format(len(rows)))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


def write_csv(df: pd.DataFrame, path: str) -> None:
    rows = df.to_dict("records")
    print("Will write {} rows".format(len(rows)))
    with open(path, "w") as f:
        writer = csv.DictWriter(f, list(df.columns))
        for r in rows:
            writer.writerow(r)


def sub_whitespace(v):
    if isinstance(v, np.int64): 
        return v
    return re.sub(re.compile(r"\s+"), " ", str(v))


def main(input_json: str, image_urls: str, output_json: str) -> None:
    df = pd.read_json(input_json, orient="records")
    image_urls = pd.read_csv(image_urls, header=None, names=IMAGE_COLS, index_col=False)

    df = clean_raw(df)
    df = df[~(df[LOT_IMAGE_URL] == NO_IMAGE_URL)]

    # drop invalid image_urls
    image_urls = image_urls[~(image_urls[LOT_IMAGE_URL] == NO_IMAGE_URL)]
    image_urls = drop_image_duplicates(image_urls, LOT_IMAGE_URL)

    # join on lot_image_url
    df_all = pd.merge(df, image_urls, on=LOT_IMAGE_URL, how="left")

    col_order = ["id"] + list(df_all.columns)

    # add an id column
    df_all["id"] = pd.Series([str(uuid4()) for _ in range(df_all.shape[0])])
    df_all = df_all[col_order]

    df_all = df_all.applymap(sub_whitespace)

    print("Writing {} rows".format(df.shape[0]))
    df_all.to_json(output_json, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_json", type=valid_path,
                        help="Path to input json with raw scrapped data, header on first row")
    parser.add_argument("image_urls", type=valid_path,
                        help="Path to a csv with image urls")
    parser.add_argument("output_json", type=str,
                        help="Path to save output json containing only cleaned data")
    args = parser.parse_args()

    main(args.input_json, args.image_urls, args.output_json)
