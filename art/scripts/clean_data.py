import argparse
from collections import namedtuple
from pathlib import Path
from uuid import uuid4

import pandas as pd

column = namedtuple("Column", ["name", "dtype", "na_allowed"])

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
IMAGE_ID = "image_id"
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
    return df[col][~df.duplicated(col)]


def main(input_csv: str, image_urls: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv, header=0, index_col=False)
    image_urls = pd.read_csv(image_urls, header=None, names=IMAGE_COLS, index_col=False)

    df = clean_raw(df)

    # drop invalid image_urls
    image_urls = image_urls[~(image_urls[LOT_IMAGE_URL] == NO_IMAGE_URL)]
    image_urls = drop_image_duplicates(image_urls, LOT_IMAGE_URL)

    # join on lot_image_url
    df_all = pd.merge(df, image_urls, on=LOT_IMAGE_URL, how="left")

    col_order = ["id"] + list(df_all.columns)

    # add an id column
    df_all["id"] = pd.Series([uuid4() for _ in range(df_all.shape[0])])
    df_all = df_all[col_order]
    df_all.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_csv", type=valid_path,
                        help="Path to input csv with raw scrapped data, header on first row")
    parser.add_argument("image_urls", type=valid_path,
                        help="Path to a csv with image urls")
    parser.add_argument("output_csv", type=str,
                        help="Path to save output csv containing only cleaned data")
    args = parser.parse_args()

    main(args.input_csv, args.image_urls, args.output_csv)
