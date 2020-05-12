import argparse
import csv
import datetime
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

column = namedtuple("Column", ["name", "dtype", "na_allowed"])

DESCRIPTION = "Clean raw tabular data"
# Columns we wish to keep. Types will be coerced to dtype if not None
COLUMNS = [
    # (Column Name, DataType Coerscion Function, Can Be NA)
    column("id", None, False),
    column("input_url", None, True),
    column("sale_location", None, True),
    column("sale_location_id", int, True),
    column("sale_christies_id", None, True),  # TODO this should be changed back to an int
    column("sale_iso_currency_code", 'category', False),
    column("sale_year", int, False),
    column("sale_month", int, False),
    column("lot_start_date", None, True),
    column("lot_end_date", None, True),
    column("lot_artist", None, True),
    column("lot_title", None, True),
    column("lot_translated_description", None, True),
    column("lot_description", None, True),
    column("lot_medium", None, True),
    column("lot_dimensions", None, True),
    column("lot_attributes", None, True),
    column("lot_image_url", None, True),
    column("lot_estimate_low_iso_currency", None, True),
    column("lot_estimate_high_iso_currency", None, True),
    column("lot_realized_price_iso_currency", float, False),
]


def build_sale_date(df: pd.DataFrame) -> pd.Series:
    return df[["sale_year", "sale_month"]].apply(lambda x: datetime.datetime(year=x[0], month=x[1], day=1), axis=1)


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def csv_to_df(path: Path) -> pd.DataFrame:
    """Pandas 1.0.3 is trash (or it's my ipython). They read duplicates for no reason. Roll your own I/O"""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            rows.append({k: v if v != "" else np.NaN for k, v in zip(header, line)})
    return pd.DataFrame(rows)


def main(input_csv: str, image_urls: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv, header=0, index_col=False)
    # Pandas is trashy shit. Drop duplicates that it seems to read.
    df = df[[c.name for c in COLUMNS]]  # select columns

    for c in COLUMNS:
        # filter out NA
        if not c.na_allowed:
            df = df[~df[c.name].isna()]

    df = df.astype({c.name: c.dtype for c in COLUMNS if c.dtype is not None})
    df.to_csv(output_csv, index=None)


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
