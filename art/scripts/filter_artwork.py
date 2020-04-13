import argparse
from pathlib import Path

import pandas as pd

DESCRIPTION = "Filter out artwork that is unsuitable for analysis according to a set of rules"
RULES = set()


def rule(func):
    """Annotation for adding a rule to the set of all rules"""
    RULES.add(func)
    return func


@rule
def is_2d(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are one of the following mediums:"""
    pass


@rule
def is_known_genre(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a known genre"""
    pass


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def main(input_csv: Path, output_csv: Path, rules: list) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_csv", type=valid_path,
                        help="Path to input csv with raw scrapped data, header on first row")
    parser.add_argument("output_csv", type=valid_path, help="Path to save output csv containing only filtered data")
    parser.add_argument("--rules", nargs="*", default="*", help="Filtering rules to apply",
                        choices=[r.__name__ for r in RULES])
    args = parser.parse_args()

    main(args.input_csv, args.output_csv, args.rules)
