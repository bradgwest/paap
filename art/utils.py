from pathlib import Path

import pandas as pd


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def valid_directory(directory_str: str) -> Path:
    directory = Path(directory_str)
    if not (directory.exists() and directory.is_dir()):
        raise ValueError("{} is not a valid directory".format(directory))
    return directory


# All data is stored as JSON, oriented as records
def read_data(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records")


def write_data(df: pd.DataFrame, path: str) -> None:
    df.to_json(path)
