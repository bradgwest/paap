import argparse

from ..defaults import Columns
from ..utils import valid_path, read_data, write_data


DESCRIPTION = "Filter dataset to a certain category"


def main(input_json: str, output_json: str, categories: list) -> None:
    df = read_data(input_json)
    out = df[df[Columns.SALE_CATEGORY].isin(categories)]
    write_data(out, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_json", help="Input JSON file with sale_number/sale_url column", type=valid_path)
    parser.add_argument("output_json", help="Output JSON file to write to")
    parser.add_argument("-c", "--categories", required=True, help="Categories to include", nargs="+")
    args = parser.parse_args()

    main(args.input_json, args.output_json, args.categories)
