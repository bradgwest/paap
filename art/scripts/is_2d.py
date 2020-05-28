import argparse
import json
import webbrowser

import pandas as pd


DESCRIPTION = "Iterate over sales in a dataframe, and determine if they are exclusively 2d artwork"
S_NUMBER = "sale_number"
S_URL = "sale_url"
S_CATEGORY = "sale_category"
IS_2D = "sale_is_2d"


def check_sale(sale_number: str, sale_url: str) -> bool:
    try:
        webbrowser.open(sale_url, new=0)
    except webbrowser.Error:
        print("Failed to open {}".format(sale_url))

    while True:
        raw = input("Sale {} has 2d artwork only? (y/n)".format(sale_number))

        if raw.lower() == "y":
            return True

        if raw.lower() == "n":
            return False

        print("Invalid input, responsd 'y' or 'n'")


def main(input_json: str, output_json: str) -> None:
    df = pd.read_json(input_json, orient="records")

    rows = []
    try:
        with open(output_json) as f:
            rows = json.load(f)
    except FileNotFoundError:
        pass

    completed = {(s[S_NUMBER], s[S_URL]) for s in rows}
    print("Found {} completed sales".format(len(completed)))

    sales = df[[S_NUMBER, S_URL, S_CATEGORY]]
    sales = sales[~sales.duplicated()]

    # Free up some memory
    del(df)

    try:
        for r in sales.iterrows():
            sale_number = r[1][0]
            sale_url = r[1][1]
            sale_category = r[1][2]

            if (sale_number, sale_url) in completed:
                continue

            if sale_category == "photographs_and_prints":
                is_2d = True
            else:
                is_2d = check_sale(sale_number, sale_url)

            rows.append({S_NUMBER: sale_number, S_URL: sale_url, IS_2D: is_2d})
    finally:
        print("Writing {} rows".format(len(rows)))
        with open(output_json, "w") as f:
            json.dump(rows, f)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("input_json", help="Input JSON file with sale_number/sale_url column")
    parser.add_argument("output_json", help="Output JSON file to write to")
    args = parser.parse_args()

    main(args.input_json, args.output_json)
