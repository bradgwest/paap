import argparse

import pandas as pd

DESCRPTION = "Calculate prices in Jan 1, 2020 USD"


def main(filter_artwork: str, exchange_rates: str, output: str) -> None:
    df = pd.read_json(filter_artwork, orient="records")

    rates = pd.read_json(exchange_rates, orient="records")
    rates.columns = ["sale_year", "sale_month", "sale_currency", "usd_equivalent", "inflation_to_2020"]

    df = pd.merge(df, rates, how="left", on=("sale_year", "sale_month", "sale_currency"))
    for c in ("lot_realized_price", "lot_estimate_low", "lot_estimate_high"):
        df[c + "_usd"] = df[c] * df["inflation_to_2020"] * df["usd_equivalent"]

    print("Writing {} rows to {}".format(df.shape[0], output))
    df.to_json(output, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRPTION)
    parser.add_argument("--filtered-artwork", default="data/filter_artwork_output.json",
                        help="Path to filtered artwork json, the output from filter_artwork")
    parser.add_argument("--exchange-rates", default="data/exchange_rates_output.json",
                        help="Path to the output of exchange rates")
    parser.add_argument("--output", default="data/calculate_modern_price_output.py",
                        help="Output path to write the dataset to")
    args = parser.parse_args()

    main(args.filtered_artwork, args.exchange_rates, args.output)
