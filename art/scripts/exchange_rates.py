import argparse
import datetime
import json
import time
from typing import Sequence, List
from collections import namedtuple

import cpi
from forex_python import converter

YEARS = range(2006, 2021)
MONTHS = range(1, 13)
DESCRIPTION = """Get the dollar equivalent of currencies at a given date, and the inflation relative to today. Output
is a csv with 5 columns: year, month, currency, dollar_equivalent, inflation.

currency is the currency which we wish to convert to dollars.
dollar_equivalent is the price of that currency, in USD at that year and month.
inflation is the multiple by which to multiply a dollar in that year/month to get it's equivalent worth at the target
date.
""".strip()


class Currency(object):
    EUR = "EUR"
    GBP = "GBP"
    HKD = "HKD"
    AUD = "AUD"
    CHF = "CHF"
    INR = "INR"
    CNY = "CNY"
    USD = "USD"
    ALL = (EUR, GBP, HKD, AUD, CHF, INR, CNY, USD)


BASE_CURRENCY = Currency.USD

ExchangeRateSnapshot = namedtuple("ExchangeRateSnapshot", ["year", "month", "currency", "dollar_equivalent", "inflation"])


def convert_to_dollars(rates: converter.CurrencyRates, year: int, month: int, currency: str) -> float:
    if currency == Currency.USD:
        return 1

    dt = datetime.datetime(year, month, 1)
    if dt > datetime.datetime.now():
        raise ValueError("{} is in the future".format(dt.strptime("%Y-%m-%d")))

    for i in range(3):
        try:
            return rates.convert(currency, BASE_CURRENCY, 1, dt)
        except converter.RatesNotAvailableError:
            time.sleep(2**i)
    else:
        print("WARNING: RatesNotAvailableError for {} in {}-{}".format(currency, year, month))
        return None


def calculate_inflation(year: int, month: int, target_date: datetime.datetime) -> float:
    try:
        return cpi.inflate(1, datetime.date(year, month, 1), to=target_date)
    except cpi.errors.CPIObjectDoesNotExist:
        return None


def make_currency_map(years: Sequence[int], months: Sequence[int], currencies: Sequence[str]) -> List[ExchangeRateSnapshot]:
    return [ExchangeRateSnapshot(y, m, c, None, None) for y in years for m in months for c in currencies]


def main(output, target_date):
    rates = converter.CurrencyRates()

    now = datetime.datetime.now()
    months = [(y, m) for y in YEARS for m in MONTHS if not (y >= now.year and m > now.month)]

    rows = []
    for y, m in months:
        # calculate how much more dollars are worth today
        inflation = calculate_inflation(y, m, target_date)
        for c in Currency.ALL:
            # calculate how many dollars, 1 currency unit is worth
            dollar_equivalent = convert_to_dollars(rates, y, m, c)
            rows.append(
                ExchangeRateSnapshot(y, m, c, dollar_equivalent, inflation)._asdict()
            )

    with open(output, "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-o", "--output", help="Output to write json to")
    parser.add_argument("-d", "--target-date", default="2020-01-01",
                        type=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    args = parser.parse_args()

    main(args.output, args.target_date)
