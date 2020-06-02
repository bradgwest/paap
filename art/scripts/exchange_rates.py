import argparse
import csv
import datetime
from typing import Sequence, List
from collections import namedtuple

import cpi
from forex_python import converter

YEARS = range(2006, 2020)
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

C = converter.CurrencyRates()

ExchangeRateSnapshot = namedtuple("ExchangeRateSnapshot", ["year", "month", "currency", "dollar_equivalent", "inflation"])


def convert_to_dollars(year: int, month: int, currency: str) -> float:
    if currency == Currency.USD:
        return 1

    dt = datetime.datetime(year, month, 1)
    if dt > datetime.datetime.now():
        raise ValueError("{} is in the future".format(dt.strptime("%Y-%m-%d")))

    # might raise converter.RatesNotAvailableError
    return C.convert(currency, BASE_CURRENCY, 1, dt)


def calculate_inflation(year: int, month: int, target_date: datetime.datetime) -> float:
    # might raise cpi.errors.CPIObjectDoesNotExist
    return cpi.inflate(1, datetime.date(year, month, 1), to=target_date)


def make_currency_map(years: Sequence[int], months: Sequence[int], currencies: Sequence[str]) -> List[ExchangeRateSnapshot]:
    return [ExchangeRateSnapshot(y, m, c, None, None) for y in years for m in months for c in currencies]


def main(output, target_date):
    rows = []
    for y in YEARS:
        for m in MONTHS:
            # calculate how much more dollars are worth today
            inflation = calculate_inflation(y, m, target_date)
            for c in Currency.ALL:
                # calculate how many dollars, 1 currency unit is worth
                dollar_equivalent = convert_to_dollars(y, m, c)
                rows.append(
                    ExchangeRateSnapshot(y, m, c, dollar_equivalent, inflation)
                )

    with open(output, "w") as f:
        writer = csv.DictWriter(f)
        writer.writeheader(ExchangeRateSnapshot._fields)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("-o", "--output")
    parser.add_argument("-d", "--target-date", default="2020-01-01",
                        type=lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

    args = parser.parse_args()

    main(args.output, args.target_date)
