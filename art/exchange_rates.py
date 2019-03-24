"""
Get exchange rates for various currencies
"""

import argparse
import csv
import datetime
import logging
import sys

from forex_python import converter
import cpi

YEARS = range(2006, 2020)
MONTHS = range(1, 13)
CURRENCIES = ("EUR", "GBP", "HKD", "AUD", "CHF", "INR", "CNY", "USD")

C = converter.CurrencyRates()

CURR_MAP = [[y, m, c, None, None] for y in YEARS for m in MONTHS for c in CURRENCIES]


def convert_to_dollars(base_currency='USD'):
    """
    mutates CURR_MAP
    :param base_currency:
    :return:
    """
    for l in CURR_MAP:
        # First of the month
        if l[1] == 1 and l[2] == CURRENCIES[0]:
            logging.info("Getting currencies for {}".format(l[0]))
        date_obj = datetime.datetime(l[0], l[1], 1)
        if l[2] == "USD":
            l[3] = 1
            continue
        elif date_obj > datetime.datetime.now():
            continue
        try:
            l[3] = C.convert(l[2], base_currency, 1, date_obj)
        except converter.RatesNotAvailableError:
            continue


def calculate_inflation(target_date=datetime.date(2019, 1, 1)):
    for l in CURR_MAP:
        if l[3] is not None:
            try:
                l[4] = cpi.inflate(l[3],
                                   datetime.date(l[0], l[1], 1),
                                   to=target_date)
            except cpi.errors.CPIObjectDoesNotExist:
                l[4] = l[3]


def parse_args(sys_args):
    parser = argparse.ArgumentParser(
        description="Convert historical currencies to a target")
    parser.add_argument("-c", "--target-currency", default="USD")
    parser.add_argument("-d", "--target-date", default='2019-01-01')
    parser.add_argument("-o", "--output", default="./currency_converter.csv")
    return parser.parse_args(sys_args)


def main():
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args(sys.argv[1:])
    convert_to_dollars(args.target_currency)
    calculate_inflation()
    with open(args.output, 'w') as f:
        csv_writer = csv.writer(f)
        for l in CURR_MAP:
            csv_writer.writerow(l)


if __name__ == "__main__":
    main()
