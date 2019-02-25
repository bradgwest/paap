"""
Clean data scraped from Christies Website.
"""

import argparse
import logging
import sys


def clean_sale(sale):
    """
    Clean a sales data,
    :param str sale: A json object, representing a sale at christies
    :returns: A list of tuples, each tuple represents a sale of piece of artwork
    :rtype: list
    """
    works = []
    return works


def parse_arguments(sys_args):
    parser = argparse.ArgumentParser(
        description="Clean json newline delimited data, scraped from "
                    "Christies into a format that can be used for predictive "
                    "analytics")
    parser.add_argument("input", help="Path to json lines file")
    parser.add_argument("output", help="Path to save output to")
    return parser.parse_args(sys_args)


def main():
    args = parse_arguments(sys.argv[1:])
    csv_output = open(args.output, "w+")
    with open(args.input) as json_lines:
        for line in json_lines:
            cleaned_sale = clean_sale(line)
            for work in cleaned_sale:
                csv_output.write(work)
    csv_output.close()


if __name__ == "__main__":
    main()
