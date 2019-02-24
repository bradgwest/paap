"""
Download images from a url
"""

import argparse
import csv
import logging
import os
import sys
import time
import urllib.error
import urllib.request


def get_images(input_csv, directory, sleep):
    if not os.path.isdir(directory):
        raise ValueError("{} is not an existing directory".format(directory))
    num_failed = 0
    with open(input_csv, newline="") as csv_file:
        reader = csv.DictReader(csv_file, fieldnames=("url", "basename"))
        for row in reader:
            path = os.path.join(directory, row["basename"])
            try:
                _, info = urllib.request.urlretrieve(row["url"], path)
            except urllib.error.URLError:
                logging.warning("Download failed for {}".format(row["url"]))
                num_failed += 1
            time.sleep(sleep)
    logging.info("Finished Downloading Images. Failures: {}".format(num_failed))


def parse_arguments(sys_args):
    parser = argparse.ArgumentParser(
        description="Downloads images at the urls specified in a CSV file"
    )
    parser.add_argument("input",
                        help="Path to the file which should be a two column "
                             "csv, url and name, no header, where url is the "
                             "url of the image and name is the basename for the "
                             "image")
    parser.add_argument("directory",
                        help="Directory to save the images to. Must already "
                             "exist")
    parser.add_argument("-s", "--sleep", type=float, default=0.001,
                        help="The amount of time in seconds to sleep (in "
                             "seconds) before making the next request")
    return parser.parse_args(sys_args)


def main():
    args = parse_arguments(sys.argv[1:])
    get_images(args.input, args.directory, args.sleep)


if __name__ == "__main__":
    main()
