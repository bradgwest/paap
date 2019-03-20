"""
Clean data scraped from Christies Website.
"""

import argparse
import csv
import json
import logging
import re
import uuid

import google.auth
from google.cloud import storage

from art import gc_utils
from art.scrape_art.spiders import christies_settings


# Lot = collections.namedtuple("Lot", christies_settings.LOT_FIELD_NAMES)


def clean_sale_number(sale_number):
    if not sale_number:
        return None
    mtch = re.search("[0-9]+", sale_number)
    if not mtch:
        return None
    return mtch.group(0)


def clean_sale_total(total_raw):
    mtch = re.search("[0-9,]+", total_raw)
    total = int(mtch.group(0).replace(",", "")) if mtch else None
    mtch = re.search("(?<=\\+)[A-Z\\s]+(?=[0-9])", total_raw)
    currency = mtch.group(0).strip() if mtch else None
    return total, currency


def clean_img_url(img_url):
    if not img_url:
        return None
    loc = img_url.find("?")
    if loc == -1:
        return img_url
    return img_url[:loc]


def clean_iso_estimate(estimate):
    amount_re = "[0-9,]+"
    low_raw, high_raw = estimate.split("-")
    code_match = re.search("^[A-Z]+", low_raw)
    code = code_match.group(0) if code_match else None
    low_match = re.search(amount_re, low_raw)
    low = low_match.group(0).replace(",", "") if low_match else None
    high_match = re.search(amount_re, high_raw)
    high = high_match.group(0).replace(",", "") if high_match else None
    return code, int(low), int(high)


def clean_secondary_estimate(estimate):
    amount_re = "[0-9,]+"
    raw = re.findall(amount_re, estimate)
    low = raw[0].replace(",", "") if raw else None
    high = raw[1].replace(",", "") if len(raw) > 1 else None
    return int(low), int(high)


def clean_realized(realized):
    found = re.findall("[0-9,]+", realized)
    if not found:
        return None
    return int(found[0].replace(",", ""))


class ChristiesSaleParser(object):

    def __init__(self):
        self.lots = []  # Individual lots

    def process(self):
        raise NotImplementedError("process must be called on a subclass")

    @staticmethod
    def add_lot_id(lot):
        lot['id'] = str(uuid.uuid4)

    def sale_to_lots(self, sale):
        if sale["category"] == "online":
            self.lots += self.add_lot_details(sale, 'js')
            return
        self.lots += self.add_lot_details(sale, 'html')

    def add_lot_details(self, sale, details_key):
        """
        Add details to a lot
        :param dict sale:
        :param str details_key: one of 'html' or 'js'
        :return:
        """
        allowed = ('html', 'js')
        if details_key not in allowed:
            raise ValueError("details_key must be in {}".format(allowed))
        details = sale.get("sale_details_{}".format(details_key))
        if not details:
            return []
        items = details if details_key == 'html' else details['items']
        for i in items:
            lot = {k: None for k in christies_settings.LOT_FIELD_NAMES}
            self.add_sale_fields_general(lot, sale)
            self.add_lot_id(lot)
            if details_key == 'html':
                self.add_html_details(lot, i)
            else:
                attrs = self.get_sale_attribute_types(details)
                self.add_sale_lot_fields_js(lot, details)
                self.add_js_details(lot, i, attrs)
            self.lots.append(lot.copy())

    @staticmethod
    def get_sale_attribute_types(details):
        return {}

    @staticmethod
    def add_sale_fields_general(lot, sale):
        lot["input_url"] = sale.get("input_url")
        lot["sale_year"] = sale.get("year")
        lot["sale_month"] = sale.get("month")
        lot["sale_category"] = sale.get("category")
        lot["sale_location"] = sale.get("location")
        lot["sale_location_id"] = sale.get("location_int")
        lot["sale_url"] = sale.get("sale_url")
        lot["sale_status"] = sale.get("sale_status")
        lot["sale_number"] = clean_sale_number(sale.get("sale_number"))
        lot["sale_total_realized_iso_currency"], sale["sale_iso_currency_code"] = \
            clean_sale_total(sale.get("sale_total_raw"))

    @staticmethod
    def add_sale_lot_fields_js(lot, details):
        lot["sale_lot_count"] = details.get("totalItemsCount")
        lot["sale_title"] = details.get("canonicalTitle")
        lot["sale_christies_id"] = details.get("saleId")

    @staticmethod
    def add_html_details(lot, itm):
        lot["lot_image_url"] = clean_img_url(itm.get("image_url"))
        lot["lot_number"] = itm.get("lot_number")
        lot["lot_artist"] = itm.get("maker")
        lot["lot_description"] = itm.get("description")
        lot["medium"] = itm.get("medium")
        lot["dimensions"] = itm.get("dimensions")
        lot["lot_realized_price_iso_currency"] = clean_realized(itm.get("realized_primary"))
        lot["lot_realized_price_usd"] = clean_realized(itm.get("realized_secondary"))
        _, iso_est_low, iso_est_high = clean_iso_estimate(itm.get["estimate_primary"])
        lot["lot_estimate_low_iso_currency"] = iso_est_low
        lot["lot_estimate_high_iso_currency"] = iso_est_high
        usd_est_low, usd_est_high = clean_secondary_estimate(itm.get["estimate_secondary"])
        lot["lot_estimate_low_usd"] = usd_est_low
        lot["lot_estimate_high_usd"] = usd_est_high

    @staticmethod
    def add_js_details(lot, itm, attrs):
        lot["item_id"] = itm.get("itemId")
        # TODO Finish me

    def write_lots(self, output):
        if output.startswith("gs://"):
            self.write_to_gcs(output)
        else:
            self.write_to_local(output)

    def write_to_gcs(self, output):
        raise NotImplementedError("Must call on a GCS based subclass")

    def write_to_local(self, output):
        with open(output, 'w') as f:
            lot_writer = csv.writer(f)
            for lot in self.lots:
                lot_writer.writerow(lot)


class GCSChristiesSaleParser(ChristiesSaleParser):

    def __init__(self, input_bucket_path=None, input_files=None, project=None):
        super().__init__()
        if input_bucket_path and input_files:
            raise ValueError("Specify only one of input_bucket_path or input_files")

        self.project = project
        credentials, project_id = google.auth.default()
        if not self.project:
            self.project = project_id
        self.client = storage.Client(self.project, credentials=credentials)

        self.blobs_to_parse = []

        if input_files:
            with open(input_files) as f:
                files_to_parse = f.readlines()
            self.blobs_to_parse = []
            for p in files_to_parse:
                if not (p.endswith(".json") and p.startswith("gs://")):
                    raise ValueError("Files to parse must start with gs:// and end with .json")
                bucket_name, blob_name = gc_utils.bucket_and_path_from_uri(input_bucket_path)
                self.bucket = self.client.get_bucket(bucket_name)
                self.blobs_to_parse.append(self.bucket.get_blob(blob_name))
        else:
            bucket_name, prefix = gc_utils.bucket_and_path_from_uri(input_bucket_path)
            self.bucket = self.client.get_bucket(bucket_name)
            for blob in self.bucket.list_blobs(prefix=prefix, delimiter="/"):
                self.blobs_to_parse.append(blob)

    def process(self):
        """
        Transforms a blob into a list of lots
        """
        for blob in self.blobs_to_parse:
            json_string = blob.download_as_string()
            sale = json.loads(json_string)
            self.sale_to_lots(sale)

    def write_to_gcs(self, output):
        pass


def parse_arguments(sys_args):
    parser = argparse.ArgumentParser(
        description="Clean json data, scraped from Christies into a format that "
                    "can be used for predictive analytics")
    parser.add_argument("input_path",
                        help="path of files to process, like gs://paap/christies/data/raw")
    parser.add_argument("input_files",
                        help="Alternative way of specifying files to process, one file per line")
    parser.add_argument("output", help="csv to save to. Can be local or GCS")
    return parser.parse_args(sys_args)


def main():
    logging.getLogger().setLevel(logging.INFO)
    pass


if __name__ == "__main__":
    main()
