import argparse
import csv
import datetime
import json
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List, NamedTuple, Tuple

DESCRIPTION = "Clean json data, scraped from Christies into a format that can be used for predictive analytics"

# Regexes for parsing
SALE_NUMBER_REGEX = re.compile(r"[0-9]+")
SALE_TOTAL_RAW_REGEX = re.compile(r"^Sale total: \+ (?P<currency>[A-Z]+) (?P<total_raw>[0-9\.,]+)")
LOT_NUMBER_REGEX = re.compile(r"^Lot (?P<number>[0-9\sA-Z]+)$")
LOT_REALIZED_PRICE = re.compile(r"^[^0-9,]+(?P<price>[0-9,]+)$")
NO_PUNCTUATION_REGEX = re.compile(r"\D")
JS_MAKER_REGEX = re.compile(r"^(?P<maker>[\w\s\.]+)\s\([\s0-9bB\-\.]+\)$")

# Date Formats
SALE_STATUS_DATE_FORMAT = "%d %b %Y"
OUTPUT_DATE_FORMAT = "%Y-%m-%d"

EXPECTED_SALE_KEYS = frozenset(
    [
        "category",
        "input_url",
        "location",
        "location_int",
        "month",
        "sale_details_html",
        "sale_details_js",
        "sale_number",
        "sale_status",
        "sale_total_raw",
        "sale_url",
        "year",
    ]
)
EXPECTED_HTML_KEYS = frozenset(
    [
        "description",
        "estimate_primary",
        "estimate_seconday",
        "image_url",
        "maker",  # May be missing on some
        "medium_dimensions",
        "number",
        "realized_primary",
        "realized_secondary",
    ]
)

EXPECTED_MISSING_HTML_OUTPUT_KEYS = frozenset(
    [
        "lot_christies_unique_id",
        "lot_title",
        "lot_item_id",
        "sale_analytics_name",
        "sale_analytics_type",
        "sale_christies_id",
        "sale_id",
        "sale_iso_currency_code",
        "sale_title",
        "sale_total_items",
        "sale_type",
    ]
)
EXPECTED_MISSING_JS_OUTPUT_KEYS = frozenset(["lot_dimensions", "lot_number", "lot_medium"])
# Output keys
OUTPUT_KEYS = [
    "sale_christies_id",
    "sale_id",
    "sale_number",
    "sale_url",
    "sale_input_url",
    "sale_is_html_type",
    "sale_year",
    "sale_month",
    "sale_date",
    "sale_location",
    "sale_location_int",
    "sale_category",
    "sale_currency",
    "sale_iso_currency_code",
    "sale_title",
    "sale_analytics_name",
    "sale_analytics_type",
    "sale_total",
    "sale_total_items",
    "sale_type",
    "lot_christies_unique_id",
    "lot_item_id",
    "lot_number",
    "lot_realized_price",
    "lot_title",
    "lot_artist",
    "lot_description",
    "lot_dimensions",
    "lot_medium",
    "lot_image_url",
]

SALE_BLACKLIST = frozenset(
    [
        "2017_7_asian_art_online_14966.json",
        "2017_9_fine_art_online_16002.json",
        "2018_2_asian_art_online_16680.json",
        "2018_2_fine_art_online_16680.json",
        "2018_5_asian_art_online_16099.json",
        "2018_5_photographs_and_prints_online_16099.json",
        "2018_7_fine_art_online_15880.json",
    ]
)


class NotSoldException(ValueError):
    pass


class ProcessFunction(NamedTuple):
    output_keys: Tuple[str]
    input_key: str
    func: Callable
    nullable: bool = False


def valid_path(path_str: str) -> Path:
    path = Path(path_str)
    if not (path.exists() and path.is_file()):
        raise ValueError("{} is not a valid path".format(path))
    return path


def assert_identical_sets(actual: set, expected: Iterable) -> None:
    assert (set(actual) - set(expected)) == set(), "Unexpected items: {}".format(set(actual) - set(expected))
    assert set(expected) - set(actual) == set(), "Missing items: {}".format(set(expected) - set(actual))


def process_sale_total_raw(raw: str) -> str:
    match = re.search(SALE_TOTAL_RAW_REGEX, raw)
    assert match, "process_sale_number - Did not find match for sale total: {}".format(raw)
    currency, total_raw = match.group("currency", "total_raw")
    total = int(re.sub(NO_PUNCTUATION_REGEX, "", total_raw))
    return currency, total


def process_sale_number(raw: str) -> int:
    match = re.search(SALE_NUMBER_REGEX, raw)  # This might fail if it's not there
    assert match, "process_sale_number - Did not find sale_number: {}".format(raw)
    return int(match.group(0))


def process_sale_status(raw: str) -> str:
    """Sale status is really just the date"""
    if "-" in raw:
        # multi-day
        raw = raw.split("-")[-1].strip()

    if not re.match(r"^[0-9]{2}", raw):
        # Pad it with a leading zero
        raw = "0" + raw
    dt = datetime.datetime.strptime(raw, SALE_STATUS_DATE_FORMAT)
    return dt.strftime(OUTPUT_DATE_FORMAT)


def process_js_maker(raw: str) -> str:
    match = re.search(JS_MAKER_REGEX, raw)
    try:
        assert match, "process_js_maker - Did not find match for maker: {}".format(raw)
    except AssertionError:
        print("ERROR - didn't find js maker: {}".format(raw))
        return None
    return match.group("maker")


def process_js_title(raw: str) -> str:
    try:
        assert raw, "process_js_title - title is Falsey: {}".format(raw)
    except AssertionError:
        print("ERROR - title is falsey: {}".format(raw))
        return None
    return raw


def apply_process_functions(raw: dict, process_functions: List[ProcessFunction]) -> dict:
    out = {}
    for pf in process_functions:
        if pf.nullable and pf.input_key not in raw:
            vals = tuple(None for _ in pf.output_keys)
        else:
            vals = pf.func(raw[pf.input_key])

        if not isinstance(vals, tuple):
            vals = (vals,)
        out.update(dict(zip(pf.output_keys, vals)))
    return out


def process_js_lot(raw_lot_details: dict) -> dict:
    if raw_lot_details.get("anyBidsPlaced") is False and not raw_lot_details.get("priceRealised"):
        raise NotSoldException("Unsold")

    process_functions = [
        ProcessFunction(("lot_item_id",), "itemId", int),
        ProcessFunction(("lot_christies_unique_id",), "christiesUniqueId", str),
        ProcessFunction(("lot_artist",), "canonicalTitle", process_js_maker),
        ProcessFunction(("lot_title",), "translatedArtist", process_js_title),
        ProcessFunction(("lot_description",), "translatedDescription", str),
        ProcessFunction(("lot_realized_price",), "priceRealised", int),
        ProcessFunction(("lot_image_url",), "imageUrl", process_image_url),
    ]

    return apply_process_functions(raw_lot_details, process_functions)


def _process_sale_details_js_wo_lot(sale_details: dict) -> dict:
    process_functions = [
        ProcessFunction(("sale_id",), "saleId", int),
        ProcessFunction(("sale_title",), "canonicalTitle", str),
        ProcessFunction(("sale_analytics_name",), "analyticsSaleName", str),
        ProcessFunction(("sale_analytics_type",), "analyticsSaleType", str),
        ProcessFunction(("sale_christies_id",), "christiesSaleId", int),
        ProcessFunction(("sale_type",), "saleType", int),
        ProcessFunction(("sale_iso_currency_code",), "isoCurrencyCode", str),
        ProcessFunction(("sale_total_items",), "totalItemsCount", int),
    ]
    return apply_process_functions(sale_details, process_functions)


def process_sale_details_js(sale: dict) -> dict:
    sale_details = sale["sale_details_js"]
    non_lot_details = _process_sale_details_js_wo_lot(sale_details)
    lots = []
    for l in sale_details["items"]:

        try:
            lot = process_js_lot(l)
        except NotSoldException:
            print("WARNING - Item not sold")
            continue
        except KeyError as e:
            if "priceRealised" in str(e):
                print("WARNING - priceRealised issue")
                continue
            else:
                raise

        lot.update(non_lot_details)
        lots.append(lot)
    return lots


def process_image_url(raw: str) -> str:
    return "".join(raw.split("?")[:-1])


def process_html_lot_number(raw: str) -> str:
    match = re.search(LOT_NUMBER_REGEX, raw)
    assert match, "process_html_lot_number - Did not find lot number: {}".format(raw)
    return match.group("number")


# TODO maybe we could do more with this if you find a pattern
def process_html_medium_dimensions(raw: str) -> str:
    if raw is None:
        return "", ""
    split = raw.split(",")
    if len(split) == 1:
        return split[0], ""
    return split[0], ";".join(m.strip() for m in split[1:])


def process_html_realized_price(raw: str) -> str:
    match = re.search(LOT_REALIZED_PRICE, raw)
    assert match, "process_html_realized_price - Did not find realized price: {}".format(raw)
    price_with_punctuation = match.group("price")
    return price_with_punctuation
    price = re.sub(NO_PUNCTUATION_REGEX, "", price_with_punctuation)
    return int(price)


def process_html_lot(raw: dict) -> dict:
    # actual_keys = set(raw.keys())
    # # Add some keys that might be missing
    # actual_keys.union({"maker", "estimate_seconday", "realized_secondary"})
    # assert_identical_sets(actual_keys, EXPECTED_HTML_KEYS)

    processor_functions = {
        ProcessFunction(("lot_image_url",), "image_url", process_image_url),
        ProcessFunction(("lot_number",), "number", process_html_lot_number),
        ProcessFunction(("lot_artist",), "maker", str, True),
        ProcessFunction(("lot_description",), "description", str),
        ProcessFunction(("lot_dimensions", "lot_medium"), "medium_dimensions", process_html_medium_dimensions, True),
        ProcessFunction(("lot_realized_price",), "realized_primary", process_html_realized_price),
    }
    return apply_process_functions(raw, processor_functions)


def process_sale_html_details(sale: dict) -> dict:
    """We're intentionally dropping primary and secondary estimates here, in the interest of time"""
    raw = sale["sale_details_html"]

    lots = []
    for l in raw:
        try:
            lots.append(process_html_lot(l))
        except KeyError as e:
            if "realized_primary" in str(e):
                print("ERROR - KeyError realized_primary: {}, {}".format(l.get("number"), sale.get("input_url")))
                continue
            raise

    return lots


def process_sale_details(sale: dict) -> dict:
    processor_functions = {
        ProcessFunction(("sale_input_url",), "input_url", str),
        ProcessFunction(("sale_year",), "year", str),
        ProcessFunction(("sale_month",), "month", str),
        ProcessFunction(("sale_category",), "category", str),
        ProcessFunction(("sale_location",), "location", str),
        ProcessFunction(("sale_location_int",), "location_int", str),
        ProcessFunction(("sale_url",), "sale_url", str),
        ProcessFunction(("sale_number",), "sale_number", process_sale_number),
        ProcessFunction(("sale_date",), "sale_status", process_sale_status),
        ProcessFunction(("sale_currency", "sale_total"), "sale_total_raw", process_sale_total_raw),
        ProcessFunction(("sale_is_html_type",), "sale_details_html", bool),
    }
    return apply_process_functions(sale, processor_functions)


def process_lot_details(sale: dict) -> dict:
    assert not (sale["sale_details_html"] and sale["sale_details_js"]), "Sale has both js and html details"
    assert not (
        sale["sale_details_html"] and sale["location"] == "online"
    ), "Sale is online but does not have js details"

    if sale["sale_details_html"]:
        return process_sale_html_details(sale)
    return process_sale_details_js(sale)


def process_raw_file(file_name: str) -> List[dict]:
    with open(file_name) as f:
        sale = json.load(f)

    assert_identical_sets(sale.keys(), EXPECTED_SALE_KEYS)

    sale_details = process_sale_details(sale)
    lot_details = process_lot_details(sale)

    for lot in lot_details:
        lot.update(sale_details)

        # Assert that we have only the keys we want
        if lot["sale_is_html_type"]:
            expected_missing = EXPECTED_MISSING_HTML_OUTPUT_KEYS
        else:
            expected_missing = EXPECTED_MISSING_JS_OUTPUT_KEYS
        actual_missing = set(OUTPUT_KEYS) - set(lot.keys())
        assert actual_missing == expected_missing, "Missing keys don't match expected: Have: {}; Want: {}".format(
            sorted(actual_missing), sorted(expected_missing)
        )

        # Add missing keys
        lot.update({k: None for k in expected_missing})

        assert set(lot.keys()) == set(OUTPUT_KEYS)

    return lot_details


def main(input_files: str, output_path: str) -> None:
    # Iterate over each file and process it
    rows = []
    for fn in input_files:
        if os.path.basename(fn) in SALE_BLACKLIST:
            print("WARNING - Will not process, file is in blacklist: {}".format(fn))
            continue

        print("Processing {}".format(fn))
        rows.extend(process_raw_file(fn))

    with open(output_path, "w") as f:
        writer = csv.DictWriter(f, OUTPUT_KEYS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-i",
        "--input-files",
        nargs="+",
        required=True,
        type=valid_path,
        help="Input newline delimited json files to process.",
    )
    parser.add_argument("-o", "--output-path", required=True, help="CSV to save to")
    args = parser.parse_args()

    main(args.input_files, args.output_path)
