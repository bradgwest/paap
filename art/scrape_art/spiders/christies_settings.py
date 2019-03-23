"""
Settings for Christies' website
"""

ART_CATEGORY_QUERY = "scids"
ART_CATEGORIES = {
    "african_oceanic_and_precolumbian": 22,
    "asian_art": 5,
    "fine_art": 7,
    "islamic_and_eastern_european": 17,
    "photographs_and_prints": 11
}
LANGUAGE_QUERY = "sc_lang"
LANGUAGE_CATEGORIES = {
    "english": "en"
}
LOCATIONS = {
    "amsterdam": 28,
    "beaune": 95,
    "dubai": 91,
    "geneva": 33,
    "hong_kong": 35,
    "london": 36,
    "milan": 40,
    "mumbai": 105,
    "new_york": 43,
    "online": 115,
    "other": 82,
    "paris": 45,
    "shanghai": 104,
    "zurich": 54
}
LOCATION_QUERY = "locations"
MONTHS = [m for m in range(1, 13)]
MONTH_QUERY = "month"
YEARS = [y for y in range(1998, 2020)]
YEAR_QUERY = "year"

URL_FORMAT = "https://www.christies.com/results?{langq}={lang}&{monthq}={month}&{yearq}={year}&{artcatq}={artcat}&{locq}={loc}"


def create_url_list():
    urls = []
    for y in YEARS:
        for m in MONTHS:
            for c in ART_CATEGORIES:
                for l in LOCATIONS:
                    url = URL_FORMAT.format(langq=LANGUAGE_QUERY,
                                            lang=LANGUAGE_CATEGORIES["english"],
                                            monthq=MONTH_QUERY,
                                            month=m,
                                            yearq=YEAR_QUERY,
                                            year=y,
                                            artcatq=ART_CATEGORY_QUERY,
                                            artcat=ART_CATEGORIES[c],
                                            locq=LOCATION_QUERY,
                                            loc=LOCATIONS[l])
                    urls.append(
                        {'year': y,
                         'month': m,
                         'category': c,
                         'location': l,
                         'location_int': LOCATIONS[l],
                         'url': url}
                    )
    return urls


TAGS = {
    "sales_or_events": "cc-sales-or-events-list"
}

LOT_FIELD_NAMES = [
    "received_timestamp",
    "id",
    "input_url",
    "sale_year",
    "sale_month",
    "sale_category",
    "sale_location",
    "sale_location_id",
    "sale_url",
    "sale_status",
    "sale_number",
    "sale_total_realized_iso_currency",
    "sale_total_realized_usd",
    "sale_lot_count",
    "sale_title",
    "sale_christies_id",
    "sale_iso_currency_code",
    "sale_secondary_currency_code",
    "lot_start_date",
    "lot_end_date",
    "lot_item_id",
    "lot_number",
    "lot_artist",
    "lot_title",
    "lot_description",
    "lot_medium",
    "lot_dimensions",
    "lot_estimate_low_iso_currency",
    "lot_estimate_low_usd",
    "lot_estimate_high_iso_currency",
    "lot_estimate_high_usd",
    "lot_realized_price_iso_currency",
    "lot_realized_price_usd",
    "lot_image_url",
    "lot_attributes",
    "lot_tags_date",
    "lot_tags_design",
    "lot_tags_material",
    "lot_tags_object",
    "lot_tags_origin_tags",
    "lot_tags_style",
    "lot_tags_subject_matter",
    "lot_tags_type",
    "lot_translated_title",
    "lot_translated_description",
    "lot_translated_artist",
    "lot_starting_bid_usd",
    "lot_current_bid_usd",
    "lot_iso_currency_starting_bid",
    "lot_iso_currency_current_bid",
    "lot_current_bid_less_than_reserve",
    "lot_any_bids_placed",
    "lot_number_available",
    "lot_is_live_auction"
]
