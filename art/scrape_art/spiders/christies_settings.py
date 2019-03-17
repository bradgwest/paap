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
    # TODO Comment me out
    YEARS = [2018]
    MONTHS = [7]
    ART_CATEGORIES = {"fine_art": 7, "photographs_and_prints": 11}
    LOCATIONS = {"london": 36}
    # ^^^ TODO DELETE
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
