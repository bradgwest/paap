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
MONTHS = [m for m in range(1, 13)]
MONTH_QUERY = "month"
YEARS = [y for y in range(1998, 2020)]
YEAR_QUERY = "year"

URL_FORMAT = "https://www.christies.com/results?{langq}={lang}&{monthq}={month}&{yearq}={year}&{artcatq}={artcat}"


def create_urls():
    urls = []
    for y in YEARS:
        for m in MONTHS:
            for c in ART_CATEGORIES:
                url = URL_FORMAT.format(langq=LANGUAGE_QUERY,
                                        lang=LANGUAGE_CATEGORIES["english"],
                                        monthq=MONTH_QUERY,
                                        month=m,
                                        yearq=YEAR_QUERY,
                                        year=y,
                                        artcatq=ART_CATEGORY_QUERY,
                                        artcat=c)
                urls.append(url)
    return urls


TAGS = {
    "sales_or_events": "cc-sales-or-events-list"
}