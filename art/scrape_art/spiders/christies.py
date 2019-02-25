"""
Scraper for Christies auction results
"""

import json
import re

import scrapy

from art.scrape_art.spiders import christies_settings


class ChristiesCrawler(scrapy.Spider):
    name = "christies"
    start_urls = christies_settings.create_urls()
    # TODO Delete me
    start_urls = ["https://www.christies.com/results?sc_lang=en&month=7&year=2018&scids=11"]

    def parse(self, response):
        """
        Entry point for parsing Christies auction results
        :param response:
        :return:
        """
        sales = response.xpath('//{}/li'.format(christies_settings.TAGS["sales_or_events"]))

        for sale in sales:
            # TODO These are brittle
            status = sale.xpath('./div/h4[contains(@class, "sale-status")]/text()').get()
            number = sale.xpath('./div/div/span[contains(@class, "sale-number")]/text()').get()
            location = sale.xpath('./div/div/span[contains(@class, "location")]/text()').get()
            total = sale.xpath('./div/div/div[contains(@class, "sale-total")]/text()').get()

            sale_details = {
                "sale_url": response.url,
                "sale_status": status,
                "sale_number": number,
                "sale_location": location,
                "sale_total": total,
                "sale_details": None
            }

            sale_page = sale.xpath('./div/div/a[text() = "View results"]/@href').get()
            if sale_page is not None:
                next_page = response.urljoin(sale_page)
                yield scrapy.Request(next_page,
                                     callback=self.parse_redirect_sale_page,
                                     meta={"sale_details": sale_details})

    def parse_redirect_sale_page(self, response):
        """
        This is a simple function to get the landing page url for the sale
        :param response:
        :return:
        """
        all_results_url = response.url + "&ShowAll=true"
        sale_details = response.meta["sale_details"]
        yield scrapy.Request(all_results_url, callback=self.parse_sale_page,
                             meta={"sale_details": sale_details})

    def parse_sale_page(self, response):
        """
        The sale page is the home page for that auction
        :param response:
        :return:
        """
        js = response.xpath('//script[@type="text/javascript"][contains(text(), "var saleName")]/text()').get()
        try:
            details = self.parse_js(js)
        except json.JSONDecodeError:
            details = None

        sale_details = response.meta["sale_details"]
        sale_details["sale_details"] = details

        yield sale_details

    @staticmethod
    def parse_js(js):
        """
        Parses javascript function to json object
        :param js:
        :return:
        """
        try:
            list_view_loc = re.search("var lotListViewModel", js).regs[0][1]
            starting_bracket = re.search("{", js[list_view_loc:]).regs[0][0]
            start = int(list_view_loc) + int(starting_bracket)
            # first double newline is end of object
            ending_bracket = re.search("}\\);", js[start:]).regs[0][0]
        except IndexError:
            return None
        except AttributeError:
            return None

        js_obj = js[start:(start + ending_bracket + 1)]
        return json.loads(js_obj)
