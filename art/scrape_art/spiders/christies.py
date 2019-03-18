"""
Scraper for Christies auction results
"""

import copy
import json
import re

import scrapy

from art.scrape_art import items
from art.scrape_art.spiders import christies_settings


class ChristiesCrawler(scrapy.Spider):

    name = "christies"

    def start_requests(self):
        months = christies_settings.create_url_list()
        # Reverse so that we get hits early
        for m in reversed(months):
            yield scrapy.Request(url=m["url"],
                                 callback=self.parse,
                                 meta={"page_meta": m})

    def parse(self, response):
        """
        Entry point for parsing Christies auction results
        :param response:
        :return:
        """
        sales = response.xpath('//{}/li'.format(christies_settings.TAGS["sales_or_events"]))
        page_meta = response.meta["page_meta"]

        for sale in sales:
            # TODO brittle, fix them
            status = sale.xpath('./div/h4[contains(@class, "sale-status")]/text()').get()
            number = sale.xpath('./div/div/span[contains(@class, "sale-number")]/text()').get()
            total = sale.xpath('./div/div/div[contains(@class, "sale-total")]/text()').get()

            sale_item = items.ChristiesItem()
            sale_item["input_url"] = page_meta["url"]
            sale_item["year"] = page_meta["year"]
            sale_item["month"] = page_meta["month"]
            sale_item["category"] = page_meta["category"]
            sale_item["location"] = page_meta["location"]
            sale_item["location_int"] = page_meta["location_int"]
            sale_item["sale_url"] = response.url
            sale_item["sale_status"] = status
            sale_item["sale_number"] = number
            sale_item["sale_total_raw"] = total
            sale_item["sale_details_js"] = []
            sale_item["sale_details_html"] = []

            sale_page = sale.xpath('./div/div/a[text() = "View results"]/@href').get()
            if sale_page is not None:
                next_page = response.urljoin(sale_page)
                yield scrapy.Request(next_page,
                                     callback=self.parse_redirect_sale_page,
                                     meta={"item": sale_item})

    def parse_redirect_sale_page(self, response):
        """
        This is a simple function to get the landing page url for the sale
        :param response:
        :return:
        """
        all_results_url = response.url + "&ShowAll=true"
        yield scrapy.Request(all_results_url, callback=self.parse_sale_page,
                             meta={"item": response.meta["item"]})

    def parse_sale_page(self, response):
        """
        The sale page is the home page for that auction
        :param response:
        :return:
        """
        sale_item = response.meta["item"]
        if sale_item["location_int"] == christies_settings.LOCATIONS["online"]:
            js = response.xpath('//script[@type="text/javascript"][contains(text(), "var saleName")]/text()').get()
            sale_item["sale_details_js"] = ChristiesCrawler.parse_js(js)
            yield sale_item
        else:
            print_lot_list_url = response.xpath('//*[@id="dvPrint"]/a/@href').get()
            if print_lot_list_url is not None:
                lot_list = response.urljoin(print_lot_list_url)
                yield scrapy.Request(lot_list,
                                     callback=self.parse_lot_list_page,
                                     meta={"item": sale_item})

    @staticmethod
    def parse_lot_list_page(response):
        sale_item = response.meta["item"]
        lots = response.xpath('//*[@id="lot-list"]/tr')
        for lot in lots:
            sale_item["sale_details_html"].append(ChristiesCrawler.parse_html_lot(lot))
        yield sale_item

    @staticmethod
    def parse_html_lot(element):
        lot = {"image_url": element.xpath('.//*[@class="thumb"]/img/@src').get()}
        lot_info = element.xpath('.//*[@class="lot-info"]')
        if lot_info:
            l = lot_info[0]
            lot["number"] = ChristiesCrawler.get_if_exists(l, './/*[@class="lot-number"]/text()')
            lot["maker"] = ChristiesCrawler.get_if_exists(l, './/*[@class="lot-maker"]/text()')
            lot["description"] = ChristiesCrawler.get_if_exists(l, './/*[@class="lot-description"]/text()')
            medium_and_size = ChristiesCrawler.get_if_exists(l, './/*[@class="medium-dimensions"]/text()', first=False)
            lot["medium"] = medium_and_size[0] if len(medium_and_size) > 0 else None
            lot["dimension"] = medium_and_size[1] if len(medium_and_size) > 1 else None

        estimate = element.xpath('.//*[@class="estimate"]')
        if estimate:
            e = estimate[0]
            try:
                lot["estimate_primary"], lot["realized_primary"] = e.xpath('.//*[@class="lot-description"]/text()').getall()
            except ValueError:
                pass
            try:
                lot["estimate_seconday"], lot["realized_secondary"] = e.xpath('.//*[@class="estimate-secondary"]/text()').getall()
            except ValueError:
                pass
        return copy.copy(lot)

    @staticmethod
    def get_if_exists(element, xpath, first=True):
        target = element.xpath(xpath).extract()
        if target and first:
            return target[0]
        elif target:
            return target
        return None

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
        try:
            return json.loads(js_obj)
        except json.JSONDecodeError:
            return None
