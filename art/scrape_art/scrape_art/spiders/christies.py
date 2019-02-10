"""
Scraper for Christies auction results
"""

import scrapy

from art.scrape_art.scrape_art.spiders import christies_settings


class ChristiesCrawler(scrapy.Spider):
    name = "christies"
    start_urls = christies_settings.create_urls()
    # TODO Delete me
    start_urls = ["https://www.christies.com/results?sc_lang=en&month=7&year=2018&scids=11"]

    def parse(self, response):
        sales = response.xpath('//{}/li'.format(christies_settings.TAGS["sales_or_events"]))

        for sale in sales:
            # TODO These are brittle
            status = sale.xpath('./div/h4[contains(@class, "sale-status")]/text()').get()
            number = sale.xpath('./div/div/span[contains(@class, "sale-number")]/text()').get()
            location = sale.xpath('./div/div/span[contains(@class, "location")]/text()').get()
            total = sale.xpath('./div/div/div[contains(@class, "sale-total")]/text()').get()

            yield {
                "sale_url": response.url,
                "sale_status": status,
                "sale_nunber": number,
                "sale_location": location,
                "sale_total": total
            }

            sale_page = sale.xpath('./div/div/a[text() = "View results"]/@href').get()
            if sale_page is not None:
                next_page = response.urljoin(sale_page)
                yield scrapy.Request(next_page, callback=self.parse_sale_page)

    def parse_sale_page(self, response):
        """
        The sale page is the home page for that auction

        :param response:
        :return:
        """
        pass
