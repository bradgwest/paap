# -*- coding: utf-8 -*-

import scrapy


class ChristiesItem(scrapy.Item):
    """
    An item for Christies data
    """
    input_url = scrapy.Field()
    year = scrapy.Field()
    month = scrapy.Field()
    category = scrapy.Field()
    location = scrapy.Field()
    location_int = scrapy.Field()
    sale_url = scrapy.Field()
    sale_status = scrapy.Field()
    sale_number = scrapy.Field()
    sale_total_raw = scrapy.Field()
    sale_details_js = scrapy.Field()
    sale_details_html = scrapy.Field()
