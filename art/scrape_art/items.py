# -*- coding: utf-8 -*-

import scrapy


class ChristiesItem(scrapy.Item):
    """
    An item for Christies data
    """
    sale_url = scrapy.Field()
    sale_status = scrapy.Field()
    sale_number = scrapy.Field()
    sale_location = scrapy.Field()
    sale_total = scrapy.Field()
    sale_details = scrapy.Field()
