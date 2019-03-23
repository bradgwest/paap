import csv
import logging

import scrapy

from art.scrape_art import settings


class ChristiesImageCrawler(scrapy.Spider):

    name = "christiesImages"
    custom_settings = {'ITEM_PIPELINES': {'scrapy.pipelines.images.ImagesPipeline': 1},
                       'IMAGES_STORE': 'gs://paap/christies/data/img/',
                       'GCS_PROJECT_ID': 'art-auction-prices'}

    def start_requests(self):
        with open(settings.IMAGES_PATH_FILE) as f:
            images = csv.reader(f)
            for i in images:
                img_url = i[0].strip() + '?mode=max&w=' + str(settings.IMAGE_MAX_WIDTH)
                logging.info("Getting {}".format(img_url))
                yield scrapy.Request(url=img_url,
                                     callback=self.parse)

    def parse(self, response):
        yield {'image_urls': [response.url]}
