import csv
import logging

import google.auth
from google.cloud import storage
import scrapy

from art.scrape_art import settings
from art import gc_utils


def get_images(path):
    if path.startswith("gs://"):
        return get_gcs_image_paths(path)
    return get_local_image_paths(path)


def get_local_image_paths(path):
    images = []
    with open(path) as f:
        image_iter = csv.reader(f)
        for i in image_iter:
            images.append(i[0].strip())
    return images


def get_gcs_image_paths(path):
    bucket_path, blob_path = gc_utils.bucket_and_path_from_uri(path)
    credentials, project = google.auth.default()
    client = storage.Client(project, credentials=credentials)
    bucket = client.bucket(bucket_path)
    blob = bucket.get_blob(blob_name=blob_path)
    csv_bytes = blob.download_as_string()
    csv_string = csv_bytes.decode('utf-8')
    return csv_string.split("\n")


class ChristiesImageCrawler(scrapy.Spider):

    name = "christiesImages"
    custom_settings = {'ITEM_PIPELINES': {'scrapy.pipelines.images.ImagesPipeline': 1},
                       'IMAGES_STORE': 'gs://paap/christies/data/img/',
                       'GCS_PROJECT_ID': 'art-auction-prices'}

    def start_requests(self):
        images = get_images(settings.IMAGES_PATH_FILE)
        for img in images:
            img_url = img + '?mode=max&w=' + str(settings.IMAGE_MAX_WIDTH)
            logging.info("Getting {}".format(img_url))
            yield scrapy.Request(url=img_url,
                                 callback=self.parse)

    def parse(self, response):
        yield {'image_urls': [response.url]}
