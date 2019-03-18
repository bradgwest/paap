# -*- coding: utf-8 -*-

"""
Defines a custom pipeline for storing results to google cloud storage
"""

import json
import logging
import re
import uuid

import google.auth
from google.cloud import storage
from scrapy.exceptions import DropItem

from art import gc_utils


class ChristiesPipeline(object):
    def process_item(self, item, spider):
        return item


class GCSPipeline(object):
    """
    Pipeline for writing results to files in Google Cloud Storage
    """

    def __init__(self, project, bucket_path):
        """
        Initializes GCS pipeline
        :param project: Google Cloud project
        :param bucket_path: GCS bucket path, like gs://bucket-name/top-dir
        """
        logging.info("Initiating GCS pipeline")
        self.project = project
        bucket_name, path = gc_utils.bucket_and_path_from_uri(bucket_path)
        print(bucket_name)
        print(path)
        self.bucket_name = bucket_name
        if not path.endswith("/"):
            path += "/"
        self.path = path

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            project=crawler.settings.get('GCS_PROJECT'),
            bucket_path=crawler.settings.get('GCS_BUCKET_PATH')
        )

    def open_spider(self, spider):
        credentials, project_id = google.auth.default()
        if not self.project:
            self.project = project_id
        self.client = storage.Client(self.project, credentials=credentials)
        self.bucket = self.client.get_bucket(self.bucket_name)

    def process_item(self, item, spider):
        # We could process the item here, but we think it's better to have the
        # raw data in storage so that we can iterate without re-initiating the
        # pipeline each time.
        sale_number_match = re.search("[0-9]+", item["sale_number"])
        if sale_number_match:
            sale_number = sale_number_match.group(0)
        else:
            sale_number = str(uuid.uuid4())[:8]
        path_id = "_".join([str(item["year"]), str(item["month"]),
                            item["category"], item["location"],
                            str(sale_number)])

        blob_path = self.path + path_id + ".json"
        blob = self.bucket.blob(blob_path, chunk_size=524288)
        js = json.dumps(dict(item))
        try:
            blob.upload_from_string(js, content_type='text/json')
        except:
            # If upload fails, then
            return item
        return DropItem("Upload to Google successful, no further processing needed")
