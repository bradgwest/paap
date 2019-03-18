"""
Utilities for Google Cloud
"""

import re


def bucket_and_path_from_uri(bucket_path):
    if not bucket_path.startswith("gs://"):
        raise ValueError("bucket_path must start with gs://")
    bucket = re.search("(?<=gs://)[a-z0-9-._]*", bucket_path).group(0)
    try:
        path = re.search("(?<=[a-z0-9]/).*", bucket_path).group(0)
    except AttributeError:
        path = None
    return bucket, path
