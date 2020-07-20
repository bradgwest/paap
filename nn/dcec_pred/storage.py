import glob
import os
import tarfile
from typing import Optional, Tuple

from google.cloud import storage


def is_gcs_blob(path: str) -> bool:
    return path.startswith("gs://")


def path_from_uri(gcs_path: str) -> Tuple[str, str]:
    return gcs_path[len("gs://"):].split("/")


def tar(src: str, dest: str) -> None:
    if not os.path.isdir(src):
        raise ValueError("Expected src to be a directory: `{}`".format(src))

    if not dest.endswith(".tar.gz"):
        raise ValueError("Expected dest to be a gzipped tarball: `{}`".format(dest))

    with tarfile.open(dest, "x:gz") as tar:
        tar.add(src)


def untar(src: str, dest: str = Optional[str]):
    if not os.path.isfile(src):
        raise ValueError("Expected src to be a file: `{}`".format(src))

    if not src.endswith(".tar.gz"):
        raise ValueError("Expected src to be a gzipped tarball: `{}`".format(src))

    if dest is None:
        dest = tarfile.open("sample.tar.gz", "r:gz")

    with tarfile.open(src, "r:gz") as tar:
        tar.extractall(dest)


# TODO add logging
# TODO support downloading as directory
# TODO test
class GCStorage(object):
    def __init__(self, bucket: Optional[str] = None):
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket) if bucket is not None else None

    @property
    def bucket(self):
        return self._bucket

    @bucket.setter
    def bucket(self, b):
        return self._client.bucket(b)

    @classmethod
    def bucket_from_blob(cls, path: str) -> str:
        return path[len("gs://"):].split("/")[0]



    def _upload_file(self, src: str, dest: str) -> None:
        """Uploads a file to the bucket."""
        if not os.path.isfile(src):
            raise ValueError("Expecting src to be a file: `{}`".format(src))

        blob = self.bucket.blob(dest)
        print(blob)
        blob.upload_from_filename(dest)

    def _upload_dir(self, src: str, dest: str) -> None:
        if not os.path.isdir(src):
            raise ValueError("Expecting src to be a directory: `{}`".format(dest))

        for fp in glob.glob(src.rstrip("/") + '/**'):
            if not os.path.isfile(fp):
                self._upload_dir(fp, os.path.join(dest, os.path.basename(fp)))
            else:
                self._upload_file(fp, os.path.join(dest, fp[1 + len(src):]))

    def upload(self, src: str, dest: str) -> None:
        if os.path.isfile(src):
            return self._upload_file(src, dest)
        return self._upload_dir(src, dest)

    def _download_file(self, src: str, dest: str) -> None:
        blob = self.bucket.blob(src)
        print(blob)
        blob.download_to_filename(dest)

    def download(self, src: str, dest: str) -> None:
        self._download_file(src, dest)
