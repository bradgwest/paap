#!/usr/bin/env bash

set -ex

# Copy data from GCS
mkdir -p data
gsutil cp gs://paap/nn/dcec_paint/data/photos_and_prints_split.tar.gz /build/data
tar -C /build/data/ -xzf /build/data/photos_and_prints_split.tar.gz
rm /build/data/photos_and_prints_split.tar.gz

# Run the model
/build/dcec/bin/python /build/DCEC.py photos_and_prints \
    --dataset-path=/build/data/photos_and_prints_split/train/ \
    --assert-gpu

# Copy the results to GCS
gsutil -m cp -r results/temp gs://paap/nn/dcec_paint/results/
