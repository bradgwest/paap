#!/usr/bin/env bash

set -ex

# echo "test" > /tmp/test.txt
# gsutil cp /tmp/test.txt gs://paap/nn/dcec/
mkdir -p data
gsutil cp gs://paap/nn/dcec_paint/data/photos_and_prints.tar.gz /build/data
tar -xzf /build/data/photos_and_prints.tar.gz
rm /build/data/photos_and_prints.tar.gz

# Run the model
/build/dcec/bin/python /build/DCEC.py photos_and_prints --assert-gpu
# Copy the results to GCS
gsutil -m cp results/temp/** gs://paap/nn/dcec_paint/results/temp
