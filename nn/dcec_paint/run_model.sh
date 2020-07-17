#!/usr/bin/env bash

set -ex

LOCAL_RESULTS="/build/results/temp/"
GCS_RESULTS="gs://paap/nn/dcec_paint/results/"
LOCAL_DATA="/build/data/"
GCS_DATA="gs://paap/nn/dcec_paint/data/"

# Copy data from GCS
mkdir -p data
gsutil cp ${GCS_DATA}photos_and_prints_split.tar.gz $LOCAL_DATA
tar -C $LOCAL_DATA -xzf ${LOCAL_DATA}photos_and_prints_split.tar.gz
rm ${LOCAL_DATA}photos_and_prints_split.tar.gz

# Get the preexisting model
mkdir -p $LOCAL_DATA
gsutil cp ${GCS_RESULTS}temp/pretrain_cae_model.h5 $LOCAL_DATA

# Run the model
/build/dcec/bin/python /build/DCEC.py photos_and_prints \
    --dataset-path=${LOCAL_DATA}photos_and_prints_split/train/ \
    --assert-gpu \
    --cae-weights ${LOCAL_DATA}pretrain_cae_model.h5

# Copy the results to GCS
gsutil -m cp -r ${LOCAL_DATA} $GCS_RESULTS
