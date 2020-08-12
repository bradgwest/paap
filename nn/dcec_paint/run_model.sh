#!/usr/bin/env bash

set -ex

LOCAL_RESULTS="/build/results/temp/"
GCS_RESULTS="gs://paap/nn/dcec_paint/results/"
LOCAL_DATA="/build/data/"
GCS_DATA="gs://paap/nn/dcec_paint/data/"
GCS_DATASET="${GCS_DATA}christies.tar.gz"

# Copy data from GCS
echo "Copying data from ${GCS_DATASET} to ${LOCAL_DATA}"
mkdir -p data
gsutil cp ${GCS_DATASET} ${LOCAL_DATA}
tar -C ${LOCAL_DATA} -xzf ${LOCAL_DATA}christies.tar.gz
rm ${LOCAL_DATA}christies.tar.gz

# Get the preexisting model
mkdir -p ${LOCAL_DATA}
gsutil cp ${GCS_RESULTS}temp/pretrain_cae_model.h5 $LOCAL_DATA

# Run the model
/build/dcec/bin/python /build/DCEC.py photos_and_prints \
    --dataset-path=${LOCAL_DATA}christies \
    --assert-gpu \
    --cae-weights ${LOCAL_DATA}pretrain_cae_model.h5

# Copy the results to GCS
gsutil -m cp -r ${LOCAL_DATA} $GCS_RESULTS
