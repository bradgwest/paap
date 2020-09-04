#! /usr/bin/bash

for i in {1..10}; do
    echo "Plotting k=${i}"
    python vis.py $i --plots \
        "center_images" \
        # "kmeans_metrics" \
        # "metrics" \
        # "loss" \
done
