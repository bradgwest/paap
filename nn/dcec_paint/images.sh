#! /usr/bin/bash

for i in {2..10}; do
    echo "Plotting k=${i}"
    # python combine_images.py $i 6
    python vis.py $i --plots \
        "tsne_artist" "tsne_genre" "tsne_photo"
        # "center_images" \
        # "final_dataset" \
        # "kmeans_metrics" \
        # "metrics" \
        # "loss" \
done
