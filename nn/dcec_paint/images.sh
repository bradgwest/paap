#! /usr/bin/bash

for i in {3..10}; do
    python vis.py $i --plots "tsne_final"
done
