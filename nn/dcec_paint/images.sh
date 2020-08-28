#! /usr/bin/bash

for i in {3..10}; do
    echo "Plotting k=${i}"
    python vis.py $i --plots "metrics" "loss"
done
