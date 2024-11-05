#!/bin/bash

# === multi output training ===#
for j in $(seq 0 9); do
python TabNet_multioutput.py "config.yaml" "[0,1,2,3,4]" "$j" &
done
echo "Done training"
