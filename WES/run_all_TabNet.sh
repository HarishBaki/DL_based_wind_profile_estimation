#!/bin/bash
# === multi output training ===#
train_years=(1 3 5 10 17)
for i in "${train_years[@]}"; do
    for j in $(seq 0 9); do
    python TabNet_multioutput.py "config_ERA5.yaml" "$i" "$j" &
    done
    wait
done
echo "Done training"
