#!/bin/bash

: '
# === multi output training obs targets ===#
for j in $(seq 0 9); do
    python TabNet_multioutput_lidar_targets.py "config_ERA5_to_obs.yaml" "$j" &
done
wait
'

# === multi output training CERRA tragets===#
train_years=17 #(1 3 5 10 17)
for i in "${train_years[@]}"; do
    for j in $(seq 0 9); do
    python TabNet_multioutput_CERRA_targets.py "config_CERRA_to_CERRA.yaml" "$i" "$j" &
    echo "Training $i years, $j ensemble"
    done
    wait
done
wait

: '
# === multi output retraining obs targets ===#
for j in $(seq 0 9); do
    python TabNet_multioutput_lidar_targets_retrain.py "config_TL_ERA5_to_CERRA_to_obs.yaml" "$j" &
done
wait
'
echo "Done training"
