#!/bin/bash
: ' 
# === single output training ===#
for j in $(seq 0 4); do
    for i in $(seq 0 9); do
        python TabNet.py "config.yaml" "[$j]"  "$i" &
    done
    wait
done
echo "Done training"
'
# === multi output training ===#
for j in $(seq 0 9); do
python TabNet_multioutput.py "config_CERRA.yaml" "[0,1,2,3,4]" "$j" &
done
echo "Done training"
