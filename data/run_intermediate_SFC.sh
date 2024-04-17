#!/bin/bash
lev_type='SFC'
vars="['t2m','si10','skt']"

# Set the batch size
batch_size=100

source_dir='/media/harish/External_1/CERRA/'

# Loop over years from 1990 to 204
for year in $(seq 2000 2004); do
    echo "Processing year: $year"

    # Glob for GRIB files
    files=(${source_dir}/${year}/*_${lev_type}.grb)

    # Get the total number of files
    total_files=${#files[@]}

    # Iterate over the list of files with an incrementing counter
    for ((i=0; i<${total_files}; i+=batch_size)); do
        # Slice the array to get the current batch of files
        batch_files=("${files[@]:$i:$batch_size}")

        # Iterate over the current batch of files
        for ((j=0; j<${#batch_files[@]}; j++)); do
            file="${batch_files[$j]}"

            echo "Processing file $i: ${file}"
            python Extracting_intermediate_CERRA.py "$file" "$lev_type" "${vars}" "$((i + j))" &
        done 
        echo "Done $i th batch"
        wait
    done
    wait
    echo "Processing completed."

    output_dir="CERRA_complete/${lev_type}"
    mkdir -p "${output_dir}"
    target_file="${output_dir}/${year}.nc"
    echo "${target_file}"
    python concat_yearly_CERRA.py "${target_file}"
    echo "Concat completed."
    
    rm -r CERRA_complete/temp_dir/*.nc
    echo "Done deleting."
    
    echo "Done ${year}"
done
