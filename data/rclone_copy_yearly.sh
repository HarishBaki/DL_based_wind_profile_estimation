source_dir=$1
year=$2
lev=$3
ls $source_dir'/'$year'/'*'_'$lev'.grb' | xargs -I{} basename {} > 'files.txt'
rclone copy --progress --transfers 1 --files-from 'files.txt' $source_dir'/'$year CERRA_complete/temp_dir/
