#!/bin/bash
#SBATCH --job-name=p30_E165_R1_cluster
#SBATCH --output=logs/p30_E165_R1_cluster.out
#SBATCH --error=logs/p30_E165_R1_cluster.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --account=kubacki.michal
#SBATCH --partition=cuda
#SBATCH --gpus=v100:1

source /beegfs/scratch/ric.broccoli/kubacki.michal/conda/etc/profile.d/conda.sh
conda activate cellpose

# export PYTHONPATH="/beegfs/scratch/ric.broccoli/kubacki.michal/SPATIAL_data/Segmentation/python-packages:$PYTHONPATH"

export FOLDER="p30-E165"
export REGION="R1"
export SAMPLE="${FOLDER}_${REGION}"
export DATA_PATH="/beegfs/scratch/ric.broccoli/kubacki.michal/SPATIAL_data/data_${FOLDER}"
export CONFIG="${SAMPLE}_default_1_ZLevel_cpsam" 

vpt --verbose --processes 3 run-segmentation \
    --segmentation-algorithm "${CONFIG}.json" \
    --input-images "${DATA_PATH}/${REGION}/images/mosaic_(?P<stain>[\\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-micron-to-mosaic "${DATA_PATH}/${REGION}/images/micron_to_mosaic_pixel_transform.csv" \
    --output-path "${CONFIG}" \
    --tile-size 2400 \
    --tile-overlap 200 \
    --overwrite