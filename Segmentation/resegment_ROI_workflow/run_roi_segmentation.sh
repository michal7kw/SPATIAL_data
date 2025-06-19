#!/bin/bash
#SBATCH --job-name=p30_E165_R1_roi_cluster
#SBATCH --output=logs/p30_E165_R1_roi_cluster.out
#SBATCH --error=logs/p30_E165_R1_roi_cluster.err
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

# -------------------
# --- CONFIGURATION ---
# -------------------
export FOLDER="p30-E165"
export REGION="R1"
export SAMPLE="${FOLDER}_${REGION}"
export DATA_PATH="/beegfs/scratch/ric.broccoli/kubacki.michal/SPATIAL_data/data_${FOLDER}"
export CONFIG_NAME="${SAMPLE}_default_1_ZLevel_cpsam"
export CONFIG_FILE_NAME="${CONFIG_NAME}.json"

export ROI_WORKFLOW_DIR="Segmentation/resegment_ROI_workflow"
export ROI_COORDS_FILE="${ROI_WORKFLOW_DIR}/roi_coords.json"
export ROI_OUTPUT_DIR="${CONFIG_NAME}_roi_analysis"
export FULL_SPEC_FILE="${ROI_OUTPUT_DIR}/segmentation_specification_full.json"
export ROI_SPEC_FILE="${ROI_OUTPUT_DIR}/segmentation_specification_roi.json"

# Create output directory
mkdir -p ${ROI_OUTPUT_DIR}
mkdir -p logs

# ---------------------------------
# --- STEP 1: PREPARE SEGMENTATION ---
# ---------------------------------
echo "--- Step 1: Preparing full segmentation specification ---"
vpt prepare-segmentation \
    --segmentation-algorithm "${CONFIG_FILE_NAME}" \
    --input-images "${DATA_PATH}/${REGION}/images/mosaic_(?P<stain>[\\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-micron-to-mosaic "${DATA_PATH}/${REGION}/images/micron_to_mosaic_pixel_transform.csv" \
    --output-path "${FULL_SPEC_FILE}" \
    --overwrite

# -------------------------------
# --- STEP 2: FILTER TILES FOR ROI ---
# -------------------------------
echo "--- Step 2: Filtering tiles for ROI ---"
python ${ROI_WORKFLOW_DIR}/filter_segmentation_specification.py \
    --input-spec "${FULL_SPEC_FILE}" \
    --input-roi "${ROI_COORDS_FILE}" \
    --output-spec "${ROI_SPEC_FILE}"

# ---------------------------------------
# --- STEP 3: RUN SEGMENTATION ON ROI TILES ---
# ---------------------------------------
echo "--- Step 3: Running segmentation on ROI tiles ---"
vpt run-segmentation-on-tile \
    --segmentation-task-spec "${ROI_SPEC_FILE}" \
    --output-path "${ROI_OUTPUT_DIR}/result_tiles" \
    --overwrite

# ---------------------------------
# --- STEP 4: COMPILE SEGMENTATION ---
# ---------------------------------
echo "--- Step 4: Compiling segmentation results ---"
vpt compile-tile-segmentation \
    --segmentation-task-spec "${ROI_SPEC_FILE}" \
    --output-path "${ROI_OUTPUT_DIR}" \
    --overwrite

# ---------------------------------------
# --- STEP 5: PARTITION TRANSCRIPTS ---
# ---------------------------------------
echo "--- Step 5: Partitioning transcripts ---"
vpt partition-transcripts \
    --input-boundaries "${ROI_OUTPUT_DIR}/cellpose_micron_space.parquet" \
    --input-transcripts "${DATA_PATH}/${REGION}/detected_transcripts.csv" \
    --output-entity-by-gene "${ROI_OUTPUT_DIR}/cell_by_gene.csv" \
    --output-transcripts "${ROI_OUTPUT_DIR}/detected_transcripts.csv" \
    --overwrite

# ---------------------------------------
# --- STEP 6: DERIVE ENTITY METADATA ---
# ---------------------------------------
echo "--- Step 6: Deriving entity metadata ---"
vpt derive-entity-metadata \
    --input-boundaries "${ROI_OUTPUT_DIR}/cellpose_micron_space.parquet" \
    --input-entity-by-gene "${ROI_OUTPUT_DIR}/cell_by_gene.csv" \
    --output-metadata "${ROI_OUTPUT_DIR}/cell_metadata.csv" \
    --overwrite

# ---------------------------------------
# --- STEP 7: UPDATE VZG FILE ---
# ---------------------------------------
echo "--- Step 7: Updating VZG file ---"
vpt update-vzg \
    --input-vzg "${DATA_PATH}/${REGION}/${SAMPLE}.vzg" \
    --input-boundaries "${ROI_OUTPUT_DIR}/cellpose_micron_space.parquet" \
    --input-entity-by-gene "${ROI_OUTPUT_DIR}/cell_by_gene.csv" \
    --input-metadata "${ROI_OUTPUT_DIR}/cell_metadata.csv" \
    --output-vzg "${ROI_OUTPUT_DIR}/${SAMPLE}_roi.vzg" \
    --overwrite

echo "--- ROI segmentation workflow finished successfully! ---"
