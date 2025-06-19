conda activate vizgen

# -------------------
# --- CONFIGURATION ---
# -------------------
export FOLDER="p30-E165"
export REGION="R1"
export SAMPLE="${FOLDER}_${REGION}"
export DATA_PATH="E:/Githubs/SPATIAL_data/data_${FOLDER}"
export CONFIG_NAME="${SAMPLE}_default_1_ZLevel_cpsam"
export CONFIG_DIR="../Vpt-segmentation"
export CONFIG_FILE_NAME="${CONFIG_DIR}/${CONFIG_NAME}.json"

export ROI_WORKFLOW_DIR="./"
export ROI_COORDS_FILE="${ROI_WORKFLOW_DIR}/roi_coords.json"
export ROI_OUTPUT_DIR="${CONFIG_NAME}_roi_analysis"
export FULL_SPEC_FILE="${ROI_OUTPUT_DIR}/segmentation_specification.json"
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
    --output-path "${ROI_OUTPUT_DIR}" \
    --overwrite


# ----------------------------------------------------
# --- STEP 2: FILTER SPEC AND GET TILE INDICES FOR ROI ---
# ----------------------------------------------------
echo "--- Step 2: Filtering spec and getting tile indices for ROI ---"
ROI_TILE_INDICES=$(python ${ROI_WORKFLOW_DIR}/filter_spec_and_get_indices.py \
    --input-spec "${FULL_SPEC_FILE}" \
    --input-roi "${ROI_COORDS_FILE}" \
    --output-spec "${ROI_SPEC_FILE}")

echo "Found tile indices: ${ROI_TILE_INDICES}"

# ---------------------------------------
# --- STEP 3: RUN SEGMENTATION ON ROI TILES ---
# ---------------------------------------
echo "--- Step 3: Running segmentation on ROI tiles ---"
for TILE_INDEX in ${ROI_TILE_INDICES}
do
    echo "--- Running segmentation on tile ${TILE_INDEX} ---"
    vpt run-segmentation-on-tile \
        --input-segmentation-parameters "${FULL_SPEC_FILE}" \
        --tile-index ${TILE_INDEX} \
        --overwrite
done

# -----------------------------------------------------
# --- STEP 3.5: CREATE SYMLINKS FOR COMPILE STEP ---
# -----------------------------------------------------
echo "--- Step 3.5: Creating symlinks for compile step ---"
i=0
for TILE_INDEX in ${ROI_TILE_INDICES}
do
    if [ -f "${ROI_OUTPUT_DIR}/result_tiles/cell_${TILE_INDEX}.parquet" ]; then
        ln -sf "cell_${TILE_INDEX}.parquet" "${ROI_OUTPUT_DIR}/result_tiles/cell_${i}.parquet"
    fi
    if [ -f "${ROI_OUTPUT_DIR}/result_tiles/nucleus_${TILE_INDEX}.parquet" ]; then
        ln -sf "nucleus_${TILE_INDEX}.parquet" "${ROI_OUTPUT_DIR}/result_tiles/nucleus_${i}.parquet"
    fi
    i=$((i+1))
done

# ---------------------------------
# --- STEP 4: COMPILE SEGMENTATION ---
# ---------------------------------
echo "--- Step 4: Compiling segmentation results ---"
vpt compile-tile-segmentation \
    --input-segmentation-parameters "${ROI_SPEC_FILE}" \
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
    --input-vzg "${DATA_PATH}/${REGION}/data.vzg2" \
    --input-boundaries "${ROI_OUTPUT_DIR}/cellpose_micron_space.parquet" \
    --input-entity-by-gene "${ROI_OUTPUT_DIR}/cell_by_gene.csv" \
    --input-metadata "${ROI_OUTPUT_DIR}/cell_metadata.csv" \
    --output-vzg "${ROI_OUTPUT_DIR}/${SAMPLE}_roi.vzg" \
    --overwrite

echo "--- ROI segmentation workflow finished successfully! ---"