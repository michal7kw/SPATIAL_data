export DATA_PATH="/e/Githubs/SPATIAL_data/data_p30-E165"

vpt --verbose --processes 12 run-segmentation \
    --segmentation-algorithm ./my_segmentation_algorithm.json \
    --input-images="../../data_p30-E165/R1/images/mosaic_(?P<stain>[\\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-micron-to-mosaic "../../data_p30-E165/R1/images/micron_to_mosaic_pixel_transform.csv" \
    --output-path ./segmentation_algorithm \
    --tile-size 2400 \
    --tile-overlap 200 \
    --overwrite

vpt --verbose partition-transcripts \
    --input-boundaries ./segmentation_algorithm/cellpose_micron_space.parquet \
    --input-transcripts "${DATA_PATH}/R1/detected_transcripts.csv" \
    --output-entity-by-gene ./segmentation_algorithm/cell_by_gene.csv \
    --output-transcripts ./segmentation_algorithm/detected_transcripts_assigned.csv \
    --overwrite

vpt --verbose derive-entity-metadata \
    --input-boundaries ./segmentation_algorithm/cellpose_micron_space.parquet \
    --input-entity-by-gene ./segmentation_algorithm/cell_by_gene.csv \
    --output-metadata ./segmentation_algorithm/cell_metadata.csv \
    --overwrite

vpt --verbose sum-signals \
    --input-images="${DATA_PATH}/R1/images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-boundaries segmentation_algorithm/cellpose_micron_space.parquet \
    --input-micron-to-mosaic "${DATA_PATH}/R1/images/micron_to_mosaic_pixel_transform.csv" \
    --output-csv segmentation_algorithm/sum_signals.csv \
    --overwrite

vpt --verbose --processes 8 update-vzg \
    --input-vzg "${DATA_PATH}/R1/data_p30-E165_region_R1.vzg2" \
    --input-boundaries segmentation_algorithm/cellpose_micron_space.parquet \
    --input-entity-by-gene segmentation_algorithm/cell_by_gene.csv \
    --input-metadata segmentation_algorithm/cell_metadata.csv \
    --output-vzg segmentation_algorithm/data_p30-E165_region_R1_resegmented.vzg2 \
    --temp-path "/e/tmp/segmentation_algorithm_$$" \
    --overwrite

// minimum_mask_size to min_size.
// min_size value from 500 to 50.
// flow_threshold from 1.0 to 0.4.
// cellprob_threshold from -5.5 to 0.0.