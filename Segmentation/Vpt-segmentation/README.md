Run using with:

```bash
conda activate vizgen
```

```bash
export DATA_PATH="/e/Githubs/SPATIAL_data/data_p30-E165"
```

```bash
export DATA_PATH="/e/Githubs/SPATIAL_data/data_p30-E165"
vpt --verbose --processes 16 run-segmentation \
    --segmentation-algorithm ./my_segmentation_algorithm.json \
    --input-images="${DATA_PATH}/R1/images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-micron-to-mosaic "${DATA_PATH}/R1/images/micron_to_mosaic_pixel_transform.csv" \
    --output-path ./analysis_outputs \
    --tile-size 2400 \
    --tile-overlap 200
```

```bash
$ vpt --verbose partition-transcripts \
    --input-boundaries ./analysis_outputs/cellpose_micron_space.parquet \
    --input-transcripts "${DATA_PATH}/detected_transcripts.csv" \
    --output-entity-by-gene ./analysis_outputs/cell_by_gene.csv \
    --output-transcripts ./analysis_outputs/detected_transcripts_assigned.csv
```