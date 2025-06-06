
# Commands

Run using with:

```bash
conda activate vizgen
```

```bash
export DATA_PATH="/e/Githubs/SPATIAL_data/data_p30-E165"
```

```bash
vpt --verbose --processes 4 run-segmentation \
    --segmentation-algorithm ./my_segmentation_algorithm.json \
    --input-images="../../data_p30-E165/R1_combined/images/mosaic_(?P<stain>[\\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-micron-to-mosaic "../../data_p30-E165/R1_combined/images/micron_to_mosaic_pixel_transform.csv" \
    --output-path ./analysis_outputs \
    --tile-size 2400 \
    --tile-overlap 200
```

```bash
vpt --verbose partition-transcripts \
    --input-boundaries ./analysis_outputs/cellpose_micron_space.parquet \
    --input-transcripts "../../data_p30-E165/R1_combined/detected_transcripts.csv" \
    --output-entity-by-gene ./analysis_outputs/cell_by_gene.csv \
    --output-transcripts ./analysis_outputs/detected_transcripts_assigned.csv
```

```bash
vpt --verbose derive-entity-metadata \
    --input-boundaries ./analysis_outputs/cellpose_micron_space.parquet \
    --input-entity-by-gene ./analysis_outputs/cell_by_gene.csv \
    --output-metadata ./analysis_outputs/cell_metadata.csv
```

```bash
vpt --verbose sum-signals \
    --input-images="${DATA_PATH}/R1/images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0-9]+).tif" \
    --input-boundaries ./analysis_outputs/cellpose_micron_space.parquet \
    --input-micron-to-mosaic "../../data_p30-E165/R1_combined/images/micron_to_mosaic_pixel_transform.csv" \
    --output-csv ./analysis_outputs/sum_signals.csv
```

```bash
vpt --verbose --processes 20 update-vzg \
    --input-vzg "../../data_p30-E165/R1/data_p30-E165_region_R1.vzg2" \
    --input-boundaries ./analysis_outputs/cellpose_micron_space.parquet \
    --input-entity-by-gene ./analysis_outputs/cell_by_gene.csv \
    --input-metadata ./analysis_outputs/cell_metadata.csv \
    --output-vzg ./analysis_outputs/data_p30-E165_region_R1_resegmented.vzg2
```