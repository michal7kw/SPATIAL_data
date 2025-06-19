## Example: Segmenting a Small Dataset Saved on a Local Hard Drive

## Before Beginning: System Set Up

- Windows 10 laptop computer (i7-1185G7 processor, 16 GB RAM)
- Using Ubuntu 20.04 through the Windows Subsystem for Linux 2 (wsl2)
- Python, pip, and venv installed in Ubuntu
- The data produced by the MERSCOPE™ Image Processing Software was downloaded to the wsl2 home directory [Download Data Files](https://vzg-web-resources.s3.amazonaws.com/202305010900_U2OS_small_set_VMSC00000.zip)
- The example segmentation algorithm json files were downloaded from github to the wsl2 home directory [Download Algorithm Files](https://github.com/Vizgen/vizgen-postprocessing/tree/develop/example_analysis_algorithm)

At the beginning of the analysis, the following data is saved in the home directory:

**User Input**

```
user@computer:~$ tree
```

**Console Output**

```
.
├── 202305010900_U2OS_small_set_VMSC00000
│   └── region_0
│       ├── 202305010900_U2OS_small_set_VMSC00000_region_0.vzg
│       ├── detected_transcripts.csv
│       ├── images
│       │   ├── manifest.json
│       │   ├── micron_to_mosaic_pixel_transform.csv
│       │   ├── mosaic_Cellbound1_z0.tif
│       │   ├── mosaic_Cellbound1_z1.tif
│       │   ├── mosaic_Cellbound1_z2.tif
│       │   ├── mosaic_Cellbound1_z3.tif
│       │   ├── mosaic_Cellbound1_z4.tif
│       │   ├── mosaic_Cellbound1_z5.tif
│       │   ├── mosaic_Cellbound1_z6.tif
│       │   ├── mosaic_Cellbound2_z0.tif
│       │   ├── mosaic_Cellbound2_z1.tif
│       │   ├── mosaic_Cellbound2_z2.tif
│       │   ├── mosaic_Cellbound2_z3.tif
│       │   ├── mosaic_Cellbound2_z4.tif
│       │   ├── mosaic_Cellbound2_z5.tif
│       │   ├── mosaic_Cellbound2_z6.tif
│       │   ├── mosaic_Cellbound3_z0.tif
│       │   ├── mosaic_Cellbound3_z1.tif
│       │   ├── mosaic_Cellbound3_z2.tif
│       │   ├── mosaic_Cellbound3_z3.tif
│       │   ├── mosaic_Cellbound3_z4.tif
│       │   ├── mosaic_Cellbound3_z5.tif
│       │   ├── mosaic_Cellbound3_z6.tif
│       │   ├── mosaic_DAPI_z0.tif
│       │   ├── mosaic_DAPI_z1.tif
│       │   ├── mosaic_DAPI_z2.tif
│       │   ├── mosaic_DAPI_z3.tif
│       │   ├── mosaic_DAPI_z4.tif
│       │   ├── mosaic_DAPI_z5.tif
│       │   ├── mosaic_DAPI_z6.tif
│       │   ├── mosaic_PolyT_z0.tif
│       │   ├── mosaic_PolyT_z1.tif
│       │   ├── mosaic_PolyT_z2.tif
│       │   ├── mosaic_PolyT_z3.tif
│       │   ├── mosaic_PolyT_z4.tif
│       │   ├── mosaic_PolyT_z5.tif
│       │   └── mosaic_PolyT_z6.tif
└── example_analysis_algorithm
    ├── cellpose_default_1_ZLevel.json
    ├── cellpose_default_3_ZLevel.json
    ├── cellpose_default_3_ZLevel_nuclei_only.json
    └── watershed_default.json

4 directories, 43 files
```

In this example workflow, all of the analysis output files will be saved to `~/analysis_outputs`.

## Step 1: Install vpt in a Virtual Environment

**User Input**

```
user@computer:~$ python3 -m venv ~/.venv/vpt_env
user@computer:~$ source .venv/vpt_env/bin/activate
(vpt_env) user@computer:~$ pip install vpt[all]
```

**Console Output**

```
[ pip installation progress trimmed for brevity ]

Successfully installed MarkupSafe-2.1.2 Pillow-9.4.0 PyWavelets-1.4.1 absl-py-1.4.0 affine-2.4.0 aiobotocore-1.4.2
aiohttp-3.8.4 aioitertools-0.11.0 aiosignal-1.3.1 astunparse-1.6.3 async-timeout-4.0.2 attrs-22.2.0 boto3-1.17.0
botocore-1.20.106 cachetools-5.3.0 cellpose-1.0.2 certifi-2022.12.7 cffi-1.15.1 charset-normalizer-3.0.1 click-8.1.3
click-plugins-1.1.1 cligj-0.7.2 cloudpickle-2.2.1 contourpy-1.0.7 csbdeep-0.7.3 cycler-0.11.0 dask-2022.9.0
decorator-5.1.1 distributed-2022.9.0 fastremap-1.13.4 fiona-1.9.1 flatbuffers-1.12 fonttools-4.38.0 frozenlist-1.3.3
fsspec-2021.10.0 gast-0.4.0 gcsfs-2021.10.0 geojson-2.5.0 geopandas-0.12.1 google-auth-2.16.1 google-auth-oauthlib-0.4.6
google-pasta-0.2.0 grpcio-1.51.3 h5py-3.7.0 heapdict-1.0.1 idna-3.4 imageio-2.25.1 jinja2-3.1.2 jmespath-0.10.0
keras-2.9.0 keras-preprocessing-1.1.2 kiwisolver-1.4.4 libclang-15.0.6.1 llvmlite-0.39.1 locket-1.0.0 markdown-3.4.1
matplotlib-3.7.0 msgpack-1.0.4 multidict-6.0.4 munch-2.5.0 natsort-8.2.0 networkx-3.0 numba-0.56.4 numpy-1.22.4
nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96
oauthlib-3.2.2 opencv-python-headless-4.6.0.66 opt-einsum-3.3.0 packaging-23.0 pandas-1.4.3 partd-1.3.0 protobuf-3.19.6
psutil-5.9.4 pyarrow-8.0.0 pyasn1-0.4.8 pyasn1-modules-0.2.8 pyclustering-0.10.1.2 pycparser-2.21 pyparsing-3.0.9
pyproj-3.4.1 python-dateutil-2.8.2 python-dotenv-0.20.0 pytz-2022.7.1 pyvips-2.2.1 pyyaml-6.0 rasterio-1.3.0
requests-2.28.2 requests-oauthlib-1.3.1 rsa-4.9 s3fs-2021.10.0 s3transfer-0.3.7 scikit-image-0.19.3 scipy-1.8.1
shapely-2.0.0 six-1.16.0 snuggs-1.4.7 sortedcontainers-2.4.0 stardist-0.8.3 tblib-1.7.0 tensorboard-2.9.1
tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.9.1 tensorflow-estimator-2.9.0
tensorflow-io-gcs-filesystem-0.30.0 termcolor-2.2.0 tifffile-2023.2.3 toolz-0.12.0 torch-1.13.1 tornado-6.1 tqdm-4.64.1
typing-extensions-4.5.0 urllib3-1.26.14 vpt-1.0.1 werkzeug-2.2.3 wheel-0.38.4 wrapt-1.14.1 yarl-1.8.2 zict-2.2.0
```

After installation, use the help function to confirm that vpt is installed correctly.

**User Input**

```
(vpt_env) user@computer:~$ vpt --help
```

**Console Output**

```
usage: vpt [OPTIONS] COMMAND [arguments]

Commands:

    run-segmentation    Top-level interface for this CLI which invokes the segmentation functionality of the tool. It is intended for users who would like to run the program with minimal
                        additional configuration. Specifically, it executes: prepare-segmentation, run-segmentation-on-tile, and compile-tile-segmentation.
    prepare-segmentation
                        Generates a segmentation specification json file to be used for cell segmentation tasks. The segmentation specification json includes specification for the algorithm
                        to run, the paths for all images for each stain for each z index, the micron to mosaic pixel transformation matrix, the number of tiles, and the window coordinates for
                        each tile.
    run-segmentation-on-tile
                        Executes the segmentation algorithm on a specific tile of the mosaic images. This functionality is intended both for visualizing a preview of the segmentation (run
                        only one tile), and for distributing jobs using an orchestration tool such as Nextflow.
    compile-tile-segmentation
                        Combines the per-tile segmentation outputs into a single, internally-consistent parquet file containing all of the segmentation boundaries found in the experiment.
    derive-entity-metadata
                        Uses the segmentation boundaries to calculate the geometric attributes of each Entity. These attributes include the position, volume, and morphological features.
    partition-transcripts
                        Uses the segmentation boundaries to determine which Entity, if any, contains each detected transcript. Outputs an Entity by gene matrix, and may optionally output a
                        detected transcript csv with an additional column indicating the containing Entity.
    sum-signals         Uses the segmentation boundaries to find the intensity of each mosaic image in each Entity. Outputs both the summed intensity of the raw images and the summed
                        intensity of high-pass filtered images (reduces the effect of background fluorescence).
    update-vzg          Updates an existing .vzg file with new segmentation boundaries and the corresponding expression matrix. NOTE: This functionality requires enough disk space to unpack
                        the existing .vzg file.
    convert-geometry    Converts entity boundaries produced by a different tool into a vpt compatible parquet file. In the process, each of the input entities is checked for geometric
                        validity, overlap with other geometries, and assigned a globally-unique EntityID to facilitate other processing steps.
    convert-to-ome      Transforms the large 16-bit mosaic tiff images produced by the MERSCOPE™ into a OME pyramidal tiff.
    convert-to-rgb-ome  Converts up to three flat tiff images into a rgb OME-tiff pyramidal images. If a rgb channel input isn’t specified, the channel will be dark (all 0’s).

Options:
--processes PROCESSES
                        Number of parallel processes to use when executing locally
--aws-profile-name AWS_PROFILE_NAME
                        Named profile for AWS access
--aws-access-key AWS_ACCESS_KEY
                        AWS access key from key / secret pair
--aws-secret-key AWS_SECRET_KEY
                        AWS secret from key / secret pair
--gcs-service-account-key GCS_SERVICE_ACCOUNT_KEY
                        Path to a google service account key json file. Not needed if google authentication is performed using gcloud
--verbose             Display progress messages during execution
--profile-execution-time PROFILE_EXECUTION_TIME
                        Path to profiler output file
--log-level LOG_LEVEL
                        Log level value. Level is specified as a number from 1 - 5, corresponding to debug, info, warning, error, crit
--log-file LOG_FILE   Path to log output file. If not provided, logs are written to standard output
-h, --help            Show this help message and exit

Run 'vpt COMMAND --help' for more information on a command.
```

## Step 2: Identify Cell Boundaries from Images (Cell Segmentation)

Before running cell segmentation, check to see if the number of z-layers in the segmentation algorithm json file matches the number of z-layers in the input data.

**User Input**

```
(vpt_env) user@computer:~$ head example_analysis_algorithm/cellpose_default_1_ZLevel.json
```

**Console Output**

```
{
"experiment_properties": {
    "all_z_indexes": [0, 1, 2, 3, 4, 5, 6],
    "z_positions_um": [1.5, 3, 4.5, 6, 7.5, 9, 10.5]
},
"segmentation_tasks": [
    {
    "task_id": 0,
    "segmentation_family": "Cellpose",
    "entity_types_detected": [
```

The images are numbered 0 - 6 *(see above)*, and `all_z_indexes` also ranges from 0 - 6.

Note

If the experimental data does not match the segmentation algorithm json file, it is important to edit the json file. The `run-segmentation` command will proceed as normal with a mismatched json file, but partitioning transcripts into cells and updating the.vzg file will produce errors.

Now that the segmentation algorithm has been confirmed to describe what should be done, it is safe to run segmentation.

This example shows some optional parameters that were set to optimize memory usage when running on a laptop:

- `--processes 4` — Each process running with Cellpose consumes > 2 GB of memory
- `--tile-size 2400` — Larger tiles require more memory per process
- `--tile-overlap 200` — The `tile-overlap` is padded outward from each tile, minimizing it reduces image size

For more information about the options and arguments that may be passed to `run-segmentation`, please see the [Command Line Interface](https://vizgen.github.io/vizgen-postprocessing/command_line_interface/index.html#command-line-interface) section of the user guide.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose --processes 4 run-segmentation \
> --segmentation-algorithm example_analysis_algorithm/cellpose_default_1_ZLevel.json \
> --input-images="202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0-9]+).tif" \
> --input-micron-to-mosaic 202305010900_U2OS_small_set_VMSC00000/region_0/images/micron_to_mosaic_pixel_transform.csv \
> --output-path analysis_outputs \
> --tile-size 2400 \
> --tile-overlap 200
```

**Console Output**

```
2023-02-22 13:59:28,176 - . - INFO - run run-segmentation with args:Namespace(segmentation_algorithm='example_analysis_algorithm/cellpose_default_1_ZLevel.json', input_images='202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_(?P<stain>[\\w|-]+)_z(?P<z>[0-9]+).tif', input_micron_to_mosaic='202305010900_U2OS_small_set_VMSC00000/region_0/images/micron_to_mosaic_pixel_transform.csv', output_path='analysis_outputs', tile_size=2400, tile_overlap=200, max_row_group_size=17500, overwrite=False)
2023-02-22 13:59:28,177 - . - INFO - run_segmentation started
2023-02-22 13:59:28,354 - . - INFO - prepare segmentation started
2023-02-22 13:59:28,419 - . - INFO - prepare segmentation finished
2023-02-22 13:59:29,842 - ./task-2 - INFO - Run segmentation on tile 2 started
2023-02-22 13:59:29,842 - ./task-3 - INFO - Run segmentation on tile 3 started
2023-02-22 13:59:29,842 - ./task-2 - INFO - Tile 2 [0, 1160, 2800, 2800]
2023-02-22 13:59:29,843 - ./task-0 - INFO - Run segmentation on tile 0 started
2023-02-22 13:59:29,843 - ./task-0 - INFO - Tile 0 [0, 0, 2800, 2800]
2023-02-22 13:59:29,843 - ./task-3 - INFO - Tile 3 [1153, 1160, 2800, 2800]
2023-02-22 13:59:29,848 - ./task-1 - INFO - Run segmentation on tile 1 started
2023-02-22 13:59:29,849 - ./task-1 - INFO - Tile 1 [1153, 0, 2800, 2800]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:02<00:00, 9.30MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.54k/3.54k [00:00<00:00, 17.2MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:04<00:00, 6.57MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:08<00:00, 3.27MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:08<00:00, 3.06MB/s]
2023-02-22 14:01:08,670 - ./task-3 - INFO - generate_polygons_from_mask
2023-02-22 14:01:08,825 - ./task-3 - INFO - get_polygons_from_mask: z=0, labels:454
2023-02-22 14:01:11,088 - ./task-1 - INFO - generate_polygons_from_mask
2023-02-22 14:01:11,242 - ./task-1 - INFO - get_polygons_from_mask: z=0, labels:441
2023-02-22 14:01:12,748 - ./task-2 - INFO - generate_polygons_from_mask
2023-02-22 14:01:12,907 - ./task-2 - INFO - get_polygons_from_mask: z=0, labels:445
2023-02-22 14:01:14,018 - ./task-0 - INFO - generate_polygons_from_mask
2023-02-22 14:01:14,172 - ./task-0 - INFO - get_polygons_from_mask: z=0, labels:450
2023-02-22 14:01:20,943 - ./task-3 - INFO - raw segmentation result contains 410 rows
2023-02-22 14:01:20,943 - ./task-3 - INFO - fuze across z
2023-02-22 14:01:21,220 - ./task-3 - INFO - remove edge polys
2023-02-22 14:01:22,735 - ./task-1 - INFO - raw segmentation result contains 403 rows
2023-02-22 14:01:22,736 - ./task-1 - INFO - fuze across z
2023-02-22 14:01:22,946 - ./task-1 - INFO - remove edge polys
4%|██████▏                                                                                                                                                | 1.03M/25.3M [00:00<00:16, 1.50MB/s]2023-02-22 14:01:25,032 - ./task-2 - INFO - raw segmentation result contains 397 rows
2023-02-22 14:01:25,033 - ./task-2 - INFO - fuze across z
10%|███████████████▊                                                                                                                                       | 2.66M/25.3M [00:01<00:09, 2.54MB/s]2023-02-22 14:01:25,609 - ./task-2 - INFO - remove edge polys
35%|████████████████████████████████████████████████████▋                                                                                                  | 8.84M/25.3M [00:03<00:03, 5.22MB/s]2023-02-22 14:01:27,075 - ./task-0 - INFO - raw segmentation result contains 414 rows
2023-02-22 14:01:27,075 - ./task-0 - INFO - fuze across z
11%|████████████████▍                                                                                                                                      | 2.75M/25.3M [00:01<00:09, 2.44MB/s]2023-02-22 14:01:27,534 - ./task-0 - INFO - remove edge polys
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:05<00:00, 4.84MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3.54k/3.54k [00:00<00:00, 15.9MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:06<00:00, 4.06MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:05<00:00, 4.83MB/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25.3M/25.3M [00:06<00:00, 4.18MB/s]
2023-02-22 14:01:59,720 - ./task-3 - INFO - generate_polygons_from_mask
2023-02-22 14:01:59,890 - ./task-3 - INFO - get_polygons_from_mask: z=0, labels:323
2023-02-22 14:02:03,450 - ./task-1 - INFO - generate_polygons_from_mask
2023-02-22 14:02:03,512 - ./task-2 - INFO - generate_polygons_from_mask
2023-02-22 14:02:03,585 - ./task-1 - INFO - get_polygons_from_mask: z=0, labels:340
2023-02-22 14:02:03,662 - ./task-2 - INFO - get_polygons_from_mask: z=0, labels:317
2023-02-22 14:02:06,188 - ./task-3 - INFO - raw segmentation result contains 320 rows
2023-02-22 14:02:06,189 - ./task-3 - INFO - fuze across z
2023-02-22 14:02:06,443 - ./task-3 - INFO - remove edge polys
2023-02-22 14:02:06,865 - ./task-0 - INFO - generate_polygons_from_mask
2023-02-22 14:02:07,058 - ./task-0 - INFO - get_polygons_from_mask: z=0, labels:333
2023-02-22 14:02:07,878 - ./task-3 - INFO - fuse_task_polygons
2023-02-22 14:02:08,174 - ./task-3 - INFO - Found 416 overlaps
2023-02-22 14:02:10,767 - ./task-2 - INFO - raw segmentation result contains 312 rows
2023-02-22 14:02:10,767 - ./task-2 - INFO - fuze across z
2023-02-22 14:02:10,774 - ./task-1 - INFO - raw segmentation result contains 336 rows
2023-02-22 14:02:10,774 - ./task-1 - INFO - fuze across z
2023-02-22 14:02:10,900 - ./task-2 - INFO - remove edge polys
2023-02-22 14:02:10,938 - ./task-1 - INFO - remove edge polys
2023-02-22 14:02:11,910 - ./task-2 - INFO - fuse_task_polygons
2023-02-22 14:02:12,081 - ./task-1 - INFO - fuse_task_polygons
2023-02-22 14:02:12,126 - ./task-2 - INFO - Found 377 overlaps
2023-02-22 14:02:12,327 - ./task-1 - INFO - Found 433 overlaps
2023-02-22 14:02:13,291 - ./task-3 - INFO - After union of large overlaps, found 102 overlaps
2023-02-22 14:02:14,051 - ./task-0 - INFO - raw segmentation result contains 329 rows
2023-02-22 14:02:14,052 - ./task-0 - INFO - fuze across z
2023-02-22 14:02:14,235 - ./task-0 - INFO - remove edge polys
2023-02-22 14:02:15,060 - ./task-3 - INFO - After both resolution steps, found 0 uncaught overlaps
2023-02-22 14:02:15,250 - ./task-0 - INFO - fuse_task_polygons
2023-02-22 14:02:15,477 - ./task-0 - INFO - Found 418 overlaps
2023-02-22 14:02:15,698 - ./task-3 - INFO - Run segmentation on tile 3 finished
2023-02-22 14:02:16,034 - ./task-2 - INFO - After union of large overlaps, found 85 overlaps
2023-02-22 14:02:16,477 - ./task-1 - INFO - After union of large overlaps, found 104 overlaps
2023-02-22 14:02:17,217 - ./task-2 - INFO - After both resolution steps, found 0 uncaught overlaps
2023-02-22 14:02:17,671 - ./task-2 - INFO - Run segmentation on tile 2 finished
2023-02-22 14:02:17,836 - ./task-1 - INFO - After both resolution steps, found 0 uncaught overlaps
2023-02-22 14:02:18,149 - ./task-1 - INFO - Run segmentation on tile 1 finished
2023-02-22 14:02:18,657 - ./task-0 - INFO - After union of large overlaps, found 89 overlaps
2023-02-22 14:02:19,494 - ./task-0 - INFO - After both resolution steps, found 0 uncaught overlaps
2023-02-22 14:02:19,836 - ./task-0 - INFO - Run segmentation on tile 0 finished
2023-02-22 14:02:21,359 - . - INFO - Compile tile segmentation started
2023-02-22 14:02:21,361 - . - INFO - Loading segmentation results
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 29.44it/s]
2023-02-22 14:02:21,499 - . - INFO - Loaded results for 4 tiles
2023-02-22 14:02:21,510 - . - INFO - Concatenated dataframes
2023-02-22 14:02:22,153 - . - INFO - Found 2061 overlaps
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2061/2061 [00:06<00:00, 305.05it/s]
2023-02-22 14:02:29,276 - . - INFO - After union of large overlaps, found 437 overlaps
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 437/437 [00:04<00:00, 108.20it/s]
2023-02-22 14:02:33,402 - . - INFO - After both resolution steps, found 0 uncaught overlaps
2023-02-22 14:02:33,562 - . - INFO - Resolved overlapping in the compiled dataframe
2023-02-22 14:02:33,622 - . - INFO - Saved compiled dataframe in micron space
2023-02-22 14:02:33,755 - . - INFO - Saved compiled dataframe in mosaic space
2023-02-22 14:02:33,755 - . - INFO - Compile tile segmentation finished
2023-02-22 14:02:33,758 - . - INFO - run_segmentation finished
```

After segmentation is complete, new files are present in the output folder, in this case `~/analysis_outputs`

**User Input**

```
(vpt_env) user@computer:~$ tree analysis_outputs/
```

**Console Output**

```
analysis_outputs/
├── cellpose_micron_space.parquet
├── cellpose_mosaic_space.parquet
├── result_tiles
│   ├── 0.parquet
│   ├── 1.parquet
│   ├── 2.parquet
│   └── 3.parquet
└── segmentation_specification.json

1 directory, 7 files
```

- `cellpose_micron_space.parquet` — The primary output of the segmentation. This table contains the EntityIDs for the cells and the geometries in units of microns. This file is used throught the rest of the vpt workflow.
- `cellpose_mosaic_space.parquet` — A secondary output of the segmentation. This table contains the EntityIDs for the cells and the geometries in units of pixels. This file can be helpful for generating plots of cell outlines with the mosaic tiff images, but is not used by vpt.
- `segmentation_specification.json` — A full specification of the segmentation process including file paths. Useful for reproducing analysis or running specific tiles of the segmentation again.
- `result_tiles` folder — The tile outputs that were combined into the `cellpose_micron_space.parquet` file. Primarily useful for troubleshooting, can be safely discarded after analysis completes successfully.

## Step 2: Partition Transcripts into Cells

Now that the cell boundaries are defined, vpt can use the boundaries to group (or partition) transcripts into cells.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose partition-transcripts \
> --input-boundaries analysis_outputs/cellpose_micron_space.parquet \
> --input-transcripts 202305010900_U2OS_small_set_VMSC00000/region_0/detected_transcripts.csv \
> --output-entity-by-gene analysis_outputs/cell_by_gene.csv \
> --output-transcripts analysis_outputs/detected_transcripts.csv
```

**Console Output**

```
2023-02-22 14:19:17,153 - . - INFO - run partition-transcripts with args:Namespace(input_boundaries='analysis_outputs/cellpose_micron_space.parquet', input_transcripts='202305010900_U2OS_small_set_VMSC00000/region_0/detected_transcripts.csv', output_entity_by_gene='analysis_outputs/cell_by_gene.csv', chunk_size=10000000, output_transcripts='analysis_outputs/detected_transcripts.csv', overwrite=False)
2023-02-22 14:19:17,161 - . - INFO - Partition transcripts started
/home/user/.venv/vpt_env/lib/python3.10/site-packages/pandas/core/indexes/base.py:6982: FutureWarning: In a future version, the Index constructor will not infer numeric dtypes when passed object-dtype sequences (matching Series behavior)
return Index(sequences[0], name=names)
2023-02-22 14:19:22,298 - . - INFO - cell by gene matrix saved as analysis_outputs/cell_by_gene.csv
2023-02-22 14:19:22,298 - . - INFO - detected transcripts saved as analysis_outputs/detected_transcripts.csv
2023-02-22 14:19:22,298 - . - INFO - Partition transcripts finished
```

After transcript processing is complete, new files are present in the output folder:

**User Input**

```
(vpt_env) user@computer:~$ tree analysis_outputs/
```

**Console Output**

```
analysis_outputs/
├── cell_by_gene.csv
├── cellpose_micron_space.parquet
├── cellpose_mosaic_space.parquet
├── detected_transcripts.csv
├── result_tiles
│   ├── 0.parquet
│   ├── 1.parquet
│   ├── 2.parquet
│   └── 3.parquet
└── segmentation_specification.json

1 directory, 9 files
```

- `cell_by_gene.csv` — The raw count of transcripts of each targeted gene in each cell
- `detected_transcripts.csv` — A copy of the original `detected_transcripts.csv` file with an added column for EntityID. Because the type of the Entity is specified as "cell" in the segmentation algorithm json file, the column name is "cell\_id."
	Printing the first 10 lines of each file demonstrates the difference:

**User Input**

```
(vpt_env) user@computer:~$ head 202305010900_U2OS_small_set_VMSC00000/region_0/detected_transcripts.csv
```

**Console Output**

```
,barcode_id,global_x,global_y,global_z,x,y,fov,gene,transcript_id
63,10,370.95007,5.520504,0.0,1611.833,110.285774,0,AKAP11,ENST00000025301.3
68,10,355.46716,6.4616404,0.0,1468.4725,119.0,0,AKAP11,ENST00000025301.3
71,10,371.31482,7.4345083,0.0,1615.2103,128.00804,0,AKAP11,ENST00000025301.3
77,10,347.71286,8.297641,0.0,1396.6736,136.0,0,AKAP11,ENST00000025301.3
81,10,389.76013,9.377641,0.0,1786.0,146.0,0,AKAP11,ENST00000025301.3
86,10,285.00012,10.13364,0.0,816.0,153.0,0,AKAP11,ENST00000025301.3
90,10,255.08412,11.429641,0.0,539.0,165.0,0,AKAP11,ENST00000025301.3
91,10,238.5847,11.699728,0.0,386.2277,167.50081,0,AKAP11,ENST00000025301.3
106,10,220.53308,14.170068,0.0,219.08304,190.37433,0,AKAP11,ENST00000025301.3
```

**User Input**

```
(vpt_env) user@computer:~$ head analysis_outputs/detected_transcripts.csv
```

**Console Output**

```
,barcode_id,global_x,global_y,global_z,x,y,fov,gene,transcript_id,cell_id
63,10,370.95007,5.520504,0.0,1611.833,110.285774,0,AKAP11,ENST00000025301.3,1535046800001100032
68,10,355.46716,6.4616404,0.0,1468.4725,119.0,0,AKAP11,ENST00000025301.3,1535046800001200009
71,10,371.31482,7.4345083,0.0,1615.2103,128.00804,0,AKAP11,ENST00000025301.3,1535046800001100032
77,10,347.71286,8.297641,0.0,1396.6736,136.0,0,AKAP11,ENST00000025301.3,1535046800001100038
81,10,389.76013,9.377641,0.0,1786.0,146.0,0,AKAP11,ENST00000025301.3,1535046800001100030
86,10,285.00012,10.13364,0.0,816.0,153.0,0,AKAP11,ENST00000025301.3,1535046800000100004
90,10,255.08412,11.429641,0.0,539.0,165.0,0,AKAP11,ENST00000025301.3,1535046800000100018
91,10,238.5847,11.699728,0.0,386.2277,167.50081,0,AKAP11,ENST00000025301.3,1535046800001100034
106,10,220.53308,14.170068,0.0,219.08304,190.37433,0,AKAP11,ENST00000025301.3,1535046800000100022
```

## Step 3: Calculate Cell Metadata

One benefit of MERSCOPE™ data is having information about each cell beyond its transcript contents. These data are summarized in a cell metadata file and sum signals file.

The cell metadata file has annotation about the location, size, and shape of each cell that can be used to identify cell neighbors, sort cells into cell types, filter low quality cells, etc.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose derive-entity-metadata \
> --input-boundaries analysis_outputs/cellpose_micron_space.parquet \
> --input-entity-by-gene analysis_outputs/cell_by_gene.csv \
> --output-metadata analysis_outputs/cell_metadata.csv
```

**Console Output**

```
2023-02-22 14:23:20,889 - . - INFO - run derive-entity-metadata with args:Namespace(input_boundaries='analysis_outputs/cellpose_micron_space.parquet', output_metadata='analysis_outputs/cell_metadata.csv', input_entity_by_gene='analysis_outputs/cell_by_gene.csv', overwrite=False)
2023-02-22 14:23:20,890 - . - INFO - Derive cell metadata started
2023-02-22 14:23:21,637 - . - INFO - Derive cell metadata finished
```

The sum signals file has information about the brightness of each mosaic tiff image within each cell. This is most useful when combined with the Vizgen MERSCOPE™ Protein Co-Detection Kit to identify cells that express the markers of interest. In experiments without protein co-detection, the sum signals output is useful to filter low-quality cells by DAPI or PolyT content.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose sum-signals \
> --input-images="202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0-9]+).tif" \
> --input-boundaries analysis_outputs/cellpose_micron_space.parquet \
> --input-micron-to-mosaic 202305010900_U2OS_small_set_VMSC00000/region_0/images/micron_to_mosaic_pixel_transform.csv \
> --output-csv analysis_outputs/sum_signals.csv
```

**Console Output**

```
2023-02-22 14:25:41,969 - . - INFO - run sum-signals with args:Namespace(input_images='202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_(?P<stain>[\\w|-]+)_z(?P<z>[0-9]+).tif', input_boundaries='analysis_outputs/cellpose_micron_space.parquet', input_micron_to_mosaic='202305010900_U2OS_small_set_VMSC00000/region_0/images/micron_to_mosaic_pixel_transform.csv', output_csv='analysis_outputs/sum_signals.csv', overwrite=False)
2023-02-22 14:25:42,026 - . - INFO - Sum signals started
2023-02-22 14:25:42,106 - . - INFO - output structures prepared
2023-02-22 14:25:42,106 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z5.tif started
2023-02-22 14:25:43,654 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z3.tif started
2023-02-22 14:25:45,243 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z2.tif started
2023-02-22 14:25:46,723 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z3.tif started
2023-02-22 14:25:48,401 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z5.tif started
2023-02-22 14:25:49,993 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z6.tif started
2023-02-22 14:25:51,545 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z0.tif started
2023-02-22 14:25:53,226 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z0.tif started
2023-02-22 14:25:54,900 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z4.tif started
2023-02-22 14:25:56,446 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z2.tif started
2023-02-22 14:25:57,995 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z0.tif started
2023-02-22 14:25:59,540 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z1.tif started
2023-02-22 14:26:01,271 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z2.tif started
2023-02-22 14:26:02,970 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z4.tif started
2023-02-22 14:26:04,576 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z2.tif started
2023-02-22 14:26:06,164 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z5.tif started
2023-02-22 14:26:07,729 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z6.tif started
2023-02-22 14:26:09,300 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z4.tif started
2023-02-22 14:26:10,923 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z3.tif started
2023-02-22 14:26:12,499 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z6.tif started
2023-02-22 14:26:14,080 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z1.tif started
2023-02-22 14:26:15,735 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z1.tif started
2023-02-22 14:26:17,334 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound1_z4.tif started
2023-02-22 14:26:18,988 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z6.tif started
2023-02-22 14:26:20,706 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound3_z1.tif started
2023-02-22 14:26:22,314 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z4.tif started
2023-02-22 14:26:23,966 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z5.tif started
2023-02-22 14:26:25,572 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z0.tif started
2023-02-22 14:26:27,162 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z1.tif started
2023-02-22 14:26:28,796 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z3.tif started
2023-02-22 14:26:30,387 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z6.tif started
2023-02-22 14:26:32,148 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_PolyT_z2.tif started
2023-02-22 14:26:33,790 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z5.tif started
2023-02-22 14:26:35,416 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_DAPI_z0.tif started
2023-02-22 14:26:37,033 - . - INFO - sum_signals.calculate for /home/user/202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_Cellbound2_z3.tif started
0%|                                                                                                                                                                     | 0/35 [00:00<?, ?it/s]2023-02-22 14:26:38,838 - . - INFO - all jobs finished
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:00<00:00, 1307.91it/s]
2023-02-22 14:26:38,865 - . - INFO - results combined
2023-02-22 14:26:38,873 - . - INFO - Sum signals finished
```

## Step 4: Update the.vzg File

In order to examine the new cell boundaries in the Vizgen MERSCOPE™ Vizualizer, vpt is able to open and modify an existing.vzg file.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose --processes 2 update-vzg \
> --input-vzg 202305010900_U2OS_small_set_VMSC00000/region_0/202305010900_U2OS_small_set_VMSC00000_region_0.vzg \
> --input-boundaries analysis_outputs/cellpose_micron_space.parquet \
>  --input-entity-by-gene analysis_outputs/cell_by_gene.csv \
> --input-metadata analysis_outputs/cell_metadata.csv \
> --output-vzg analysis_outputs/202305010900_U2OS_small_set_VMSC00000_region_0_CellPose_PolyT.vzg
```

**Console Output**

```
2023-02-22 14:30:49,640 - . - INFO - run update-vzg with args:Namespace(input_vzg='202305010900_U2OS_small_set_VMSC00000/region_0/202305010900_U2OS_small_set_VMSC00000_region_0.vzg', input_boundaries='analysis_outputs/cellpose_micron_space.parquet', input_entity_by_gene='analysis_outputs/cell_by_gene.csv', output_vzg='analysis_outputs/202305010900_U2OS_small_set_VMSC00000_region_0_CellPose_PolyT.vzg', input_metadata='analysis_outputs/cell_metadata.csv', temp_path='/home/wiggin/vzg_build_temp', overwrite=False)
202305010900_U2OS_small_set_VMSC00000/region_0/202305010900_U2OS_small_set_VMSC00000_region_0.vzg unpacked!
2023-02-22 14:30:50,294 - . - INFO - Running cell assembly in 2 processes
Done fov 0
2023-02-22 14:31:12,175 - . - INFO - Cells binaries generation completed
Start calculating expression matrices
Start calculating coloring arrays
Finish calculating
2023-02-22 14:31:12,413 - . - INFO - Assembler data binaries generation complected
new vzg file created
temp files deleted
2023-02-22 14:31:15,982 - . - INFO - Update VZG completed
```

Note

Updating the.vzg file requires enough hard drive space to decompress and compress the file. For large experiments this may be a significant amount of data (>50 GiB), so ensure that the vpt compute environment has sufficient disk space.

Once the vzg file is updated, it is possible to explore the data in the Vizgen MERSCOPE™ Vizualizer as usual

[![An image of cells in the Vizgen MERSCOPE™ Vizualizer](https://vizgen.github.io/vizgen-postprocessing/_images/cellpose_segmented_small_dataset.png)](https://vizgen.github.io/vizgen-postprocessing/_images/cellpose_segmented_small_dataset.png)

To download a copy of the Vizgen MERSCOPE™ Vizualizer, visit: [https://portal.vizgen.com/](https://portal.vizgen.com/)



## Example: Re-segmenting a MERSCOPE Heart Dataset with a Machine Learning Model Customized with Manual Annotations

## Introduction

MERSCOPE maps the precise locations of hundreds of millions of transcripts from hundreds of genes across intact tissue slices using MERFISH technology. The high-resolution, highly multiplexed measurements enable mapping the cellular composition of tissues with fine cell-type resolution. This single-cell analysis is greatly facilitated by MERSCOPEs cell boundary stain and onboard cell segmentation algorithms that provide single-cell analysis results alongside the list of detected transcripts. However, cells in some tissues, such as multinucleated muscle tissues or tissues with abnormal cell shape may not be segmented well by the MERSCOPE onboard cell segmentation algorithms. For these tissues that are more challenging to segment, you can improve the segmentation results by hand annotating a small number of tiles, following the steps described in this workflow.

The workflow includes several steps – the first part consists of evaluating the initial segmentation to determine the need to improve the segmentation results. Next, identifying problematic regions using MERSCOPE Vizualizer and extracting corresponding image patches with the Vizgen Postprocessing Tool (VPT). Then, loading the image patches into Cellpose 2 (Pachitariu & Stringer, Nature Methods, 2022) and annotating the boundaries of the observed cells to retrain the Cellpose2 model. Lastly, using VPT to resegment the full dataset and evaluate the improved segmentation results. Here we show how the workflow can be applied to heart tissue to generate improved segmentation where the resegmented cells better visually match the cell boundary stain, have a much larger volume consistent with the larger volume of heart cells, and a substantially larger fraction of transcripts are assigned to a cell compared to the original cell segmentation results. The heart dataset used has an imaged area of 27.5 square millimeters and 205,596,450 transcripts.

[![An image of cellpose2 workflow](https://vizgen.github.io/vizgen-postprocessing/_images/cellpose2_workflow.png)](https://vizgen.github.io/vizgen-postprocessing/_images/cellpose2_workflow.png)

  

**Workflow inputs**

This workflow processes the output data from the MERSCOPE instrument. For a dataset named %EXPERIMENT\_NAME%, the output data is found on the MERSCOPE instrument under z:\\merfish\_output\\%EXPERIMENT\_NAME%\\%REGION\_NAME%\\. Within this directory, you can find the following files that are required as inputs for this workflow.

| Input | File Name | Description |
| --- | --- | --- |
| Mosaic tiff images | images/mosaic\_{stain name}\_z{z level}.tif | Every image channel acquired during a MERFISH experiment that is not decoded as a MERFISH  bit, will be output as a mosaic tiff image in this folder. This includes DAPI, PolyT, Cellbound  stains (if applicable), and subsequent round stains (if applicable). The raw data images from  the MERFISH experiment are stitched together based on the alignment of fiducial beads to create  a mosaic that minimizes the appearance of seams between fields of view. The images themselves  are single channel, single plane, 16-bit grayscale tiff files, with the naming convention  `mosaic_{stain name}_z{z level}.tif` |
| Micron to mosaic pixel transformation matrix | micron\_to\_mosaic\_pixel\_transform.csv | An affine transformation matrix describing translation and a scaling to convert from micron  units (used for transcirpt locations) to pixel units (of the mosaic images). This file helps convert  the coordinates of the pixels in the mosaic tiff images to real world micron coordinates. |
| List of detected transcripts | detected\_transcripts.csv | The `detected_transcripts.csv` file is a standard comma separated values (csv) formatted  text file that lists all of the transcripts detected in the MERSCOPE run, include the gene identity  of each transcripts and it’s x, y, and z location within the sample. |
| VZG file | {experiment\_name}\_region\_{region\_index}.vzg | The VZG file contains a representation of the dataset that can be opened with the  MERSCOPE Vizualizer. It contains all the information needed to interactively visualize  the transcript locations, cell boundaries, and a compressed version of the mosaic image  channels (e.g. DAPI, PolyT, Cellbound stains). |

**Workflow Summary (6 hours 30 minutes)**

Note

We ran the VPT re-segmentation on a more powerful computer than we used for the rest of the workflow. We utilized a compute instance with 32 cores and 256GB of RAM. The rest of the steps were completed on a machine with 4 cores and 16GB of RAM.

| Step | Summary | Time Estimate (for heart dataset) |
| --- | --- | --- |
| System setup | Identify the location of the required input files. Install MERSCOPE Vizualizer, Vizgen  Postprocessing Tool (VPT), and Cellpose 2. | 15 minutes |
| Evaluate baseline segmentation | Load the VZG file for the dataset into the MERSCOPE Vizualizer and evaluate the initial  segmentation results. | 15 minutes |
| Identify regions to target for segmentation model retraining  and extract image patches | Identify regions that need improved segmentation using MERSCOPE Vizualizer. Extract  corresponding image patches from the mosaic tiff images using VPT. For this example we  extracted 20 patches, each 108 x 108 um. | 30 minutes |
| Annotate cell boundaries on extracted image patches | Load the extracted patches into the Cellpose UI and use the Cellpose tools to annotate the  boundaries on each image. | 3 hours, 30 minutes |
| Retrain the machine learning model using the annotations | Use the Cellpose UI to retrain the base model using the manual annotations. Here we  retrained with 100 epochs. | 30 minutes |
| Re-segment the full MERSCOPE dataset using the  retrained model | Use VPT to re-segment the full dataset, generating a new VZG file, cell metadata, and  cell by gene matrix with the new segmentation results. | 1 hour |
| Evaluate the new segmentation | Load the new VZG file into MERSCOPE Vizualizer to qualitatively examine the new  segmentation and use VPT to generate a quantitative segmentation report. | 30 minutes |

## System Setup

**Requirements Summary**

- Computer:
- Software:
	> - MERSCOPE Vizualizer
	> - Python >=3.9 and <3.11 with virtual environments configured for:
	> 	> - `vpt >= 1.2.0`
	> 	> - `cellpose >= 2.0.0`

**MERSCOPE Vizualizer**

The MERSCOPE Vizualizer is a software tool for interactively exploring MERSCOPE output data and is available to any MERSCOPE user. MacOS and Windows versions can be downloaded from [here.](https://portal.vizgen.com/resources/software)

**Python**

VPT and Cellpose2 are Python libraries and require a version of Python between 3.9 and 3.11 to be installed. Python can be downloaded from [Download Python.](https://www.python.org/downloads/) Once python is installed, pip and venv modules should be installed before proceeding to installing VPT and Cellpose2.

**VPT (with cellpose2 plugin)**

Vizgen postprocessing tool (VPT) is a command line tool that facilitates re-segmenting full MERSCOPE output datasets with customized segmentation parameters. To install VPT, follow the instructions at [Installation](https://vizgen.github.io/vizgen-postprocessing/installation.html#installation). This workflow requires `vpt >= 1.2.0`. If you don’t have the latest version if VPT, it should be upgraded to the latest version using the command:

```bash
pip install --upgrade vpt
```

The Cellpose2 plugin is available as a Python package and can be installed using pip:

```bash
pip install vpt-plugin-cellpose2
```

Note

For the plugin to be recognized, it must be installed in the same environment as VPT

**Cellpose2**

Cellpose2 is widely used segmentation tool created by Professor Carsen Stringer’s lab. Cellpose2 contains tools for interactively annotating images and retraining the base Cellpose2 models. For additional resources, please visit [PyPi](https://pypi.org/project/cellpose/) or the [cellpose installation page](https://cellpose.readthedocs.io/en/latest/installation.html). To prepare cellpose2 for this workflow:

1. Create a virtual environment and activate it
2. Install `cellpose >= 2.0.0` with the GUI into the virtual environment

**User Input**

```
user@computer:~$ python3 -m venv ~/.venv/cellpose2
user@computer:~$ source .venv/cellpose2/bin/activate
(cellpose2) user@computer:~$ pip install cellpose[gui]
```

**Console Output**

```
[ pip installation progress trimmed for brevity ]

Successfully installed MarkupSafe-2.1.3 PyQt6-Qt6-6.6.1 cachetools-5.3.2 cellpose-2.2.3 certifi-2023.11.17 charset-normalizer-3.3.2 colorama-0.4.6 fastremap-1.14.0 filelock-3.13.1 fsspec-2023.12.2 google-api-core-2.15.0 google-auth-2.25.2 google-cloud-core-2.4.1 google-cloud-storage-2.14.0 google-crc32c-1.5.0 google-resumable-media-2.7.0 googleapis-common-protos-1.62.0 idna-3.6 imagecodecs-2023.9.18 jinja2-3.1.2 llvmlite-0.41.1 mpmath-1.3.0 natsort-8.4.0 networkx-3.2.1 numba-0.58.1 numpy-1.26.2 opencv-python-headless-4.8.1.78 packaging-23.2 protobuf-4.25.1 pyasn1-0.5.1 pyasn1-modules-0.3.0 pygments-2.17.2 pyqt6-6.6.1 pyqt6.sip-13.6.0 pyqtgraph-0.13.3 qtpy-2.4.1 requests-2.31.0 roifile-2023.8.30 rsa-4.9 scipy-1.11.4 superqt-0.6.1 sympy-1.12 tifffile-2023.12.9 torch-2.1.2 tqdm-4.66.1 typing-extensions-4.9.0 urllib3-2.1.0
```

We recommend confirming that Cellpose has been installed properly and that the Cellpose2 UI can be opened using the command:

```
(cellpose2) user@computer:~$ python -m cellpose
```

## Step 1: Evaluate Baseline Segmentation

To evaluate whether the segmentation may benefit from a machine learning model retrained with manual annotations, we begin by qualitatively and quantitatively evaluating the out-of-the-box segmentation. We find that retraining the machine learning model can substantially improve the cell segmentation on samples if either cells visually present in the DAPI or cell boundary stain images but not detected in the out-of-the-box segmentation or the cells have atypical cell morphology that is not well traced by the out of the box segmentation.

**Qualitative segmentation evaluation with MERSCOPE Vizualizer**

To qualitatively explore the initial segmentation results, we loaded the VZG file from the mouse heart experiment into the MERSCOPE Vizualizer software and examined the segmentation boundaries overlaid on the detected transcripts and the DAPI and cellpoa portion o. In the image below, we overlay the cell boundaries on top of the mosaic. The DAPI is colored blue, the cell boundary stain is colored green, the transcripts are overlaid as points, and the segmented cell boundaries are shown as cyan lines. Immediately, it can be seen that the geometries do not closely follow the clear cell boundaries and many transcripts fall outside of the cell boundaries.

[![An image of cellpose1](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image1.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image1.png)

  

**Quantitative segmentation evaluation with MERSCOPE segmentation summary report**

To MERSCOPE run summary report contains a segmentation summary that can facilitate quantitative evaluation of the segmentation quality. For experiments ran prior to MERSCOPE instrument control software version 234, the segmentation summary can be generated using the VPT command, `generate-segmentation-metrics`. For more information about the options and arguments that may be passed to `generate-segmentation-metrics`, please see the [Command Line Interface](https://vizgen.github.io/vizgen-postprocessing/command_line_interface/index.html#command-line-interface) section of the user guide.

An example of the segmentation report for the mouse heart dataset is shown below. From this segmentation summary, both the tissue area covered by cells and transcripts within a cell appear low (38% and 64% respectively), consistent with the qualitative evaluation using the MERSCOPE Vizualizer.

[![An image of cellpose1 report](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image2.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image2.png) [![An image of cellpose1 report](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image3.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image3.png) [![An image of cellpose1 report](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image4.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image4.png)

  

## Step 2: Identify Regions to Target for Segmentation Model Retraining and Extract Image Patches

To retrain the machine learning model to achieve improved cell segmentation results, we first have to identify regions of the tissue to extract images to manually annotate. Since MERSCOPE Vizualizer enables interactively exploring a MERSCOPE output dataset, it is an ideal tool for identify regions of interest across the sample. To identify regions of the heart to extract for manual annotation, we opened the MERSCOPE Vizualizer, loaded the VZG file for the experiment, and identified regions where the segmentation boundaries did not match the expectations based on the DAPI and cell boundary stain images. If there are diverse cell morphologies across different regions of the tissue, we recommend extracting a diversity of regions covering the diversity of cell morphologies to avoid over-optimizing the model against a subset of the tissue.

Once a region is identified, the following steps allow you to extract the corresponding image patch that can be loaded into Cellpose2.

1. Select the “Toggle info panel” button in the top left corner of the window (highlighted in red below).
2. Zoom in to the area of interest and use the “Window center (um)” or use the live cursor coordinates named “Cursor position (um)” from the info panel in the bottom left corner of the window (highlighted in red below) to find the (x, y) coordinate in micron space of the patch center you want to extract.
[![An image of Vizualizer](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image5.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image5.png)

  

1. Record the (x,y) center coordinates of the selected region (center\_x=4316.0, center\_y=2512.0 here).
2. Use VPT to extract the corresponding patch from the mosaic images using the `extract-image-patch` command. For more information about the options and arguments that may be passed to `extract-image-patch`, please see the [Command Line Interface](https://vizgen.github.io/vizgen-postprocessing/command_line_interface/index.html#command-line-interface) section of the user guide. This generates an RGB PNG image patch wherever you have specified the output.
	> 1. To minimize file transfer later on in the workflow, we recommend saving all output PNGs to the same folder.
	> 2. Note for this example heart dataset, the MERSCOPE Cell Boundary Stain was used and the Cellbound1 and Cellbound3 images were included in the output patch. For experiments where the MERSCOPE Cell Boundary Stain wasn’t used, DAPI and PolyT stains are still available for segmentation.
3. Repeat steps 2 through 4 for each of the regions selected for manual annotation.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose --log-level 1 extract-image-patch \
> --input-images MsHeart/region_0/images/ \
> --input-micron-to-mosaic MsHeart/region_0/images/micron_to_mosaic_pixel_transform.csv \
> --output-patch analysis_outputs/patch_4316_2512.png \
> --center-x 4316.0 \
> --center-y 2512.0 \
> --size-x 108 \
> --size-y 108 \
> --input-z-index 3 \
> --red-stain-name Cellbound1 \
> --green-stain-name Cellbound3 \
> --blue-stain-name DAPI \
> --normalization CLAHE
```

**Console Output**

```
2023-12-06 16:53:22,352 - . - INFO - run extract-image-patch with args:Namespace(input_images='MsHeart/region_0/images/', input_micron_to_mosaic='MsHeart/region_0/images/micron_to_mosaic_pixel_transform.csv', output_patch='analysis_outputs/patch_4316_2512.png', center_x=4316.0, center_y=2512.0, size_x=108.0, size_y=108.0, input_z_index=3, red_stain_name='Cellbound2', green_stain_name='Cellbound3', blue_stain_name='DAPI', normalization='CLAHE', overwrite=False)
2023-12-06 16:53:23,500 - . - INFO - extract image patch started
2023-12-06 16:54:38,346 - . - INFO - extract image patch finished
```

An example of an output RGB PNG is shown below:

[![An image of a patch](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image6.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image6.png)

  

## Step3: Annotate Cell Boundaries on Extracted Image Patches

Retraining the machine learning model requires manually indicating the ideal location of the cell boundaries within each of the selected image patches. This is achieved by loading the image patches into the Cellpose2 UI and using the Cellpose2 tools to indicate the ideal cell boundaries. To do this, follow these steps:

1. Activate the cellpose 2.0 environment created during setup

```
user@computer:~$ source .venv/cellpose2/bin/activate
```

1. Launch the Cellpose UI (the UI should immediately pop up):

**User Input**

```
(cellpose2) user@computer:~$ python -m cellpose
```

**Console Output**

```
2023-12-21 15:57:00,717 [INFO] WRITING LOG OUTPUT TO user\.cellpose\run.log
2023-12-21 15:57:00,717 [INFO]
cellpose version:       2.2.3
platform:               win32
python version:         3.9.13
torch version:          2.0.0+cpu
2023-12-21 15:57:01,681 [INFO] TORCH CUDA version not installed/working.
```

[![An image of Cellpose GUI](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image7.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image7.png)

  

1. Load the PNG image that was saved in the previous step via File → Load image (\*.tif, \*.png, \*.jpg).
[![An image of Cellpose GUI](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image8.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image8.png)

  

1. Modify the segmentation parameters on the left panel and select the desired segmentation channels.
	> 1. The cyto2 model we used here accepts a channel to segment and an optional nuclear channel. Here we selected channel 0 as “chan to segment” and channel 3 as “chan2 (optional).”
	> 2. A more detailed description of each parameter can be found in the [Cellpose API](https://cellpose.readthedocs.io/en/latest/api.html#).
2. To facilitate the hand annotation, we first ran a baseline segmentation model on the image patch to generate preliminary cell boundaries to manually adjust by hand. The baseline model can either be one from the Cellpose2 “model zoo” or a previously trained custom model.
	> 1. We suggest evaluating the models in the model zoo to determine which gives the best baseline segmentation. The best baseline segmentation will both require the fewest manual edits and likely be the best base model to use for the following retraining step.
3. Modify the baseline segmentation masks by following the instructions in the Cellpose2 instruction video: [Cellpose2: human-in-the-loop model training (2x speed)](https://www.youtube.com/watch?v=3Y1VKcxjNy4).
4. After any modification, a save via File → Save masks and image (as \*\_seg.npy) OR Ctrl+S will save the new annotation NPY file in the current working directory with the same name as the image with a “\_seg.npy” tagged on the end.
[![An image of Cellpose GUI](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image9.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image9.png)

  

## Step 4: Retrain the Machine Learning Model Using the Annotations

Once the manual annotations are created, Cellpose2 facilitates retraining the base model with the new annotations. To train on the series of ROI images that were just annotated, ensure all PNG images and associated NPY files are in a common folder and ensure that folder is set as the current working directory (can be seen at the top of the cellpose2 GUI). Once all manual annotations are completed, we retrained the model following the steps below:

1. In the menu bar along the top of the window, select Models → Train new model with images+masks in folder
2. Enter the name of the new model, the base model the new model should be derived from, and the training parameters and run the model.
	> 1. The model to be trained on top of is the model whose weights you wish to adjust. We recommend using the built-in cellpose model that was just used to generate the baseline for the manual adjustments. If no built-in model provided reasonable baseline segmentation, you may wish instead to train a model from scratch by selecting “scratch”.
	> 2. Model training parameters
	> 	> 1. **Learning Rate:** The size of the steps taken during gradient descent (used to scale the magnitude of parameter updates). A higher rate can speed up learning, but risks not minimizing the loss function, while a lower rate may lead to slow convergence.
	> 	> 2. **Weight Decay:** A regularization technique that penalizes large weights in the model. This can help to prevent overfitting by discouraging overly complex models.
	> 	> 3. **Number of Epochs:** The number the total passes through the training data. Here we used 100 epochs.
3. The model will get saved to the \\models folder in the current working directory and/or wherever you installed cellpose and specified the model locations (typically a \\.cellpose\\models\\ folder in the \\Users directory).
4. To evaluate the new model, import an image patch, modify the segmentation parameters to match the settings used for training, and select the custom model in the “custom models” section. Select “run model” as highlighted in red in the image below and examine the results of the model on your image.
5. If the results of the retrained model do not closely meet your expectations, we recommend either including additional image patches, adjusting the segmentation parameters, or changing the base model.
[![An image of Cellpose GUI](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image10.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image10.png)

  

## Step 5: Re-segment Full MERSCOPE Dataset Using Retrained Model

Once the refined Cellpose2 model was created using the manual annotations, we need to run the retrained model across the full MERSCOPE dataset to regenerate the cell by gene matrix to use for downstream single cell analysis. To do this processing, we use VPT to resegment the original images using the newly trained model.

The segmentation algorithm for VPT is specified through an algorithm JSON file. Example algorithm JSON files for Cellpose2 can be found in the “example\_analysis\_algorithm” folder within the vpt-plugin-cellpose2 repository: [https://github.com/Vizgen/vpt-plugin-cellpose2/tree/develop/example\_analysis\_algorithm](https://github.com/Vizgen/vpt-plugin-cellpose2/tree/develop/example_analysis_algorithm). These can be used as a template for customizing to match the parameters specified within the Cellpose2 UI.

1. Files with “custom” are examples using customs models and not built-in models.
2. Files with “2task” are examples uisng multiple segmentation tasks whose results get harmonized. Typically one task segments the cell boundaries and the other segments nuclei.

In the algorithm JSON file, there are some fields that need to be updated. This includes the path to the newly saved custom model and the channel colors to the proper stain in the “segmentation\_properties” section:

```javascript
"segmentation_properties": {
    "model": null,
    "model_dimensions": "2D",
    "custom_weights": "CP_20230830_093420",
    "channel_map": {
    "red": "Cellbound1",
    "green": "Cellbound3",
    "blue": "DAPI"
    }
},
```

and the channel names, cell diameter, and thresholds in the “segmentation\_parameters” section:

```javascript
"segmentation_parameters": {
    "nuclear_channel": "DAPI",
    "entity_fill_channel": "all",
    "diameter": 137.76,
    "flow_threshold": 0.95,
    "cellprob_threshold": -5.5,
    "minimum_mask_size": 500
},
```

To achieve equivalent results to what was observed earlier in the Cellpose2 UI, the diameter parameter should be set equal to the expected one from the cellpose2 GUI. To get this value, load the newly trained model in the “custom models” section and read the value filled in the “cell diameter” field, as indicated in the image below:

[![An image of Cellpose GUI](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image11.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image11.png)

  

With the segmentation parameter file configured, we used VPTs run-segmentation command to run the new model at scale by utilizing the cellpose2 plugin. This generates a new parquet file with the segmentation boundaries determined using the newly trained model across the full dataset. Note that we ran the VPT re-segmentation on a more powerful computer than we used for the rest of the workflow. We utilized a compute instance with 32 cores and 256GB of RAM.

**User Input**

```javascript
(vpt_env) user@computer:~$ vpt --verbose --processes 32 run-segmentation \
> --segmentation-algorithm example_analysis_algorithm/cellpose2_custom_2task.json \
> --input-images="MsHeart/region_0/images/mosaic_(?P<stain>[\w|-]+)_z(?P<z>[0-9]+).tif" \
> --input-micron-to-mosaic MsHeart/region_0/images/micron_to_mosaic_pixel_transform.csv \
> --output-path analysis_outputs \
> --tile-size 2400 \
> --tile-overlap 200
```

**Console Output**

```javascript
2024-01-03 16:45:11,103 - . - INFO - run run-segmentation with args:Namespace(segmentation_algorithm='202305010900_U2OS_small_set_VMSC00000/region_0/cellpose2_2task_optimized.json', input_images='202305010900_U2OS_small_set_VMSC00000/region_0/images/mosaic_(?P<stain>[\\w|-]+[0-9]?)_z(?P<z>[0-9]+).tif', input_micron_to_mosaic='202305010900_U2OS_small_set_VMSC00000/region_0/images/micron_to_mosaic_pixel_transform.csv', output_path='202305010900_U2OS_small_set_VMSC00000/cellpose2/', tile_size=1000, tile_overlap=200, max_row_group_size=17500, overwrite=True)
2024-01-03 16:45:11,600 - . - INFO - run_segmentation started
2024-01-03 16:45:11,797 - . - INFO - prepare segmentation started
2024-01-03 16:46:08,556 - . - INFO - prepare segmentation finished
2024-01-03 16:46:16,445 - ./task-368 - INFO - Run segmentation on tile 368 started
2024-01-03 16:46:16,477 - ./task-344 - INFO - Run segmentation on tile 344 started
2024-01-03 16:46:16,497 - ./task-396 - INFO - Run segmentation on tile 396 started
2024-01-03 16:46:16,497 - ./task-364 - INFO - Run segmentation on tile 364 started
2024-01-03 16:46:16,517 - ./task-380 - INFO - Run segmentation on tile 380 started
2024-01-03 16:46:16,517 - ./task-376 - INFO - Run segmentation on tile 376 started
2024-01-03 16:46:16,547 - ./task-308 - INFO - Run segmentation on tile 308 started
2024-01-03 16:46:16,547 - ./task-284 - INFO - Run segmentation on tile 284 started
2024-01-03 16:46:16,547 - ./task-312 - INFO - Run segmentation on tile 312 started
2024-01-03 16:46:16,547 - ./task-372 - INFO - Run segmentation on tile 372 started
2024-01-03 16:46:16,547 - ./task-360 - INFO - Run segmentation on tile 360 started
2024-01-03 16:46:16,547 - ./task-316 - INFO - Run segmentation on tile 316 started
2024-01-03 16:46:16,579 - ./task-300 - INFO - Run segmentation on tile 300 started
.
.
.
[ run-segmentation progress trimmed ]
.
.
.
2024-01-03 16:46:17,099 - ./task-376 - INFO - Tile 376 [22400, 11200, 1600, 1600]
2024-01-03 16:46:17,172 - ./task-368 - INFO - Tile 368 [11200, 11200, 1600, 1600]
2024-01-03 16:46:17,172 - ./task-284 - INFO - Tile 284 [19600, 8400, 1600, 1600]
2024-01-03 16:46:17,180 - ./task-308 - INFO - Tile 308 [53200, 8400, 1600, 1600]
2024-01-03 16:46:17,180 - ./task-312 - INFO - Tile 312 [58800, 8400, 1600, 1600]
2024-01-03 16:46:17,202 - ./task-372 - INFO - Tile 372 [16800, 11200, 1600, 1600]
2024-01-03 16:46:17,221 - ./task-316 - INFO - Tile 316 [1400, 9800, 1600, 1600]
2024-01-03 16:46:17,240 - ./task-344 - INFO - Tile 344 [40600, 9800, 1600, 1600]
2024-01-03 16:46:17,240 - ./task-364 - INFO - Tile 364 [5600, 11200, 1600, 1600]
2024-01-03 16:46:17,261 - ./task-300 - INFO - Tile 300 [42000, 8400, 1600, 1600]
.
.
.
[ run-segmentation progress trimmed ]
.
.
.
2024-01-03 16:48:21,020 - ./task-344 - INFO - generate_polygons_from_mask
2024-01-03 16:48:21,100 - ./task-344 - INFO - get_polygons_from_mask: z=0, labels:96
2024-01-03 16:48:21,171 - ./task-300 - INFO - generate_polygons_from_mask
2024-01-03 16:48:21,232 - ./task-300 - INFO - get_polygons_from_mask: z=0, labels:133
2024-01-03 16:48:21,294 - ./task-380 - INFO - generate_polygons_from_mask
2024-01-03 16:48:21,371 - ./task-380 - INFO - get_polygons_from_mask: z=0, labels:110
2024-01-03 16:48:21,404 - ./task-384 - INFO - generate_polygons_from_mask
2024-01-03 16:48:21,481 - ./task-384 - INFO - get_polygons_from_mask: z=0, labels:110
2024-01-03 16:48:21,549 - ./task-392 - INFO - generate_polygons_from_mask
2024-01-03 16:48:21,626 - ./task-336 - INFO - generate_polygons_from_mask
2024-01-03 16:48:21,637 - ./task-392 - INFO - get_polygons_from_mask: z=0, labels:127
2024-01-03 16:48:21,704 - ./task-336 - INFO - get_polygons_from_mask: z=0, labels:94
2024-01-03 16:48:22,262 - ./task-364 - INFO - raw segmentation result contains 36 rows
2024-01-03 16:48:22,263 - ./task-364 - INFO - fuze across z
2024-01-03 16:48:22,408 - ./task-372 - INFO - generate_polygons_from_mask
2024-01-03 16:48:22,410 - ./task-364 - INFO - remove edge polys
2024-01-03 16:48:22,486 - ./task-372 - INFO - get_polygons_from_mask: z=0, labels:90
2024-01-03 16:48:23,529 - ./task-284 - INFO - raw segmentation result contains 84 rows
2024-01-03 16:48:23,529 - ./task-284 - INFO - fuze across z
2024-01-03 16:48:23,716 - ./task-284 - INFO - remove edge polys
2024-01-03 16:48:25,166 - ./task-344 - INFO - raw segmentation result contains 94 rows
2024-01-03 16:48:25,166 - ./task-344 - INFO - fuze across z
.
.
.
[ run-segmentation progress trimmed ]
.
.
.
2024-01-03 17:23:19,716 - . - INFO - After both resolution steps, found 0 uncaught overlaps
2024-01-03 17:23:49,330 - . - INFO - Resolved overlapping in the compiled dataframe
2024-01-03 17:23:56,102 - . - INFO - Saved compiled dataframe for entity cell in micron space
2024-01-03 17:24:14,568 - . - INFO - Saved compiled dataframe for entity cell in mosaic space
2024-01-03 17:24:14,569 - . - INFO - Compile tile segmentation finished
2024-01-03 17:24:15,509 - . - INFO - run_segmentation finished
```

Along with the cell boundaries, we also regenerated the cell by gene matrix (the number of times a transcript from each of the targetted genes appears within each of the segmented cell boundaries), the cell metadata (containing coordinates, volume, and transcript counts for each cell), and updated the vzg file to include the new segmentation boundaries using the following commands in VPT:

**User Input**

```javascript
(vpt_env) user@computer:~$ vpt --verbose partition-transcripts \
> --input-boundaries analysis_outputs/cellpose2_micron_space.parquet \
> --input-transcripts MsHeart/region_0/detected_transcripts.csv \
> --output-entity-by-gene analysis_outputs/cell_by_gene.csv
```

**Console Output**

```javascript
2023-12-22 18:12:22,915 - . - INFO - run partition-transcripts with args:Namespace(input_boundaries='analysis_outputs/cellpose2_micron_space.parque', input_transcripts='MsHeart/region_0/detected_transcripts.csv', output_entity_by_gene='analysis_outputs/cell_by_gene.csv', chunk_size=10000000, output_transcripts=None, overwrite=False)
2023-12-22 18:12:23,023 - . - INFO - Partition transcripts started
2023-12-22 19:36:15,115 - . - INFO - cell by gene matrix saved as analysis_outputs/cell_by_gene.csv
2023-12-22 19:36:15,115 - . - INFO - Partition transcripts finished
```

**User Input**

```javascript
(vpt_env) user@computer:~$ vpt --verbose derive-entity-metadata \
> --input-boundaries analysis_outputs/cellpose2_micron_space.parquet \
> --output-metadata analysis_outputs/cell_metadata.csv
```

**Console Output**

```javascript
2023-12-22 21:09:19,721 - . - INFO - run derive-entity-metadata with args:Namespace(input_boundaries='analysis_outputs/cellpose2_micron_space.parquet', output_metadata='analysis_outputs/cell_metadata.csv', input_entity_by_gene=None, overwrite=False)
2023-12-22 21:09:19,828 - . - INFO - Derive cell metadata started
2023-12-22 21:12:58,070 - . - INFO - Derive cell metadata finished
```

**User Input**

```javascript
(vpt_env) user@computer:~$ vpt --verbose --processes 8 update-vzg \
> --input-vzg MsHeart/region_0/MsHeart_region_0.vzg \
> --input-boundaries analysis_outputs/cellpose2_micron_space.parquet \
> --input-entity-by-gene analysis_outputs/cell_by_gene.csv \
> --output-vzg analysis_outputs/MsHeart_region_0_cellpose2.vzg
```

**Console Output**

```javascript
2024-01-04 15:53:07,072 - . - INFO - run update-vzg with args:Namespace(input_vzg='MsHeart/region_0/MsHeart_region_0.vzg', input_boundaries='analysis_outputs/cellpose2_micron_space.parquet', input_entity_by_gene='analysis_outputs/cell_by_gene.csv', output_vzg='analysis_outputs/MsHeart_region_0_cellpose2.vzg', input_metadata=None, input_entity_type=None, overwrite=False)
2024-01-04 15:53:07,162 - . - INFO - Unpacking vzg file
2024-01-04 15:54:22,797 - . - INFO - MsHeart/region_0/MsHeart_region_0.vzg unpacked!
2024-01-04 15:54:22,800 - . - INFO - Dataset folder: vzg_build_temp/vzg_2024-01-04T15_53_07_162250/MsHeart_region_0
2024-01-04 15:54:22,800 - . - INFO - Number of input genes: 635
2024-01-04 15:54:27,800 - . - INFO - There is no cell metadata on input, start creating
2024-01-04 15:55:56,412 - . - INFO - Cell metadata file created
2024-01-04 15:55:57,133 - . - INFO - Running cell assembly in 8 processes for feature cell
2024-01-04 15:55:59,528 - ./task-0 - INFO - running cells processing for fovs
2024-01-04 15:55:59,528 - ./task-5 - INFO - running cells processing for fovs
2024-01-04 15:55:59,529 - ./task-7 - INFO - running cells processing for fovs
2024-01-04 15:55:59,529 - ./task-1 - INFO - running cells processing for fovs
2024-01-04 15:55:59,529 - ./task-3 - INFO - running cells processing for fovs
2024-01-04 15:55:59,531 - ./task-4 - INFO - running cells processing for fovs
2024-01-04 15:55:59,546 - ./task-6 - INFO - running cells processing for fovs
2024-01-04 15:55:59,548 - ./task-2 - INFO - running cells processing for fovs
2024-01-04 16:03:13,625 - ./task-6 - INFO - Done fov 6
2024-01-04 16:04:03,978 - ./task-0 - INFO - Done fov 0
2024-01-04 16:05:21,948 - ./task-2 - INFO - Done fov 2
2024-01-04 16:05:25,018 - ./task-3 - INFO - Done fov 3
2024-01-04 16:05:26,003 - ./task-1 - INFO - Done fov 1
2024-01-04 16:05:44,222 - ./task-4 - INFO - Done fov 4
2024-01-04 16:05:48,758 - ./task-5 - INFO - Done fov 5
2024-01-04 16:06:20,875 - . - INFO - Cells binaries generation completed for feature cell
2024-01-04 16:06:24,141 - . - INFO - Start calculating expression matrices
2024-01-04 16:07:52,520 - . - INFO - Start calculating coloring arrays
2024-01-04 16:07:54,998 - . - INFO - Finish calculating
2024-01-04 16:07:55,047 - . - INFO - Assembler data binaries generation complected for feature cell
2024-01-04 16:12:29,691 - . - INFO - new vzg file created
2024-01-04 16:12:30,654 - . - INFO - temp files deleted
2024-01-04 16:12:30,654 - . - INFO - Update VZG completed
```

## Step 6: Evaluate New Segmentation

With the newly generated segmentation, we next evaluated the performance improvement from the re-annotated dataset, both qualitatively and quantitatively as we did with the original segmentation results.

For qualitative evaluation, we loaded the newly generated VZG file into the MERSCOPE Vizualizer software and examined the segmentation boundaries overlaid on the transcripts and images. With the retrained model, we can see that the segmented cells much more closely follow the elongated shape expected for the muscle cells in this region of the heart.

[![An image of Vizualizer](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image12.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image12.png)

  

With VPT, we also regenerated the segmentation summary using generate-segmentation-metrics directly to generate a new quantitative report and metrics csv file for the full-scale segmentation run. This command requires the detected\_transcripts.csv as well as the cell\_by\_gene.csv, and cell\_metadata.csv, generated from the previous VPT commands.

**User Input**

```
(vpt_env) user@computer:~$ vpt --verbose generate-segmentation-metrics \
> --input-entity-by-gene analysis_outputs/cell_by_gene.csv \
> --input-metadata analysis_outputs/cell_metadata.csv \
> --input-transcripts MsHeart/region_0/detected_transcripts.csv \
> --output-csv analysis_outputs/segmentation_metrics.csv \
> --experiment-name MsHeart_cellpose2 \
> --output-report analysis_outputs/segmentation_report.html \
> --output-clustering analysis_outputs/ \
> --input-images MsHeart/region_0/images/ \
> --input-boundaries MsHeart/cellpose2_micron_space.parquet \
> --input-micron-to-mosaic MsHeart/region_0/images/micron_to_mosaic_pixel_transform.csv \
> --input-z-index 0 \
> --red-stain-name Cellbound1 \
> --green-stain-name Cellbound3 \
> --blue-stain-name DAPI \
> --normalization CLAHE \
> --transcript-count-filter-threshold 100 \
> --volume-filter-threshold 200 \
```

**Console Output**

```
2023-12-06 12:10:32,525 - . - INFO - run generate-segmentation-metrics with args:Namespace(input_entity_by_gene='analysis_outputs/cell_by_gene.csv', input_metadata='analysis_outputs/cell_metadata.csv', input_transcripts='MsHeart/region_0/detected_transcripts.csv', output_csv='analysis_outputs/segmentation_metrics.csv', experiment_name='MsHeart_cellpose2', output_report='analysis_outputs/segmentation_report.html', output_clustering='analysis_outputs/', input_images='MsHeart/region_0/images/', input_boundaries='MsHeart/cellpose2_micron_space.parquet', input_micron_to_mosaic='MsHeart/region_0/images/micron_to_mosaic_pixel_transform.csv', input_z_index=0, red_stain_name='Cellbound1', green_stain_name='Cellbound3', blue_stain_name='DAPI', normalization='CLAHE', transcript_count_filter_threshold=100, volume_filter_threshold=200, overwrite=False)
2023-12-06 12:10:51,642 - . - INFO - Generate segmentation metrics started
2023-12-06 12:10:53,518 - . - INFO - Cell clustering started
2023-12-06 12:11:17,369 - . - INFO - Cell clustering finished
2023-12-06 12:11:17,370 - . - INFO - Making html report started
2023-12-06 12:12:02,489 - . - INFO - Making html report finished
2023-12-06 12:12:02,499 - . - INFO - Generate segmentation metrics finished
```

Opening of the summary html (shown below), the metrics and distributions can be directly compared to the previous segmentation. From the new summary report, we can see that for this example heart dataset, the retrained model significantly improved the cell segmentation metrics: cell size increased from 730.3 um^3 to 1423.3 um^3, transcripts per cell increased from 891 to 1514, and the percent of transcripts found within a segmented cell increased from 64.1% to 78.6%. For this sample with atypical cell morphology, manually annotating a few small regions and retraining the machine-learning model yielded significantly improved single-cell quantification for downstream biological analysis.

[![An image of cellpose2 report](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image13.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image13.png)

  

[![An image of cellpose2 report](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image14.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image14.png)

  

[![An image of cellpose2 report](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image15.png)](https://vizgen.github.io/vizgen-postprocessing/_images/workflow_image15.png)