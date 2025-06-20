import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from shapely import geometry
from vpt_core import log
from vpt_core.io.vzgfs import initialize_filesystem, io_with_retries

from vpt.utils.boundaries import Boundaries
from vpt.utils.cellsreader import CellsReader, cell_reader_factory


def polygons_generator(geometries):
    for planeGeometry in geometries:
        yield from planeGeometry.geoms


def anisotropy_calculation(multi_poly):
    bounding_rectangle = multi_poly.minimum_rotated_rectangle
    rectangle_points = list(zip(*bounding_rectangle.exterior.xy))

    sides = []
    for p1, p2 in zip(rectangle_points[:-1], rectangle_points[1:]):
        distance = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        sides.append(distance)

    major_axis = np.max(sides)
    minor_axis = np.min(sides)
    anisotropy = major_axis / minor_axis
    return anisotropy


def create_input_metadata(build_folder: str, input_boundaries: str, feature_name: str):
    log.info("There is no cell metadata on input, start creating")
    initialize_filesystem()
    input_metadata = os.path.join(build_folder, f"{feature_name}_metadata.csv")
    cells_reader: CellsReader = cell_reader_factory(input_boundaries)
    cell_bounds = Boundaries(cells_reader)
    metadata_table = create_metadata_table(cell_bounds, cells_reader.get_z_depth_per_level())

    io_with_retries(input_metadata, "w", lambda f: metadata_table.to_csv(f, sep=","))
    log.info("Cell metadata file created")
    return input_metadata


def create_metadata_table(
    bnds: Boundaries, zDepthList: Union[List, np.ndarray], barcodesCountPerCell: Optional[np.ndarray] = None
) -> pd.DataFrame:
    cell_metadata = []
    min_x, min_y, max_x, max_y, center_x, center_y = 0, 0, 0, 0, 0, 0

    for featureIdx, feature in enumerate(bnds.features):
        barcodeCount = np.nan if barcodesCountPerCell is None else barcodesCountPerCell[featureIdx]
        cell_volume = 0
        firstPoly = True
        polysCount = 0
        paRatio = 0
        for polyIdx, poly in enumerate(feature.shapes):
            if poly is None or poly.is_empty or poly.area < 1e-9:
                continue
            if firstPoly:
                min_x = poly.bounds[0]
                min_y = poly.bounds[1]
                max_x = poly.bounds[2]
                max_y = poly.bounds[3]
                center_x = poly.centroid.x
                center_y = poly.centroid.y
                firstPoly = False
            else:
                min_x = min(min_x, poly.bounds[0])
                min_y = min(min_y, poly.bounds[1])
                max_x = max(max_x, poly.bounds[2])
                max_y = max(max_y, poly.bounds[3])
                center_x += poly.centroid.x
                center_y += poly.centroid.y

            cell_volume += poly.area * zDepthList[polyIdx]
            paRatio += poly.length / poly.area
            polysCount += 1

        # If an empty feature is found, append a blank row to the output
        if polysCount == 0:
            meta = {
                "fov": np.nan,
                "EntityID": np.int64(feature.get_feature_id()),
                "volume": np.nan,
                "center_x": np.nan,
                "center_y": np.nan,
                "min_x": np.nan,
                "min_y": np.nan,
                "max_x": np.nan,
                "max_y": np.nan,
                "anisotropy": np.nan,
                "transcript_count": np.nan,
                "perimeter_area_ratio": np.nan,
                "solidity": np.nan,
            }
            cell_metadata.append(meta)
            continue

        allPlanesMultipolygon = geometry.MultiPolygon(polygons_generator(feature.get_true_polygons()))

        try:
            anisotropy = anisotropy_calculation(allPlanesMultipolygon)
        except AttributeError:
            log.info(f"Anisotropy of Entity {feature.get_feature_id()} could not be calculated")
            anisotropy = np.nan

        try:
            convex_hull_area = sum(z_poly.convex_hull.area for z_poly in feature.get_true_polygons())
            z_polygons_area = sum(z_poly.area for z_poly in feature.get_true_polygons())
            solidity = z_polygons_area / convex_hull_area
        except ZeroDivisionError:
            log.info(f"Solidity of Entity {feature.get_feature_id()} could not be calculated")
            solidity = np.nan

        meta = {
            "fov": np.nan,
            "EntityID": np.int64(feature.get_feature_id()),
            "volume": cell_volume,
            "center_x": center_x / polysCount,
            "center_y": center_y / polysCount,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "anisotropy": anisotropy,
            "transcript_count": barcodeCount,
            "perimeter_area_ratio": paRatio / polysCount,
            "solidity": solidity,
        }
        cell_metadata.append(meta)

    if cell_metadata:
        output = pd.DataFrame(cell_metadata)
    else:
        output = pd.DataFrame(
            columns=[
                "fov",
                "EntityID",
                "volume",
                "center_x",
                "center_y",
                "min_x",
                "min_y",
                "max_x",
                "max_y",
                "anisotropy",
                "transcript_count",
                "perimeter_area_ratio",
                "solidity",
            ]
        )

    output.set_index("EntityID", inplace=True)
    output = output.sort_index()
    return output
