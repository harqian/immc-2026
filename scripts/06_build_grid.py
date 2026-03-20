#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, WGS84, validate_geojson, write_geojson


BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
GRID_PATH = OUTPUTS_DIR / "grid.geojson"
GRID_CENTROIDS_PATH = OUTPUTS_DIR / "grid_centroids.geojson"
GRID_SIZE_M = 5000
METRIC_CRS = "EPSG:32733"
GRID_VERSION = "etosha_5km_v1"


def load_boundary() -> gpd.GeoDataFrame:
    return validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "processed boundary")


def generate_grid(boundary: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    boundary_metric = boundary.to_crs(METRIC_CRS)
    xmin, ymin, xmax, ymax = boundary_metric.total_bounds
    x_start = math.floor(xmin / GRID_SIZE_M) * GRID_SIZE_M
    y_start = math.floor(ymin / GRID_SIZE_M) * GRID_SIZE_M
    x_stop = math.ceil(xmax / GRID_SIZE_M) * GRID_SIZE_M
    y_stop = math.ceil(ymax / GRID_SIZE_M) * GRID_SIZE_M

    rows: list[dict[str, object]] = []
    row_index = 0
    y = y_start
    while y < y_stop:
        col_index = 0
        x = x_start
        while x < x_stop:
            cell = box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M)
            if not cell.intersects(boundary_metric.geometry.iloc[0]):
                x += GRID_SIZE_M
                col_index += 1
                continue
            clipped = cell.intersection(boundary_metric.geometry.iloc[0])
            if clipped.is_empty:
                x += GRID_SIZE_M
                col_index += 1
                continue
            centroid = clipped.centroid
            rows.append(
                {
                    "cell_id": f"{GRID_VERSION}_r{row_index:03d}_c{col_index:03d}",
                    "grid_size_m": GRID_SIZE_M,
                    "metric_crs": METRIC_CRS,
                    "grid_version": GRID_VERSION,
                    "cell_area_m2": clipped.area,
                    "centroid_x_m": centroid.x,
                    "centroid_y_m": centroid.y,
                    "row_index": row_index,
                    "col_index": col_index,
                    "geometry": clipped,
                }
            )
            x += GRID_SIZE_M
            col_index += 1
        y += GRID_SIZE_M
        row_index += 1

    grid_metric = gpd.GeoDataFrame(rows, geometry="geometry", crs=METRIC_CRS)
    grid = grid_metric.to_crs(WGS84)
    grid_centroids_metric = gpd.GeoDataFrame(
        grid_metric.drop(columns="geometry"),
        geometry=gpd.points_from_xy(grid_metric["centroid_x_m"], grid_metric["centroid_y_m"]),
        crs=METRIC_CRS,
    )
    grid_centroids = grid_centroids_metric.to_crs(WGS84)
    return grid, grid_centroids


def write_outputs(grid: gpd.GeoDataFrame, centroids: gpd.GeoDataFrame) -> None:
    write_geojson(grid, GRID_PATH)
    write_geojson(centroids, GRID_CENTROIDS_PATH)


def check_outputs() -> None:
    grid = validate_geojson(
        GRID_PATH,
        [
            "cell_id",
            "grid_size_m",
            "metric_crs",
            "grid_version",
            "cell_area_m2",
            "centroid_x_m",
            "centroid_y_m",
            "row_index",
            "col_index",
        ],
        "analysis grid",
    )
    centroids = validate_geojson(
        GRID_CENTROIDS_PATH,
        [
            "cell_id",
            "grid_size_m",
            "metric_crs",
            "grid_version",
            "cell_area_m2",
            "centroid_x_m",
            "centroid_y_m",
            "row_index",
            "col_index",
        ],
        "grid centroids",
    )
    if grid["cell_id"].isna().any() or centroids["cell_id"].isna().any():
        raise ValueError("grid outputs contain null cell_id values")
    if len(grid) != len(centroids):
        raise ValueError("grid polygons and centroids do not contain the same number of cells")
    if not grid["cell_id"].equals(centroids["cell_id"]):
        raise ValueError("grid polygons and centroids are not aligned by cell_id")
    print(f"grid build check passed for {len(grid)} cells")


def main() -> int:
    parser = argparse.ArgumentParser(description="build a 5 km Etosha analysis grid in a metric CRS")
    parser.add_argument("--check", action="store_true", help="validate existing grid outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    boundary = load_boundary()
    grid, centroids = generate_grid(boundary)
    write_outputs(grid, centroids)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
