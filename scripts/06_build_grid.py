#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import geopandas as gpd
from shapely.geometry import Polygon

from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, WGS84, validate_geojson, write_geojson


BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
GRID_PATH = OUTPUTS_DIR / "grid.geojson"
GRID_CENTROIDS_PATH = OUTPUTS_DIR / "grid_centroids.geojson"
GRID_TARGET_AREA_M2 = 25_000_000.0
METRIC_CRS = "EPSG:32733"
GRID_VERSION = "etosha_hex_25km2_v1"


def load_boundary() -> gpd.GeoDataFrame:
    return validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "processed boundary")


def hex_side_length_m(target_area_m2: float) -> float:
    return math.sqrt((2.0 * target_area_m2) / (3.0 * math.sqrt(3.0)))


def make_flat_top_hexagon(center_x_m: float, center_y_m: float, side_length_m: float) -> Polygon:
    vertices = [
        (
            center_x_m + side_length_m * math.cos(math.radians(angle_deg)),
            center_y_m + side_length_m * math.sin(math.radians(angle_deg)),
        )
        for angle_deg in range(0, 360, 60)
    ]
    return Polygon(vertices)


def generate_grid(boundary: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    boundary_metric = boundary.to_crs(METRIC_CRS)
    boundary_shape = boundary_metric.geometry.union_all()
    side_length_m = hex_side_length_m(GRID_TARGET_AREA_M2)
    horizontal_spacing_m = 1.5 * side_length_m
    vertical_spacing_m = math.sqrt(3.0) * side_length_m
    xmin, ymin, xmax, ymax = boundary_metric.total_bounds
    x_start = math.floor((xmin - 2.0 * side_length_m) / horizontal_spacing_m) * horizontal_spacing_m
    y_start = math.floor((ymin - vertical_spacing_m) / vertical_spacing_m) * vertical_spacing_m
    x_stop = math.ceil((xmax + 2.0 * side_length_m) / horizontal_spacing_m) * horizontal_spacing_m
    y_stop = math.ceil((ymax + vertical_spacing_m) / vertical_spacing_m) * vertical_spacing_m

    rows: list[dict[str, object]] = []
    cell_index = 0
    col_index = 0
    x_center_m = x_start
    while x_center_m <= x_stop:
        y_offset_m = vertical_spacing_m / 2.0 if col_index % 2 else 0.0
        row_index = 0
        y_center_m = y_start + y_offset_m
        while y_center_m <= y_stop:
            cell = make_flat_top_hexagon(x_center_m, y_center_m, side_length_m)
            if not cell.intersects(boundary_shape):
                y_center_m += vertical_spacing_m
                row_index += 1
                continue
            clipped = cell.intersection(boundary_shape)
            if clipped.is_empty or clipped.area <= 0:
                y_center_m += vertical_spacing_m
                row_index += 1
                continue
            centroid = clipped.centroid
            rows.append(
                {
                    "cell_id": f"{GRID_VERSION}_{cell_index:04d}",
                    "metric_crs": METRIC_CRS,
                    "grid_version": GRID_VERSION,
                    "cell_target_area_m2": GRID_TARGET_AREA_M2,
                    "hex_side_length_m": side_length_m,
                    "cell_area_m2": clipped.area,
                    "centroid_x_m": centroid.x,
                    "centroid_y_m": centroid.y,
                    "geometry": clipped,
                }
            )
            cell_index += 1
            y_center_m += vertical_spacing_m
            row_index += 1
        x_center_m += horizontal_spacing_m
        col_index += 1

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
            "metric_crs",
            "grid_version",
            "cell_target_area_m2",
            "hex_side_length_m",
            "cell_area_m2",
            "centroid_x_m",
            "centroid_y_m",
        ],
        "analysis grid",
    )
    centroids = validate_geojson(
        GRID_CENTROIDS_PATH,
        [
            "cell_id",
            "metric_crs",
            "grid_version",
            "cell_target_area_m2",
            "hex_side_length_m",
            "cell_area_m2",
            "centroid_x_m",
            "centroid_y_m",
        ],
        "grid centroids",
    )
    if grid["cell_id"].isna().any() or centroids["cell_id"].isna().any():
        raise ValueError("grid outputs contain null cell_id values")
    if len(grid) != len(centroids):
        raise ValueError("grid polygons and centroids do not contain the same number of cells")
    if not grid["cell_id"].equals(centroids["cell_id"]):
        raise ValueError("grid polygons and centroids are not aligned by cell_id")
    if (grid["cell_area_m2"] <= 0).any() or (centroids["cell_area_m2"] <= 0).any():
        raise ValueError("grid outputs contain non-positive cell areas")
    if grid.geometry.is_empty.any() or centroids.geometry.is_empty.any():
        raise ValueError("grid outputs contain empty geometries")
    print(f"grid build check passed for {len(grid)} cells")


def main() -> int:
    parser = argparse.ArgumentParser(description="build a hexagonal Etosha analysis grid in a metric CRS")
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
