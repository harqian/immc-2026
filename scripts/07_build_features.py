#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np

from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet


GRID_PATH = OUTPUTS_DIR / "grid.geojson"
GRID_CENTROIDS_PATH = OUTPUTS_DIR / "grid_centroids.geojson"
FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
ROADS_PATH = PROCESSED_DIR / "roads.geojson"
TOURIST_ROADS_PATH = PROCESSED_DIR / "tourist_roads.geojson"
GATES_PATH = PROCESSED_DIR / "gates.geojson"
CAMPS_PATH = PROCESSED_DIR / "camps.geojson"
WATERHOLES_PATH = PROCESSED_DIR / "waterholes.geojson"
PAN_PATH = PROCESSED_DIR / "pan_polygon.geojson"
WILDFIRES_PATH = PROCESSED_DIR / "wildfire_history.geojson"
METRIC_CRS = "EPSG:32733"


def load_inputs() -> dict[str, gpd.GeoDataFrame]:
    return {
        "grid": validate_geojson(
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
        ),
        "centroids": validate_geojson(
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
        ),
        "boundary": validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "processed boundary"),
        "roads": validate_geojson(ROADS_PATH, ["name", "ref", "fclass", "source"], "roads"),
        "tourist_roads": validate_geojson(
            TOURIST_ROADS_PATH,
            ["name", "ref", "fclass", "source"],
            "tourist roads",
        ),
        "gates": validate_geojson(GATES_PATH, ["name", "kind", "source", "source_detail"], "gates"),
        "camps": validate_geojson(CAMPS_PATH, ["name", "kind", "source", "source_detail"], "camps"),
        "waterholes": validate_geojson(
            WATERHOLES_PATH,
            ["name", "kind", "source", "source_detail"],
            "waterholes",
        ),
        "pan": validate_geojson(PAN_PATH, ["name", "source", "source_detail", "notes"], "pan"),
        "wildfires": validate_geojson(
            WILDFIRES_PATH,
            ["event_id", "title", "observation_date", "magnitude_ha", "magnitude_unit", "source"],
            "wildfire history",
        ),
    }


def min_distance(centroids_metric: gpd.GeoDataFrame, target_metric: gpd.GeoDataFrame) -> np.ndarray:
    union = target_metric.geometry.union_all()
    return centroids_metric.geometry.distance(union).to_numpy()


def classify_terrain(features: gpd.GeoDataFrame) -> list[str]:
    classes: list[str] = []
    for _, row in features.iterrows():
        if row["pan_overlap_ratio"] >= 0.2:
            classes.append("pan")
        elif row["dist_to_pan_m"] <= 10000:
            classes.append("pan_margin")
        elif row["dist_to_boundary_m"] <= 10000:
            classes.append("boundary_edge")
        else:
            classes.append("interior_savanna")
    return classes


def build_features(inputs: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    grid = inputs["grid"].copy()
    centroids = inputs["centroids"].copy()

    grid_metric = grid.to_crs(METRIC_CRS)
    centroids_metric = centroids.to_crs(METRIC_CRS)
    boundary_metric = inputs["boundary"].to_crs(METRIC_CRS)
    roads_metric = inputs["roads"].to_crs(METRIC_CRS)
    tourist_roads_metric = inputs["tourist_roads"].to_crs(METRIC_CRS)
    gates_metric = inputs["gates"].to_crs(METRIC_CRS)
    camps_metric = inputs["camps"].to_crs(METRIC_CRS)
    waterholes_metric = inputs["waterholes"].to_crs(METRIC_CRS)
    pan_metric = inputs["pan"].to_crs(METRIC_CRS)
    wildfires_metric = inputs["wildfires"].to_crs(METRIC_CRS)

    features = grid_metric[
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
            "geometry",
        ]
    ].copy()

    boundary_edge = gpd.GeoDataFrame(geometry=[boundary_metric.geometry.iloc[0].boundary], crs=METRIC_CRS)
    features["dist_to_boundary_m"] = min_distance(centroids_metric, boundary_edge)
    features["dist_to_fence_proxy_m"] = features["dist_to_boundary_m"]
    features["dist_to_road_m"] = min_distance(centroids_metric, roads_metric)
    features["dist_to_tourist_road_m"] = min_distance(centroids_metric, tourist_roads_metric)
    features["dist_to_gate_m"] = min_distance(centroids_metric, gates_metric)
    features["dist_to_camp_m"] = min_distance(centroids_metric, camps_metric)
    features["dist_to_waterhole_m"] = min_distance(centroids_metric, waterholes_metric)
    features["dist_to_pan_m"] = min_distance(centroids_metric, pan_metric)

    pan_intersection_area = gpd.overlay(
        features[["cell_id", "geometry"]],
        pan_metric[["geometry"]],
        how="intersection",
    )
    pan_intersection_area["overlap_area_m2"] = pan_intersection_area.geometry.area
    pan_intersection_area = pan_intersection_area.groupby("cell_id")["overlap_area_m2"].sum()
    features["pan_overlap_m2"] = features["cell_id"].map(pan_intersection_area).fillna(0.0)
    features["pan_overlap_ratio"] = features["pan_overlap_m2"] / features["cell_area_m2"]
    features["pan_overlap_ratio"] = features["pan_overlap_ratio"].clip(0.0, 1.0)

    wildfire_counts = gpd.sjoin(
        features[["cell_id", "geometry"]],
        wildfires_metric[["event_id", "geometry"]],
        how="left",
        predicate="intersects",
    ).groupby("cell_id")["event_id"].count()
    features["historical_fire_event_count"] = (
        features["cell_id"].map(wildfire_counts).fillna(0).astype(int)
    )

    features["terrain_class"] = classify_terrain(features)

    centroid_points_wgs84 = centroids_metric.to_crs("EPSG:4326").geometry
    features["centroid_lon"] = centroid_points_wgs84.x.to_numpy()
    features["centroid_lat"] = centroid_points_wgs84.y.to_numpy()

    return features.to_crs("EPSG:4326")


def write_features(features: gpd.GeoDataFrame) -> None:
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)


def check_outputs() -> None:
    features = validate_parquet(
        FEATURES_PATH,
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
            "dist_to_boundary_m",
            "dist_to_fence_proxy_m",
            "dist_to_road_m",
            "dist_to_tourist_road_m",
            "dist_to_gate_m",
            "dist_to_camp_m",
            "dist_to_waterhole_m",
            "dist_to_pan_m",
            "pan_overlap_m2",
            "pan_overlap_ratio",
            "historical_fire_event_count",
            "terrain_class",
            "centroid_lon",
            "centroid_lat",
            "geometry",
        ],
        "grid features",
    )
    if features["cell_id"].isna().any():
        raise ValueError("grid features contain null cell_id values")
    distance_columns = [
        "dist_to_boundary_m",
        "dist_to_fence_proxy_m",
        "dist_to_road_m",
        "dist_to_tourist_road_m",
        "dist_to_gate_m",
        "dist_to_camp_m",
        "dist_to_waterhole_m",
        "dist_to_pan_m",
    ]
    for column in distance_columns:
        if not np.issubdtype(features[column].dtype, np.number):
            raise ValueError(f"{column} is not numeric")
        if (features[column] < 0).any():
            raise ValueError(f"{column} contains negative values")
    if not features["terrain_class"].isin(
        ["pan", "pan_margin", "boundary_edge", "interior_savanna"]
    ).all():
        raise ValueError("terrain_class contains unexpected labels")
    if not ((features["pan_overlap_ratio"] >= 0) & (features["pan_overlap_ratio"] <= 1)).all():
        raise ValueError("pan_overlap_ratio must stay within [0, 1]")
    print(f"feature build check passed for {len(features)} cells")


def main() -> int:
    parser = argparse.ArgumentParser(description="build reusable per-cell Etosha spatial features")
    parser.add_argument("--check", action="store_true", help="validate existing feature outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    inputs = load_inputs()
    features = build_features(inputs)
    write_features(features)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
