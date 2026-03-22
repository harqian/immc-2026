#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
from scipy.stats import gaussian_kde

from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet


GRID_PATH = OUTPUTS_DIR / "grid.geojson"
FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
SPECIES_OUTPUT_PATH = OUTPUTS_DIR / "species_layers.parquet"
ELEPHANTS_PATH = PROCESSED_DIR / "elephant_density_points.parquet"
LIONS_PATH = PROCESSED_DIR / "lion_zones.geojson"
RHINOS_PATH = PROCESSED_DIR / "rhino_reference_areas.geojson"
WATERHOLES_PATH = PROCESSED_DIR / "waterholes.geojson"
METRIC_CRS = "EPSG:32733"
GRID_SCHEMA_COLUMNS = [
    "cell_id",
    "metric_crs",
    "grid_version",
    "cell_target_area_m2",
    "hex_side_length_m",
    "cell_area_m2",
    "centroid_x_m",
    "centroid_y_m",
]
FEATURE_SCHEMA_COLUMNS = [
    *GRID_SCHEMA_COLUMNS,
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
]


def load_inputs() -> dict[str, gpd.GeoDataFrame]:
    return {
        "grid": validate_geojson(
            GRID_PATH,
            GRID_SCHEMA_COLUMNS,
            "analysis grid",
        ),
        "features": validate_parquet(
            FEATURES_PATH,
            FEATURE_SCHEMA_COLUMNS,
            "grid features",
        ),
        "elephants": validate_parquet(
            ELEPHANTS_PATH,
            ["gbif_id", "scientific_name", "event_date", "basis_of_record", "source", "geometry"],
            "elephant points",
        ),
        "lions": validate_geojson(LIONS_PATH, ["zone_id", "source", "point_count", "notes"], "lion zones"),
        "rhinos": validate_geojson(
            RHINOS_PATH,
            ["name", "source", "detection_count", "notes"],
            "rhino reference areas",
        ),
        "waterholes": validate_geojson(
            WATERHOLES_PATH,
            ["name", "kind", "source", "source_detail"],
            "waterholes",
        ),
    }


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    if maximum == minimum:
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def build_elephant_density(features_metric: gpd.GeoDataFrame, elephants_metric: gpd.GeoDataFrame) -> np.ndarray:
    samples = np.vstack([elephants_metric.geometry.x.to_numpy(), elephants_metric.geometry.y.to_numpy()])
    kde = gaussian_kde(samples, bw_method=0.18)
    centroids = np.vstack([features_metric["centroid_x_m"].to_numpy(), features_metric["centroid_y_m"].to_numpy()])
    return kde(centroids)


def build_rhino_support(
    features_metric: gpd.GeoDataFrame,
    rhino_metric: gpd.GeoDataFrame,
    waterholes_metric: gpd.GeoDataFrame,
) -> np.ndarray:
    rhino_union = rhino_metric.geometry.union_all()
    waterhole_union = waterholes_metric.geometry.union_all()
    centroids = features_metric.geometry.centroid
    dist_to_rhino = centroids.distance(rhino_union).to_numpy()
    dist_to_water = centroids.distance(waterhole_union).to_numpy()
    inside_rhino = centroids.within(rhino_union).astype(float).to_numpy()
    pan_penalty = 1.0 - features_metric["pan_overlap_ratio"].to_numpy()
    support = (inside_rhino * 1.0 + np.exp(-dist_to_rhino / 12000.0) * 0.7 + np.exp(-dist_to_water / 15000.0) * 0.3)
    return support * pan_penalty


def build_lion_support(features_metric: gpd.GeoDataFrame, lion_metric: gpd.GeoDataFrame) -> np.ndarray:
    lion_union = lion_metric.geometry.union_all()
    centroids = features_metric.geometry.centroid
    dist_to_lion = centroids.distance(lion_union).to_numpy()
    inside_lion = centroids.within(lion_union).astype(float).to_numpy()
    pan_penalty = 1.0 - 0.9 * features_metric["pan_overlap_ratio"].to_numpy()
    support = inside_lion * 1.0 + np.exp(-dist_to_lion / 10000.0) * 0.6
    return support * pan_penalty


def build_herbivore_support(features_metric: gpd.GeoDataFrame) -> np.ndarray:
    water_support = np.exp(-features_metric["dist_to_waterhole_m"].to_numpy() / 18000.0)
    pan_penalty = 1.0 - features_metric["pan_overlap_ratio"].to_numpy()
    fire_penalty = 1.0 - 0.25 * np.minimum(features_metric["historical_fire_event_count"].to_numpy(), 1)
    terrain_bonus = np.where(features_metric["terrain_class"].to_numpy() == "interior_savanna", 1.0, 0.8)
    terrain_bonus = np.where(features_metric["terrain_class"].to_numpy() == "pan_margin", 1.1, terrain_bonus)
    terrain_bonus = np.where(features_metric["terrain_class"].to_numpy() == "pan", 0.2, terrain_bonus)
    return water_support * pan_penalty * fire_penalty * terrain_bonus


def build_species_layers(inputs: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    grid = inputs["grid"].copy()
    features = inputs["features"].copy()
    elephants = inputs["elephants"].to_crs(METRIC_CRS)
    lions = inputs["lions"].to_crs(METRIC_CRS)
    rhinos = inputs["rhinos"].to_crs(METRIC_CRS)
    waterholes = inputs["waterholes"].to_crs(METRIC_CRS)

    species = grid.merge(
        features.drop(columns="geometry"),
        on=GRID_SCHEMA_COLUMNS,
        how="inner",
    )
    species_metric = species.to_crs(METRIC_CRS)

    species_metric["elephant_density_raw"] = build_elephant_density(species_metric, elephants)
    species_metric["rhino_support_raw"] = build_rhino_support(species_metric, rhinos, waterholes)
    species_metric["lion_support_raw"] = build_lion_support(species_metric, lions)
    species_metric["herbivore_support_raw"] = build_herbivore_support(species_metric)

    species_metric["elephant_density_norm"] = normalize(species_metric["elephant_density_raw"].to_numpy())
    species_metric["rhino_support_norm"] = normalize(species_metric["rhino_support_raw"].to_numpy())
    species_metric["lion_support_norm"] = normalize(species_metric["lion_support_raw"].to_numpy())
    species_metric["herbivore_support_norm"] = normalize(species_metric["herbivore_support_raw"].to_numpy())

    species_metric["elephant_source"] = "gbif_elephant_points_kde"
    species_metric["rhino_source"] = "digitized_rhino_reference_distribution"
    species_metric["lion_source"] = "digitized_lion_detection_zones"
    species_metric["herbivore_source"] = "waterhole_pan_terrain_support_inference"

    return species_metric.to_crs("EPSG:4326")


def write_species_layers(species: gpd.GeoDataFrame) -> None:
    SPECIES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    species.to_parquet(SPECIES_OUTPUT_PATH, index=False)


def check_outputs() -> None:
    species = validate_parquet(
        SPECIES_OUTPUT_PATH,
        [
            "cell_id",
            "metric_crs",
            "grid_version",
            "cell_target_area_m2",
            "hex_side_length_m",
            "cell_area_m2",
            "centroid_x_m",
            "centroid_y_m",
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
            "elephant_density_raw",
            "elephant_density_norm",
            "rhino_support_raw",
            "rhino_support_norm",
            "lion_support_raw",
            "lion_support_norm",
            "herbivore_support_raw",
            "herbivore_support_norm",
            "elephant_source",
            "rhino_source",
            "lion_source",
            "herbivore_source",
            "geometry",
        ],
        "species layers",
    )
    if species["cell_id"].isna().any():
        raise ValueError("species layers contain null cell_id values")
    for column in [
        "elephant_density_raw",
        "elephant_density_norm",
        "rhino_support_raw",
        "rhino_support_norm",
        "lion_support_raw",
        "lion_support_norm",
        "herbivore_support_raw",
        "herbivore_support_norm",
    ]:
        values = species[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
        if (values < 0).any():
            raise ValueError(f"{column} contains negative values")
    for column in [
        "elephant_density_norm",
        "rhino_support_norm",
        "lion_support_norm",
        "herbivore_support_norm",
    ]:
        values = species[column].to_numpy()
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} is outside [0, 1]")
    print(f"species layer check passed for {len(species)} cells")


def main() -> int:
    parser = argparse.ArgumentParser(description="build per-cell species density and support layers")
    parser.add_argument("--check", action="store_true", help="validate existing species-layer outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    inputs = load_inputs()
    species = build_species_layers(inputs)
    write_species_layers(species)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
