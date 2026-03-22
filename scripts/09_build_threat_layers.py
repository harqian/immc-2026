#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np

from _spatial_common import OUTPUTS_DIR, validate_parquet


FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
SPECIES_PATH = OUTPUTS_DIR / "species_layers.parquet"
THREAT_OUTPUT_PATH = OUTPUTS_DIR / "threat_layers.parquet"

POACHING_WEIGHTS = {
    "gate_access": 0.24,
    "boundary_access": 0.24,
    "road_access": 0.18,
    "tourist_road_access": 0.08,
    "surveillance_gap": 0.08,
    "rhino_value": 0.18,
}
TOURISM_WEIGHTS = {
    "tourist_road_access": 0.35,
    "camp_access": 0.25,
    "gate_access": 0.15,
    "waterhole_access": 0.25,
}


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    features = validate_parquet(
        FEATURES_PATH,
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
            "geometry",
        ],
        "grid features",
    )
    species = validate_parquet(
        SPECIES_PATH,
        [
            "cell_id",
            "elephant_density_norm",
            "rhino_support_norm",
            "lion_support_norm",
            "herbivore_support_norm",
            "elephant_source",
            "rhino_source",
            "lion_source",
            "herbivore_source",
            "geometry",
        ],
        "species layers",
    )
    return features, species


def bounded_access(distance_m: np.ndarray, scale_m: float) -> np.ndarray:
    return np.exp(-distance_m / scale_m)


def build_poaching(df: gpd.GeoDataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    components = {
        "gate_access": bounded_access(df["dist_to_gate_m"].to_numpy(), 15000.0),
        "boundary_access": bounded_access(df["dist_to_boundary_m"].to_numpy(), 12000.0),
        "road_access": bounded_access(df["dist_to_road_m"].to_numpy(), 12000.0),
        "tourist_road_access": bounded_access(df["dist_to_tourist_road_m"].to_numpy(), 9000.0),
        "surveillance_gap": 1.0 - bounded_access(df["dist_to_camp_m"].to_numpy(), 18000.0),
        "rhino_value": df["rhino_support_norm"].to_numpy(),
    }
    score = sum(POACHING_WEIGHTS[name] * values for name, values in components.items())
    return score.clip(0.0, 1.0), components


def build_wildfire(df: gpd.GeoDataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    recent_fire_suppression = 1.0 - 0.7 * np.minimum(df["historical_fire_event_count"].to_numpy(), 1).astype(float)
    flammable_land = (1.0 - df["pan_overlap_ratio"].to_numpy()).clip(0.0, 1.0)
    water_remoteness = 1.0 - bounded_access(df["dist_to_waterhole_m"].to_numpy(), 18000.0)
    components = {
        "recent_fire_suppression": recent_fire_suppression,
        "flammable_land": flammable_land,
        "water_remoteness": water_remoteness,
    }
    base = 0.60 * flammable_land + 0.40 * water_remoteness
    score = base * recent_fire_suppression
    return score.clip(0.0, 1.0), components


def build_tourism(df: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    components = {
        "tourist_road_access": bounded_access(df["dist_to_tourist_road_m"].to_numpy(), 7000.0),
        "camp_access": bounded_access(df["dist_to_camp_m"].to_numpy(), 12000.0),
        "gate_access": bounded_access(df["dist_to_gate_m"].to_numpy(), 20000.0),
        "waterhole_access": bounded_access(df["dist_to_waterhole_m"].to_numpy(), 10000.0),
    }
    pressure = sum(TOURISM_WEIGHTS[name] * values for name, values in components.items()).clip(0.0, 1.0)
    wildlife_presence = (
        0.45 * df["elephant_density_norm"].to_numpy()
        + 0.25 * df["lion_support_norm"].to_numpy()
        + 0.30 * df["herbivore_support_norm"].to_numpy()
    ).clip(0.0, 1.0)
    interaction = (pressure * wildlife_presence).clip(0.0, 1.0)
    return pressure, interaction, components


def build_threat_layers(features: gpd.GeoDataFrame, species: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    merged = features.merge(
        species.drop(columns="geometry"),
        on="cell_id",
        how="inner",
        suffixes=("", "_species"),
    )

    poaching_score, poaching_components = build_poaching(merged)
    wildfire_score, wildfire_components = build_wildfire(merged)
    tourism_pressure, tourism_interaction, tourism_components = build_tourism(merged)

    merged["poaching_threat_norm"] = poaching_score
    merged["wildfire_threat_norm"] = wildfire_score
    merged["tourism_pressure_norm"] = tourism_pressure
    merged["tourism_threat_norm"] = tourism_interaction

    for name, values in poaching_components.items():
        merged[f"poaching_{name}"] = values
    for name, values in wildfire_components.items():
        merged[f"wildfire_{name}"] = values
    for name, values in tourism_components.items():
        merged[f"tourism_{name}"] = values

    merged["poaching_formula"] = "0.24*gate + 0.24*boundary + 0.18*road + 0.08*tourist_road + 0.08*surveillance_gap + 0.18*rhino"
    merged["wildfire_formula"] = "wildfire=(0.60*flammable_land + 0.40*water_remoteness) * recent_fire_suppression"
    merged["tourism_formula"] = "pressure=(0.35*tourist_road + 0.25*camp + 0.15*gate + 0.25*waterhole); threat=pressure*wildlife_presence"

    return merged


def write_threat_layers(threats: gpd.GeoDataFrame) -> None:
    THREAT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    threats.to_parquet(THREAT_OUTPUT_PATH, index=False)


def check_outputs() -> None:
    threats = validate_parquet(
        THREAT_OUTPUT_PATH,
        [
            "cell_id",
            "poaching_threat_norm",
            "wildfire_threat_norm",
            "tourism_pressure_norm",
            "tourism_threat_norm",
            "poaching_formula",
            "wildfire_formula",
            "tourism_formula",
            "geometry",
        ],
        "threat layers",
    )
    if threats["cell_id"].isna().any():
        raise ValueError("threat layers contain null cell_id values")
    for column in ["poaching_threat_norm", "wildfire_threat_norm", "tourism_pressure_norm", "tourism_threat_norm"]:
        values = threats[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite values")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} is outside [0, 1]")
    print(f"threat layer check passed for {len(threats)} cells")


def main() -> int:
    parser = argparse.ArgumentParser(description="build per-cell poaching, wildfire, and tourism threat layers")
    parser.add_argument("--check", action="store_true", help="validate existing threat-layer outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    features, species = load_inputs()
    threats = build_threat_layers(features, species)
    write_threat_layers(threats)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
