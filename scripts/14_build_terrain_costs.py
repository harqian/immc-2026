#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from _optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from _spatial_common import PROCESSED_DIR, OUTPUTS_DIR, validate_geojson, validate_parquet
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from scripts._spatial_common import PROCESSED_DIR, OUTPUTS_DIR, validate_geojson, validate_parquet


GRID_FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
COMPOSITE_RISK_PATH = OUTPUTS_DIR / "composite_risk.geojson"
TERRAIN_COSTS_PATH = PROCESSED_DIR / "terrain_cost_surface.parquet"
TERRAIN_BASE_ROUGHNESS = {
    "pan": 0.10,
    "pan_margin": 0.35,
    "boundary_edge": 0.60,
    "interior_savanna": 0.45,
}
HABITAT_CLASS_MAP = {
    "pan": "open_pan",
    "pan_margin": "grass_shrub_transition",
    "boundary_edge": "edge_scrub",
    "interior_savanna": "savanna_woodland",
}


def normalize_series(series: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max <= value_min:
        return np.zeros(len(values), dtype=float)
    return (values - value_min) / (value_max - value_min)


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    features = validate_parquet(
        GRID_FEATURES_PATH,
        [
            "cell_id",
            "dist_to_camp_m",
            "dist_to_road_m",
            "dist_to_waterhole_m",
            "pan_overlap_ratio",
            "historical_fire_event_count",
            "terrain_class",
            "geometry",
        ],
        "grid features",
    )
    composite = validate_geojson(
        COMPOSITE_RISK_PATH,
        [
            "cell_id",
            "elephant_density_norm",
            "rhino_support_norm",
            "lion_support_norm",
            "herbivore_support_norm",
            "poaching_risk_norm",
            "wildfire_risk_norm",
            "tourism_risk_norm",
            "composite_risk_norm",
        ],
        "composite risk output",
    )
    return features, composite


def load_scenario(scenario_id: str) -> dict[str, object]:
    bundle = validate_config_bundle()
    if scenario_id not in bundle.scenarios_by_id:
        raise ValueError(f"scenario_id {scenario_id} not found in {SCENARIO_CONFIG_PATH.name}")
    return bundle.scenarios_by_id[scenario_id]


def derive_terrain_costs(features: gpd.GeoDataFrame, composite: gpd.GeoDataFrame, scenario: dict[str, object]) -> gpd.GeoDataFrame:
    merged = features.merge(composite.drop(columns="geometry"), on="cell_id", how="inner")
    if len(merged) != len(features):
        raise ValueError("terrain cost surface requires exactly one composite risk record per grid cell")

    merged["habitat_class"] = merged["terrain_class"].map(HABITAT_CLASS_MAP)
    if merged["habitat_class"].isna().any():
        raise ValueError("terrain_class contains values without habitat_class mappings")

    distance_road_norm = normalize_series(merged["dist_to_road_m"])
    distance_camp_norm = normalize_series(merged["dist_to_camp_m"])
    distance_waterhole_norm = normalize_series(merged["dist_to_waterhole_m"])
    fire_history_norm = normalize_series(merged["historical_fire_event_count"])
    terrain_base = merged["terrain_class"].map(TERRAIN_BASE_ROUGHNESS).to_numpy(dtype=float)
    openness = np.where(
        merged["terrain_class"].eq("pan"),
        1.0,
        np.where(
            merged["terrain_class"].eq("pan_margin"),
            0.82,
            np.where(merged["terrain_class"].eq("interior_savanna"), 0.67, 0.55),
        ),
    )

    merged["waterhole_influence_score"] = np.exp(
        -merged["dist_to_waterhole_m"].to_numpy(dtype=float)
        / float(scenario["waterhole_influence_radius_m"])
    )
    merged["terrain_roughness_score"] = np.clip(
        terrain_base + 0.20 * distance_road_norm + 0.10 * (1.0 - openness) + 0.10 * fire_history_norm,
        0.0,
        1.0,
    )
    merged["slope_mean_deg"] = 1.5 + 11.5 * merged["terrain_roughness_score"]

    protection = scenario["protection_benefit"]
    wildlife_weights = protection["wildlife_weight_columns"]
    threat_weights = protection["threat_weight_columns"]
    wildlife_total = sum(float(weight) for weight in wildlife_weights.values())
    threat_total = sum(float(weight) for weight in threat_weights.values())
    wildlife_value = sum(merged[column] * float(weight) for column, weight in wildlife_weights.items()) / wildlife_total
    threat_value = sum(merged[column] * float(weight) for column, weight in threat_weights.items()) / threat_total
    intervention_leverage = 0.65 * merged["waterhole_influence_score"] + 0.35 * normalize_series(merged["rhino_support_norm"])
    merged["protection_benefit"] = np.clip(
        float(protection["minimum_positive_floor"])
        + float(protection["leverage_scale"]) * wildlife_value * (0.5 + threat_value) * (1.0 + 0.4 * intervention_leverage),
        float(protection["minimum_positive_floor"]),
        None,
    )

    operability = scenario["human_operability_penalty"]
    abundance_component = normalize_series(merged[operability["abundance_column"]])
    distance_component = normalize_series(merged[operability["camp_distance_column"]])
    configured_roughness = normalize_series(merged[operability["terrain_roughness_column"]])
    roughness_component = np.clip(0.65 * merged["terrain_roughness_score"] + 0.35 * configured_roughness, 0.0, 1.0)
    weighted_penalty = (
        float(operability["abundance_weight"]) * abundance_component
        + float(operability["camp_distance_weight"]) * distance_component
        + float(operability["terrain_roughness_weight"]) * roughness_component
    )
    weight_total = (
        float(operability["abundance_weight"])
        + float(operability["camp_distance_weight"])
        + float(operability["terrain_roughness_weight"])
    )
    penalty_unit = weighted_penalty / weight_total
    merged["human_operability_penalty"] = np.clip(
        float(operability["penalty_floor"])
        + penalty_unit * (float(operability["penalty_ceiling"]) - float(operability["penalty_floor"])),
        float(operability["penalty_floor"]),
        float(operability["penalty_ceiling"]),
    )

    merged["foot_speed_factor"] = np.clip(
        1.05 - 0.55 * merged["terrain_roughness_score"] - 0.30 * penalty_unit,
        0.18,
        1.05,
    )
    merged["car_speed_factor"] = np.clip(
        1.10 - 0.65 * merged["terrain_roughness_score"] - 0.15 * penalty_unit - 0.20 * distance_road_norm,
        0.15,
        1.10,
    )
    merged["drone_speed_factor"] = np.clip(
        1.12 - 0.10 * merged["terrain_roughness_score"] - 0.05 * fire_history_norm,
        0.70,
        1.12,
    )
    merged["camera_visibility_factor"] = np.clip(
        0.55 + 0.55 * openness + 0.20 * merged["waterhole_influence_score"] - 0.10 * merged["terrain_roughness_score"],
        0.20,
        1.25,
    )
    merged["source"] = "phase3_proxy_from_grid_features_and_composite_risk"
    return gpd.GeoDataFrame(
        merged[
            [
                "cell_id",
                "terrain_class",
                "habitat_class",
                "slope_mean_deg",
                "foot_speed_factor",
                "car_speed_factor",
                "drone_speed_factor",
                "camera_visibility_factor",
                "waterhole_influence_score",
                "terrain_roughness_score",
                "protection_benefit",
                "human_operability_penalty",
                "elephant_density_norm",
                "rhino_support_norm",
                "lion_support_norm",
                "herbivore_support_norm",
                "wildfire_risk_norm",
                "dist_to_camp_m",
                "dist_to_road_m",
                "dist_to_waterhole_m",
                "historical_fire_event_count",
                "source",
                "geometry",
            ]
        ].copy(),
        geometry="geometry",
        crs=features.crs,
    )


def write_terrain_costs(terrain_costs: gpd.GeoDataFrame) -> None:
    TERRAIN_COSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    terrain_costs.to_parquet(TERRAIN_COSTS_PATH, index=False)


def check_outputs() -> None:
    terrain_costs = validate_parquet(
        TERRAIN_COSTS_PATH,
        [
            "cell_id",
            "terrain_class",
            "habitat_class",
            "slope_mean_deg",
            "foot_speed_factor",
            "car_speed_factor",
            "drone_speed_factor",
            "camera_visibility_factor",
            "protection_benefit",
            "human_operability_penalty",
            "source",
            "geometry",
        ],
        "terrain cost surface",
    )
    if terrain_costs["cell_id"].duplicated().any():
        raise ValueError("terrain cost surface must contain exactly one record per cell_id")
    numeric_columns = [
        "slope_mean_deg",
        "foot_speed_factor",
        "car_speed_factor",
        "drone_speed_factor",
        "camera_visibility_factor",
        "protection_benefit",
        "human_operability_penalty",
        "waterhole_influence_score",
        "terrain_roughness_score",
        "wildfire_risk_norm",
        "dist_to_camp_m",
        "dist_to_road_m",
        "dist_to_waterhole_m",
    ]
    for column in numeric_columns:
        values = terrain_costs[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for every grid cell")
    if (terrain_costs["protection_benefit"] <= 0).any():
        raise ValueError("protection_benefit must stay strictly positive")
    print(f"terrain-cost build check passed for {len(terrain_costs)} cells")


def main() -> int:
    parser = argparse.ArgumentParser(description="build terrain and mobility proxy factors for surveillance optimization")
    parser.add_argument("--scenario-id", default="etosha_placeholder_baseline", help="optimization scenario id")
    parser.add_argument("--check", action="store_true", help="validate existing terrain cost output")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    features, composite = load_inputs()
    scenario = load_scenario(args.scenario_id)
    terrain_costs = derive_terrain_costs(features, composite, scenario)
    write_terrain_costs(terrain_costs)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
