#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from _optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from _spatial_common import PROCESSED_DIR, validate_geojson, validate_parquet, write_geojson
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from scripts._spatial_common import PROCESSED_DIR, validate_geojson, validate_parquet, write_geojson


CANDIDATE_SITES_PATH = PROCESSED_DIR / "surveillance_candidate_sites.geojson"
TERRAIN_COSTS_PATH = PROCESSED_DIR / "terrain_cost_surface.parquet"
WATERHOLE_INTERVENTIONS_PATH = PROCESSED_DIR / "waterhole_interventions.geojson"
MAX_INTERVENTION_SITES = 12
INTERVENTION_MIN_DISTANCE_TO_EXISTING_WATERHOLE_M = 5000.0
INTERVENTION_DEDUP_DISTANCE_M = 7500.0
METRIC_CRS = "EPSG:32733"


def load_scenario(scenario_id: str) -> dict[str, object]:
    bundle = validate_config_bundle()
    if scenario_id not in bundle.scenarios_by_id:
        raise ValueError(f"scenario_id {scenario_id} not found in {SCENARIO_CONFIG_PATH.name}")
    return bundle.scenarios_by_id[scenario_id]


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    sites = validate_geojson(
        CANDIDATE_SITES_PATH,
        [
            "site_id",
            "scenario_id",
            "site_kind",
            "supports_cameras",
            "waterhole_influence_radius_m",
            "nearest_cell_id",
        ],
        "surveillance candidate sites",
    )
    terrain = validate_parquet(
        TERRAIN_COSTS_PATH,
        [
            "cell_id",
            "protection_benefit",
            "human_operability_penalty",
            "elephant_density_norm",
            "rhino_support_norm",
            "wildfire_risk_norm",
            "dist_to_waterhole_m",
            "geometry",
        ],
        "terrain cost surface",
    )
    return sites, terrain


def deduplicate_interventions(candidates: gpd.GeoDataFrame, keep_count: int) -> gpd.GeoDataFrame:
    metric = candidates.to_crs(METRIC_CRS).sort_values(
        ["intervention_priority", "protection_benefit_gain"],
        ascending=[False, False],
    )
    kept: list[pd.Series] = []
    kept_geometries = []
    for _, row in metric.iterrows():
        if len(kept) >= keep_count:
            break
        if any(row.geometry.distance(geom) < INTERVENTION_DEDUP_DISTANCE_M for geom in kept_geometries):
            continue
        kept.append(row.copy())
        kept_geometries.append(row.geometry)
    return gpd.GeoDataFrame(kept, crs=METRIC_CRS).to_crs(candidates.crs)


def build_waterhole_interventions(sites: gpd.GeoDataFrame, terrain: gpd.GeoDataFrame, scenario: dict[str, object]) -> gpd.GeoDataFrame:
    config = scenario["artificial_waterhole_interventions"]
    if not bool(config["enabled"]):
        raise ValueError("artificial_waterhole_interventions.enabled must be true for phase 4")

    waterhole_sites = sites[sites["site_kind"] == "waterhole"][["site_id", "geometry"]].to_crs(METRIC_CRS)
    terrain_metric = terrain.to_crs(METRIC_CRS).copy()
    existing_waterholes = waterhole_sites.geometry.union_all()
    terrain_metric["distance_to_existing_waterhole_m"] = terrain_metric.geometry.distance(existing_waterholes)

    if (terrain_metric["distance_to_existing_waterhole_m"] < 0).any():
        raise ValueError("distance_to_existing_waterhole_m must be non-negative")

    remote_candidates = terrain_metric[
        terrain_metric["distance_to_existing_waterhole_m"] >= INTERVENTION_MIN_DISTANCE_TO_EXISTING_WATERHOLE_M
    ].copy()
    if remote_candidates.empty:
        raise ValueError("no remote candidate cells available for artificial waterhole interventions")

    distance_norm = remote_candidates["distance_to_existing_waterhole_m"] / remote_candidates["distance_to_existing_waterhole_m"].max()
    wildlife_density_proxy = (
        0.55 * remote_candidates["elephant_density_norm"] + 0.45 * remote_candidates["rhino_support_norm"]
    )
    remote_candidates["intervention_priority"] = (
        remote_candidates["protection_benefit"] * (0.65 + 0.35 * distance_norm) * (0.75 + 0.25 * wildlife_density_proxy)
    )
    remote_candidates["wildlife_density_proxy_before"] = wildlife_density_proxy
    remote_candidates["wildlife_density_proxy_after"] = np.clip(
        wildlife_density_proxy * (1.0 - 0.45 * float(config["expected_density_dispersion_benefit"])),
        0.0,
        None,
    )
    remote_candidates["protection_benefit_gain"] = (
        remote_candidates["protection_benefit"]
        * float(config["expected_density_dispersion_benefit"])
        * (0.8 + 0.2 * distance_norm)
    )
    remote_candidates["protection_benefit_after_intervention"] = (
        remote_candidates["protection_benefit"] + remote_candidates["protection_benefit_gain"]
    )

    selected = deduplicate_interventions(remote_candidates, MAX_INTERVENTION_SITES)
    selected = selected.reset_index(drop=True)
    selected["intervention_site_id"] = [f"artificial-waterhole-{index + 1:02d}" for index in range(len(selected))]
    selected["kind"] = "artificial_waterhole"
    selected["capital_cost"] = float(config["capital_cost"])
    selected["tourism_cost"] = float(config["tourism_cost"])
    selected["expected_density_dispersion_benefit"] = float(config["expected_density_dispersion_benefit"])
    selected["influence_radius_m"] = float(scenario["waterhole_influence_radius_m"])
    selected["source"] = "phase4_proxy_from_terrain_cost_surface"
    return selected[
        [
            "intervention_site_id",
            "kind",
            "capital_cost",
            "tourism_cost",
            "expected_density_dispersion_benefit",
            "cell_id",
            "distance_to_existing_waterhole_m",
            "wildlife_density_proxy_before",
            "wildlife_density_proxy_after",
            "protection_benefit",
            "protection_benefit_gain",
            "protection_benefit_after_intervention",
            "intervention_priority",
            "human_operability_penalty",
            "wildfire_risk_norm",
            "influence_radius_m",
            "source",
            "geometry",
        ]
    ].copy()


def check_waterhole_interventions() -> None:
    interventions = validate_geojson(
        WATERHOLE_INTERVENTIONS_PATH,
        [
            "intervention_site_id",
            "kind",
            "capital_cost",
            "tourism_cost",
            "expected_density_dispersion_benefit",
            "geometry",
        ],
        "waterhole interventions",
    )
    if not interventions["intervention_site_id"].is_unique:
        raise ValueError("waterhole interventions must have unique intervention_site_id values")
    for column in [
        "capital_cost",
        "tourism_cost",
        "expected_density_dispersion_benefit",
        "wildlife_density_proxy_before",
        "wildlife_density_proxy_after",
        "protection_benefit",
        "protection_benefit_gain",
        "protection_benefit_after_intervention",
    ]:
        values = interventions[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for every intervention candidate")
    if (interventions["capital_cost"] <= 0).any() or (interventions["tourism_cost"] <= 0).any():
        raise ValueError("every intervention candidate must have explicit positive monetary and tourism-cost terms")
    if (interventions["expected_density_dispersion_benefit"] <= 0).any():
        raise ValueError("every intervention candidate must have an explicit positive density-dispersion benefit term")
    if (interventions["wildlife_density_proxy_after"] < 0).any():
        raise ValueError("intervention effects must not create negative wildlife density proxies")
    if (interventions["protection_benefit_after_intervention"] <= 0).any():
        raise ValueError("intervention effects must preserve positive protection benefit")
    print(f"waterhole intervention check passed for {len(interventions)} candidates")


def main() -> int:
    parser = argparse.ArgumentParser(description="build proactive waterhole interventions and surveillance matrices")
    parser.add_argument("--scenario-id", default="etosha_placeholder_baseline", help="optimization scenario id")
    parser.add_argument("--check", action="store_true", help="validate existing intervention outputs")
    args = parser.parse_args()

    if args.check:
        check_waterhole_interventions()
        return 0

    sites, terrain = load_inputs()
    scenario = load_scenario(args.scenario_id)
    interventions = build_waterhole_interventions(sites, terrain, scenario)
    write_geojson(interventions, WATERHOLE_INTERVENTIONS_PATH)
    check_waterhole_interventions()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
