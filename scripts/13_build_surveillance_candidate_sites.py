#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from _optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from _spatial_common import PROCESSED_DIR, OUTPUTS_DIR, WGS84, validate_geojson, validate_parquet, write_geojson
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from scripts._spatial_common import (
        PROCESSED_DIR,
        OUTPUTS_DIR,
        WGS84,
        validate_geojson,
        validate_parquet,
        write_geojson,
    )


CANDIDATE_SITES_PATH = PROCESSED_DIR / "surveillance_candidate_sites.geojson"
BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
WATERHOLES_PATH = PROCESSED_DIR / "waterholes.geojson"
CAMPS_PATH = PROCESSED_DIR / "camps.geojson"
GATES_PATH = PROCESSED_DIR / "gates.geojson"
ROADS_PATH = PROCESSED_DIR / "roads.geojson"
COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
GRID_CENTROIDS_PATH = OUTPUTS_DIR / "grid_centroids.geojson"
GRID_FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
METRIC_CRS = "EPSG:32733"
ADJACENT_BOUNDARY_BUFFER_M = 5000.0
CAR_ACCESS_DISTANCE_THRESHOLD_M = 6000.0
SITE_KIND_PRIORITY = {
    "camp": 1.0,
    "gate": 0.96,
    "waterhole": 0.9,
    "high_risk_cell": 0.6,
}
SITE_FIXED_COST = {
    "camp": 0.0,
    "gate": 0.0,
    "waterhole": 75.0,
    "high_risk_cell": 150.0,
}


@dataclass
class ScenarioSettings:
    scenario_id: str
    top_high_risk_count: int
    merge_distance_m: float
    waterhole_influence_radius_m: float


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unnamed"


def load_inputs() -> dict[str, gpd.GeoDataFrame]:
    return {
        "boundary": validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "boundary"),
        "waterholes": validate_geojson(WATERHOLES_PATH, ["name", "kind", "source", "source_detail"], "waterholes"),
        "camps": validate_geojson(CAMPS_PATH, ["name", "kind", "source", "source_detail"], "camps"),
        "gates": validate_geojson(GATES_PATH, ["name", "kind", "source", "source_detail"], "gates"),
        "roads": validate_geojson(ROADS_PATH, ["fclass", "source"], "roads"),
        "composite": validate_geojson(
            COMPOSITE_PATH,
            ["cell_id", "composite_risk_norm", "poaching_risk_norm", "wildfire_risk_norm", "tourism_risk_norm"],
            "composite risk output",
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
        "features": validate_parquet(
            GRID_FEATURES_PATH,
            [
                "cell_id",
                "dist_to_road_m",
                "dist_to_camp_m",
                "dist_to_waterhole_m",
                "terrain_class",
                "geometry",
            ],
            "grid features",
        ),
    }


def load_scenario_settings(scenario_id: str) -> ScenarioSettings:
    bundle = validate_config_bundle()
    if scenario_id not in bundle.scenarios_by_id:
        raise ValueError(f"scenario_id {scenario_id} not found in {SCENARIO_CONFIG_PATH.name}")
    scenario = bundle.scenarios_by_id[scenario_id]
    return ScenarioSettings(
        scenario_id=scenario_id,
        top_high_risk_count=int(scenario["top_site_count"]),
        merge_distance_m=float(scenario["merge_distance_m"]),
        waterhole_influence_radius_m=float(scenario["waterhole_influence_radius_m"]),
    )


def make_existing_site_records(
    gdf: gpd.GeoDataFrame,
    site_kind: str,
    settings: ScenarioSettings,
) -> gpd.GeoDataFrame:
    records = gdf.copy()
    records["site_kind"] = site_kind
    records["site_name"] = records["name"].fillna(site_kind)
    records["source"] = records["source"].astype(str)
    records["source_detail"] = records["source_detail"].astype(str)
    records["origin_id"] = records["site_kind"] + ":" + records["site_name"].map(slugify)
    records["composite_risk_norm"] = np.nan
    records["site_priority_score"] = SITE_KIND_PRIORITY[site_kind]
    records["base_cost_fixed"] = SITE_FIXED_COST[site_kind]
    records["waterhole_influence_radius_m"] = (
        settings.waterhole_influence_radius_m if site_kind == "waterhole" else 0.0
    )
    return records[
        [
            "origin_id",
            "site_name",
            "site_kind",
            "source",
            "source_detail",
            "composite_risk_norm",
            "site_priority_score",
            "base_cost_fixed",
            "waterhole_influence_radius_m",
            "geometry",
        ]
    ].copy()


def make_high_risk_site_records(inputs: dict[str, gpd.GeoDataFrame], settings: ScenarioSettings) -> gpd.GeoDataFrame:
    centroids = inputs["centroids"]
    composite = inputs["composite"].drop(columns="geometry")
    features = inputs["features"].drop(columns="geometry")
    merged = centroids.merge(composite, on="cell_id", how="inner").merge(features, on="cell_id", how="inner")
    top_risk = merged.nlargest(settings.top_high_risk_count, "composite_risk_norm").copy()
    top_risk["site_name"] = top_risk["cell_id"]
    top_risk["site_kind"] = "high_risk_cell"
    top_risk["source"] = "composite_risk_top_cells"
    top_risk["source_detail"] = "top composite_risk_norm grid centroids"
    top_risk["origin_id"] = top_risk["site_kind"] + ":" + top_risk["cell_id"].astype(str)
    top_risk["site_priority_score"] = SITE_KIND_PRIORITY["high_risk_cell"] + top_risk["composite_risk_norm"]
    top_risk["base_cost_fixed"] = SITE_FIXED_COST["high_risk_cell"]
    top_risk["waterhole_influence_radius_m"] = 0.0
    return top_risk[
        [
            "origin_id",
            "site_name",
            "site_kind",
            "source",
            "source_detail",
            "composite_risk_norm",
            "site_priority_score",
            "base_cost_fixed",
            "waterhole_influence_radius_m",
            "geometry",
        ]
    ].copy()


def deduplicate_sites(candidates: gpd.GeoDataFrame, merge_distance_m: float) -> gpd.GeoDataFrame:
    metric = candidates.to_crs(METRIC_CRS).copy()
    metric = metric.sort_values(
        by=["site_priority_score", "composite_risk_norm", "site_kind", "site_name"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

    kept_rows: list[pd.Series] = []
    kept_geometries = []
    merged_labels: list[list[str]] = []
    for _, row in metric.iterrows():
        geometry = row.geometry
        merge_target = None
        for index, kept_geometry in enumerate(kept_geometries):
            if geometry.distance(kept_geometry) <= merge_distance_m:
                merge_target = index
                break
        if merge_target is None:
            kept_rows.append(row.copy())
            kept_geometries.append(geometry)
            merged_labels.append([row["origin_id"]])
        else:
            merged_labels[merge_target].append(row["origin_id"])

    deduped = gpd.GeoDataFrame(kept_rows, crs=METRIC_CRS).to_crs(WGS84)
    deduped["merged_origin_ids"] = ["|".join(values) for values in merged_labels]
    deduped["merged_site_count"] = [len(values) for values in merged_labels]
    return deduped


def assign_support_flags(candidates: gpd.GeoDataFrame, asset_types: list[dict[str, object]]) -> gpd.GeoDataFrame:
    flagged = candidates.copy()
    flagged["supports_people"] = False
    flagged["supports_cars"] = False
    flagged["supports_drones"] = False
    flagged["supports_cameras"] = False
    car_profile_columns: set[str] = set()

    for asset in asset_types:
        asset_type = str(asset["asset_type"])
        eligible_kinds = {str(kind) for kind in asset["site_eligibility"]}
        support_column = {
            "person": "supports_people",
            "car": "supports_cars",
            "drone": "supports_drones",
            "camera": "supports_cameras",
        }.get(asset_type)
        if support_column is None:
            continue
        flagged[support_column] = flagged["site_kind"].isin(eligible_kinds)
        if str(asset["terrain_modifier_profile"]) == "car":
            car_profile_columns.add(support_column)

    for support_column in sorted(car_profile_columns):
        flagged[support_column] = flagged[support_column] & (
            flagged["dist_to_road_m"] <= CAR_ACCESS_DISTANCE_THRESHOLD_M
        )
    return flagged


def finalize_candidates(
    deduped: gpd.GeoDataFrame,
    inputs: dict[str, gpd.GeoDataFrame],
    settings: ScenarioSettings,
) -> gpd.GeoDataFrame:
    roads_metric = inputs["roads"].to_crs(METRIC_CRS)
    boundary_metric = inputs["boundary"].to_crs(METRIC_CRS)
    features = inputs["features"].drop(columns="geometry")
    composite = inputs["composite"].drop(columns="geometry")

    metric = deduped.to_crs(METRIC_CRS).copy()
    road_union = roads_metric.geometry.union_all()
    boundary_shape = boundary_metric.geometry.iloc[0]
    metric["dist_to_road_m"] = metric.geometry.distance(road_union)
    metric["within_boundary"] = metric.geometry.within(boundary_shape)
    metric["adjacent_to_boundary"] = metric.geometry.within(boundary_shape.buffer(ADJACENT_BOUNDARY_BUFFER_M))

    nearest_features = gpd.sjoin_nearest(
        metric[["origin_id", "geometry"]],
        inputs["features"].to_crs(METRIC_CRS)[["cell_id", "dist_to_camp_m", "dist_to_waterhole_m", "terrain_class", "geometry"]],
        how="left",
        distance_col="nearest_cell_distance_m",
    ).drop(columns=["index_right"]).rename(columns={"cell_id": "nearest_cell_id"})
    metric = metric.merge(
        nearest_features.drop(columns="geometry"),
        on="origin_id",
        how="left",
    )

    nearest_composite = gpd.sjoin_nearest(
        metric[["origin_id", "geometry"]],
        inputs["composite"].to_crs(METRIC_CRS)[["cell_id", "composite_risk_norm", "geometry"]],
        how="left",
        distance_col="nearest_risk_cell_distance_m",
    ).drop(columns=["index_right"])
    metric = metric.drop(columns=["composite_risk_norm"]).merge(
        nearest_composite.drop(columns="geometry"),
        on="origin_id",
        how="left",
    )
    bundle = validate_config_bundle()
    flagged = assign_support_flags(metric.to_crs(WGS84), bundle.asset_types)
    flagged = flagged.sort_values(
        by=["site_priority_score", "composite_risk_norm", "site_kind", "site_name"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    flagged["candidate_rank"] = np.arange(1, len(flagged) + 1)

    site_ids: list[str] = []
    seen_site_ids: set[str] = set()
    for _, row in flagged.iterrows():
        base = f"{row['site_kind']}-{slugify(str(row['site_name']))}"
        candidate = base
        suffix = 2
        while candidate in seen_site_ids:
            candidate = f"{base}-{suffix}"
            suffix += 1
        seen_site_ids.add(candidate)
        site_ids.append(candidate)
    flagged["site_id"] = site_ids
    flagged["scenario_id"] = settings.scenario_id
    flagged["source_config_merge_distance_m"] = settings.merge_distance_m
    return flagged[
        [
            "site_id",
            "scenario_id",
            "site_name",
            "site_kind",
            "source",
            "source_detail",
            "candidate_rank",
            "supports_people",
            "supports_cars",
            "supports_drones",
            "supports_cameras",
            "base_cost_fixed",
            "waterhole_influence_radius_m",
            "composite_risk_norm",
            "dist_to_road_m",
            "dist_to_camp_m",
            "dist_to_waterhole_m",
            "nearest_cell_id",
            "nearest_cell_distance_m",
            "nearest_risk_cell_distance_m",
            "terrain_class",
            "within_boundary",
            "adjacent_to_boundary",
            "merged_site_count",
            "merged_origin_ids",
            "source_config_merge_distance_m",
            "geometry",
        ]
    ].copy()


def build_candidate_sites(scenario_id: str) -> gpd.GeoDataFrame:
    inputs = load_inputs()
    settings = load_scenario_settings(scenario_id)
    candidate_frames = [
        make_existing_site_records(inputs["waterholes"], "waterhole", settings),
        make_existing_site_records(inputs["camps"], "camp", settings),
        make_existing_site_records(inputs["gates"], "gate", settings),
        make_high_risk_site_records(inputs, settings),
    ]
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=WGS84)
    deduped = deduplicate_sites(candidates, settings.merge_distance_m)
    finalized = finalize_candidates(deduped, inputs, settings)
    return finalized


def check_outputs() -> None:
    candidates = validate_geojson(
        CANDIDATE_SITES_PATH,
        [
            "site_id",
            "scenario_id",
            "site_kind",
            "source",
            "candidate_rank",
            "supports_people",
            "supports_cars",
            "supports_drones",
            "supports_cameras",
            "base_cost_fixed",
            "waterhole_influence_radius_m",
        ],
        "surveillance candidate sites",
    )
    if candidates["site_id"].duplicated().any():
        raise ValueError("surveillance candidate sites contain duplicate site_id values")
    support_columns = ["supports_people", "supports_cars", "supports_drones", "supports_cameras"]
    if not candidates[support_columns].any(axis=1).all():
        raise ValueError("every candidate site must allow at least one asset type")
    boundary = validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "boundary").to_crs(METRIC_CRS)
    candidate_metric = candidates.to_crs(METRIC_CRS)
    allowable_zone = boundary.geometry.iloc[0].buffer(ADJACENT_BOUNDARY_BUFFER_M)
    if not candidate_metric.geometry.within(allowable_zone).all():
        raise ValueError("candidate sites must lie within or plausibly adjacent to the park boundary")
    if not candidates["candidate_rank"].is_monotonic_increasing:
        raise ValueError("candidate_rank must be monotonic increasing")
    print(f"candidate-site build check passed for {len(candidates)} sites")


def main() -> int:
    parser = argparse.ArgumentParser(description="build finite surveillance candidate sites for optimization")
    parser.add_argument("--scenario-id", default="etosha_placeholder_baseline", help="optimization scenario id")
    parser.add_argument("--check", action="store_true", help="validate existing candidate-site output")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    candidates = build_candidate_sites(args.scenario_id)
    write_geojson(candidates, CANDIDATE_SITES_PATH)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
