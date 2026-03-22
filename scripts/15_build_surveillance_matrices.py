#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from _optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet, write_geojson
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._optimization_common import SCENARIO_CONFIG_PATH, validate_config_bundle
    from scripts._spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet, write_geojson


CANDIDATE_SITES_PATH = PROCESSED_DIR / "surveillance_candidate_sites.geojson"
TERRAIN_COSTS_PATH = PROCESSED_DIR / "terrain_cost_surface.parquet"
WATERHOLE_INTERVENTIONS_PATH = PROCESSED_DIR / "waterhole_interventions.geojson"
GRID_CENTROIDS_PATH = OUTPUTS_DIR / "grid_centroids.geojson"
ASSET_COVERAGE_MATRIX_PATH = PROCESSED_DIR / "coverage_matrix.parquet"
RESPONSE_TIME_MATRIX_PATH = PROCESSED_DIR / "response_time_matrix.parquet"
FIRE_DELAY_BREAKPOINTS_PATH = PROCESSED_DIR / "fire_delay_breakpoints.parquet"
MAX_INTERVENTION_SITES = 12
INTERVENTION_MIN_DISTANCE_TO_EXISTING_WATERHOLE_M = 5000.0
INTERVENTION_DEDUP_DISTANCE_M = 7500.0
METRIC_CRS = "EPSG:32733"
FIRE_BREAKPOINT_STEPS_MIN = [0.0, 15.0, 30.0, 60.0, 90.0]
MIN_RESPONSE_TIME_MIN = 0.5
FIRE_PENALTY_PLATEAU_AFTER_THRESHOLD_MIN = 90.0
FIRE_PENALTY_CEILING_DELAY_MIN = 60.0
FIRE_SIGMOID_STEEPNESS = 10.0


def load_bundle_parts(scenario_id: str) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    bundle = validate_config_bundle()
    if scenario_id not in bundle.scenarios_by_id:
        raise ValueError(f"scenario_id {scenario_id} not found in {SCENARIO_CONFIG_PATH.name}")
    return (
        bundle.scenarios_by_id[scenario_id],
        bundle.availability_by_scenario[scenario_id],
        {str(asset["asset_type"]): asset for asset in bundle.asset_types},
    )


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
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
    return sites, terrain, centroids


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


def support_column_for_asset(asset_type: str) -> str:
    support_columns = {
        "person": "supports_people",
        "car": "supports_cars",
        "drone": "supports_drones",
        "camera": "supports_cameras",
    }
    if asset_type not in support_columns:
        raise ValueError(f"unsupported asset_type: {asset_type}")
    return support_columns[asset_type]


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    value_min = float(values.min())
    value_max = float(values.max())
    if value_max <= value_min:
        return np.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def mobile_terrain_factor(
    asset_type: str,
    asset: dict[str, object],
    terrain_by_cell: pd.DataFrame,
    human_penalty_norm: np.ndarray,
    road_distance_norm: np.ndarray,
    *,
    response_mode: bool,
    site_road_norm: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    profile = str(asset["terrain_modifier_profile"])
    base_factor = float(asset["terrain_modifier_parameters"]["base_factor"])
    operability_penalty = float(asset["terrain_modifier_parameters"]["operability_penalty"])

    if profile == "foot":
        factor = np.clip(
            base_factor
            * terrain_by_cell["foot_speed_factor"].to_numpy()
            * (1.0 - operability_penalty * human_penalty_norm),
            0.05 if response_mode else 0.0,
            None if response_mode else 1.5,
        )
        return factor, "foot_proxy"

    if profile == "car":
        factor = (
            base_factor
            * terrain_by_cell["car_speed_factor"].to_numpy()
            * (1.0 - operability_penalty * human_penalty_norm)
            * (1.0 - 0.35 * road_distance_norm)
        )
        if response_mode and site_road_norm is not None:
            factor = factor * (1.0 - 0.20 * site_road_norm[:, None])
        factor = np.clip(factor, 0.05 if response_mode else 0.0, None if response_mode else 1.5)
        label = "car_proxy_euclidean" if asset_type == "car" else "ranger_vehicle_proxy_euclidean"
        return factor, label

    if profile == "drone":
        factor = np.clip(
            base_factor
            * terrain_by_cell["drone_speed_factor"].to_numpy()
            * (
                1.0
                - float(asset["terrain_modifier_parameters"]["roughness_penalty"])
                * terrain_by_cell["terrain_roughness_score"].to_numpy()
            ),
            0.05 if response_mode else 0.0,
            None if response_mode else 1.5,
        )
        return factor, "drone_proxy_euclidean"

    raise ValueError(f"unsupported terrain_modifier_profile for {asset_type}: {profile}")


def bounded_fire_delay_penalty(
    response_time_min: float | np.ndarray,
    tau_fire_min: float,
    beta_fire: float,
    plateau_time_min: float,
) -> float | np.ndarray:
    values = np.asarray(response_time_min, dtype=float)
    plateau_time_min = max(float(plateau_time_min), tau_fire_min + 1e-6)
    plateau_penalty = np.expm1(beta_fire * min(FIRE_PENALTY_CEILING_DELAY_MIN, plateau_time_min - tau_fire_min))

    scaled_delay = np.clip((values - tau_fire_min) / (plateau_time_min - tau_fire_min), 0.0, 1.0)
    sigmoid = 1.0 / (1.0 + np.exp(-FIRE_SIGMOID_STEEPNESS * (scaled_delay - 0.5)))
    sigmoid_floor = 1.0 / (1.0 + np.exp(FIRE_SIGMOID_STEEPNESS / 2.0))
    sigmoid_ceiling = 1.0 / (1.0 + np.exp(-FIRE_SIGMOID_STEEPNESS / 2.0))
    normalized = np.clip((sigmoid - sigmoid_floor) / (sigmoid_ceiling - sigmoid_floor), 0.0, 1.0)
    penalty = np.where(values <= tau_fire_min, 0.0, plateau_penalty * normalized)
    if np.isscalar(response_time_min):
        return float(penalty)
    return penalty


def build_coverage_matrix(
    sites: gpd.GeoDataFrame,
    terrain: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    scenario: dict[str, object],
    asset_types: dict[str, dict[str, object]],
) -> pd.DataFrame:
    site_metric = sites.to_crs(METRIC_CRS).copy()
    cell_metric = centroids.to_crs(METRIC_CRS).copy()
    terrain_by_cell = terrain.drop(columns="geometry").set_index("cell_id").loc[cell_metric["cell_id"]].reset_index()

    site_coords = np.column_stack([site_metric.geometry.x.to_numpy(), site_metric.geometry.y.to_numpy()])
    cell_coords = np.column_stack([cell_metric.geometry.x.to_numpy(), cell_metric.geometry.y.to_numpy()])
    distance_matrix = np.sqrt(
        (site_coords[:, None, 0] - cell_coords[None, :, 0]) ** 2
        + (site_coords[:, None, 1] - cell_coords[None, :, 1]) ** 2
    )

    human_penalty_norm = normalize(terrain_by_cell["human_operability_penalty"].to_numpy())
    road_distance_norm = normalize(terrain_by_cell["dist_to_road_m"].to_numpy())
    waterhole_influence = terrain_by_cell["waterhole_influence_score"].to_numpy()
    camera_gain_factor = float(scenario["protection_benefit"]["camera_gain_factor"])

    frames: list[pd.DataFrame] = []
    for asset_type, asset in asset_types.items():
        support_column = support_column_for_asset(asset_type)
        site_supported = site_metric[support_column].to_numpy(dtype=bool)[:, None]
        coverage_radius = float(asset["coverage_radius_m"])
        attenuation = np.clip(1.0 - distance_matrix / max(coverage_radius, 1.0), 0.0, 1.0)

        if asset_type in {"person", "car", "drone"}:
            terrain_factor, _ = mobile_terrain_factor(
                asset_type,
                asset,
                terrain_by_cell,
                human_penalty_norm,
                road_distance_norm,
                response_mode=False,
            )
            score_matrix = attenuation * terrain_factor[None, :]
            coverable_matrix = site_supported & (distance_matrix <= coverage_radius) & (score_matrix > 0)
            protection_gain = np.zeros_like(score_matrix)
        elif asset_type == "camera":
            score_matrix = np.zeros_like(distance_matrix, dtype=float)
            waterhole_site = site_metric["site_kind"].eq("waterhole").to_numpy(dtype=bool)[:, None]
            influence_radius = np.maximum(site_metric["waterhole_influence_radius_m"].to_numpy(dtype=float)[:, None], 1.0)
            local_influence = np.exp(-distance_matrix / influence_radius) * waterhole_influence[None, :]
            protection_gain = (
                site_supported
                & waterhole_site
                & (distance_matrix <= influence_radius)
            ) * local_influence * float(asset["risk_suppression_factor"]) * camera_gain_factor
            coverable_matrix = np.zeros_like(distance_matrix, dtype=bool)
        else:  # pragma: no cover - protected by config validation
            raise ValueError(f"unexpected asset_type: {asset_type}")

        frame = pd.DataFrame(
            {
                "site_id": np.repeat(site_metric["site_id"].to_numpy(), len(cell_metric)),
                "cell_id": np.tile(cell_metric["cell_id"].to_numpy(), len(site_metric)),
                "asset_type": asset_type,
                "is_coverable": coverable_matrix.reshape(-1),
                "effective_coverage_score": np.where(
                    coverable_matrix.reshape(-1),
                    score_matrix.reshape(-1),
                    0.0,
                ),
                "protection_gain_influence": protection_gain.reshape(-1),
                "camera_lockdown_eligible": (
                    np.repeat(
                        (site_supported[:, 0] & site_metric["site_kind"].eq("waterhole").to_numpy(dtype=bool)),
                        len(cell_metric),
                    )
                    if asset_type == "camera"
                    else np.zeros(len(site_metric) * len(cell_metric), dtype=bool)
                ),
                "distance_m": distance_matrix.reshape(-1),
                "scenario_id": site_metric["scenario_id"].iloc[0],
                "source": "phase5_surveillance_matrix_proxy",
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_response_time_matrix(
    sites: gpd.GeoDataFrame,
    terrain: gpd.GeoDataFrame,
    centroids: gpd.GeoDataFrame,
    availability: dict[str, object],
    asset_types: dict[str, dict[str, object]],
) -> pd.DataFrame:
    site_metric = sites.to_crs(METRIC_CRS).copy()
    cell_metric = centroids.to_crs(METRIC_CRS).copy()
    terrain_by_cell = terrain.drop(columns="geometry").set_index("cell_id").loc[cell_metric["cell_id"]].reset_index()

    site_coords = np.column_stack([site_metric.geometry.x.to_numpy(), site_metric.geometry.y.to_numpy()])
    cell_coords = np.column_stack([cell_metric.geometry.x.to_numpy(), cell_metric.geometry.y.to_numpy()])
    distance_matrix_m = np.sqrt(
        (site_coords[:, None, 0] - cell_coords[None, :, 0]) ** 2
        + (site_coords[:, None, 1] - cell_coords[None, :, 1]) ** 2
    )
    distance_matrix_km = distance_matrix_m / 1000.0

    human_penalty_norm = normalize(terrain_by_cell["human_operability_penalty"].to_numpy())
    terrain_roughness = terrain_by_cell["terrain_roughness_score"].to_numpy()
    road_distance_norm = normalize(terrain_by_cell["dist_to_road_m"].to_numpy())
    site_road_norm = normalize(site_metric["dist_to_road_m"].to_numpy())

    frames: list[pd.DataFrame] = []
    for asset_type in ["person", "car", "drone"]:
        asset = asset_types[asset_type]
        support_column = support_column_for_asset(asset_type)
        site_supported = site_metric[support_column].to_numpy(dtype=bool)[:, None]
        speed_factor, travel_mode = mobile_terrain_factor(
            asset_type,
            asset,
            terrain_by_cell,
            human_penalty_norm,
            road_distance_norm,
            response_mode=True,
            site_road_norm=site_road_norm,
        )

        effective_speed_kmh = float(asset["response_speed_kmh"]) * speed_factor
        response_time_min = np.where(
            site_supported,
            np.maximum((distance_matrix_km / effective_speed_kmh) * 60.0, MIN_RESPONSE_TIME_MIN),
            np.nan,
        )
        frames.append(
            pd.DataFrame(
                {
                    "site_id": np.repeat(site_metric["site_id"].to_numpy(), len(cell_metric)),
                    "cell_id": np.tile(cell_metric["cell_id"].to_numpy(), len(site_metric)),
                    "asset_type": asset_type,
                    "response_time_min": response_time_min.reshape(-1),
                    "response_feasible": np.repeat(site_supported[:, 0], len(cell_metric)),
                    "travel_mode": travel_mode,
                    "tau_fire_min": float(availability["tau_fire_min"]),
                    "beta_fire": float(availability["beta_fire"]),
                    "lambda_fire": float(availability["lambda_fire"]),
                    "source": "phase5_surveillance_matrix_proxy",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_fire_delay_breakpoints(response_time_matrix: pd.DataFrame, availability: dict[str, object], scenario_id: str) -> pd.DataFrame:
    feasible_times = response_time_matrix.loc[response_time_matrix["response_feasible"], "response_time_min"].dropna()
    if feasible_times.empty:
        raise ValueError("cannot build fire-delay breakpoints without feasible response times")
    tau_fire_min = float(availability["tau_fire_min"])
    beta_fire = float(availability["beta_fire"])
    lambda_fire = float(availability["lambda_fire"])
    max_time = max(
        min(float(feasible_times.quantile(0.99)), tau_fire_min + 180.0),
        tau_fire_min + FIRE_BREAKPOINT_STEPS_MIN[-1],
    )
    plateau_time = min(max_time, tau_fire_min + FIRE_PENALTY_PLATEAU_AFTER_THRESHOLD_MIN)
    raw_breakpoints = [0.0, tau_fire_min] + [tau_fire_min + step for step in FIRE_BREAKPOINT_STEPS_MIN[1:]] + [max_time]
    breakpoints: list[float] = []
    for point in raw_breakpoints:
        if not breakpoints or point > breakpoints[-1]:
            breakpoints.append(point)
    penalties = [bounded_fire_delay_penalty(point, tau_fire_min, beta_fire, plateau_time) for point in breakpoints]
    return pd.DataFrame(
        {
            "scenario_id": scenario_id,
            "breakpoint_index": np.arange(len(breakpoints)),
            "response_time_min": breakpoints,
            "penalty_value": penalties,
            "tau_fire_min": tau_fire_min,
            "beta_fire": beta_fire,
            "lambda_fire": lambda_fire,
            "source": "phase5_fire_delay_piecewise_proxy",
        }
    )


def write_parquet(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


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


def check_coverage_matrix(sites: gpd.GeoDataFrame, centroids: gpd.GeoDataFrame) -> None:
    coverage = pd.read_parquet(ASSET_COVERAGE_MATRIX_PATH)
    required_columns = [
        "site_id",
        "cell_id",
        "asset_type",
        "is_coverable",
        "effective_coverage_score",
        "protection_gain_influence",
        "source",
    ]
    missing = [column for column in required_columns if column not in coverage.columns]
    if missing:
        raise ValueError(f"coverage matrix is missing required columns: {', '.join(missing)}")
    expected_rows = len(sites) * len(centroids) * 4
    if len(coverage) != expected_rows:
        raise ValueError(f"coverage matrix row count mismatch: expected {expected_rows}, found {len(coverage)}")
    if not np.isfinite(coverage["effective_coverage_score"].to_numpy(dtype=float)).all():
        raise ValueError("effective_coverage_score must be finite for every row")
    if not np.isfinite(coverage["protection_gain_influence"].to_numpy(dtype=float)).all():
        raise ValueError("protection_gain_influence must be finite for every row")
    camera_rows = coverage[coverage["asset_type"] == "camera"].copy()
    non_camera_rows = coverage[coverage["asset_type"] != "camera"]
    if (non_camera_rows["protection_gain_influence"] != 0).any():
        raise ValueError("camera protection gain influence must be absent for non-camera assets")
    waterhole_sites = set(sites.loc[sites["site_kind"] == "waterhole", "site_id"])
    positive_camera_sites = set(camera_rows.loc[camera_rows["protection_gain_influence"] > 0, "site_id"])
    if not positive_camera_sites.issubset(waterhole_sites):
        raise ValueError("camera protection gain must only appear at eligible waterhole sites")
    if not positive_camera_sites:
        raise ValueError("camera protection gain influence must be present for at least one waterhole site")
    print(f"coverage matrix check passed for {len(coverage)} rows")


def check_response_time_matrix(sites: gpd.GeoDataFrame, centroids: gpd.GeoDataFrame) -> None:
    response = pd.read_parquet(RESPONSE_TIME_MATRIX_PATH)
    required_columns = [
        "site_id",
        "cell_id",
        "asset_type",
        "response_time_min",
        "response_feasible",
        "travel_mode",
        "tau_fire_min",
        "beta_fire",
        "lambda_fire",
        "source",
    ]
    missing = [column for column in required_columns if column not in response.columns]
    if missing:
        raise ValueError(f"response time matrix is missing required columns: {', '.join(missing)}")
    expected_rows = len(sites) * len(centroids) * 3
    if len(response) != expected_rows:
        raise ValueError(f"response matrix row count mismatch: expected {expected_rows}, found {len(response)}")
    if set(response["asset_type"].unique()) != {"person", "car", "drone"}:
        raise ValueError("response time matrix must exclude cameras cleanly")
    feasible = response[response["response_feasible"]].copy()
    if feasible.empty:
        raise ValueError("response time matrix must include feasible mobile asset/site/cell combinations")
    if not np.isfinite(feasible["response_time_min"].to_numpy(dtype=float)).all():
        raise ValueError("response_time_min must be finite where response is feasible")
    if (feasible["response_time_min"] <= 0).any():
        raise ValueError("response_time_min must be positive where response is feasible")
    print(f"response matrix check passed for {len(response)} rows")


def check_fire_delay_breakpoints() -> None:
    breakpoints = pd.read_parquet(FIRE_DELAY_BREAKPOINTS_PATH)
    required_columns = [
        "scenario_id",
        "breakpoint_index",
        "response_time_min",
        "penalty_value",
        "tau_fire_min",
        "beta_fire",
        "lambda_fire",
        "source",
    ]
    missing = [column for column in required_columns if column not in breakpoints.columns]
    if missing:
        raise ValueError(f"fire delay breakpoints are missing required columns: {', '.join(missing)}")
    if not breakpoints["response_time_min"].is_monotonic_increasing:
        raise ValueError("fire delay breakpoints must be monotonic increasing")
    if not breakpoints["penalty_value"].is_monotonic_increasing:
        raise ValueError("fire delay penalties must be monotonic increasing")
    tau_fire_min = float(breakpoints["tau_fire_min"].iloc[0])
    below_threshold = breakpoints[breakpoints["response_time_min"] <= tau_fire_min]
    if (below_threshold["penalty_value"] > 1e-9).any():
        raise ValueError("fire delay penalties must stay zero at or below tau_fire_min")
    above_threshold = breakpoints[breakpoints["response_time_min"] > tau_fire_min]
    if above_threshold.empty or not (above_threshold["penalty_value"] > 0).any():
        raise ValueError("fire delay penalties must increase above tau_fire_min")
    print(f"fire delay breakpoint check passed for {len(breakpoints)} breakpoints")


def main() -> int:
    parser = argparse.ArgumentParser(description="build proactive waterhole interventions and surveillance matrices")
    parser.add_argument("--scenario-id", default="etosha_placeholder_baseline", help="optimization scenario id")
    parser.add_argument("--check", action="store_true", help="validate existing intervention outputs")
    args = parser.parse_args()

    if args.check:
        check_waterhole_interventions()
        sites, _, centroids = load_inputs()
        check_coverage_matrix(sites, centroids)
        check_response_time_matrix(sites, centroids)
        check_fire_delay_breakpoints()
        return 0

    sites, terrain, centroids = load_inputs()
    scenario, availability, asset_types = load_bundle_parts(args.scenario_id)
    interventions = build_waterhole_interventions(sites, terrain, scenario)
    write_geojson(interventions, WATERHOLE_INTERVENTIONS_PATH)
    coverage_matrix = build_coverage_matrix(sites, terrain, centroids, scenario, asset_types)
    response_matrix = build_response_time_matrix(sites, terrain, centroids, availability, asset_types)
    fire_delay_breakpoints = build_fire_delay_breakpoints(response_matrix, availability, args.scenario_id)
    write_parquet(coverage_matrix, ASSET_COVERAGE_MATRIX_PATH)
    write_parquet(response_matrix, RESPONSE_TIME_MATRIX_PATH)
    write_parquet(fire_delay_breakpoints, FIRE_DELAY_BREAKPOINTS_PATH)
    check_waterhole_interventions()
    check_coverage_matrix(sites, centroids)
    check_response_time_matrix(sites, centroids)
    check_fire_delay_breakpoints()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
