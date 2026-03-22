from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from _spatial_common import OUTPUTS_DIR, PROJECT_ROOT, ensure_columns, validate_geojson, validate_parquet
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._spatial_common import (
        OUTPUTS_DIR,
        PROJECT_ROOT,
        ensure_columns,
        validate_geojson,
        validate_parquet,
    )


CONFIG_DIR = PROJECT_ROOT / "data/configs"
ASSET_CONFIG_PATH = CONFIG_DIR / "asset_types.yaml"
AVAILABILITY_CONFIG_PATH = CONFIG_DIR / "daily_asset_availability.yaml"
SCENARIO_CONFIG_PATH = CONFIG_DIR / "optimization_scenarios.yaml"
GRID_FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
COMPOSITE_RISK_PATH = OUTPUTS_DIR / "composite_risk.geojson"

REQUIRED_ASSET_KEYS = [
    "asset_type",
    "unit_cost",
    "coverage_radius_m",
    "response_speed_kmh",
    "site_eligibility",
    "terrain_modifier_profile",
    "max_units_per_site",
    "counts_toward_budget",
    "camera_bundle_size",
    "risk_suppression_factor",
]
REQUIRED_AVAILABILITY_KEYS = [
    "scenario_id",
    "budget_total",
    "max_people",
    "included_people",
    "max_cars",
    "included_cars",
    "max_drones",
    "included_drones",
    "max_cameras",
    "included_cameras",
    "tau_fire_min",
    "beta_fire",
    "lambda_fire",
]
REQUIRED_SCENARIO_KEYS = [
    "scenario_id",
    "description",
    "active_asset_types",
    "alpha_values",
    "top_site_count",
    "merge_distance_m",
    "waterhole_influence_radius_m",
    "protection_benefit",
    "human_operability_penalty",
    "artificial_waterhole_interventions",
]
REQUIRED_PROTECTION_KEYS = [
    "wildlife_weight_columns",
    "threat_weight_columns",
    "leverage_scale",
    "minimum_positive_floor",
    "camera_gain_factor",
]
REQUIRED_OPERABILITY_KEYS = [
    "abundance_column",
    "abundance_weight",
    "camp_distance_column",
    "camp_distance_weight",
    "terrain_roughness_column",
    "terrain_roughness_weight",
    "penalty_floor",
    "penalty_ceiling",
]
REQUIRED_INTERVENTION_KEYS = [
    "enabled",
    "capital_cost",
    "tourism_cost",
    "expected_density_dispersion_benefit",
]
MOBILE_ASSET_CAP_KEYS = {
    "person": "max_people",
    "car": "max_cars",
    "drone": "max_drones",
    "camera": "max_cameras",
}
REQUIRED_GRID_FEATURE_COLUMNS = [
    "cell_id",
    "dist_to_camp_m",
    "historical_fire_event_count",
    "geometry",
]
REQUIRED_COMPOSITE_COLUMNS = [
    "cell_id",
    "elephant_density_norm",
    "rhino_support_norm",
    "lion_support_norm",
    "herbivore_support_norm",
    "poaching_risk_norm",
    "wildfire_risk_norm",
    "tourism_risk_norm",
    "composite_risk_norm",
    "geometry",
]


@dataclass
class OptimizationConfigBundle:
    asset_types: list[dict[str, Any]]
    availability_by_scenario: dict[str, dict[str, Any]]
    scenarios_by_id: dict[str, dict[str, Any]]


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a top-level mapping")
    return loaded


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    return value


def require_keys(mapping: dict[str, Any], required_keys: list[str], label: str) -> None:
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required keys: {joined}")


def require_numeric(value: Any, label: str, *, allow_zero: bool = True) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if allow_zero:
        if numeric < 0:
            raise ValueError(f"{label} must be non-negative")
    elif numeric <= 0:
        raise ValueError(f"{label} must be positive")
    return numeric


def require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def validate_asset_types(config: dict[str, Any]) -> list[dict[str, Any]]:
    asset_types = require_list(config.get("asset_types"), "asset_types.yaml asset_types")
    seen_asset_types: set[str] = set()
    for asset in asset_types:
        mapping = require_mapping(asset, "asset_types entry")
        require_keys(mapping, REQUIRED_ASSET_KEYS, f"asset type {mapping.get('asset_type', '<unknown>')}")
        asset_type = mapping["asset_type"]
        if not isinstance(asset_type, str) or not asset_type:
            raise ValueError("asset_type must be a non-empty string")
        if asset_type in seen_asset_types:
            raise ValueError(f"duplicate asset_type in asset_types.yaml: {asset_type}")
        seen_asset_types.add(asset_type)

        require_numeric(mapping["unit_cost"], f"{asset_type}.unit_cost")
        require_numeric(mapping["coverage_radius_m"], f"{asset_type}.coverage_radius_m")
        require_numeric(mapping["response_speed_kmh"], f"{asset_type}.response_speed_kmh")
        require_list(mapping["site_eligibility"], f"{asset_type}.site_eligibility")
        if not all(isinstance(site_kind, str) and site_kind for site_kind in mapping["site_eligibility"]):
            raise ValueError(f"{asset_type}.site_eligibility must contain non-empty strings")
        if not isinstance(mapping["terrain_modifier_profile"], str) or not mapping["terrain_modifier_profile"]:
            raise ValueError(f"{asset_type}.terrain_modifier_profile must be a non-empty string")
        terrain_parameters = require_mapping(
            mapping.get("terrain_modifier_parameters"),
            f"{asset_type}.terrain_modifier_parameters",
        )
        if not terrain_parameters:
            raise ValueError(f"{asset_type}.terrain_modifier_parameters must not be empty")
        for parameter_name, parameter_value in terrain_parameters.items():
            require_numeric(parameter_value, f"{asset_type}.terrain_modifier_parameters.{parameter_name}")
        require_numeric(mapping["max_units_per_site"], f"{asset_type}.max_units_per_site")
        require_bool(mapping["counts_toward_budget"], f"{asset_type}.counts_toward_budget")
        bundle_size = require_numeric(mapping["camera_bundle_size"], f"{asset_type}.camera_bundle_size")
        suppression = require_numeric(mapping["risk_suppression_factor"], f"{asset_type}.risk_suppression_factor")
        if asset_type == "camera" and bundle_size <= 0:
            raise ValueError("camera.camera_bundle_size must be positive")
        if asset_type != "camera" and suppression != 0:
            raise ValueError(f"{asset_type}.risk_suppression_factor must be 0 for non-camera assets")
    return asset_types


def validate_daily_availability(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = require_list(config.get("scenarios"), "daily_asset_availability.yaml scenarios")
    availability_by_scenario: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        mapping = require_mapping(scenario, "daily asset availability entry")
        require_keys(mapping, REQUIRED_AVAILABILITY_KEYS, f"availability scenario {mapping.get('scenario_id', '<unknown>')}")
        scenario_id = mapping["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("daily availability scenario_id must be a non-empty string")
        if scenario_id in availability_by_scenario:
            raise ValueError(f"duplicate daily availability scenario_id: {scenario_id}")
        require_numeric(mapping["budget_total"], f"{scenario_id}.budget_total")
        for cap_key in [
            "max_people",
            "included_people",
            "max_cars",
            "included_cars",
            "max_drones",
            "included_drones",
            "max_cameras",
            "included_cameras",
        ]:
            require_numeric(mapping[cap_key], f"{scenario_id}.{cap_key}")
        if float(mapping["included_people"]) > float(mapping["max_people"]):
            raise ValueError(f"{scenario_id}.included_people must be <= max_people")
        if float(mapping["included_cars"]) > float(mapping["max_cars"]):
            raise ValueError(f"{scenario_id}.included_cars must be <= max_cars")
        if float(mapping["included_drones"]) > float(mapping["max_drones"]):
            raise ValueError(f"{scenario_id}.included_drones must be <= max_drones")
        if float(mapping["included_cameras"]) > float(mapping["max_cameras"]):
            raise ValueError(f"{scenario_id}.included_cameras must be <= max_cameras")
        require_numeric(mapping["tau_fire_min"], f"{scenario_id}.tau_fire_min")
        require_numeric(mapping["beta_fire"], f"{scenario_id}.beta_fire")
        require_numeric(mapping["lambda_fire"], f"{scenario_id}.lambda_fire")
        availability_by_scenario[scenario_id] = mapping
    return availability_by_scenario


def validate_optimization_scenarios(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = require_list(config.get("scenarios"), "optimization_scenarios.yaml scenarios")
    scenarios_by_id: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        mapping = require_mapping(scenario, "optimization scenario entry")
        require_keys(mapping, REQUIRED_SCENARIO_KEYS, f"optimization scenario {mapping.get('scenario_id', '<unknown>')}")
        scenario_id = mapping["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("optimization scenario_id must be a non-empty string")
        if scenario_id in scenarios_by_id:
            raise ValueError(f"duplicate optimization scenario_id: {scenario_id}")
        if not isinstance(mapping["description"], str) or not mapping["description"]:
            raise ValueError(f"{scenario_id}.description must be a non-empty string")
        require_list(mapping["active_asset_types"], f"{scenario_id}.active_asset_types")
        require_list(mapping["alpha_values"], f"{scenario_id}.alpha_values")
        for alpha in mapping["alpha_values"]:
            alpha_value = require_numeric(alpha, f"{scenario_id}.alpha_values[]")
            if alpha_value <= 0 or alpha_value > 1:
                raise ValueError(f"{scenario_id}.alpha_values entries must stay within (0, 1]")
        require_numeric(mapping["top_site_count"], f"{scenario_id}.top_site_count", allow_zero=False)
        require_numeric(mapping["merge_distance_m"], f"{scenario_id}.merge_distance_m")
        require_numeric(mapping["waterhole_influence_radius_m"], f"{scenario_id}.waterhole_influence_radius_m")

        protection = require_mapping(mapping["protection_benefit"], f"{scenario_id}.protection_benefit")
        require_keys(protection, REQUIRED_PROTECTION_KEYS, f"{scenario_id}.protection_benefit")
        wildlife_weights = require_mapping(
            protection["wildlife_weight_columns"], f"{scenario_id}.protection_benefit.wildlife_weight_columns"
        )
        threat_weights = require_mapping(
            protection["threat_weight_columns"], f"{scenario_id}.protection_benefit.threat_weight_columns"
        )
        if not wildlife_weights or not threat_weights:
            raise ValueError(f"{scenario_id}.protection_benefit weight mappings must not be empty")
        for key, value in {**wildlife_weights, **threat_weights}.items():
            require_numeric(value, f"{scenario_id}.protection_benefit weight {key}")
        require_numeric(protection["leverage_scale"], f"{scenario_id}.protection_benefit.leverage_scale")
        require_numeric(
            protection["minimum_positive_floor"],
            f"{scenario_id}.protection_benefit.minimum_positive_floor",
        )
        require_numeric(protection["camera_gain_factor"], f"{scenario_id}.protection_benefit.camera_gain_factor")

        operability = require_mapping(mapping["human_operability_penalty"], f"{scenario_id}.human_operability_penalty")
        require_keys(operability, REQUIRED_OPERABILITY_KEYS, f"{scenario_id}.human_operability_penalty")
        for key in ["abundance_column", "camp_distance_column", "terrain_roughness_column"]:
            if not isinstance(operability[key], str) or not operability[key]:
                raise ValueError(f"{scenario_id}.human_operability_penalty.{key} must be a non-empty string")
        for key in [
            "abundance_weight",
            "camp_distance_weight",
            "terrain_roughness_weight",
            "penalty_floor",
            "penalty_ceiling",
        ]:
            require_numeric(operability[key], f"{scenario_id}.human_operability_penalty.{key}")
        if float(operability["penalty_ceiling"]) < float(operability["penalty_floor"]):
            raise ValueError(f"{scenario_id}.human_operability_penalty.penalty_ceiling must be >= penalty_floor")

        interventions = require_mapping(
            mapping["artificial_waterhole_interventions"],
            f"{scenario_id}.artificial_waterhole_interventions",
        )
        require_keys(interventions, REQUIRED_INTERVENTION_KEYS, f"{scenario_id}.artificial_waterhole_interventions")
        require_bool(interventions["enabled"], f"{scenario_id}.artificial_waterhole_interventions.enabled")
        require_numeric(
            interventions["capital_cost"], f"{scenario_id}.artificial_waterhole_interventions.capital_cost"
        )
        require_numeric(
            interventions["tourism_cost"], f"{scenario_id}.artificial_waterhole_interventions.tourism_cost"
        )
        require_numeric(
            interventions["expected_density_dispersion_benefit"],
            f"{scenario_id}.artificial_waterhole_interventions.expected_density_dispersion_benefit",
        )

        scenarios_by_id[scenario_id] = mapping
    return scenarios_by_id


def validate_config_bundle() -> OptimizationConfigBundle:
    asset_types = validate_asset_types(load_yaml(ASSET_CONFIG_PATH))
    availability_by_scenario = validate_daily_availability(load_yaml(AVAILABILITY_CONFIG_PATH))
    scenarios_by_id = validate_optimization_scenarios(load_yaml(SCENARIO_CONFIG_PATH))

    asset_type_names = {asset["asset_type"] for asset in asset_types}
    for scenario_id, scenario in scenarios_by_id.items():
        if scenario_id not in availability_by_scenario:
            raise ValueError(f"optimization scenario {scenario_id} is missing matching daily availability")
        unknown_asset_types = sorted(set(scenario["active_asset_types"]) - asset_type_names)
        if unknown_asset_types:
            joined = ", ".join(unknown_asset_types)
            raise ValueError(f"scenario {scenario_id} references unknown asset types: {joined}")
        availability = availability_by_scenario[scenario_id]
        for asset_type in scenario["active_asset_types"]:
            cap_key = MOBILE_ASSET_CAP_KEYS.get(asset_type)
            if cap_key is None:
                raise ValueError(f"scenario {scenario_id} uses unsupported asset type: {asset_type}")
            require_numeric(availability[cap_key], f"{scenario_id}.{cap_key}")
        interventions = scenario["artificial_waterhole_interventions"]
        if "capital_cost" not in interventions or "tourism_cost" not in interventions:
            raise ValueError(
                f"scenario {scenario_id} must define both capital_cost and tourism_cost for artificial waterholes"
            )

    return OptimizationConfigBundle(
        asset_types=asset_types,
        availability_by_scenario=availability_by_scenario,
        scenarios_by_id=scenarios_by_id,
    )


def validate_phase1_input_contract(bundle: OptimizationConfigBundle, scenario_id: str) -> None:
    if scenario_id not in bundle.scenarios_by_id:
        raise ValueError(f"scenario_id {scenario_id} not found in optimization_scenarios.yaml")
    scenario = bundle.scenarios_by_id[scenario_id]
    composite = validate_geojson(COMPOSITE_RISK_PATH, REQUIRED_COMPOSITE_COLUMNS, "composite risk output")
    features = validate_parquet(GRID_FEATURES_PATH, REQUIRED_GRID_FEATURE_COLUMNS, "grid features output")
    if len(composite) != len(features):
        raise ValueError("grid features and composite risk outputs have different row counts")

    protection = scenario["protection_benefit"]
    ensure_columns(
        composite,
        list(protection["wildlife_weight_columns"].keys()) + list(protection["threat_weight_columns"].keys()),
        "composite risk output for protection_benefit",
    )
    operability = scenario["human_operability_penalty"]
    ensure_columns(
        features,
        [
            operability["camp_distance_column"],
            operability["terrain_roughness_column"],
        ],
        "grid features output for human_operability_penalty",
    )
    ensure_columns(
        composite,
        [operability["abundance_column"]],
        "composite risk output for human_operability_penalty abundance",
    )


def summarize_bundle(bundle: OptimizationConfigBundle) -> str:
    scenario_ids = ", ".join(sorted(bundle.scenarios_by_id))
    asset_types = ", ".join(sorted(asset["asset_type"] for asset in bundle.asset_types))
    return f"validated optimization configs for scenarios [{scenario_ids}] with asset types [{asset_types}]"
