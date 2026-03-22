#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyomo.environ as pyo

try:
    from _optimization_common import (
        summarize_bundle,
        validate_config_bundle,
        validate_phase1_input_contract,
    )
    from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet, write_geojson
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._optimization_common import (
        summarize_bundle,
        validate_config_bundle,
        validate_phase1_input_contract,
    )
    from scripts._spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet, write_geojson


CANDIDATE_SITES_PATH = PROCESSED_DIR / "surveillance_candidate_sites.geojson"
TERRAIN_COSTS_PATH = PROCESSED_DIR / "terrain_cost_surface.parquet"
WATERHOLE_INTERVENTIONS_PATH = PROCESSED_DIR / "waterhole_interventions.geojson"
COVERAGE_MATRIX_PATH = PROCESSED_DIR / "coverage_matrix.parquet"
RESPONSE_MATRIX_PATH = PROCESSED_DIR / "response_time_matrix.parquet"
FIRE_DELAY_BREAKPOINTS_PATH = PROCESSED_DIR / "fire_delay_breakpoints.parquet"
COMPOSITE_RISK_PATH = OUTPUTS_DIR / "composite_risk.geojson"
FRONTIER_PATH = OUTPUTS_DIR / "optimization_frontier.csv"
SOLUTION_PATH = OUTPUTS_DIR / "optimization_solution.geojson"
CELLS_PATH = OUTPUTS_DIR / "optimization_cells.parquet"
SUMMARY_PATH = OUTPUTS_DIR / "optimization_summary.json"
RESPONSE_ARC_TOP_K = 3
MAX_RESPONSE_HORIZON_MIN = 225.0
MOBILE_ASSETS = ("person", "car", "drone")
CAMERA_ASSET = "camera"
SELECTED_ALPHA_INDEX = 1
METRIC_CRS = "EPSG:32733"
FIRE_PENALTY_PLATEAU_AFTER_THRESHOLD_MIN = 90.0
FIRE_PENALTY_CEILING_DELAY_MIN = 60.0
FIRE_SIGMOID_STEEPNESS = 10.0


INCLUDED_ASSET_KEYS = {
    "person": "included_people",
    "car": "included_cars",
    "drone": "included_drones",
    "camera": "included_cameras",
}


@dataclass
class OptimizationData:
    scenario_id: str
    alpha_values: list[float]
    selected_sites: gpd.GeoDataFrame
    terrain: gpd.GeoDataFrame
    composite: gpd.GeoDataFrame
    interventions: gpd.GeoDataFrame
    coverage_rows: pd.DataFrame
    response_rows: pd.DataFrame
    fire_breakpoints: pd.DataFrame
    asset_types: dict[str, dict[str, object]]
    availability: dict[str, object]
    site_pairs: list[tuple[str, str]]
    waterhole_camera_sites: list[str]
    cells: list[str]
    dummy_response_time: float
    intervention_ids: list[str]
    response_arc_ids: list[tuple[str, str, str]]
    response_arc_data: dict[tuple[str, str, str], tuple[str, str, float]]
    coverage_by_cell: dict[str, list[tuple[str, str]]]
    camera_gain_by_cell: dict[str, list[tuple[str, float]]]
    intervention_gain_by_cell: dict[str, list[tuple[str, float]]]
    base_protection: dict[str, float]
    human_operability_penalty: dict[str, float]
    wildfire_risk: dict[str, float]
    composite_risk: dict[str, float]
    fire_bp_ids: list[int]
    fire_bp_time: dict[int, float]
    fire_bp_penalty: dict[int, float]


class FrontierSolveTimeoutError(RuntimeError):
    pass


def load_optimization_inputs(scenario_id: str) -> OptimizationData:
    bundle = validate_config_bundle()
    validate_phase1_input_contract(bundle, scenario_id)
    scenario = bundle.scenarios_by_id[scenario_id]
    availability = bundle.availability_by_scenario[scenario_id]
    asset_types = {str(asset["asset_type"]): asset for asset in bundle.asset_types}

    selected_sites = select_sites_for_optimization(
        validate_geojson(
            CANDIDATE_SITES_PATH,
            [
                "site_id",
                "scenario_id",
                "site_kind",
                "candidate_rank",
                "supports_people",
                "supports_cars",
                "supports_drones",
                "supports_cameras",
                "base_cost_fixed",
                "waterhole_influence_radius_m",
            ],
            "surveillance candidate sites",
        ),
        scenario,
        scenario_id,
    )
    terrain = validate_parquet(
        TERRAIN_COSTS_PATH,
        [
            "cell_id",
            "protection_benefit",
            "human_operability_penalty",
            "wildfire_risk_norm",
            "geometry",
        ],
        "terrain cost surface",
    )
    composite = validate_geojson(
        COMPOSITE_RISK_PATH,
        ["cell_id", "composite_risk_norm", "wildfire_risk_norm"],
        "composite risk layer",
    )
    interventions = validate_geojson(
        WATERHOLE_INTERVENTIONS_PATH,
        [
            "intervention_site_id",
            "capital_cost",
            "tourism_cost",
            "protection_benefit_gain",
            "influence_radius_m",
        ],
        "waterhole interventions",
    )
    coverage = pd.read_parquet(COVERAGE_MATRIX_PATH)
    response = pd.read_parquet(RESPONSE_MATRIX_PATH)
    fire_breakpoints = pd.read_parquet(FIRE_DELAY_BREAKPOINTS_PATH)

    selected_site_ids = set(selected_sites["site_id"])
    active_asset_types = set(str(asset_type) for asset_type in scenario["active_asset_types"])
    coverage = coverage[
        coverage["site_id"].isin(selected_site_ids) & coverage["asset_type"].isin(active_asset_types)
    ].copy()
    response = response[
        response["site_id"].isin(selected_site_ids) & response["asset_type"].isin(active_asset_types)
    ].copy()

    merged = terrain.merge(
        composite[["cell_id", "composite_risk_norm"]].rename(columns={"composite_risk_norm": "composite_risk"}),
        on="cell_id",
        how="inner",
    ).copy()
    if len(merged) != len(terrain):
        raise ValueError("terrain and composite outputs must align one-to-one by cell_id")
    cells = merged["cell_id"].tolist()

    site_pairs = build_site_asset_pairs(selected_sites, asset_types, active_asset_types)
    response_rows = filter_response_rows(response, cells)
    response_arc_ids = list(response_rows.index)
    response_arc_data = {
        index: (row["site_id"], row["asset_type"], float(row["response_time_min"]))
        for index, row in response_rows.iterrows()
    }

    coverage_by_cell = build_coverage_lookup(coverage, cells)
    camera_gain_by_cell = build_camera_gain_lookup(coverage, cells)
    intervention_gain_by_cell = build_intervention_gain_lookup(interventions, merged)

    fire_breakpoints = augment_fire_breakpoints(
        fire_breakpoints,
        response_rows,
        float(availability["tau_fire_min"]),
        float(availability["beta_fire"]),
        float(availability["lambda_fire"]),
        scenario_id,
    )
    fire_bp_ids = fire_breakpoints["breakpoint_index"].astype(int).tolist()

    dummy_response_time = float(fire_breakpoints["response_time_min"].max())
    waterhole_camera_sites = selected_sites.loc[
        selected_sites["site_kind"].eq("waterhole") & selected_sites["supports_cameras"],
        "site_id",
    ].tolist()
    return OptimizationData(
        scenario_id=scenario_id,
        alpha_values=sorted([float(value) for value in scenario["alpha_values"]], reverse=True),
        selected_sites=selected_sites,
        terrain=merged,
        composite=composite,
        interventions=interventions,
        coverage_rows=coverage,
        response_rows=response_rows.reset_index(drop=False),
        fire_breakpoints=fire_breakpoints,
        asset_types=asset_types,
        availability=availability,
        site_pairs=site_pairs,
        waterhole_camera_sites=waterhole_camera_sites,
        cells=cells,
        dummy_response_time=dummy_response_time,
        intervention_ids=interventions["intervention_site_id"].tolist(),
        response_arc_ids=response_arc_ids,
        response_arc_data=response_arc_data,
        coverage_by_cell=coverage_by_cell,
        camera_gain_by_cell=camera_gain_by_cell,
        intervention_gain_by_cell=intervention_gain_by_cell,
        base_protection=merged.set_index("cell_id")["protection_benefit"].to_dict(),
        human_operability_penalty=merged.set_index("cell_id")["human_operability_penalty"].to_dict(),
        wildfire_risk=merged.set_index("cell_id")["wildfire_risk_norm"].to_dict(),
        composite_risk=merged.set_index("cell_id")["composite_risk"].to_dict(),
        fire_bp_ids=fire_bp_ids,
        fire_bp_time=fire_breakpoints.set_index("breakpoint_index")["response_time_min"].to_dict(),
        fire_bp_penalty=fire_breakpoints.set_index("breakpoint_index")["penalty_value"].to_dict(),
    )


def select_sites_for_optimization(
    sites: gpd.GeoDataFrame,
    scenario: dict[str, object],
    scenario_id: str,
) -> gpd.GeoDataFrame:
    scoped = sites[sites["scenario_id"] == scenario_id].copy()
    if scoped.empty:
        raise ValueError(f"no candidate sites found for scenario_id {scenario_id}")
    anchors = scoped[scoped["site_kind"] != "high_risk_cell"].copy()
    high_risk = scoped[scoped["site_kind"] == "high_risk_cell"].nsmallest(int(scenario["top_site_count"]), "candidate_rank")
    selected = pd.concat([anchors, high_risk], ignore_index=True)
    if selected["site_id"].duplicated().any():
        raise ValueError("selected optimization candidate sites must have unique site_id values")
    return selected


def build_site_asset_pairs(
    sites: gpd.GeoDataFrame,
    asset_types: dict[str, dict[str, object]],
    active_asset_types: set[str],
) -> list[tuple[str, str]]:
    support_columns = {
        "person": "supports_people",
        "car": "supports_cars",
        "drone": "supports_drones",
        "camera": "supports_cameras",
    }
    pairs: list[tuple[str, str]] = []
    for _, row in sites.iterrows():
        for asset_type in active_asset_types:
            if bool(row[support_columns[asset_type]]):
                pairs.append((str(row["site_id"]), asset_type))
    if not pairs:
        raise ValueError("no feasible site/asset pairs are available for optimization")
    return sorted(pairs)


def filter_response_rows(response: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    filtered = response[
        response["response_feasible"] & response["cell_id"].isin(cells) & (response["response_time_min"] <= MAX_RESPONSE_HORIZON_MIN)
    ].copy()
    filtered = filtered.sort_values(["cell_id", "asset_type", "response_time_min", "site_id"])
    filtered = filtered.groupby(["cell_id", "asset_type"], group_keys=False).head(RESPONSE_ARC_TOP_K).copy()
    filtered["arc_id"] = (
        filtered["cell_id"].astype(str) + "|" + filtered["site_id"].astype(str) + "|" + filtered["asset_type"].astype(str)
    )
    if filtered.empty:
        raise ValueError("filtered response arc set is empty")
    return filtered.set_index("arc_id")


def build_coverage_lookup(coverage_rows: pd.DataFrame, cells: list[str]) -> dict[str, list[tuple[str, str]]]:
    mobile = coverage_rows[
        coverage_rows["asset_type"].isin(MOBILE_ASSETS) & coverage_rows["is_coverable"] & (coverage_rows["effective_coverage_score"] > 0)
    ]
    lookup = {
        cell_id: list(zip(group["site_id"].astype(str), group["asset_type"].astype(str)))
        for cell_id, group in mobile.groupby("cell_id")
    }
    for cell_id in cells:
        lookup.setdefault(cell_id, [])
    return lookup


def build_camera_gain_lookup(coverage_rows: pd.DataFrame, cells: list[str]) -> dict[str, list[tuple[str, float]]]:
    camera = coverage_rows[
        coverage_rows["asset_type"].eq(CAMERA_ASSET) & (coverage_rows["protection_gain_influence"] > 0)
    ].copy()
    lookup = {
        cell_id: list(
            zip(group["site_id"].astype(str), group["protection_gain_influence"].astype(float))
        )
        for cell_id, group in camera.groupby("cell_id")
    }
    for cell_id in cells:
        lookup.setdefault(cell_id, [])
    return lookup


def build_intervention_gain_lookup(interventions: gpd.GeoDataFrame, terrain: gpd.GeoDataFrame) -> dict[str, list[tuple[str, float]]]:
    intervention_metric = interventions.to_crs(METRIC_CRS)
    terrain_metric = terrain.to_crs(METRIC_CRS)
    lookup: dict[str, list[tuple[str, float]]] = {cell_id: [] for cell_id in terrain["cell_id"]}
    for _, row in intervention_metric.iterrows():
        distance = terrain_metric.geometry.distance(row.geometry)
        influence_radius = max(float(row["influence_radius_m"]), 1.0)
        gain = np.where(
            distance <= influence_radius,
            float(row["protection_benefit_gain"]) * np.exp(-distance / influence_radius),
            0.0,
        )
        for cell_id, gain_value in zip(terrain_metric["cell_id"], gain):
            if gain_value > 0:
                lookup[str(cell_id)].append((str(row["intervention_site_id"]), float(gain_value)))
    return lookup


def augment_fire_breakpoints(
    breakpoints: pd.DataFrame,
    response_rows: pd.DataFrame,
    tau_fire_min: float,
    beta_fire: float,
    lambda_fire: float,
    scenario_id: str,
) -> pd.DataFrame:
    def bounded_fire_delay_penalty(response_time_min: float, plateau_time_min: float) -> float:
        plateau_time_min = max(float(plateau_time_min), tau_fire_min + 1e-6)
        plateau_penalty = np.expm1(beta_fire * min(FIRE_PENALTY_CEILING_DELAY_MIN, plateau_time_min - tau_fire_min))
        scaled_delay = np.clip((response_time_min - tau_fire_min) / (plateau_time_min - tau_fire_min), 0.0, 1.0)
        sigmoid = 1.0 / (1.0 + np.exp(-FIRE_SIGMOID_STEEPNESS * (scaled_delay - 0.5)))
        sigmoid_floor = 1.0 / (1.0 + np.exp(FIRE_SIGMOID_STEEPNESS / 2.0))
        sigmoid_ceiling = 1.0 / (1.0 + np.exp(-FIRE_SIGMOID_STEEPNESS / 2.0))
        normalized = np.clip((sigmoid - sigmoid_floor) / (sigmoid_ceiling - sigmoid_floor), 0.0, 1.0)
        if response_time_min <= tau_fire_min:
            return 0.0
        return float(plateau_penalty * normalized)

    max_response = max(float(response_rows["response_time_min"].max()), MAX_RESPONSE_HORIZON_MIN)
    rows = breakpoints.copy()
    last_time = float(rows["response_time_min"].max())
    if max_response > last_time:
        plateau_time = min(max_response, tau_fire_min + FIRE_PENALTY_PLATEAU_AFTER_THRESHOLD_MIN)
        penalty = bounded_fire_delay_penalty(max_response, plateau_time)
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    {
                        "scenario_id": [scenario_id],
                        "breakpoint_index": [int(rows["breakpoint_index"].max()) + 1],
                        "response_time_min": [max_response],
                        "penalty_value": [penalty],
                        "tau_fire_min": [tau_fire_min],
                        "beta_fire": [beta_fire],
                        "lambda_fire": [lambda_fire],
                        "source": ["phase6_dynamic_fire_breakpoint"],
                    }
                ),
            ],
            ignore_index=True,
        )
    rows = rows.sort_values("response_time_min").reset_index(drop=True)
    rows["breakpoint_index"] = np.arange(len(rows))
    return rows


def build_model(
    data: OptimizationData,
    *,
    mode: str,
    coverage_floor: float | None = None,
) -> tuple[pyo.ConcreteModel, object]:
    model = pyo.ConcreteModel(name=f"surveillance_{mode}")
    model.CELLS = pyo.Set(initialize=data.cells, ordered=True)
    model.SITE_ASSETS = pyo.Set(dimen=2, initialize=data.site_pairs, ordered=True)
    model.SITES = pyo.Set(initialize=sorted(data.selected_sites["site_id"].astype(str).tolist()), ordered=True)
    model.RESPONSE_ARCS = pyo.Set(initialize=data.response_arc_ids, ordered=True)
    model.FIRE_BPS = pyo.Set(initialize=data.fire_bp_ids, ordered=True)
    model.INTERVENTIONS = pyo.Set(initialize=data.intervention_ids, ordered=True)
    model.WATERHOLE_SITES = pyo.Set(initialize=data.waterhole_camera_sites, ordered=True)

    model.site_active = pyo.Var(model.SITES, domain=pyo.Binary)
    model.asset_active = pyo.Var(model.SITE_ASSETS, domain=pyo.Binary)
    model.x = pyo.Var(model.SITE_ASSETS, domain=pyo.NonNegativeIntegers)
    model.excess_assets = pyo.Var(sorted(INCLUDED_ASSET_KEYS), domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.CELLS, domain=pyo.Binary)
    model.z = pyo.Var(model.RESPONSE_ARCS, domain=pyo.Binary)
    model.dummy = pyo.Var(model.CELLS, domain=pyo.Binary)
    model.t = pyo.Var(model.CELLS, domain=pyo.NonNegativeReals, bounds=(0.0, data.dummy_response_time))
    model.fire_penalty = pyo.Var(model.CELLS, domain=pyo.NonNegativeReals)
    model.fire_lambda = pyo.Var(model.CELLS, model.FIRE_BPS, domain=pyo.NonNegativeReals)
    model.u = pyo.Var(model.INTERVENTIONS, domain=pyo.Binary)
    model.lockdown = pyo.Var(model.WATERHOLE_SITES, domain=pyo.Binary)

    site_rows = data.selected_sites.set_index("site_id")

    def site_link_rule(model: pyo.ConcreteModel, site_id: str, asset_type: str):
        max_units = int(data.asset_types[asset_type]["max_units_per_site"])
        return model.x[site_id, asset_type] <= max_units * model.asset_active[site_id, asset_type]

    model.asset_upper = pyo.Constraint(model.SITE_ASSETS, rule=site_link_rule)

    def asset_lower_rule(model: pyo.ConcreteModel, site_id: str, asset_type: str):
        if asset_type == CAMERA_ASSET:
            return pyo.Constraint.Skip
        return model.x[site_id, asset_type] >= model.asset_active[site_id, asset_type]

    model.asset_lower = pyo.Constraint(model.SITE_ASSETS, rule=asset_lower_rule)

    def site_active_rule(model: pyo.ConcreteModel, site_id: str, asset_type: str):
        return model.asset_active[site_id, asset_type] <= model.site_active[site_id]

    model.site_asset_link = pyo.Constraint(model.SITE_ASSETS, rule=site_active_rule)

    bundle_size = int(data.asset_types[CAMERA_ASSET]["camera_bundle_size"])

    def camera_bundle_rule(model: pyo.ConcreteModel, site_id: str):
        return model.x[site_id, CAMERA_ASSET] == bundle_size * model.lockdown[site_id]

    model.camera_bundle = pyo.Constraint(model.WATERHOLE_SITES, rule=camera_bundle_rule)

    def camera_active_rule(model: pyo.ConcreteModel, site_id: str):
        return model.asset_active[site_id, CAMERA_ASSET] == model.lockdown[site_id]

    model.camera_active = pyo.Constraint(model.WATERHOLE_SITES, rule=camera_active_rule)

    budget_terms = []
    model.excess_asset_balance = pyo.ConstraintList()
    for asset_type, included_key in INCLUDED_ASSET_KEYS.items():
        applicable = [(site_id, a) for site_id, a in data.site_pairs if a == asset_type]
        if not applicable:
            model.excess_asset_balance.add(model.excess_assets[asset_type] >= 0.0)
            continue
        total_units = sum(model.x[pair] for pair in applicable)
        included_units = float(data.availability[included_key])
        model.excess_asset_balance.add(model.excess_assets[asset_type] >= total_units - included_units)
        unit_cost = float(data.asset_types[asset_type]["unit_cost"])
        budget_terms.append(unit_cost * model.excess_assets[asset_type])
    for site_id in model.SITES:
        budget_terms.append(float(site_rows.loc[site_id, "base_cost_fixed"]) * model.site_active[site_id])
    for intervention_id in model.INTERVENTIONS:
        tourism_row = data.interventions.set_index("intervention_site_id").loc[intervention_id]
        budget_terms.append(float(tourism_row["capital_cost"]) * model.u[intervention_id])
    model.budget = pyo.Constraint(expr=sum(budget_terms) <= float(data.availability["budget_total"]))

    cap_map = {
        "person": int(data.availability["max_people"]),
        "car": int(data.availability["max_cars"]),
        "drone": int(data.availability["max_drones"]),
        "camera": int(data.availability["max_cameras"]),
    }
    model.asset_caps = pyo.ConstraintList()
    for asset_type, cap in cap_map.items():
        applicable = [(site_id, a) for site_id, a in data.site_pairs if a == asset_type]
        if not applicable:
            continue
        model.asset_caps.add(sum(model.x[pair] for pair in applicable) <= cap)

    def coverage_rule(model: pyo.ConcreteModel, cell_id: str):
        cover_pairs = data.coverage_by_cell[cell_id]
        if not cover_pairs:
            return model.y[cell_id] == 0
        return model.y[cell_id] <= sum(model.asset_active[site_id, asset_type] for site_id, asset_type in cover_pairs)

    model.coverage_feasibility = pyo.Constraint(model.CELLS, rule=coverage_rule)

    response_arcs_by_cell: dict[str, list[str]] = {cell_id: [] for cell_id in data.cells}
    for arc_id, (site_id, asset_type, _) in data.response_arc_data.items():
        cell_id = arc_id.split("|", 1)[0]
        response_arcs_by_cell[cell_id].append(arc_id)

    def response_assign_rule(model: pyo.ConcreteModel, cell_id: str):
        arcs = response_arcs_by_cell[cell_id]
        return sum(model.z[arc_id] for arc_id in arcs) + model.dummy[cell_id] == 1

    model.response_assignment = pyo.Constraint(model.CELLS, rule=response_assign_rule)

    def response_active_rule(model: pyo.ConcreteModel, arc_id: str):
        site_id, asset_type, _ = data.response_arc_data[arc_id]
        return model.z[arc_id] <= model.asset_active[site_id, asset_type]

    model.response_active = pyo.Constraint(model.RESPONSE_ARCS, rule=response_active_rule)

    def response_time_rule(model: pyo.ConcreteModel, cell_id: str):
        arcs = response_arcs_by_cell[cell_id]
        return model.t[cell_id] == sum(
            data.response_arc_data[arc_id][2] * model.z[arc_id] for arc_id in arcs
        ) + data.dummy_response_time * model.dummy[cell_id]

    model.response_time = pyo.Constraint(model.CELLS, rule=response_time_rule)

    def fire_simplex_rule(model: pyo.ConcreteModel, cell_id: str):
        return sum(model.fire_lambda[cell_id, bp_id] for bp_id in model.FIRE_BPS) == 1

    model.fire_simplex = pyo.Constraint(model.CELLS, rule=fire_simplex_rule)

    def fire_time_interp_rule(model: pyo.ConcreteModel, cell_id: str):
        return model.t[cell_id] == sum(
            data.fire_bp_time[bp_id] * model.fire_lambda[cell_id, bp_id] for bp_id in model.FIRE_BPS
        )

    model.fire_time_interp = pyo.Constraint(model.CELLS, rule=fire_time_interp_rule)

    def fire_penalty_interp_rule(model: pyo.ConcreteModel, cell_id: str):
        return model.fire_penalty[cell_id] == sum(
            data.fire_bp_penalty[bp_id] * model.fire_lambda[cell_id, bp_id] for bp_id in model.FIRE_BPS
        )

    model.fire_penalty_interp = pyo.Constraint(model.CELLS, rule=fire_penalty_interp_rule)

    def protection_expr(model: pyo.ConcreteModel):
        base_term = sum(data.base_protection[cell_id] * model.y[cell_id] for cell_id in model.CELLS)
        camera_term = sum(
            gain * model.lockdown[site_id]
            for cell_id in data.cells
            for site_id, gain in data.camera_gain_by_cell[cell_id]
        )
        intervention_term = sum(
            gain * model.u[intervention_id]
            for cell_id in data.cells
            for intervention_id, gain in data.intervention_gain_by_cell[cell_id]
        )
        return base_term + camera_term + intervention_term

    model.protection_expr = pyo.Expression(rule=protection_expr)
    tourism_costs = data.interventions.set_index("intervention_site_id")["tourism_cost"].to_dict()
    model.response_expr = pyo.Expression(
        expr=
        sum(data.composite_risk[cell_id] * model.t[cell_id] for cell_id in model.CELLS)
        + float(data.availability["lambda_fire"])
        * sum(data.wildfire_risk[cell_id] * model.fire_penalty[cell_id] for cell_id in model.CELLS)
        + sum(float(tourism_costs[intervention_id]) * model.u[intervention_id] for intervention_id in model.INTERVENTIONS)
    )

    if coverage_floor is not None:
        model.coverage_floor = pyo.Constraint(expr=model.protection_expr >= coverage_floor)

    if mode == "coverage":
        model.objective = pyo.Objective(expr=model.protection_expr, sense=pyo.maximize)
    elif mode == "response":
        model.objective = pyo.Objective(expr=model.response_expr, sense=pyo.minimize)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unsupported optimization mode: {mode}")
    return model, model.objective


def solve_model(model: pyo.ConcreteModel) -> tuple[str, bool]:
    solver = pyo.SolverFactory("highs")
    solver.options["time_limit"] = 180.0
    solver.options["mip_rel_gap"] = 0.05
    result = solver.solve(model, tee=False)
    termination = str(result.solver.termination_condition).lower()
    has_solution = len(result.solution) > 0
    if "maxtimelimit" in termination or "timelimit" in termination:
        raise FrontierSolveTimeoutError(
            f"optimization solve hit time limit with termination={result.solver.termination_condition} "
            f"and has_solution={has_solution}"
        )
    if "optimal" not in termination and "feasible" not in termination:
        raise RuntimeError(f"optimization solve failed with termination condition: {result.solver.termination_condition}")
    return termination, has_solution


def extract_solution(
    data: OptimizationData,
    model: pyo.ConcreteModel,
    *,
    alpha: float,
    coverage_max: float,
) -> dict[str, object]:
    site_rows = data.selected_sites.copy()
    site_rows["selected"] = False
    site_rows["site_active"] = False
    site_rows["people_count"] = 0
    site_rows["car_count"] = 0
    site_rows["drone_count"] = 0
    site_rows["camera_count"] = 0
    site_rows["camera_lockdown"] = False

    for site_id, asset_type in data.site_pairs:
        count = int(round(pyo.value(model.x[site_id, asset_type])))
        active = bool(round(pyo.value(model.asset_active[site_id, asset_type])))
        if count > 0 or active:
            site_rows.loc[site_rows["site_id"] == site_id, "selected"] = True
            site_rows.loc[site_rows["site_id"] == site_id, "site_active"] = bool(round(pyo.value(model.site_active[site_id])))
        column = {
            "person": "people_count",
            "car": "car_count",
            "drone": "drone_count",
            "camera": "camera_count",
        }[asset_type]
        site_rows.loc[site_rows["site_id"] == site_id, column] = count
    for site_id in data.waterhole_camera_sites:
        site_rows.loc[site_rows["site_id"] == site_id, "camera_lockdown"] = bool(round(pyo.value(model.lockdown[site_id])))

    cell_frame = data.terrain.copy()
    cell_frame["covered"] = [bool(round(pyo.value(model.y[cell_id]))) for cell_id in data.cells]
    cell_frame["response_time_min"] = [float(pyo.value(model.t[cell_id])) for cell_id in data.cells]
    cell_frame["fire_delay_penalty"] = [max(0.0, float(pyo.value(model.fire_penalty[cell_id]))) for cell_id in data.cells]
    cell_frame["protection_benefit_base"] = cell_frame["cell_id"].map(data.base_protection)

    selected_lockdowns = {site_id for site_id in data.waterhole_camera_sites if bool(round(pyo.value(model.lockdown[site_id])))}
    selected_interventions = {intervention_id for intervention_id in data.intervention_ids if bool(round(pyo.value(model.u[intervention_id])))}
    cell_frame["camera_gain_applied"] = [
        sum(gain for site_id, gain in data.camera_gain_by_cell[cell_id] if site_id in selected_lockdowns)
        for cell_id in data.cells
    ]
    cell_frame["intervention_gain_applied"] = [
        sum(gain for intervention_id, gain in data.intervention_gain_by_cell[cell_id] if intervention_id in selected_interventions)
        for cell_id in data.cells
    ]
    cell_frame["protection_benefit_effective"] = (
        cell_frame["protection_benefit_base"]
        + cell_frame["camera_gain_applied"]
        + cell_frame["intervention_gain_applied"]
    )

    selected_responder: dict[str, str] = {}
    selected_response_site: dict[str, str] = {}
    for cell_id in data.cells:
        chosen_arc = None
        for arc_id in data.response_arc_ids:
            if arc_id.startswith(f"{cell_id}|") and pyo.value(model.z[arc_id]) > 0.5:
                chosen_arc = arc_id
                break
        if chosen_arc is None:
            selected_responder[cell_id] = "dummy"
            selected_response_site[cell_id] = "none"
        else:
            site_id, asset_type, _ = data.response_arc_data[chosen_arc]
            selected_responder[cell_id] = asset_type
            selected_response_site[cell_id] = site_id
    cell_frame["selected_responder_asset"] = cell_frame["cell_id"].map(selected_responder)
    cell_frame["selected_responder_site_id"] = cell_frame["cell_id"].map(selected_response_site)

    selected_solution = site_rows[site_rows["selected"]].copy()
    budget_total = 0.0
    for _, row in selected_solution.iterrows():
        budget_total += float(row["base_cost_fixed"])
    asset_totals = {
        "person": int(selected_solution["people_count"].sum()),
        "car": int(selected_solution["car_count"].sum()),
        "drone": int(selected_solution["drone_count"].sum()),
        "camera": int(selected_solution["camera_count"].sum()),
    }
    for asset_type, total_units in asset_totals.items():
        included_units = int(data.availability[INCLUDED_ASSET_KEYS[asset_type]])
        chargeable_units = max(0, total_units - included_units)
        budget_total += chargeable_units * float(data.asset_types[asset_type]["unit_cost"])
    selected_intervention_frame = data.interventions[data.interventions["intervention_site_id"].isin(selected_interventions)].copy()
    budget_total += float(selected_intervention_frame["capital_cost"].sum())
    tourism_total = float(selected_intervention_frame["tourism_cost"].sum())

    frontier_row = {
        "scenario_id": data.scenario_id,
        "alpha": alpha,
        "coverage_target": alpha * coverage_max,
        "achieved_protection": float(pyo.value(model.protection_expr)),
        "response_objective": float(pyo.value(model.response_expr)),
        "budget_used": budget_total,
        "tourism_penalty_used": tourism_total,
        "selected_site_count": int(selected_solution["site_id"].nunique()),
        "selected_people": int(selected_solution["people_count"].sum()),
        "selected_cars": int(selected_solution["car_count"].sum()),
        "selected_drones": int(selected_solution["drone_count"].sum()),
        "selected_cameras": int(selected_solution["camera_count"].sum()),
        "locked_down_waterholes": int(selected_solution["camera_lockdown"].sum()),
        "selected_interventions": int(len(selected_intervention_frame)),
    }
    return {
        "sites": selected_solution,
        "cells": cell_frame,
        "interventions": selected_intervention_frame,
        "frontier_row": frontier_row,
    }


def choose_recommended_alpha(alpha_values: list[float]) -> float:
    ordered = sorted(alpha_values, reverse=True)
    return ordered[min(SELECTED_ALPHA_INDEX, len(ordered) - 1)]


def write_outputs(
    frontier_rows: list[dict[str, object]],
    chosen_solution: dict[str, object],
    data: OptimizationData,
    recommended_alpha: float,
    *,
    frontier_status: str = "complete",
    requested_recommended_alpha: float | None = None,
) -> None:
    frontier = pd.DataFrame(frontier_rows).sort_values("alpha", ascending=False)
    frontier.to_csv(FRONTIER_PATH, index=False)

    solution_sites = gpd.GeoDataFrame(chosen_solution["sites"], geometry="geometry", crs=data.selected_sites.crs)
    write_geojson(solution_sites, SOLUTION_PATH)
    chosen_solution["cells"].to_parquet(CELLS_PATH, index=False)

    summary = {
        "scenario_id": data.scenario_id,
        "recommended_alpha": recommended_alpha,
        "requested_recommended_alpha": (
            recommended_alpha if requested_recommended_alpha is None else requested_recommended_alpha
        ),
        "frontier_status": frontier_status,
        "available_budget": float(data.availability["budget_total"]),
        "available_caps": {
            "people": int(data.availability["max_people"]),
            "cars": int(data.availability["max_cars"]),
            "drones": int(data.availability["max_drones"]),
            "cameras": int(data.availability["max_cameras"]),
        },
        "frontier_points": frontier_rows,
        "chosen_solution": frontier.loc[np.isclose(frontier["alpha"], recommended_alpha)].iloc[0].to_dict(),
        "selected_interventions": chosen_solution["interventions"][
            ["intervention_site_id", "capital_cost", "tourism_cost", "protection_benefit_gain"]
        ].to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def check_outputs() -> None:
    if not FRONTIER_PATH.is_file():
        raise FileNotFoundError(f"missing output: {FRONTIER_PATH}")
    if not SOLUTION_PATH.is_file():
        raise FileNotFoundError(f"missing output: {SOLUTION_PATH}")
    if not CELLS_PATH.is_file():
        raise FileNotFoundError(f"missing output: {CELLS_PATH}")
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"missing output: {SUMMARY_PATH}")

    frontier = pd.read_csv(FRONTIER_PATH)
    if frontier.empty:
        raise ValueError("optimization frontier is empty")
    if not frontier["coverage_target"].is_monotonic_decreasing:
        raise ValueError("frontier coverage targets must be monotone decreasing")
    if (frontier["response_objective"] <= 0).any():
        raise ValueError("frontier response objective values must be positive")

    solution = validate_geojson(
        SOLUTION_PATH,
        ["site_id", "selected", "people_count", "car_count", "drone_count", "camera_count", "camera_lockdown"],
        "optimization solution",
    )
    if (solution["camera_count"] > 0).any():
        invalid_camera_sites = solution.loc[
            solution["camera_count"] > 0, "site_kind"
        ].ne("waterhole")
        if invalid_camera_sites.any():
            raise ValueError("camera deployments must only appear at eligible waterhole sites")

    cells = validate_parquet(
        CELLS_PATH,
        [
            "cell_id",
            "covered",
            "response_time_min",
            "protection_benefit_base",
            "protection_benefit_effective",
            "human_operability_penalty",
            "fire_delay_penalty",
            "camera_gain_applied",
            "intervention_gain_applied",
            "geometry",
        ],
        "optimization cells",
    )
    if (cells["response_time_min"] <= 0).any():
        raise ValueError("optimization cell response times must stay positive")
    if (cells["fire_delay_penalty"] < -1e-8).any():
        raise ValueError("fire delay penalty must stay non-negative")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    chosen = summary.get("chosen_solution", {})
    if float(chosen.get("budget_used", 0.0)) > float(summary["available_budget"]) + 1e-6:
        raise ValueError("chosen solution exceeds the available budget")
    if float(chosen.get("selected_people", 0)) > int(summary["available_caps"]["people"]):
        raise ValueError("chosen solution exceeds max_people")
    if float(chosen.get("selected_cars", 0)) > int(summary["available_caps"]["cars"]):
        raise ValueError("chosen solution exceeds max_cars")
    if float(chosen.get("selected_drones", 0)) > int(summary["available_caps"]["drones"]):
        raise ValueError("chosen solution exceeds max_drones")
    if float(chosen.get("selected_cameras", 0)) > int(summary["available_caps"]["cameras"]):
        raise ValueError("chosen solution exceeds max_cameras")
    print("optimization output check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="solve the Etosha surveillance optimization frontier")
    parser.add_argument("--scenario-id", default="etosha_placeholder_baseline", help="scenario id to optimize")
    parser.add_argument("--validate-only", action="store_true", help="validate configs and upstream inputs without solving")
    args = parser.parse_args()

    bundle = validate_config_bundle()
    validate_phase1_input_contract(bundle, args.scenario_id)
    if args.validate_only:
        print(f"{summarize_bundle(bundle)}; phase 1 input contract passed for {args.scenario_id}")
        return 0

    data = load_optimization_inputs(args.scenario_id)
    print(
        f"loaded optimization data for {data.scenario_id}: "
        f"{len(data.selected_sites)} sites, {len(data.cells)} cells, {len(data.response_arc_ids)} response arcs"
    )
    print("solving coverage-maximization model")
    coverage_model, _ = build_model(data, mode="coverage")
    solve_model(coverage_model)
    coverage_max = float(pyo.value(coverage_model.protection_expr))
    if coverage_max <= 0:
        raise ValueError("maximum achievable protection must be positive")
    print(f"coverage max = {coverage_max:.3f}")

    frontier_rows: list[dict[str, object]] = []
    chosen_solution: dict[str, object] | None = None
    last_solution: dict[str, object] | None = None
    recommended_alpha = choose_recommended_alpha(data.alpha_values)
    for alpha in data.alpha_values:
        print(f"solving response-minimization frontier point alpha={alpha:.2f}")
        response_model, _ = build_model(data, mode="response", coverage_floor=alpha * coverage_max)
        try:
            solve_model(response_model)
        except FrontierSolveTimeoutError as exc:
            print(f"frontier solve timed out at alpha={alpha:.2f}: {exc}")
            if last_solution is not None:
                fallback_alpha = float(last_solution["frontier_row"]["alpha"])
                write_outputs(
                    frontier_rows,
                    chosen_solution or last_solution,
                    data,
                    fallback_alpha if chosen_solution is None else recommended_alpha,
                    frontier_status="partial_timeout",
                    requested_recommended_alpha=recommended_alpha,
                )
                print(f"wrote partial frontier outputs through alpha={fallback_alpha:.2f}")
            return 2
        extracted = extract_solution(data, response_model, alpha=alpha, coverage_max=coverage_max)
        frontier_rows.append(extracted["frontier_row"])
        last_solution = extracted
        if np.isclose(alpha, recommended_alpha):
            chosen_solution = extracted
    if chosen_solution is None:
        raise RuntimeError("recommended frontier solution was not produced")

    write_outputs(frontier_rows, chosen_solution, data, recommended_alpha)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
