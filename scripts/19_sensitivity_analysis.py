#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from _optimization_common import validate_config_bundle, validate_phase1_input_contract
    from _spatial_common import OUTPUTS_DIR, validate_geojson, validate_parquet
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._optimization_common import validate_config_bundle, validate_phase1_input_contract
    from scripts._spatial_common import OUTPUTS_DIR, validate_geojson, validate_parquet


SCRIPTS_DIR = Path(__file__).resolve().parent
SCENARIO_SCREENING_PATH = OUTPUTS_DIR / "sensitivity_screening.csv"
SENSITIVITY_RESULTS_PATH = OUTPUTS_DIR / "sensitivity_results.csv"
SENSITIVITY_SUMMARY_PATH = OUTPUTS_DIR / "sensitivity_summary.json"
SENSITIVITY_TORNADO_PATH = OUTPUTS_DIR / "sensitivity_tornado.png"

FEATURES_PATH = OUTPUTS_DIR / "grid_features.parquet"
COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
GRID_CENTROIDS_PATH = OUTPUTS_DIR / "grid_centroids.geojson"
CANDIDATE_SITES_PATH = Path(__file__).resolve().parent.parent / "data/processed/surveillance_candidate_sites.geojson"

SCREENING_MULTIPLIERS = [0.5, 2.0]
FULL_MULTIPLIERS = [0.5, 0.75, 4.0 / 3.0, 2.0]
DEFAULT_TOP_N = 5


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    category: str

    def apply(self, scenario: dict[str, object], availability: dict[str, object], asset_types: dict[str, dict[str, object]], multiplier: float) -> None:
        if self.name == "budget_total":
            availability["budget_total"] = float(availability["budget_total"]) * multiplier
            return
        if self.name == "max_people":
            scaled = max(1, int(round(float(availability["max_people"]) * multiplier)))
            availability["max_people"] = scaled
            availability["included_people"] = min(scaled, max(0, int(round(float(availability["included_people"]) * multiplier))))
            return
        if self.name == "max_drones":
            scaled = max(1, int(round(float(availability["max_drones"]) * multiplier)))
            availability["max_drones"] = scaled
            availability["included_drones"] = min(scaled, max(0, int(round(float(availability["included_drones"]) * multiplier))))
            return
        if self.name == "max_cameras":
            scaled = max(1, int(round(float(availability["max_cameras"]) * multiplier)))
            availability["max_cameras"] = scaled
            availability["included_cameras"] = min(scaled, max(0, int(round(float(availability["included_cameras"]) * multiplier))))
            return
        if self.name == "tau_fire_min":
            availability["tau_fire_min"] = max(1.0, float(availability["tau_fire_min"]) * multiplier)
            return
        if self.name == "beta_fire":
            availability["beta_fire"] = max(1e-4, float(availability["beta_fire"]) * multiplier)
            return
        if self.name == "lambda_fire":
            availability["lambda_fire"] = max(1e-4, float(availability["lambda_fire"]) * multiplier)
            return
        if self.name == "protection_leverage_scale":
            section = scenario["protection_benefit"]
            section["leverage_scale"] = max(1e-4, float(section["leverage_scale"]) * multiplier)
            return
        if self.name == "camera_gain_factor":
            section = scenario["protection_benefit"]
            section["camera_gain_factor"] = max(1e-4, float(section["camera_gain_factor"]) * multiplier)
            return
        if self.name == "waterhole_dispersion_benefit":
            section = scenario["artificial_waterhole_interventions"]
            section["expected_density_dispersion_benefit"] = max(
                1e-4,
                float(section["expected_density_dispersion_benefit"]) * multiplier,
            )
            return
        if self.name == "drone_response_speed_kmh":
            drone = asset_types["drone"]
            drone["response_speed_kmh"] = max(1.0, float(drone["response_speed_kmh"]) * multiplier)
            return
        raise ValueError(f"unsupported parameter name: {self.name}")


@dataclass
class RawInputs:
    scenario_id: str
    scenario: dict[str, object]
    availability: dict[str, object]
    asset_types: dict[str, dict[str, object]]
    features: gpd.GeoDataFrame
    composite: gpd.GeoDataFrame
    centroids: gpd.GeoDataFrame
    candidate_sites: gpd.GeoDataFrame


PARAMETER_SPECS = [
    ParameterSpec("budget_total", "budget total", "resources"),
    ParameterSpec("max_people", "people cap", "resources"),
    ParameterSpec("max_drones", "drone cap", "resources"),
    ParameterSpec("max_cameras", "camera cap", "resources"),
    ParameterSpec("tau_fire_min", "fire threshold tau", "fire"),
    ParameterSpec("beta_fire", "fire penalty slope beta", "fire"),
    ParameterSpec("lambda_fire", "fire penalty weight lambda", "fire"),
    ParameterSpec("protection_leverage_scale", "protection leverage scale", "protection"),
    ParameterSpec("camera_gain_factor", "camera gain factor", "protection"),
    ParameterSpec("waterhole_dispersion_benefit", "waterhole dispersion benefit", "interventions"),
    ParameterSpec("drone_response_speed_kmh", "drone response speed", "mobility"),
]


def load_script_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


OPTIMIZE = load_script_module("optimize_surveillance_16", SCRIPTS_DIR / "16_optimize_surveillance.py")
TERRAIN = load_script_module("build_terrain_costs_14", SCRIPTS_DIR / "14_build_terrain_costs.py")
MATRICES = load_script_module("build_surveillance_matrices_15", SCRIPTS_DIR / "15_build_surveillance_matrices.py")


def load_raw_inputs(scenario_id: str) -> RawInputs:
    bundle = validate_config_bundle()
    validate_phase1_input_contract(bundle, scenario_id)
    return RawInputs(
        scenario_id=scenario_id,
        scenario=copy.deepcopy(bundle.scenarios_by_id[scenario_id]),
        availability=copy.deepcopy(bundle.availability_by_scenario[scenario_id]),
        asset_types={str(asset["asset_type"]): copy.deepcopy(asset) for asset in bundle.asset_types},
        features=validate_parquet(FEATURES_PATH, ["cell_id", "terrain_class", "dist_to_road_m", "dist_to_camp_m", "dist_to_waterhole_m", "historical_fire_event_count", "geometry"], "grid features"),
        composite=validate_geojson(COMPOSITE_PATH, ["cell_id", "composite_risk_norm", "wildfire_risk_norm"], "composite risk"),
        centroids=validate_geojson(GRID_CENTROIDS_PATH, ["cell_id", "geometry"], "grid centroids"),
        candidate_sites=validate_geojson(
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
                "dist_to_road_m",
                "geometry",
            ],
            "surveillance candidate sites",
        ),
    )


def build_variant_data(raw: RawInputs, spec: ParameterSpec | None, multiplier: float) -> object:
    scenario = copy.deepcopy(raw.scenario)
    availability = copy.deepcopy(raw.availability)
    asset_types = copy.deepcopy(raw.asset_types)
    if spec is not None:
        spec.apply(scenario, availability, asset_types, multiplier)

    candidate_sites = raw.candidate_sites.copy()
    candidate_sites["waterhole_influence_radius_m"] = float(scenario["waterhole_influence_radius_m"])
    selected_sites = OPTIMIZE.select_sites_for_optimization(candidate_sites, scenario, raw.scenario_id)
    terrain = TERRAIN.derive_terrain_costs(raw.features, raw.composite, scenario)
    interventions = MATRICES.build_waterhole_interventions(candidate_sites, terrain, scenario)
    coverage = MATRICES.build_coverage_matrix(selected_sites, terrain, raw.centroids, scenario, asset_types)
    response = MATRICES.build_response_time_matrix(selected_sites, terrain, raw.centroids, availability, asset_types)
    fire_breakpoints = MATRICES.build_fire_delay_breakpoints(response, availability, raw.scenario_id)
    return OPTIMIZE.prepare_optimization_data(
        scenario_id=raw.scenario_id,
        scenario=scenario,
        availability=availability,
        asset_types=asset_types,
        selected_sites=selected_sites,
        terrain=terrain,
        composite=raw.composite,
        interventions=interventions,
        coverage=coverage,
        response=response,
        fire_breakpoints=fire_breakpoints,
    )


def summarize_solution(extracted: dict[str, object], baseline: bool = False) -> dict[str, float | int | str]:
    frontier_row = dict(extracted["frontier_row"])
    cells = extracted["cells"]
    sites = extracted["sites"]
    summary = {
        **frontier_row,
        "mean_response_time_min": float(cells["response_time_min"].mean()),
        "p90_response_time_min": float(cells["response_time_min"].quantile(0.90)),
        "covered_cell_share": float(cells["covered"].mean()),
        "mean_fire_delay_penalty": float(cells["fire_delay_penalty"].mean()),
        "selected_high_risk_sites": int(sites["site_kind"].eq("high_risk_cell").sum()),
        "status": "baseline" if baseline else "ok",
    }
    return summary


def solve_variant(raw: RawInputs, spec: ParameterSpec | None, multiplier: float) -> dict[str, object]:
    parameter_name = "baseline" if spec is None else spec.name
    parameter_label = "baseline" if spec is None else spec.label
    try:
        data = build_variant_data(raw, spec, multiplier)
        coverage_model, _ = OPTIMIZE.build_model(data, mode="coverage")
        OPTIMIZE.solve_model(coverage_model)
        coverage_max = float(OPTIMIZE.pyo.value(coverage_model.protection_expr))
        if coverage_max <= 0:
            raise ValueError("coverage max must stay positive during sensitivity analysis")

        alpha = OPTIMIZE.choose_recommended_alpha(data.alpha_values)
        response_model, _ = OPTIMIZE.build_model(data, mode="response", coverage_floor=alpha * coverage_max)
        OPTIMIZE.solve_model(response_model)
        extracted = OPTIMIZE.extract_solution(data, response_model, alpha=alpha, coverage_max=coverage_max)
        result = summarize_solution(extracted, baseline=(spec is None))
        result.update(
            {
                "parameter_name": parameter_name,
                "parameter_label": parameter_label,
                "parameter_category": "baseline" if spec is None else spec.category,
                "multiplier": float(multiplier),
                "coverage_max": coverage_max,
                "recommended_alpha": alpha,
            }
        )
        return result
    except Exception as exc:  # pragma: no cover - failure reporting path
        return {
            "parameter_name": parameter_name,
            "parameter_label": parameter_label,
            "parameter_category": "baseline" if spec is None else spec.category,
            "multiplier": float(multiplier),
            "status": "error",
            "error": str(exc),
        }


def relative_change(value: float, baseline_value: float, floor: float = 1.0) -> float:
    scale = max(abs(float(baseline_value)), floor)
    return abs(float(value) - float(baseline_value)) / scale


def score_parameter(screening_rows: pd.DataFrame, baseline: pd.Series, baseline_caps: dict[str, float]) -> float:
    successes = screening_rows[screening_rows["status"] == "ok"].copy()
    if successes.empty:
        return -np.inf
    scores = []
    for _, row in successes.iterrows():
        row_score = np.mean(
            [
                relative_change(row["achieved_protection"], baseline["achieved_protection"]),
                relative_change(row["response_objective"], baseline["response_objective"]),
                relative_change(row["covered_cell_share"], baseline["covered_cell_share"], floor=1e-3),
                relative_change(row["selected_site_count"], baseline["selected_site_count"]),
                abs(float(row["selected_people"]) - float(baseline["selected_people"])) / max(baseline_caps["people"], 1.0),
                abs(float(row["selected_drones"]) - float(baseline["selected_drones"])) / max(baseline_caps["drones"], 1.0),
                abs(float(row["selected_cameras"]) - float(baseline["selected_cameras"])) / max(baseline_caps["cameras"], 1.0),
                abs(float(row["selected_interventions"]) - float(baseline["selected_interventions"])) / 12.0,
            ]
        )
        scores.append(row_score)
    return float(np.mean(scores))


def identify_top_parameters(screening: pd.DataFrame, baseline: pd.Series, raw: RawInputs, top_n: int) -> list[str]:
    baseline_caps = {
        "people": float(raw.availability["max_people"]),
        "drones": float(raw.availability["max_drones"]),
        "cameras": float(raw.availability["max_cameras"]),
    }
    scores = []
    for spec in PARAMETER_SPECS:
        scoped = screening[screening["parameter_name"] == spec.name].copy()
        score = score_parameter(scoped, baseline, baseline_caps)
        scores.append({"parameter_name": spec.name, "parameter_label": spec.label, "importance_score": score})
    ranking = pd.DataFrame(scores).sort_values("importance_score", ascending=False)
    return ranking.head(top_n)["parameter_name"].tolist()


def add_percent_changes(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    enriched = frame.copy()
    for metric in ["achieved_protection", "response_objective", "covered_cell_share", "mean_response_time_min"]:
        if metric in enriched.columns:
            enriched[f"{metric}_pct_change"] = 100.0 * (
                enriched[metric].astype(float) - float(baseline[metric])
            ) / max(abs(float(baseline[metric])), 1e-9)
    return enriched


def render_tornado(results: pd.DataFrame, baseline: pd.Series) -> None:
    success = results[results["status"] == "ok"].copy()
    metrics = [
        ("achieved_protection_pct_change", "achieved protection change (%)", False),
        ("response_objective_pct_change", "response objective change (%)", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    multiplier_colors = {0.5: "#c44536", 0.75: "#dd6b4d", 4.0 / 3.0: "#4c956c", 2.0: "#2a6f97"}

    for ax, (metric, title, use_symlog) in zip(axes, metrics):
        grouped = success.groupby("parameter_label")[metric].agg(["min", "max"])
        ordered_labels = grouped.assign(width=lambda frame: frame["max"] - frame["min"]).sort_values("width").index.tolist()
        y_positions = np.arange(len(ordered_labels))
        ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        for y_pos, label in zip(y_positions, ordered_labels):
            scoped = success[success["parameter_label"] == label].copy().sort_values("multiplier")
            ax.hlines(y=y_pos, xmin=scoped[metric].min(), xmax=scoped[metric].max(), color="#8f8f8f", linewidth=3.0)
            for _, row in scoped.iterrows():
                ax.scatter(
                    row[metric],
                    y_pos,
                    color=multiplier_colors[row["multiplier"]],
                    s=60,
                    zorder=3,
                    label=f"{row['multiplier']:.2f}x",
                )
        ax.set_title(title)
        ax.set_xlabel("change from baseline (%)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ordered_labels)
        ax.grid(axis="x", alpha=0.2)
        if use_symlog:
            ax.set_xscale("symlog", linthresh=10.0)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=f"{multiplier:g}x")
        for multiplier, color in multiplier_colors.items()
    ]
    axes[1].legend(handles=handles, title="multiplier", loc="lower right")
    fig.suptitle(f"sensitivity tornado for recommended alpha = {baseline['recommended_alpha']:.2f}")
    fig.tight_layout()
    fig.savefig(SENSITIVITY_TORNADO_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def check_outputs() -> None:
    screening = pd.read_csv(SCENARIO_SCREENING_PATH)
    results = pd.read_csv(SENSITIVITY_RESULTS_PATH)
    summary = json.loads(SENSITIVITY_SUMMARY_PATH.read_text(encoding="utf-8"))
    if screening.empty:
        raise ValueError("sensitivity screening output is empty")
    if results.empty:
        raise ValueError("sensitivity results output is empty")
    if not SENSITIVITY_TORNADO_PATH.is_file() or SENSITIVITY_TORNADO_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty tornado plot: {SENSITIVITY_TORNADO_PATH}")
    if not summary.get("top_parameters"):
        raise ValueError("sensitivity summary is missing top_parameters")
    print("sensitivity analysis output check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="run one-at-a-time sensitivity analysis on the surveillance model")
    parser.add_argument("--scenario-id", default="etosha_placeholder_baseline", help="scenario id to analyze")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="number of parameters to carry from screening into the tornado analysis")
    parser.add_argument("--check", action="store_true", help="validate existing sensitivity outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    raw = load_raw_inputs(args.scenario_id)
    baseline_row = solve_variant(raw, spec=None, multiplier=1.0)
    if baseline_row["status"] != "baseline":
        raise RuntimeError(f"baseline sensitivity solve failed: {baseline_row.get('error', 'unknown error')}")
    baseline = pd.Series(baseline_row)

    screening_rows = []
    for spec in PARAMETER_SPECS:
        for multiplier in SCREENING_MULTIPLIERS:
            print(f"screening {spec.name} at {multiplier:g}x")
            screening_rows.append(solve_variant(raw, spec, multiplier))
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(SCENARIO_SCREENING_PATH, index=False)

    top_parameter_names = identify_top_parameters(screening, baseline, raw, args.top_n)
    selected_specs = [spec for spec in PARAMETER_SPECS if spec.name in top_parameter_names]

    result_rows = []
    for spec in selected_specs:
        for multiplier in FULL_MULTIPLIERS:
            print(f"running {spec.name} at {multiplier:g}x")
            result_rows.append(solve_variant(raw, spec, multiplier))
    results = add_percent_changes(pd.DataFrame(result_rows), baseline)
    results.to_csv(SENSITIVITY_RESULTS_PATH, index=False)
    render_tornado(results, baseline)

    ranking = (
        screening.groupby(["parameter_name", "parameter_label"], as_index=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "importance_score": score_parameter(
                        frame,
                        baseline,
                        {
                            "people": float(raw.availability["max_people"]),
                            "drones": float(raw.availability["max_drones"]),
                            "cameras": float(raw.availability["max_cameras"]),
                        },
                    )
                }
            )
        )
        .reset_index(drop=True)
        .sort_values("importance_score", ascending=False)
    )
    summary = {
        "scenario_id": args.scenario_id,
        "baseline": baseline_row,
        "top_parameters": ranking.head(args.top_n).to_dict(orient="records"),
        "screening_multipliers": SCREENING_MULTIPLIERS,
        "full_multipliers": FULL_MULTIPLIERS,
        "successful_result_count": int((results["status"] == "ok").sum()),
        "error_result_count": int((results["status"] == "error").sum()),
    }
    SENSITIVITY_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
