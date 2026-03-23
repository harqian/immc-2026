#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet


SUMMARY_PATH = OUTPUTS_DIR / "optimization_summary.json"
FRONTIER_PATH = OUTPUTS_DIR / "optimization_frontier.csv"
SOLUTION_PATH = OUTPUTS_DIR / "optimization_solution.geojson"
CELLS_PATH = OUTPUTS_DIR / "optimization_cells.parquet"
COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
ASSET_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data/configs/asset_types.yaml"
AVAILABILITY_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data/configs/daily_asset_availability.yaml"
SCENARIO_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data/configs/optimization_scenarios.yaml"

COMPONENTS_DIR = OUTPUTS_DIR / "optimization_components"
MODEL_FIG_PATH = COMPONENTS_DIR / "01_model_structure.png"
FRONTIER_FIG_PATH = COMPONENTS_DIR / "02_frontier_tradeoff.png"
RESOURCE_FIG_PATH = COMPONENTS_DIR / "03_resource_summary.png"
SPATIAL_FIG_PATH = COMPONENTS_DIR / "04_spatial_solution.png"
METRICS_FIG_PATH = COMPONENTS_DIR / "05_cell_metrics.png"
MARKDOWN_PATH = COMPONENTS_DIR / "optimization_components.md"

ASSET_ORDER = ["car", "drone", "camera"]
ASSET_LABELS = {"car": "cars", "drone": "drones", "camera": "cameras"}
RESPONDER_ORDER = ["car", "drone"]
RESPONDER_LABELS = {
    "car": "car",
    "drone": "drone",
}
RESPONDER_COLORS = {"car": "#2b9348", "drone": "#bc6c25"}


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a top-level mapping")
    return loaded


def build_context() -> dict[str, object]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    scenario_id = str(summary["scenario_id"])
    frontier = pd.read_csv(FRONTIER_PATH).sort_values("alpha", ascending=False).reset_index(drop=True)
    solution = validate_geojson(
        SOLUTION_PATH,
        ["site_id", "site_kind", "car_count", "drone_count", "camera_count", "geometry"],
        "optimization solution",
    )
    cells = validate_parquet(
        CELLS_PATH,
        [
            "cell_id",
            "covered",
            "response_time_min",
            "fire_delay_penalty",
            "protection_benefit_base",
            "protection_benefit_effective",
            "selected_responder_asset",
            "composite_risk",
            "wildfire_risk_norm",
            "geometry",
        ],
        "optimization cells",
    )
    composite = validate_geojson(
        COMPOSITE_PATH,
        ["cell_id", "composite_risk_norm", "geometry"],
        "composite risk layer",
    )
    boundary = validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "boundary")

    asset_config = load_yaml(ASSET_CONFIG_PATH)
    availability_config = load_yaml(AVAILABILITY_CONFIG_PATH)
    scenario_config = load_yaml(SCENARIO_CONFIG_PATH)

    asset_types = {
        str(row["asset_type"]): row
        for row in asset_config.get("asset_types", [])
        if isinstance(row, dict) and row.get("asset_type")
    }
    availability_by_scenario = {
        str(row["scenario_id"]): row
        for row in availability_config.get("scenarios", [])
        if isinstance(row, dict) and row.get("scenario_id")
    }
    scenario_by_id = {
        str(row["scenario_id"]): row
        for row in scenario_config.get("scenarios", [])
        if isinstance(row, dict) and row.get("scenario_id")
    }

    chosen = summary["chosen_solution"]
    selected_units = {
        "car": int(chosen["selected_cars"]),
        "drone": int(chosen["selected_drones"]),
        "camera": int(chosen["selected_cameras"]),
    }
    solve_caps = {asset: int(summary["available_caps"][ASSET_LABELS[asset]]) for asset in ASSET_ORDER}
    current_availability = availability_by_scenario.get(scenario_id)
    current_caps = None
    current_included = None
    if current_availability is not None:
        current_caps = {
            "car": int(current_availability["max_cars"]),
            "drone": int(current_availability["max_drones"]),
            "camera": int(current_availability["max_cameras"]),
        }
        current_included = {
            "car": int(current_availability["included_cars"]),
            "drone": int(current_availability["included_drones"]),
            "camera": int(current_availability["included_cameras"]),
        }

    selected_site_kind_counts = solution["site_kind"].value_counts().to_dict()
    responder_counts = cells["selected_responder_asset"].value_counts().to_dict()
    coverage_counts = cells["covered"].value_counts().to_dict()
    response_stats = cells["response_time_min"].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
    gain_delta = cells["protection_benefit_effective"] - cells["protection_benefit_base"]
    resource_table = pd.DataFrame(
        {
            "asset_type": ASSET_ORDER,
            "selected_units": [selected_units[asset] for asset in ASSET_ORDER],
            "solve_time_cap": [solve_caps[asset] for asset in ASSET_ORDER],
            "unit_cost": [float(asset_types[asset]["unit_cost"]) for asset in ASSET_ORDER],
        }
    )
    if current_caps is not None:
        resource_table["current_config_cap"] = [current_caps[asset] for asset in ASSET_ORDER]
    if current_included is not None:
        resource_table["current_included_baseline"] = [current_included[asset] for asset in ASSET_ORDER]

    return {
        "scenario_id": scenario_id,
        "summary": summary,
        "chosen": chosen,
        "frontier": frontier,
        "solution": solution,
        "cells": cells,
        "composite": composite,
        "boundary": boundary,
        "asset_types": asset_types,
        "scenario": scenario_by_id.get(scenario_id),
        "current_availability": current_availability,
        "selected_units": selected_units,
        "solve_caps": solve_caps,
        "current_caps": current_caps,
        "current_included": current_included,
        "selected_site_kind_counts": selected_site_kind_counts,
        "responder_counts": responder_counts,
        "coverage_counts": coverage_counts,
        "response_stats": response_stats,
        "gain_delta": gain_delta,
        "resource_table": resource_table,
    }


def compare_caps(context: dict[str, object]) -> list[dict[str, object]]:
    current_caps = context["current_caps"]
    if current_caps is None:
        return []
    rows: list[dict[str, object]] = []
    for asset in ASSET_ORDER:
        solve_cap = int(context["solve_caps"][asset])
        current_cap = int(current_caps[asset])
        if solve_cap != current_cap:
            rows.append(
                {
                    "asset_type": asset,
                    "solve_time_cap": solve_cap,
                    "current_config_cap": current_cap,
                }
            )
    return rows


def ordered_responder_series(context: dict[str, object]) -> pd.Series:
    responder_counts = pd.Series(context["responder_counts"], dtype=float)
    responder_counts = responder_counts.reindex([a for a in RESPONDER_ORDER if a in responder_counts.index]).fillna(0)
    return responder_counts


def summarize_responder_counts(context: dict[str, object]) -> str:
    ordered = ordered_responder_series(context)
    return ", ".join(f"{RESPONDER_LABELS.get(asset, asset)}={int(count)}" for asset, count in ordered.items())


def summarize_selected_assets(chosen: dict[str, object]) -> str:
    parts = []
    asset_values = [
        ("cars", int(chosen["selected_cars"])),
        ("drones", int(chosen["selected_drones"])),
        ("cameras", int(chosen["selected_cameras"])),
    ]
    for label, value in asset_values:
        if value > 0:
            parts.append(f"{value} {label}")
    if not parts:
        return "no deployable assets selected"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def add_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, body: str, facecolor: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor="#2f2f2f",
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + 0.02, xy[1] + height - 0.05, title, fontsize=12, fontweight="bold", va="top")
    ax.text(xy[0] + 0.02, xy[1] + height - 0.12, body, fontsize=10, va="top", linespacing=1.45)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#3f3f46")
    ax.add_patch(arrow)


def render_model_structure(context: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    chosen = context["chosen"]
    scenario = context["scenario"] or {}
    coverage_true = int(context["coverage_counts"].get(True, 0))
    coverage_false = int(context["coverage_counts"].get(False, 0))
    box_specs = [
        (
            (0.04, 0.60),
            0.24,
            0.27,
            "1. scenario inputs",
            "\n".join(
                [
                    f"scenario_id: {context['scenario_id']}",
                    f"active assets: {', '.join(scenario.get('active_asset_types', [])) or 'unknown'}",
                    f"alpha values: {', '.join(f'{value:.2f}' for value in context['frontier']['alpha'])}",
                    f"solve-time budget: {context['summary']['available_budget']:.0f}",
                    f"solve-time caps: {context['summary']['available_caps']}",
                ]
            ),
            "#fef3c7",
        ),
        (
            (0.38, 0.60),
            0.24,
            0.27,
            "2. prepared data",
            "\n".join(
                [
                    f"cells: {len(context['cells'])}",
                    f"selected sites in artifact: {len(context['solution'])}",
                    f"frontier points solved: {len(context['frontier'])}",
                    f"covered / uncovered cells: {coverage_true} / {coverage_false}",
                    f"recommended alpha: {context['summary']['recommended_alpha']:.2f}",
                ]
            ),
            "#dbeafe",
        ),
        (
            (0.72, 0.60),
            0.24,
            0.27,
            "3. decision variables",
            "\n".join(
                [
                    "x[site, asset]: unit counts",
                    "asset_active / site_active: binary deployment flags",
                    "y[cell]: coverage achieved",
                    "z[arc], t[cell]: responder assignment and travel time",
                    "u[intervention], lockdown[site]: optional upgrades",
                ]
            ),
            "#dcfce7",
        ),
        (
            (0.18, 0.18),
            0.28,
            0.27,
            "4. constraints",
            "\n".join(
                [
                    "budget counts only units above included baselines",
                    "total selected units must stay below max caps",
                    "cameras deploy only at waterholes in bundles",
                    "each cell gets a responder arc assignment",
                    "fire penalty is piecewise-linear in response delay",
                ]
            ),
            "#fae8ff",
        ),
        (
            (0.54, 0.18),
            0.34,
            0.27,
            "5. frontier solve",
            "\n".join(
                [
                    "stage a: maximize protection_expr to get coverage_max",
                    "stage b: for each alpha, minimize response_expr",
                    "extra frontier constraint: protection_expr >= alpha * coverage_max",
                    f"chosen artifact: alpha={chosen['alpha']:.2f}, protection={chosen['achieved_protection']:.3f}, response={chosen['response_objective']:.1f}",
                ]
            ),
            "#fee2e2",
        ),
    ]

    for xy, width, height, title, body, facecolor in box_specs:
        add_box(ax, xy, width, height, title, body, facecolor)

    add_arrow(ax, (0.28, 0.735), (0.38, 0.735))
    add_arrow(ax, (0.62, 0.735), (0.72, 0.735))
    add_arrow(ax, (0.50, 0.60), (0.34, 0.45))
    add_arrow(ax, (0.78, 0.60), (0.70, 0.45))

    fig.suptitle("how the surveillance optimization is assembled", fontsize=17, fontweight="bold", y=0.96)
    fig.savefig(MODEL_FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_frontier(context: dict[str, object]) -> None:
    frontier = context["frontier"]
    chosen_alpha = float(context["summary"]["recommended_alpha"])
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.plot(frontier["achieved_protection"], frontier["response_objective"], marker="o", linewidth=2.2, color="#b45309")
    chosen_row = frontier.loc[np.isclose(frontier["alpha"], chosen_alpha)].iloc[0]

    for _, row in frontier.iterrows():
        marker_size = 110 if np.isclose(row["alpha"], chosen_alpha) else 55
        color = "#7c2d12" if np.isclose(row["alpha"], chosen_alpha) else "#f59e0b"
        ax.scatter(row["achieved_protection"], row["response_objective"], s=marker_size, color=color, zorder=3)
        ax.annotate(
            f"alpha={row['alpha']:.2f}",
            (row["achieved_protection"], row["response_objective"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    x_achieved = float(chosen_row["achieved_protection"])
    y_chosen = float(chosen_row["response_objective"])
    ax.annotate(
        f"chosen point\nalpha={chosen_alpha:.2f}",
        (x_achieved, y_chosen),
        textcoords="offset points",
        xytext=(10, -32),
        fontsize=10,
        color="#7c2d12",
    )
    ax.set_xlabel("achieved protection")
    ax.set_ylabel("response objective")
    ax.set_title("frontier tradeoff used to choose the recommended solution")
    ax.grid(alpha=0.25)
    fig.savefig(FRONTIER_FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_resource_summary(context: dict[str, object]) -> None:
    chosen = context["chosen"]
    solution = context["solution"]
    resource_table = context["resource_table"]
    site_kind_counts = pd.Series(context["selected_site_kind_counts"]).sort_values(ascending=False)
    responder_counts = ordered_responder_series(context)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    x = np.arange(len(ASSET_ORDER))
    ax.bar(x - 0.18, resource_table["selected_units"], width=0.36, color="#15803d", label="selected units")
    ax.bar(x + 0.18, resource_table["solve_time_cap"], width=0.36, color="#93c5fd", label="solve-time cap")
    ax.set_xticks(x)
    ax.set_xticklabels([ASSET_LABELS[asset] for asset in ASSET_ORDER])
    ax.set_ylabel("units")
    ax.set_title("selected assets versus solve-time caps")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    ax.bar(site_kind_counts.index, site_kind_counts.values, color="#7c3aed")
    ax.set_ylabel("selected sites")
    ax.set_title("selected site mix")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    ax.bar(responder_counts.index, responder_counts.values, color=[RESPONDER_COLORS[key] for key in responder_counts.index])
    ax.set_ylabel("cells assigned")
    ax.set_title("which responder actually serves each cell")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    ax.axis("off")
    current_caps = context["current_caps"]
    current_included = context["current_included"]
    lines = [
        f"budget used in chosen artifact: {chosen['budget_used']:.0f} / {context['summary']['available_budget']:.0f}",
        f"selected site count: {chosen['selected_site_count']}",
        f"selected interventions: {chosen['selected_interventions']}",
        f"locked down waterholes: {chosen['locked_down_waterholes']}",
        f"mean response time across cells: {context['cells']['response_time_min'].mean():.1f} min",
        f"median response time across cells: {context['cells']['response_time_min'].median():.1f} min",
    ]
    if current_caps is not None and current_included is not None:
        lines.extend(
            [
                "",
                "current config snapshot:",
                f"caps: {current_caps}",
                f"included baselines: {current_included}",
            ]
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    ax.set_title("resource and artifact summary")

    fig.tight_layout()
    fig.savefig(RESOURCE_FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_spatial_solution(context: dict[str, object]) -> None:
    composite = context["composite"]
    cells = context["cells"]
    solution = context["solution"]
    boundary = context["boundary"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    composite.plot(ax=axes[0, 0], column="composite_risk_norm", cmap="YlOrRd", linewidth=0.05, edgecolor="#d4d4d8")
    boundary.boundary.plot(ax=axes[0, 0], color="black", linewidth=1.0)
    solution.plot(ax=axes[0, 0], color="#0f766e", markersize=18)
    axes[0, 0].set_title("selected sites over composite risk")
    axes[0, 0].set_axis_off()

    coverage_colors = cells["covered"].map({True: "#15803d", False: "#d4d4d8"})
    cells.plot(ax=axes[0, 1], color=coverage_colors, linewidth=0.02, edgecolor="#f5f5f5")
    boundary.boundary.plot(ax=axes[0, 1], color="black", linewidth=1.0)
    axes[0, 1].set_title("coverage decision y[cell]")
    axes[0, 1].set_axis_off()
    axes[0, 1].legend(
        handles=[
            Line2D([0], [0], marker="s", color="w", label="covered", markerfacecolor="#15803d", markersize=10),
            Line2D([0], [0], marker="s", color="w", label="not covered", markerfacecolor="#d4d4d8", markersize=10),
        ],
        loc="lower left",
        frameon=True,
    )

    cells.plot(ax=axes[1, 0], column="response_time_min", cmap="YlGnBu", linewidth=0.02, edgecolor="#f5f5f5")
    boundary.boundary.plot(ax=axes[1, 0], color="black", linewidth=1.0)
    axes[1, 0].set_title("cell response times after responder assignment")
    axes[1, 0].set_axis_off()
    sm = plt.cm.ScalarMappable(
        cmap="YlGnBu",
        norm=plt.Normalize(
            vmin=cells["response_time_min"].min(),
            vmax=cells["response_time_min"].max(),
        ),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes[1, 0], fraction=0.03, pad=0.01, shrink=0.85)
    cbar.set_label("response time (min)")

    responder_palette = {asset: RESPONDER_COLORS[asset] for asset in RESPONDER_COLORS}
    responder_colors = cells["selected_responder_asset"].map(responder_palette).fillna("#d4d4d8")
    cells.plot(ax=axes[1, 1], color=responder_colors, linewidth=0.02, edgecolor="#f5f5f5")
    boundary.boundary.plot(ax=axes[1, 1], color="black", linewidth=1.0)
    axes[1, 1].set_title("which asset type is assigned to each cell")
    axes[1, 1].set_axis_off()
    axes[1, 1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=RESPONDER_LABELS.get(key, key),
                markerfacecolor=RESPONDER_COLORS[key],
                markersize=10,
            )
            for key in ordered_responder_series(context).index
        ],
        loc="lower left",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(SPATIAL_FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_cell_metrics(context: dict[str, object]) -> None:
    cells = context["cells"]
    gain_delta = context["gain_delta"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    scatter = axes[0, 0].scatter(
        cells["composite_risk"],
        cells["response_time_min"],
        c=cells["covered"].map({True: 1, False: 0}),
        cmap="viridis",
        alpha=0.65,
        s=np.clip(cells["wildfire_risk_norm"].fillna(0) * 180 + 12, 12, 150),
        edgecolors="none",
    )
    axes[0, 0].set_xlabel("composite risk")
    axes[0, 0].set_ylabel("response time (min)")
    axes[0, 0].set_title("response burden versus risk")
    axes[0, 0].grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=axes[0, 0], fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["not covered", "covered"])

    axes[0, 1].hist(gain_delta, bins=30, color="#7c3aed", alpha=0.85)
    axes[0, 1].set_title("incremental protection from cameras and interventions")
    axes[0, 1].set_xlabel("effective protection - base protection")
    axes[0, 1].set_ylabel("cells")
    axes[0, 1].grid(axis="y", alpha=0.2)

    axes[1, 0].hist(cells["fire_delay_penalty"], bins=30, color="#dc2626", alpha=0.82)
    axes[1, 0].set_title("fire delay penalty distribution")
    axes[1, 0].set_xlabel("fire delay penalty")
    axes[1, 0].set_ylabel("cells")
    axes[1, 0].grid(axis="y", alpha=0.2)

    frontier = context["frontier"]
    axes[1, 1].plot(frontier["alpha"], frontier["selected_site_count"], marker="o", color="#0369a1", label="selected sites")
    axes[1, 1].plot(frontier["alpha"], frontier["selected_cars"], marker="o", color="#2b9348", label="cars")
    axes[1, 1].plot(frontier["alpha"], frontier["selected_drones"], marker="o", color="#bc6c25", label="drones")
    axes[1, 1].plot(frontier["alpha"], frontier["selected_cameras"], marker="o", color="#7c3aed", label="cameras")
    axes[1, 1].invert_xaxis()
    axes[1, 1].set_title("how frontier points change resource usage")
    axes[1, 1].set_xlabel("alpha")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(METRICS_FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, divider, *rows])


def write_markdown(context: dict[str, object]) -> None:
    chosen = context["chosen"]
    frontier = context["frontier"]
    scenario = context["scenario"] or {}
    compare_rows = compare_caps(context)
    response_stats = context["response_stats"]
    coverage_counts = context["coverage_counts"]
    responder_counts = context["responder_counts"]

    frontier_table = frontier[
        [
            "alpha",
            "coverage_target",
            "achieved_protection",
            "response_objective",
            "budget_used",
            "selected_site_count",
            "selected_cars",
            "selected_drones",
            "selected_cameras",
            "selected_interventions",
        ]
    ].copy()
    resource_table = context["resource_table"].copy()
    resource_table["asset_type"] = resource_table["asset_type"].map(ASSET_LABELS)
    if "current_config_cap" in resource_table.columns:
        resource_table = resource_table[
            ["asset_type", "selected_units", "solve_time_cap", "current_config_cap", "current_included_baseline", "unit_cost"]
        ]
    mismatch_block = ""
    if compare_rows:
        mismatch_frame = pd.DataFrame(compare_rows)
        mismatch_frame["asset_type"] = mismatch_frame["asset_type"].map(ASSET_LABELS)
        mismatch_block = "\n".join(
            [
                "## config/output mismatch warning",
                "",
                "the current `daily_asset_availability.yaml` does not match the solved artifact embedded in `optimization_summary.json`.",
                "that means the figures below explain the current optimization outputs on disk, not a freshly re-solved run of the live config.",
                "",
                markdown_table(mismatch_frame),
                "",
            ]
        )

    md = f"""# optimization components walkthrough

this document is generated by `scripts/visualize_optimization_components.py`.
it explains how the optimization in `scripts/16_optimize_surveillance.py` works using the current output artifacts in `outputs/`.

{mismatch_block}## what problem the model is solving

the optimization is a two-stage mixed-integer model over Etosha surveillance deployment.
it first finds the maximum achievable protection score, then solves a response-minimization problem at multiple `alpha` levels while forcing protection to stay above a fraction of that maximum.

- scenario id: `{context["scenario_id"]}`
- scenario description: `{scenario.get("description", "unknown")}`
- active asset types: `{", ".join(scenario.get("active_asset_types", []))}`
- alpha values: `{", ".join(f"{value:.2f}" for value in frontier["alpha"])}`
- solve-time budget from artifact: `{context["summary"]["available_budget"]:.0f}`
- recommended alpha: `{context["summary"]["recommended_alpha"]:.2f}`

![model structure](01_model_structure.png)

## the main data objects

the model works over several linked object sets:

- sites: candidate deployment locations chosen earlier in the pipeline
- site-asset pairs: only site/asset combinations that are physically eligible
- cells: the gridded landscape units being protected and responded to
- response arcs: the top few feasible site-to-cell responder routes
- interventions: optional waterhole actions with capital cost and tourism penalty
- waterhole camera sites: the subset of sites that can host camera lockdown bundles

for the current artifact:

- cells in optimization output: `{len(context["cells"])}`
- selected sites in chosen solution artifact: `{len(context["solution"])}`
- covered cells: `{int(coverage_counts.get(True, 0))}`
- uncovered cells: `{int(coverage_counts.get(False, 0))}`
- responder assignment counts: `{summarize_responder_counts(context)}`

## decision variables and what they mean

the important variables in the Pyomo model are:

- `x[site, asset]`: how many units of an asset to place at a site
- `asset_active[site, asset]`: whether that site/asset combination is turned on
- `site_active[site]`: whether a site incurs its fixed activation cost
- `y[cell]`: whether a cell is considered covered
- `z[arc]`: which real responder arc is assigned to a cell
- `t[cell]`: realized response time for the cell
- `fire_lambda[cell, breakpoint]` and `fire_penalty[cell]`: the piecewise-linear wildfire delay approximation
- `u[intervention]`: whether a waterhole intervention is purchased
- `lockdown[site]`: whether a camera bundle is activated at an eligible waterhole site

## key constraints

the model structure matters more than the exact coefficients:

1. site/asset linkage:
   `x` is upper-bounded by `max_units_per_site * asset_active`, and mobile assets also have a lower link so an active asset implies at least one unit.
2. camera bundles:
   cameras are not free-form counts. if a waterhole is locked down, the camera count is forced to exactly `camera_bundle_size * lockdown`.
3. budget:
   the budget includes fixed site activation cost, capital cost for interventions, and only the asset units above the configured `included_*` baseline.
4. asset caps:
   total units by asset type must stay below `max_cars`, `max_drones`, and `max_cameras`.
5. coverage feasibility:
   a cell can only be marked covered if at least one eligible mobile asset is active on one of the site-asset pairs that covers it.
6. response assignment:
   every cell is assigned a response arc.
7. response-time interpolation:
   `t[cell]` equals the selected response arc time.
8. wildfire penalty interpolation:
   `fire_lambda` forms a simplex over breakpoints so the model can linearly represent the fire delay penalty curve.

## objective functions

the model uses two objective expressions:

- `protection_expr`
  - base protection for covered cells
  - plus camera gain at locked-down waterholes
  - plus intervention protection gain

- `response_expr`
  - composite-risk-weighted response time
  - plus `lambda_fire * wildfire_risk * fire_delay_penalty`
  - plus tourism penalty from selected interventions

the frontier algorithm is:

1. maximize `protection_expr` with no coverage floor to get `coverage_max`
2. for each `alpha`, solve a second model minimizing `response_expr`
3. add the frontier constraint `protection_expr >= alpha * coverage_max`
4. keep the recommended point at the configured alpha index

![frontier tradeoff](02_frontier_tradeoff.png)

## frontier results from the current artifact

{markdown_table(frontier_table)}

the chosen artifact is the `alpha={chosen["alpha"]:.2f}` point:

- achieved protection: `{chosen["achieved_protection"]:.3f}`
- response objective: `{chosen["response_objective"]:.3f}`
- budget used: `{chosen["budget_used"]:.1f}`
- selected sites: `{chosen["selected_site_count"]}`
- selected cars: `{chosen["selected_cars"]}`
- selected drones: `{chosen["selected_drones"]}`
- selected cameras: `{chosen["selected_cameras"]}`
- selected interventions: `{chosen["selected_interventions"]}`

## resource interpretation

the table below combines the chosen artifact counts with solve-time caps from `optimization_summary.json`.
if the current config is different, that difference is shown explicitly so you can tell whether you are looking at stale outputs.

{markdown_table(resource_table)}

notes:

- `selected_units` comes from the chosen frontier point in the output artifact.
- `solve_time_cap` comes from the saved summary generated by the solve that produced the artifact.
- `current_config_cap` and `current_included_baseline` come from the live `daily_asset_availability.yaml` in the workspace right now.
- `unit_cost` comes from `asset_types.yaml`.
- the exact budget decomposition cannot always be reconstructed from the output artifact alone because the summary saves caps but not all included baseline values used at solve time.

![resource summary](03_resource_summary.png)

## spatial interpretation of the chosen solution

the optimization does not just choose counts; it chooses where those resources sit and which cells they actually serve.

- the upper-left panel shows selected sites on top of the composite risk field.
- the upper-right panel shows which cells satisfied the binary coverage logic.
- the lower-left panel shows realized response time after the responder assignment step.
- the lower-right panel shows which asset type serves each cell.

![spatial solution](04_spatial_solution.png)

## cell-level behavior

the cell diagnostics highlight how the objective is assembled:

- high-risk cells with long response times are expensive in `response_expr`
- camera and intervention gains raise effective protection above base protection
- fire delay penalty adds extra cost when response times cross the configured threshold region
- different alpha points change the selected site count and the car/drone/camera mix

artifact summary statistics:

- response time min / p25 / median / p75 / max:
  - `{response_stats["min"]:.2f}` / `{response_stats["25%"]:.2f}` / `{response_stats["50%"]:.2f}` / `{response_stats["75%"]:.2f}` / `{response_stats["max"]:.2f}`
- mean incremental protection gain:
  - `{context["gain_delta"].mean():.4f}`
- max incremental protection gain:
  - `{context["gain_delta"].max():.4f}`

![cell metrics](05_cell_metrics.png)

## what this walkthrough says about the current solve

- the artifact is clearly a frontier solve, not a single-objective solve: response gets much better when alpha drops from `1.00` to `0.95`, while protection only drops slightly.
- the chosen artifact is driven by `{summarize_selected_assets(chosen)}`.
- because the current config and current output artifact disagree on some caps, this walkthrough should be read as an explanation of the saved artifact, not as proof that the current config would reproduce the same answer.

## how to regenerate

run:

```bash
./.venv/bin/python scripts/visualize_optimization_components.py
```

this will regenerate:

- `outputs/optimization_components/01_model_structure.png`
- `outputs/optimization_components/02_frontier_tradeoff.png`
- `outputs/optimization_components/03_resource_summary.png`
- `outputs/optimization_components/04_spatial_solution.png`
- `outputs/optimization_components/05_cell_metrics.png`
- `outputs/optimization_components/optimization_components.md`
"""
    MARKDOWN_PATH.write_text(md, encoding="utf-8")


def check_outputs() -> None:
    required = [
        MODEL_FIG_PATH,
        FRONTIER_FIG_PATH,
        RESOURCE_FIG_PATH,
        SPATIAL_FIG_PATH,
        METRICS_FIG_PATH,
        MARKDOWN_PATH,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty output: {path}")
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    for image_name in [
        MODEL_FIG_PATH.name,
        FRONTIER_FIG_PATH.name,
        RESOURCE_FIG_PATH.name,
        SPATIAL_FIG_PATH.name,
        METRICS_FIG_PATH.name,
    ]:
        if image_name not in markdown:
            raise ValueError(f"markdown does not reference {image_name}")
    print("optimization component visualization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="generate optimization explainer figures and markdown")
    parser.add_argument("--check", action="store_true", help="validate existing optimization component outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
    context = build_context()
    render_model_structure(context)
    render_frontier(context)
    render_resource_summary(context)
    render_spatial_solution(context)
    render_cell_metrics(context)
    write_markdown(context)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
