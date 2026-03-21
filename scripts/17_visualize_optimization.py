#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet
except ModuleNotFoundError:  # pragma: no cover - import style depends on invocation mode
    from scripts._spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet


COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
CANDIDATE_SITES_PATH = PROCESSED_DIR / "surveillance_candidate_sites.geojson"
INTERVENTIONS_PATH = PROCESSED_DIR / "waterhole_interventions.geojson"
SOLUTION_PATH = OUTPUTS_DIR / "optimization_solution.geojson"
CELLS_PATH = OUTPUTS_DIR / "optimization_cells.parquet"
FRONTIER_PATH = OUTPUTS_DIR / "optimization_frontier.csv"
SUMMARY_PATH = OUTPUTS_DIR / "optimization_summary.json"
FRONTIER_PNG_PATH = OUTPUTS_DIR / "optimization_frontier.png"
DIAGNOSTICS_PNG_PATH = OUTPUTS_DIR / "optimization_diagnostics.png"
INTERACTIVE_MAP_PATH = OUTPUTS_DIR / "optimization_map.html"
BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"


def load_inputs() -> dict[str, object]:
    return {
        "composite": validate_geojson(
            COMPOSITE_PATH,
            ["cell_id", "composite_risk_norm"],
            "composite risk layer",
        ),
        "boundary": validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "boundary"),
        "candidate_sites": validate_geojson(
            CANDIDATE_SITES_PATH,
            ["site_id", "site_kind", "candidate_rank", "supports_cameras"],
            "candidate sites",
        ),
        "solution_sites": validate_geojson(
            SOLUTION_PATH,
            ["site_id", "selected", "people_count", "car_count", "drone_count", "camera_count", "camera_lockdown"],
            "optimization solution",
        ),
        "cells": validate_parquet(
            CELLS_PATH,
            [
                "cell_id",
                "covered",
                "response_time_min",
                "protection_benefit_base",
                "protection_benefit_effective",
                "camera_gain_applied",
                "intervention_gain_applied",
                "fire_delay_penalty",
                "geometry",
            ],
            "optimization cells",
        ),
        "interventions": validate_geojson(
            INTERVENTIONS_PATH,
            ["intervention_site_id", "capital_cost", "tourism_cost", "geometry"],
            "waterhole interventions",
        ),
        "frontier": pd.read_csv(FRONTIER_PATH),
        "summary": json.loads(SUMMARY_PATH.read_text(encoding="utf-8")),
    }


def render_frontier(frontier: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frontier["achieved_protection"], frontier["response_objective"], marker="o", color="#b23a48")
    for _, row in frontier.iterrows():
        ax.annotate(f"alpha={row['alpha']:.2f}", (row["achieved_protection"], row["response_objective"]), fontsize=8)
    ax.set_xlabel("achieved protection")
    ax.set_ylabel("response objective")
    ax.set_title("coverage-response frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FRONTIER_PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_diagnostics(inputs: dict[str, object]) -> None:
    composite = inputs["composite"]
    cells = inputs["cells"]
    solution_sites = inputs["solution_sites"]
    interventions = inputs["interventions"]
    boundary = inputs["boundary"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    composite.plot(ax=axes[0, 0], column="composite_risk_norm", cmap="YlOrRd", linewidth=0.05, edgecolor="#d9d9d9")
    boundary.boundary.plot(ax=axes[0, 0], color="black", linewidth=1.0)
    solution_sites.plot(ax=axes[0, 0], color="#0b6e4f", markersize=28)
    axes[0, 0].set_title("selected deployment sites on composite risk")
    axes[0, 0].set_axis_off()

    cells.plot(ax=axes[0, 1], column="protection_benefit_base", cmap="YlGn", linewidth=0.05, edgecolor="#d9d9d9")
    boundary.boundary.plot(ax=axes[0, 1], color="black", linewidth=1.0)
    axes[0, 1].set_title("baseline protection benefit")
    axes[0, 1].set_axis_off()

    cells.plot(ax=axes[1, 0], column="protection_benefit_effective", cmap="YlGn", linewidth=0.05, edgecolor="#d9d9d9")
    boundary.boundary.plot(ax=axes[1, 0], color="black", linewidth=1.0)
    interventions.plot(ax=axes[1, 0], color="#1d4e89", markersize=18, marker="^")
    axes[1, 0].set_title("post-camera and intervention protection")
    axes[1, 0].set_axis_off()

    cells.plot(ax=axes[1, 1], column="fire_delay_penalty", cmap="OrRd", linewidth=0.05, edgecolor="#d9d9d9")
    boundary.boundary.plot(ax=axes[1, 1], color="black", linewidth=1.0)
    axes[1, 1].set_title("wildfire delay penalty")
    axes[1, 1].set_axis_off()

    fig.tight_layout()
    fig.savefig(DIAGNOSTICS_PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def site_popup(row: pd.Series) -> str:
    return (
        f"{row['site_id']}<br>"
        f"people={row['people_count']} car={row['car_count']} drone={row['drone_count']} camera={row['camera_count']}<br>"
        f"lockdown={bool(row['camera_lockdown'])}"
    )


def render_interactive(inputs: dict[str, object]) -> None:
    boundary = inputs["boundary"]
    candidate_sites = inputs["candidate_sites"]
    solution_sites = inputs["solution_sites"]
    cells = inputs["cells"]
    interventions = inputs["interventions"]
    summary = inputs["summary"]

    bounds = boundary.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    response_scale = bcm.linear.YlOrRd_09.scale(
        float(cells["response_time_min"].min()),
        float(cells["response_time_min"].quantile(0.95)),
    )
    response_scale.caption = "response time (min)"
    response_scale.add_to(fmap)

    folium.GeoJson(
        cells.to_json(),
        name="response cells",
        style_function=lambda feature: {
            "fillColor": response_scale(min(feature["properties"]["response_time_min"], response_scale.vmax)),
            "color": "#666666",
            "weight": 0.15,
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "cell_id",
                "covered",
                "response_time_min",
                "protection_benefit_base",
                "protection_benefit_effective",
                "camera_gain_applied",
                "intervention_gain_applied",
                "fire_delay_penalty",
            ],
            aliases=["cell", "covered", "response min", "base protection", "effective protection", "camera gain", "intervention gain", "fire penalty"],
        ),
    ).add_to(fmap)

    folium.GeoJson(
        boundary.to_json(),
        name="boundary",
        style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0},
    ).add_to(fmap)

    candidate_group = folium.FeatureGroup(name="candidate sites")
    for _, row in candidate_sites.iterrows():
        geom = row.geometry
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=2,
            color="#7a7a7a",
            fill=True,
            fill_opacity=0.5,
            popup=f"{row['site_id']} ({row['site_kind']})",
        ).add_to(candidate_group)
    candidate_group.add_to(fmap)

    selected_group = folium.FeatureGroup(name="selected sites")
    for _, row in solution_sites.iterrows():
        geom = row.geometry
        color = "#0b6e4f"
        if row["camera_count"] > 0:
            color = "#1d4e89"
        elif row["drone_count"] > 0:
            color = "#7f5539"
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=site_popup(row),
        ).add_to(selected_group)
    selected_group.add_to(fmap)

    intervention_group = folium.FeatureGroup(name="interventions")
    for _, row in interventions.iterrows():
        geom = row.geometry.centroid
        folium.Marker(
            location=[geom.y, geom.x],
            popup=f"{row['intervention_site_id']} capital={row['capital_cost']} tourism={row['tourism_cost']}",
            icon=folium.Icon(color="purple", icon="tint", prefix="fa"),
        ).add_to(intervention_group)
    intervention_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    folium.map.Marker(
        center,
        icon=folium.DivIcon(
            html=f"<div style='font-size:12px;background:white;padding:6px;border:1px solid #bbb;'>recommended alpha={summary['recommended_alpha']}</div>"
        ),
    ).add_to(fmap)
    fmap.save(str(INTERACTIVE_MAP_PATH))


def check_outputs() -> None:
    for path in [FRONTIER_PNG_PATH, DIAGNOSTICS_PNG_PATH, INTERACTIVE_MAP_PATH]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty visualization output: {path}")
    html = INTERACTIVE_MAP_PATH.read_text(encoding="utf-8")
    if "leaflet" not in html.lower():
        raise ValueError("optimization map does not look renderable")
    print("optimization visualization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="render optimization diagnostics and interactive map")
    parser.add_argument("--check", action="store_true", help="validate existing optimization visualizations")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    inputs = load_inputs()
    render_frontier(inputs["frontier"])
    render_diagnostics(inputs)
    render_interactive(inputs)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
