#!/usr/bin/env python3
from __future__ import annotations

import argparse

import branca.colormap as bcm
import folium
import geopandas as gpd
import matplotlib.pyplot as plt

from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet


COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
THREATS_PATH = OUTPUTS_DIR / "threat_layers.parquet"
HEATMAPS_PATH = OUTPUTS_DIR / "risk_heatmaps.png"
INTERACTIVE_MAP_PATH = OUTPUTS_DIR / "interactive_map.html"
BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
PAN_PATH = PROCESSED_DIR / "pan_polygon.geojson"
GATES_PATH = PROCESSED_DIR / "gates.geojson"
CAMPS_PATH = PROCESSED_DIR / "camps.geojson"
ROADS_PATH = PROCESSED_DIR / "tourist_roads.geojson"
WATERHOLES_PATH = PROCESSED_DIR / "waterholes.geojson"


def load_inputs() -> dict[str, gpd.GeoDataFrame]:
    return {
        "composite": validate_geojson(
            COMPOSITE_PATH,
            [
                "cell_id",
                "poaching_risk_norm",
                "wildfire_risk_norm",
                "tourism_risk_norm",
                "composite_risk_norm",
            ],
            "composite risk layer",
        ),
        "threats": validate_parquet(
            THREATS_PATH,
            [
                "cell_id",
                "poaching_threat_norm",
                "wildfire_threat_norm",
                "tourism_pressure_norm",
                "tourism_threat_norm",
                "geometry",
            ],
            "threat layers",
        ),
        "boundary": validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "boundary"),
        "pan": validate_geojson(PAN_PATH, ["name", "source", "source_detail", "notes"], "pan"),
        "gates": validate_geojson(GATES_PATH, ["name", "kind", "source", "source_detail"], "gates"),
        "camps": validate_geojson(CAMPS_PATH, ["name", "kind", "source", "source_detail"], "camps"),
        "roads": validate_geojson(ROADS_PATH, ["name", "ref", "fclass", "source"], "tourist roads"),
        "waterholes": validate_geojson(WATERHOLES_PATH, ["name", "kind", "source", "source_detail"], "waterholes"),
    }


def render_static(inputs: dict[str, gpd.GeoDataFrame]) -> None:
    composite = inputs["composite"]
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    gates = inputs["gates"]
    camps = inputs["camps"]
    roads = inputs["roads"]
    waterholes = inputs["waterholes"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plots = [
        ("composite_risk_norm", "composite risk", "YlOrRd"),
        ("poaching_risk_norm", "poaching risk", "YlOrRd"),
        ("wildfire_risk_norm", "wildfire risk", "YlOrRd"),
        ("tourism_risk_norm", "tourism risk", "YlOrRd"),
    ]
    for ax, (column, title, cmap) in zip(axes.ravel(), plots):
        composite.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            linewidth=0.1,
            edgecolor="#d0d0d0",
            legend=True,
            legend_kwds={"shrink": 0.7},
        )
        roads.plot(ax=ax, color="#666666", linewidth=0.4, alpha=0.35)
        boundary.boundary.plot(ax=ax, color="black", linewidth=1.0)
        pan.plot(ax=ax, color="#d7eef6", edgecolor="#8ec7d9", linewidth=0.5, alpha=0.8)
        gates.plot(ax=ax, color="black", markersize=14)
        camps.plot(ax=ax, color="white", edgecolor="black", marker="s", markersize=18)
        waterholes.plot(ax=ax, color="#1f77b4", markersize=5, alpha=0.45)
        ax.set_title(title)
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(HEATMAPS_PATH, dpi=180, bbox_inches="tight")


def add_point_markers(group: folium.FeatureGroup, gdf: gpd.GeoDataFrame, color: str, icon: str) -> None:
    for _, row in gdf.iterrows():
        geom = row.geometry
        folium.Marker(
            location=[geom.y, geom.x],
            popup=str(row.get("name", "")),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(group)


def render_interactive(inputs: dict[str, gpd.GeoDataFrame]) -> None:
    composite = inputs["composite"]
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    gates = inputs["gates"]
    camps = inputs["camps"]
    roads = inputs["roads"]
    waterholes = inputs["waterholes"]
    bounds = boundary.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    fmap = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")

    cmap = bcm.linear.YlOrRd_09.scale(0, 1)
    cmap.caption = "Composite Risk"
    cmap.add_to(fmap)

    folium.GeoJson(
        composite.to_json(),
        name="Composite Risk",
        style_function=lambda feature: {
            "fillColor": cmap(feature["properties"]["composite_risk_norm"]),
            "color": "#666666",
            "weight": 0.2,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["cell_id", "composite_risk_norm", "poaching_risk_norm", "wildfire_risk_norm", "tourism_risk_norm"],
            aliases=["Cell", "Composite", "Poaching", "Wildfire", "Tourism"],
        ),
    ).add_to(fmap)

    folium.GeoJson(
        boundary.to_json(),
        name="Boundary",
        style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0},
    ).add_to(fmap)
    folium.GeoJson(
        pan.to_json(),
        name="Pan",
        style_function=lambda _: {"color": "#5aa7c7", "weight": 1, "fillColor": "#d7eef6", "fillOpacity": 0.6},
    ).add_to(fmap)
    folium.GeoJson(
        roads.to_json(),
        name="Tourist Roads",
        style_function=lambda _: {"color": "#6b6b6b", "weight": 1, "opacity": 0.5},
    ).add_to(fmap)

    gate_group = folium.FeatureGroup(name="Gates")
    add_point_markers(gate_group, gates, "red", "road")
    gate_group.add_to(fmap)

    camp_group = folium.FeatureGroup(name="Camps")
    add_point_markers(camp_group, camps, "blue", "bed")
    camp_group.add_to(fmap)

    waterhole_group = folium.FeatureGroup(name="Waterholes")
    for _, row in waterholes.iterrows():
        geom = row.geometry
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=3,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.8,
            popup=str(row.get("name", "")),
        ).add_to(waterhole_group)
    waterhole_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(str(INTERACTIVE_MAP_PATH))


def check_outputs() -> None:
    if not HEATMAPS_PATH.is_file():
        raise FileNotFoundError(f"missing output: {HEATMAPS_PATH}")
    if not INTERACTIVE_MAP_PATH.is_file():
        raise FileNotFoundError(f"missing output: {INTERACTIVE_MAP_PATH}")
    if HEATMAPS_PATH.stat().st_size == 0:
        raise ValueError("risk heatmaps output is empty")
    if INTERACTIVE_MAP_PATH.stat().st_size == 0:
        raise ValueError("interactive map output is empty")
    print("visualization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="render static and interactive risk maps")
    parser.add_argument("--check", action="store_true", help="validate existing visualization outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    inputs = load_inputs()
    render_static(inputs)
    render_interactive(inputs)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
