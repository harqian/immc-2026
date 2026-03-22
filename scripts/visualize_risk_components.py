#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd
import matplotlib.pyplot as plt

from _spatial_common import OUTPUTS_DIR, PROCESSED_DIR, validate_geojson, validate_parquet


DIAGNOSTIC_DIR = OUTPUTS_DIR / "risk_diagnostics"
DATA_LAYERS_PATH = DIAGNOSTIC_DIR / "01_data_layers_overview.png"
SPECIES_LAYERS_PATH = DIAGNOSTIC_DIR / "02_species_layers_overview.png"
POACHING_COMPONENTS_PATH = DIAGNOSTIC_DIR / "03_poaching_components.png"
WILDFIRE_COMPONENTS_PATH = DIAGNOSTIC_DIR / "04_wildfire_components.png"
TOURISM_COMPONENTS_PATH = DIAGNOSTIC_DIR / "05_tourism_components.png"
RISK_ASSEMBLY_PATH = DIAGNOSTIC_DIR / "06_risk_assembly_overview.png"

BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
PAN_PATH = PROCESSED_DIR / "pan_polygon.geojson"
ROADS_PATH = PROCESSED_DIR / "tourist_roads.geojson"
GATES_PATH = PROCESSED_DIR / "gates.geojson"
CAMPS_PATH = PROCESSED_DIR / "camps.geojson"
WATERHOLES_PATH = PROCESSED_DIR / "waterholes.geojson"
WILDFIRES_PATH = PROCESSED_DIR / "wildfire_history.geojson"
ELEPHANTS_PATH = PROCESSED_DIR / "elephant_density_points.parquet"
LIONS_PATH = PROCESSED_DIR / "lion_zones.geojson"
RHINOS_PATH = PROCESSED_DIR / "rhino_reference_areas.geojson"

SPECIES_PATH = OUTPUTS_DIR / "species_layers.parquet"
THREATS_PATH = OUTPUTS_DIR / "threat_layers.parquet"
COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"

RISK_HEATMAP_WEIGHTS = {
    "poaching": 1.0,
    "wildfire": 1.5,
    "tourism": 0.5,
}


def load_inputs() -> dict[str, gpd.GeoDataFrame]:
    return {
        "boundary": validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "boundary"),
        "pan": validate_geojson(PAN_PATH, ["name", "source", "source_detail", "notes"], "pan"),
        "roads": validate_geojson(ROADS_PATH, ["name", "ref", "fclass", "source"], "tourist roads"),
        "gates": validate_geojson(GATES_PATH, ["name", "kind", "source", "source_detail"], "gates"),
        "camps": validate_geojson(CAMPS_PATH, ["name", "kind", "source", "source_detail"], "camps"),
        "waterholes": validate_geojson(WATERHOLES_PATH, ["name", "kind", "source", "source_detail"], "waterholes"),
        "wildfires": validate_geojson(
            WILDFIRES_PATH,
            ["event_id", "title", "observation_date", "magnitude_ha", "magnitude_unit", "source"],
            "wildfires",
        ),
        "elephants": validate_parquet(
            ELEPHANTS_PATH,
            ["gbif_id", "scientific_name", "event_date", "basis_of_record", "source", "geometry"],
            "elephant points",
        ),
        "lions": validate_geojson(LIONS_PATH, ["zone_id", "source", "point_count", "notes"], "lion zones"),
        "rhinos": validate_geojson(
            RHINOS_PATH,
            ["name", "source", "detection_count", "notes"],
            "rhino reference areas",
        ),
        "species": validate_parquet(
            SPECIES_PATH,
            [
                "cell_id",
                "elephant_density_norm",
                "rhino_support_norm",
                "lion_support_norm",
                "herbivore_support_norm",
                "geometry",
            ],
            "species layers",
        ),
        "threats": validate_parquet(
            THREATS_PATH,
            [
                "cell_id",
                "poaching_threat_norm",
                "wildfire_threat_norm",
                "tourism_pressure_norm",
                "tourism_threat_norm",
                "poaching_gate_access",
                "poaching_boundary_access",
                "poaching_road_access",
                "poaching_tourist_road_access",
                "poaching_surveillance_gap",
                "poaching_rhino_value",
                "wildfire_recent_fire_suppression",
                "wildfire_flammable_land",
                "wildfire_water_remoteness",
                "tourism_tourist_road_access",
                "tourism_camp_access",
                "tourism_gate_access",
                "tourism_waterhole_access",
                "geometry",
            ],
            "threat layers",
        ),
        "composite": validate_geojson(
            COMPOSITE_PATH,
            [
                "cell_id",
                "poaching_risk_norm",
                "wildfire_risk_norm",
                "tourism_risk_norm",
                "composite_risk_norm",
                "geometry",
            ],
            "composite risk",
        ),
    }


def style_axes(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")


def add_reference_layers(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    pan: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None = None,
) -> None:
    if roads is not None:
        roads.plot(ax=ax, color="#7a7a7a", linewidth=0.25, alpha=0.25)
    pan.plot(ax=ax, color="#d9eef8", edgecolor="#8ec7d9", linewidth=0.4, alpha=0.75)
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.7)


def plot_surface(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    boundary: gpd.GeoDataFrame,
    pan: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    *,
    cmap: str = "YlOrRd",
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
) -> None:
    gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        linewidth=0.05,
        edgecolor="#d3d3d3",
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={"shrink": 0.72},
    )
    add_reference_layers(ax, boundary, pan, roads)
    ax.set_title(title, fontsize=10)
    style_axes(ax)


def render_data_layers(inputs: dict[str, gpd.GeoDataFrame]) -> None:
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    roads = inputs["roads"]
    gates = inputs["gates"]
    camps = inputs["camps"]
    waterholes = inputs["waterholes"]
    wildfires = inputs["wildfires"]
    elephants = inputs["elephants"]
    lions = inputs["lions"]
    rhinos = inputs["rhinos"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    add_reference_layers(axes[0, 0], boundary, pan)
    axes[0, 0].set_title("boundary and pan", fontsize=10)
    style_axes(axes[0, 0])

    add_reference_layers(axes[0, 1], boundary, pan, roads)
    gates.plot(ax=axes[0, 1], color="#c62828", markersize=22, marker="o")
    camps.plot(ax=axes[0, 1], color="white", edgecolor="black", markersize=34, marker="s")
    axes[0, 1].set_title("tourist roads, gates, camps", fontsize=10)
    style_axes(axes[0, 1])

    add_reference_layers(axes[0, 2], boundary, pan, roads)
    waterholes.plot(ax=axes[0, 2], color="#1f77b4", markersize=12, alpha=0.7)
    wildfires.plot(ax=axes[0, 2], color="#ff7f0e", markersize=30, marker="x", linewidth=1.1)
    axes[0, 2].set_title("waterholes and wildfire events", fontsize=10)
    style_axes(axes[0, 2])

    add_reference_layers(axes[1, 0], boundary, pan, roads)
    elephants.plot(ax=axes[1, 0], color="#2ca02c", markersize=4, alpha=0.3)
    axes[1, 0].set_title("elephant occurrence points", fontsize=10)
    style_axes(axes[1, 0])

    add_reference_layers(axes[1, 1], boundary, pan, roads)
    lions.plot(ax=axes[1, 1], color="#c9a227", edgecolor="#8d6e00", linewidth=0.6, alpha=0.55)
    axes[1, 1].set_title("lion support zones", fontsize=10)
    style_axes(axes[1, 1])

    add_reference_layers(axes[1, 2], boundary, pan, roads)
    rhinos.plot(ax=axes[1, 2], color="#6a8f6b", edgecolor="#325c38", linewidth=0.6, alpha=0.55)
    axes[1, 2].set_title("rhino reference areas", fontsize=10)
    style_axes(axes[1, 2])

    fig.suptitle("risk-map source and normalized spatial layers", fontsize=14)
    fig.tight_layout()
    fig.savefig(DATA_LAYERS_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_species_layers(inputs: dict[str, gpd.GeoDataFrame]) -> None:
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    roads = inputs["roads"]
    species = inputs["species"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        ("elephant_density_norm", "elephant density"),
        ("rhino_support_norm", "rhino support"),
        ("lion_support_norm", "lion support"),
        ("herbivore_support_norm", "herbivore support"),
    ]
    for ax, (column, title) in zip(axes.ravel(), panels):
        plot_surface(ax, species, column, title, boundary, pan, roads)

    fig.suptitle("species-side inputs to the risk tensor", fontsize=14)
    fig.tight_layout()
    fig.savefig(SPECIES_LAYERS_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_diagnostic_frames(inputs: dict[str, gpd.GeoDataFrame]) -> dict[str, gpd.GeoDataFrame]:
    threats = inputs["threats"].copy()
    composite = inputs["composite"][
        ["cell_id", "poaching_risk_norm", "wildfire_risk_norm", "tourism_risk_norm", "composite_risk_norm"]
    ].copy()
    diagnostics = threats.merge(composite, on="cell_id", how="inner")

    diagnostics["poaching_weighted_sum"] = (
        0.24 * diagnostics["poaching_gate_access"]
        + 0.24 * diagnostics["poaching_boundary_access"]
        + 0.18 * diagnostics["poaching_road_access"]
        + 0.08 * diagnostics["poaching_tourist_road_access"]
        + 0.08 * diagnostics["poaching_surveillance_gap"]
        + 0.18 * diagnostics["poaching_rhino_value"]
    )
    diagnostics["wildfire_base_exposure"] = (
        0.60 * diagnostics["wildfire_flammable_land"] + 0.40 * diagnostics["wildfire_water_remoteness"]
    )
    diagnostics["tourism_wildlife_presence"] = (
        0.45 * diagnostics["elephant_density_norm"]
        + 0.25 * diagnostics["lion_support_norm"]
        + 0.30 * diagnostics["herbivore_support_norm"]
    )
    diagnostics["composite_recomputed"] = (
        RISK_HEATMAP_WEIGHTS["poaching"] * diagnostics["poaching_risk_norm"]
        + RISK_HEATMAP_WEIGHTS["wildfire"] * diagnostics["wildfire_risk_norm"]
        + RISK_HEATMAP_WEIGHTS["tourism"] * diagnostics["tourism_risk_norm"]
    ) / sum(RISK_HEATMAP_WEIGHTS.values())
    return {"diagnostics": diagnostics}


def render_poaching_components(inputs: dict[str, gpd.GeoDataFrame], diagnostics: gpd.GeoDataFrame) -> None:
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    roads = inputs["roads"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    panels = [
        ("poaching_gate_access", "gate access"),
        ("poaching_boundary_access", "boundary access"),
        ("poaching_road_access", "road access"),
        ("poaching_tourist_road_access", "tourist-road access"),
        ("poaching_surveillance_gap", "surveillance gap"),
        ("poaching_rhino_value", "rhino value"),
        ("poaching_weighted_sum", "weighted poaching score"),
        ("poaching_threat_norm", "poaching threat"),
        ("poaching_risk_norm", "poaching risk"),
    ]
    for ax, (column, title) in zip(axes.ravel(), panels):
        plot_surface(ax, diagnostics, column, title, boundary, pan, roads)

    fig.suptitle("poaching threat assembly", fontsize=14)
    fig.tight_layout()
    fig.savefig(POACHING_COMPONENTS_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_wildfire_components(inputs: dict[str, gpd.GeoDataFrame], diagnostics: gpd.GeoDataFrame) -> None:
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    roads = inputs["roads"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    panels = [
        ("wildfire_recent_fire_suppression", "recent-fire suppression"),
        ("wildfire_flammable_land", "flammable land"),
        ("wildfire_water_remoteness", "water remoteness"),
        ("wildfire_base_exposure", "base wildfire exposure"),
        ("wildfire_threat_norm", "wildfire threat"),
        ("wildfire_risk_norm", "wildfire risk"),
    ]
    for ax, (column, title) in zip(axes.ravel(), panels):
        plot_surface(ax, diagnostics, column, title, boundary, pan, roads)

    fig.suptitle("wildfire threat assembly", fontsize=14)
    fig.tight_layout()
    fig.savefig(WILDFIRE_COMPONENTS_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_tourism_components(inputs: dict[str, gpd.GeoDataFrame], diagnostics: gpd.GeoDataFrame) -> None:
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    roads = inputs["roads"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    panels = [
        ("tourism_tourist_road_access", "tourist-road access"),
        ("tourism_camp_access", "camp access"),
        ("tourism_gate_access", "gate access"),
        ("tourism_waterhole_access", "waterhole access"),
        ("tourism_pressure_norm", "tourism pressure"),
        ("tourism_wildlife_presence", "wildlife presence"),
        ("tourism_threat_norm", "tourism interaction"),
        ("tourism_risk_norm", "tourism risk"),
    ]
    for ax, (column, title) in zip(axes.ravel(), panels):
        plot_surface(ax, diagnostics, column, title, boundary, pan, roads)

    fig.suptitle("tourism threat assembly", fontsize=14)
    fig.tight_layout()
    fig.savefig(TOURISM_COMPONENTS_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_risk_assembly(inputs: dict[str, gpd.GeoDataFrame], diagnostics: gpd.GeoDataFrame) -> None:
    boundary = inputs["boundary"]
    pan = inputs["pan"]
    roads = inputs["roads"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    panels = [
        ("poaching_risk_norm", "poaching risk"),
        ("wildfire_risk_norm", "wildfire risk"),
        ("tourism_risk_norm", "tourism risk"),
        ("composite_risk_norm", "composite risk"),
    ]
    for ax, (column, title) in zip(axes.ravel(), panels):
        plot_surface(ax, diagnostics, column, title, boundary, pan, roads)

    fig.suptitle("risk assembly from threat-specific surfaces to final heatmap", fontsize=14)
    fig.tight_layout()
    fig.savefig(RISK_ASSEMBLY_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_all() -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    diagnostic_frames = build_diagnostic_frames(inputs)
    diagnostics = diagnostic_frames["diagnostics"]
    render_data_layers(inputs)
    render_species_layers(inputs)
    render_poaching_components(inputs, diagnostics)
    render_wildfire_components(inputs, diagnostics)
    render_tourism_components(inputs, diagnostics)
    render_risk_assembly(inputs, diagnostics)


def check_outputs() -> None:
    required = [
        DATA_LAYERS_PATH,
        SPECIES_LAYERS_PATH,
        POACHING_COMPONENTS_PATH,
        WILDFIRE_COMPONENTS_PATH,
        TOURISM_COMPONENTS_PATH,
        RISK_ASSEMBLY_PATH,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"missing diagnostic output: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"diagnostic output is empty: {path}")
    print(f"risk diagnostic check passed for {len(required)} images in {DIAGNOSTIC_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="render diagnostic views for risk-map component assembly")
    parser.add_argument("--check", action="store_true", help="validate existing diagnostic image outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    render_all()
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
