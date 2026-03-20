#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from _spatial_common import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, WGS84, validate_geojson, write_geojson


BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
ROADS_SOURCE_PATH = f"zip://{PROJECT_ROOT}/data/raw/roads/namibia-latest-free.shp.zip!gis_osm_roads_free_1.shp"
WATERHOLES_RAW_PATH = RAW_DIR / "waterholes/etosha_waterholes_geofabrik.csv"
CAMPS_RAW_PATH = RAW_DIR / "camps/etosha_camps_osm.csv"
GATES_RAW_PATH = RAW_DIR / "gates/etosha_gates_osm.csv"
ROADS_OUTPUT_PATH = PROCESSED_DIR / "roads.geojson"
TOURIST_ROADS_OUTPUT_PATH = PROCESSED_DIR / "tourist_roads.geojson"
WATERHOLES_OUTPUT_PATH = PROCESSED_DIR / "waterholes.geojson"
CAMPS_OUTPUT_PATH = PROCESSED_DIR / "camps.geojson"
GATES_OUTPUT_PATH = PROCESSED_DIR / "gates.geojson"
ROAD_CLASSES = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service", "track"}
TOURIST_ROAD_CLASSES = {"primary", "secondary", "tertiary", "unclassified"}


def load_boundary() -> gpd.GeoDataFrame:
    return validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "processed boundary")


def build_roads(boundary: gpd.GeoDataFrame) -> None:
    roads = gpd.read_file(ROADS_SOURCE_PATH, bbox=tuple(boundary.total_bounds)).to_crs(WGS84)
    roads = roads[roads["fclass"].isin(ROAD_CLASSES)].copy()
    roads = gpd.clip(roads, boundary[["geometry"]])
    roads = roads[["name", "ref", "fclass", "geometry"]].copy()
    roads["source"] = "geofabrik_osm_roads"
    write_geojson(roads, ROADS_OUTPUT_PATH)

    tourist_roads = roads[roads["fclass"].isin(TOURIST_ROAD_CLASSES)].copy()
    if tourist_roads.empty:
        tourist_roads = roads.copy()
    write_geojson(tourist_roads, TOURIST_ROADS_OUTPUT_PATH)


def classify_waterhole_kind(name: object) -> str:
    if not isinstance(name, str):
        return "unknown"
    lowered = name.lower()
    if "seasonal" in lowered:
        return "seasonal"
    if "man-made" in lowered or "man made" in lowered:
        return "man-made"
    if "natural" in lowered:
        return "natural"
    return "unknown"


def points_from_csv(path: Path, required_columns: list[str], source_label: str) -> gpd.GeoDataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required columns: {joined}")
    gdf = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame["longitude"], frame["latitude"]),
        crs=WGS84,
    )
    gdf["source"] = gdf["source"].fillna(source_label) if "source" in gdf.columns else source_label
    return gdf


def build_points(boundary: gpd.GeoDataFrame) -> None:
    waterholes_frame = pd.read_csv(WATERHOLES_RAW_PATH)
    waterholes_frame = waterholes_frame.rename(columns={"lon": "longitude", "lat": "latitude"})
    waterholes_frame["kind"] = waterholes_frame["name"].map(classify_waterhole_kind)
    waterholes_frame["source"] = "geofabrik_pois"
    waterholes_frame["source_detail"] = "namibia-latest-free.shp.zip:gis_osm_pois_free_1.shp"
    waterholes = gpd.GeoDataFrame(
        waterholes_frame,
        geometry=gpd.points_from_xy(waterholes_frame["longitude"], waterholes_frame["latitude"]),
        crs=WGS84,
    )
    waterholes = waterholes[waterholes.within(boundary.geometry.iloc[0])].copy()
    waterholes = waterholes[["name", "kind", "source", "source_detail", "geometry"]].drop_duplicates(subset=["name"])
    write_geojson(waterholes, WATERHOLES_OUTPUT_PATH)

    camps = points_from_csv(
        CAMPS_RAW_PATH,
        ["name", "kind", "source", "source_detail", "latitude", "longitude"],
        "curated_camp_table",
    )
    camps = camps[["name", "kind", "source", "source_detail", "geometry"]]
    write_geojson(camps, CAMPS_OUTPUT_PATH)

    gates = points_from_csv(
        GATES_RAW_PATH,
        ["name", "kind", "source", "source_detail", "latitude", "longitude"],
        "curated_gate_table",
    )
    gates = gates[["name", "kind", "source", "source_detail", "geometry"]]
    write_geojson(gates, GATES_OUTPUT_PATH)


def check_outputs() -> None:
    validate_geojson(ROADS_OUTPUT_PATH, ["fclass", "source"], "roads layer")
    validate_geojson(TOURIST_ROADS_OUTPUT_PATH, ["fclass", "source"], "tourist roads layer")
    validate_geojson(WATERHOLES_OUTPUT_PATH, ["name", "kind", "source", "source_detail"], "waterholes layer")
    validate_geojson(CAMPS_OUTPUT_PATH, ["name", "kind", "source", "source_detail"], "camps layer")
    validate_geojson(GATES_OUTPUT_PATH, ["name", "kind", "source", "source_detail"], "gates layer")
    print("infrastructure normalization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="normalize Etosha roads, waterholes, camps, and gates")
    parser.add_argument("--check", action="store_true", help="validate existing processed outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    boundary = load_boundary()
    build_roads(boundary)
    build_points(boundary)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
