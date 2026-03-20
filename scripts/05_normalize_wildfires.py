#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd

from _spatial_common import PROCESSED_DIR, RAW_DIR, WGS84, validate_geojson, write_geojson


BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
RAW_WILDFIRE_PATH = RAW_DIR / "wildfires/eonet_wildfires_bbox.json"
WILDFIRE_OUTPUT_PATH = PROCESSED_DIR / "wildfire_history.geojson"


def load_boundary() -> gpd.GeoDataFrame:
    return validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "processed boundary")


def build_wildfires(boundary: gpd.GeoDataFrame) -> None:
    raw = json.loads(Path(RAW_WILDFIRE_PATH).read_text())
    rows: list[dict[str, object]] = []
    for event in raw.get("events", []):
        for geometry in event.get("geometry", []):
            coordinates = geometry.get("coordinates", [])
            if geometry.get("type") != "Point" or len(coordinates) != 2:
                continue
            rows.append(
                {
                    "event_id": event["id"],
                    "title": event["title"],
                    "observation_date": geometry.get("date"),
                    "magnitude_ha": geometry.get("magnitudeValue"),
                    "magnitude_unit": geometry.get("magnitudeUnit"),
                    "source": "nasa_eonet_wildfires",
                    "geometry": gpd.points_from_xy([coordinates[0]], [coordinates[1]], crs=WGS84)[0],
                }
            )
    wildfire_history = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84)
    wildfire_history = wildfire_history[wildfire_history.within(boundary.geometry.iloc[0])].copy()
    write_geojson(wildfire_history, WILDFIRE_OUTPUT_PATH)


def check_outputs() -> None:
    validate_geojson(
        WILDFIRE_OUTPUT_PATH,
        ["event_id", "title", "observation_date", "magnitude_ha", "magnitude_unit", "source"],
        "wildfire history",
    )
    print("wildfire normalization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="normalize wildfire history into a geospatial event layer")
    parser.add_argument("--check", action="store_true", help="validate existing processed outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    boundary = load_boundary()
    build_wildfires(boundary)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
