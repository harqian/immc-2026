#!/usr/bin/env python3
from __future__ import annotations

import argparse

import geopandas as gpd

from _spatial_common import PROCESSED_DIR, RAW_DIR, WGS84, read_layer, validate_geojson, write_geojson


RAW_BOUNDARY_PATH = RAW_DIR / "etosha_boundary/etosha_boundary_nominatim.geojson"
RAW_PAN_PATH = RAW_DIR / "etosha_boundary/etosha_pan_nominatim.geojson"
BOUNDARY_OUTPUT_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
PAN_OUTPUT_PATH = PROCESSED_DIR / "pan_polygon.geojson"


def build_boundary() -> None:
    raw_boundary = read_layer(RAW_BOUNDARY_PATH)
    boundary = gpd.GeoDataFrame(
        {
            "name": ["Etosha National Park"],
            "source": ["nominatim_osm_relation"],
            "source_detail": ["https://nominatim.openstreetmap.org/search?q=Etosha+National+Park%2C+Namibia"],
        },
        geometry=[raw_boundary.union_all()],
        crs=WGS84,
    )
    write_geojson(boundary, BOUNDARY_OUTPUT_PATH)


def build_pan() -> None:
    boundary = validate_geojson(BOUNDARY_OUTPUT_PATH, ["name", "source", "source_detail"], "processed boundary")
    raw_pan = read_layer(RAW_PAN_PATH)
    pan = gpd.GeoDataFrame(
        {
            "name": ["Etosha Pan"],
            "source": ["nominatim_osm_relation"],
            "source_detail": ["https://nominatim.openstreetmap.org/search?q=Etosha+Pan%2C+Namibia"],
            "notes": ["raw pan geometry is stored as a nominatim geojson export backed by the OSM Etosha Pan relation"],
        },
        geometry=[raw_pan.union_all()],
        crs=WGS84,
    )
    pan = gpd.overlay(pan, boundary[["geometry"]], how="intersection")
    write_geojson(pan, PAN_OUTPUT_PATH)


def check_outputs() -> None:
    validate_geojson(BOUNDARY_OUTPUT_PATH, ["name", "source", "source_detail"], "processed boundary")
    validate_geojson(PAN_OUTPUT_PATH, ["name", "source", "source_detail", "notes"], "processed pan polygon")
    print("boundary normalization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="normalize the Etosha boundary and pan polygon")
    parser.add_argument("--check", action="store_true", help="validate existing processed outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    build_boundary()
    build_pan()
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
