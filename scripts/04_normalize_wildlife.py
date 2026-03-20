#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
import geopandas as gpd

from _spatial_common import PROCESSED_DIR, RAW_DIR, WGS84, validate_geojson, validate_parquet, write_geojson


BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
PAN_PATH = PROCESSED_DIR / "pan_polygon.geojson"
WATERHOLES_PATH = PROCESSED_DIR / "waterholes.geojson"
ELEPHANT_RAW_PATH = RAW_DIR / "elephants/gbif_elephant_occurrences.csv"
LION_RAW_PATH = RAW_DIR / "carnivore_reference/lion_detections_digitized.csv"
RHINO_RAW_PATH = RAW_DIR / "rhino_reference/rhino_detections_digitized.csv"
ELEPHANT_OUTPUT_PATH = PROCESSED_DIR / "elephant_density_points.parquet"
LION_OUTPUT_PATH = PROCESSED_DIR / "lion_zones.geojson"
RHINO_OUTPUT_PATH = PROCESSED_DIR / "rhino_reference_areas.geojson"
METRIC_CRS = "EPSG:32733"


def load_support_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    boundary = validate_geojson(BOUNDARY_PATH, ["name", "source", "source_detail"], "processed boundary")
    pan = validate_geojson(PAN_PATH, ["name", "source", "source_detail", "notes"], "processed pan polygon")
    waterholes = validate_geojson(WATERHOLES_PATH, ["name", "kind", "source", "source_detail"], "waterholes layer")
    return boundary, pan, waterholes


def gbif_points(path: str, species_label: str) -> gpd.GeoDataFrame:
    frame = pd.read_csv(path)
    required = ["key", "scientificName", "decimalLatitude", "decimalLongitude", "eventDate", "basisOfRecord"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing GBIF columns: {joined}")
    frame = frame.dropna(subset=["decimalLatitude", "decimalLongitude"]).copy()
    gdf = gpd.GeoDataFrame(
        {
            "gbif_id": frame["key"],
            "scientific_name": frame["scientificName"],
            "event_date": frame["eventDate"],
            "basis_of_record": frame["basisOfRecord"],
            "source": species_label,
        },
        geometry=gpd.points_from_xy(frame["decimalLongitude"], frame["decimalLatitude"]),
        crs=WGS84,
    )
    return gdf


def build_elephants(boundary: gpd.GeoDataFrame) -> None:
    elephants = gbif_points(str(ELEPHANT_RAW_PATH), "gbif_elephant_snapshot")
    elephants = elephants[elephants.within(boundary.geometry.iloc[0])].copy()
    elephants.to_parquet(ELEPHANT_OUTPUT_PATH, index=False)


def digitized_points(path, species_label: str) -> gpd.GeoDataFrame:
    frame = pd.read_csv(path)
    required = [
        "detection_id",
        "species",
        "source_image",
        "longitude",
        "latitude",
        "within_boundary_buffer",
        "georeference_method",
        "georeference_rmse_deg",
        "control_point_count",
        "notes",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing digitized columns: {joined}")
    filtered = frame[frame["species"] == species_label].copy()
    gdf = gpd.GeoDataFrame(
        filtered,
        geometry=gpd.points_from_xy(filtered["longitude"], filtered["latitude"]),
        crs=WGS84,
    )
    return gdf


def build_lions(boundary: gpd.GeoDataFrame, pan: gpd.GeoDataFrame) -> None:
    lions = digitized_points(str(LION_RAW_PATH), "lion")
    boundary_buffer = boundary.to_crs(METRIC_CRS).buffer(15000).to_crs(WGS84).iloc[0]
    lions = lions[lions.geometry.within(boundary_buffer)].copy()
    lions_metric = lions.to_crs(METRIC_CRS)
    buffered = gpd.GeoDataFrame(geometry=lions_metric.buffer(10000), crs=METRIC_CRS)
    merged = gpd.GeoDataFrame(geometry=[buffered.union_all()], crs=METRIC_CRS).explode(index_parts=False).reset_index(drop=True)
    merged["zone_id"] = [f"lion_zone_{index+1}" for index in range(len(merged))]
    merged["source"] = "orc_lion_detection_map_digitized"
    merged["point_count"] = len(lions)
    merged["notes"] = "zones derived by buffering digitized lion playback detections from the ORC public map image"
    merged = merged.to_crs(WGS84)
    merged = gpd.overlay(merged, boundary[["geometry"]], how="intersection")
    merged = gpd.overlay(merged, pan[["geometry"]], how="difference")
    write_geojson(merged, LION_OUTPUT_PATH)


def build_rhino_reference(boundary: gpd.GeoDataFrame, pan: gpd.GeoDataFrame, waterholes: gpd.GeoDataFrame) -> None:
    _ = waterholes
    rhinos = digitized_points(str(RHINO_RAW_PATH), "rhino")
    boundary_buffer = boundary.to_crs(METRIC_CRS).buffer(10000).to_crs(WGS84).iloc[0]
    rhinos = rhinos[rhinos.geometry.within(boundary_buffer)].copy()
    rhino_metric = rhinos.to_crs(METRIC_CRS)
    habitat = gpd.GeoDataFrame(geometry=[rhino_metric.buffer(6000).union_all()], crs=METRIC_CRS).to_crs(WGS84)
    habitat = gpd.overlay(habitat, boundary[["geometry"]], how="intersection")
    habitat = gpd.overlay(habitat, pan[["geometry"]], how="difference")
    habitat["name"] = "rhino_reference_distribution"
    habitat["source"] = "rhino_resource_center_figure_6_digitized"
    habitat["detection_count"] = len(rhinos)
    habitat["notes"] = "distribution support area from buffered approximate rhino detections digitized from figure 6 of the cited aerial survey report"
    write_geojson(habitat[["name", "source", "detection_count", "notes", "geometry"]], RHINO_OUTPUT_PATH)


def check_outputs() -> None:
    validate_parquet(
        ELEPHANT_OUTPUT_PATH,
        ["gbif_id", "scientific_name", "event_date", "basis_of_record", "source", "geometry"],
        "elephant points",
    )
    validate_geojson(LION_OUTPUT_PATH, ["zone_id", "source", "point_count", "notes"], "lion zones")
    validate_geojson(RHINO_OUTPUT_PATH, ["name", "source", "detection_count", "notes"], "rhino reference areas")
    print("wildlife normalization check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="normalize elephant, lion, and rhino-support wildlife layers")
    parser.add_argument("--check", action="store_true", help="validate existing processed outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    boundary, pan, waterholes = load_support_layers()
    build_elephants(boundary)
    build_lions(boundary, pan)
    build_rhino_reference(boundary, pan, waterholes)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
