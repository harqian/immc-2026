#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data/raw/manifest.csv"
FIELDNAMES = [
    "dataset_id",
    "source_url",
    "acquisition_method",
    "local_path",
    "license_or_terms",
    "date_acquired",
    "scripted",
    "manual_steps",
    "notes",
]
SOURCE_DEFINITIONS = [
    {
        "dataset_id": "etosha_boundary",
        "source_url": "https://www.openstreetmap.org/relation/2982497",
        "acquisition_method": "scripted_download",
        "local_path": "data/raw/etosha_boundary/etosha_boundary_nominatim.geojson",
        "license_or_terms": "OpenStreetMap contributors, ODbL 1.0",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "rerun the nominatim relation fetch and keep the raw geojson plus the public map image in the same directory",
        "notes": "raw boundary geometry is stored as a nominatim geojson export backed by the OSM Etosha relation",
    },
    {
        "dataset_id": "etosha_pan",
        "source_url": "https://nominatim.openstreetmap.org/search?q=Etosha%20Pan&format=geojsonv2&polygon_geojson=1",
        "acquisition_method": "scripted_download",
        "local_path": "data/raw/etosha_boundary/etosha_pan_nominatim.geojson",
        "license_or_terms": "OpenStreetMap contributors, ODbL 1.0",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "refresh the nominatim Etosha Pan geometry export if the pan polygon needs to be re-derived",
        "notes": "the processed pan polygon now comes from the OSM Etosha Pan relation rather than a hand-drawn approximation",
    },
    {
        "dataset_id": "etosha_public_map",
        "source_url": "https://etoshanationalpark.org/map-of-etosha-national-park",
        "acquisition_method": "manual_download",
        "local_path": "data/raw/etosha_boundary/etosha_map_2025.jpg",
        "license_or_terms": "respect site terms; retained as a visual reference for manual review only",
        "date_acquired": "2026-03-19",
        "scripted": "false",
        "manual_steps": "store the public Etosha map image used for manual camp, gate, and pan review",
        "notes": "the east and west map extracts are companion inspection artifacts kept alongside the main public map image",
    },
    {
        "dataset_id": "roads",
        "source_url": "https://download.geofabrik.de/africa/namibia-latest-free.shp.zip",
        "acquisition_method": "scripted_download",
        "local_path": "data/raw/roads/namibia-latest-free.shp.zip",
        "license_or_terms": "OpenStreetMap contributors, ODbL 1.0 via Geofabrik extract",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "refresh the geofabrik Namibia shapefile zip and rerun the infrastructure normalization script",
        "notes": "roads and many poi-derived waterholes come from the geofabrik OSM shapefile extract",
    },
    {
        "dataset_id": "waterholes",
        "source_url": "https://download.geofabrik.de/africa/namibia-latest-free.shp.zip",
        "acquisition_method": "scripted_extraction",
        "local_path": "data/raw/waterholes/etosha_waterholes_geofabrik.csv",
        "license_or_terms": "OpenStreetMap contributors, ODbL 1.0 via Geofabrik extract",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "rerun the geofabrik poi extraction inside the Etosha boundary and review duplicates before committing the csv",
        "notes": "the current raw csv contains 54 named waterholes or water points extracted from the geofabrik poi layer",
    },
    {
        "dataset_id": "gates",
        "source_url": "https://www.openstreetmap.org/relation/2982497",
        "acquisition_method": "scripted_derivation",
        "local_path": "data/raw/gates/etosha_gates_osm.csv",
        "license_or_terms": "OpenStreetMap contributors, ODbL 1.0",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "rebuild the gate csv from boundary-road intersections if the park boundary or OSM road network changes; keep the official map label note for Von Lindequist Gate",
        "notes": "all gate points are derived from OSM road intersections with the Etosha boundary, eliminating the earlier manual east-gate estimate",
    },
    {
        "dataset_id": "camps",
        "source_url": "https://download.geofabrik.de/africa/namibia-latest-free.shp.zip",
        "acquisition_method": "manual_curation",
        "local_path": "data/raw/camps/etosha_camps_osm.csv",
        "license_or_terms": "OpenStreetMap contributors, ODbL 1.0 via Geofabrik and Nominatim",
        "date_acquired": "2026-03-19",
        "scripted": "false",
        "manual_steps": "refresh the six main camp coordinates from geofabrik poi features where available and nominatim where geofabrik lacks a named camp point",
        "notes": "the current camp table is fully source-backed by OSM features, with geofabrik poi points for four camps and nominatim OSM search results for Dolomite and Onkoshi",
    },
    {
        "dataset_id": "elephants",
        "source_url": "https://api.gbif.org/v1/occurrence/search?scientificName=Loxodonta+africana&decimalLatitude=-19.49,-18.45&decimalLongitude=14.35,17.15&hasCoordinate=true",
        "acquisition_method": "scripted_download",
        "local_path": "data/raw/elephants/gbif_elephant_occurrences.csv",
        "license_or_terms": "GBIF occurrence data terms apply; retain citation metadata if expanded later",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "refresh the full paginated GBIF elephant pull or replace it with the richer Movebank telemetry package if it becomes available",
        "notes": "this mvp currently uses a full 984-record GBIF pull because the public API path was immediately usable; Movebank would still be the higher-fidelity upgrade",
    },
    {
        "dataset_id": "rhino_reference",
        "source_url": "http://rhinoresourcecenter.com/wp-content/uploads/2022/01/1642693511.pdf",
        "acquisition_method": "manual_download",
        "local_path": "data/raw/rhino_reference/rhino_resource_center_reference.pdf",
        "license_or_terms": "respect publication terms; derived ecological inference only, not sensitive location release",
        "date_acquired": "2026-03-19",
        "scripted": "false",
        "manual_steps": "store the cited rhino reference document and the extracted figure page image used for digitization",
        "notes": "the repo keeps the report pdf plus page extracts, then digitizes approximate rhino detections from figure 6 with explicit georeferencing caveats",
    },
    {
        "dataset_id": "carnivore_reference",
        "source_url": "https://orc.eco/counting-carnivores-in-the-greater-etosha-landscape-cont/",
        "acquisition_method": "manual_download",
        "local_path": "data/raw/carnivore_reference/orc_detection_1.webp",
        "license_or_terms": "respect page and image terms; use as a public figure for approximate digitization only",
        "date_acquired": "2026-03-19",
        "scripted": "false",
        "manual_steps": "store the ORC detection image that contains the lion panel and rerun the digitization script if the image changes",
        "notes": "the current lion layer is derived from approximate digitization of the ORC playback-detection map rather than buffered GBIF occurrences",
    },
    {
        "dataset_id": "lion_detections_digitized",
        "source_url": "https://orc.eco/counting-carnivores-in-the-greater-etosha-landscape-cont/",
        "acquisition_method": "scripted_digitization",
        "local_path": "data/raw/carnivore_reference/lion_detections_digitized.csv",
        "license_or_terms": "derived approximate coordinates from a public figure; not survey-grade point release",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "rerun scripts/04_digitize_reference_maps.py after updating the ORC source image or its GCP table and review the detection count and GCP fit metrics",
        "notes": "the csv stores approximate lion detections georeferenced from the ORC map image with a quadratic GCP-based warp and explicit uncertainty fields",
    },
    {
        "dataset_id": "lion_georeference_gcps",
        "source_url": "https://orc.eco/counting-carnivores-in-the-greater-etosha-landscape-cont/",
        "acquisition_method": "manual_curation",
        "local_path": "data/raw/carnivore_reference/lion_georeference_gcps.csv",
        "license_or_terms": "repo-authored control points referencing public map features",
        "date_acquired": "2026-03-19",
        "scripted": "false",
        "manual_steps": "update the image-space control points if the source image changes or if a better georeferencing fit is needed",
        "notes": "these GCPs tie visible lion-map features to known Etosha boundary and pan coordinates for the digitization warp",
    },
    {
        "dataset_id": "rhino_detections_digitized",
        "source_url": "http://rhinoresourcecenter.com/wp-content/uploads/2022/01/1642693511.pdf",
        "acquisition_method": "scripted_digitization",
        "local_path": "data/raw/rhino_reference/rhino_detections_digitized.csv",
        "license_or_terms": "derived approximate coordinates from a public figure; retained only as buffered support-area input",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "rerun scripts/04_digitize_reference_maps.py after updating the rhino figure extract or its GCP table and review the detection count and GCP fit metrics",
        "notes": "the csv stores approximate rhino detections digitized from figure 6 of the aerial survey report and georeferenced with a quadratic GCP-based warp",
    },
    {
        "dataset_id": "rhino_georeference_gcps",
        "source_url": "http://rhinoresourcecenter.com/wp-content/uploads/2022/01/1642693511.pdf",
        "acquisition_method": "manual_curation",
        "local_path": "data/raw/rhino_reference/rhino_georeference_gcps.csv",
        "license_or_terms": "repo-authored control points referencing public figure features",
        "date_acquired": "2026-03-19",
        "scripted": "false",
        "manual_steps": "update the image-space control points if the figure extract changes or if a better georeferencing fit is needed",
        "notes": "these GCPs tie visible rhino-map features to known Etosha boundary and pan coordinates for the digitization warp",
    },
    {
        "dataset_id": "wildfires",
        "source_url": "https://eonet.gsfc.nasa.gov/api/v3/events?category=wildfires&bbox=14.35,-19.49,17.15,-18.45&status=all&limit=200",
        "acquisition_method": "scripted_download",
        "local_path": "data/raw/wildfires/eonet_wildfires_bbox.json",
        "license_or_terms": "NASA EONET open data terms apply",
        "date_acquired": "2026-03-19",
        "scripted": "true",
        "manual_steps": "refresh the EONET bbox query and confirm that the resulting events still fall within or near the park before reuse",
        "notes": "the current wildfire raw artifact contains five bbox-matched EONET wildfire events, four of which fall inside the processed park boundary",
    },
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for definition in SOURCE_DEFINITIONS:
        row = {field: definition.get(field, "") for field in FIELDNAMES}
        missing = [field for field, value in row.items() if value == "" and field not in {"date_acquired"}]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{definition['dataset_id']} is missing required manifest fields: {joined}")
        rows.append(row)
    return rows


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        normalized_fieldnames = [(field or "").strip() for field in (reader.fieldnames or [])]
        if normalized_fieldnames != FIELDNAMES:
            raise ValueError(
                f"manifest columns do not match expected schema: {normalized_fieldnames!r} != {FIELDNAMES!r}"
            )
        return list(reader)


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def check_manifest(rows: list[dict[str, str]]) -> int:
    if not MANIFEST_PATH.is_file():
        print(f"manifest is missing: {MANIFEST_PATH}", file=sys.stderr)
        return 1

    current_rows = read_manifest()
    if current_rows != rows:
        print("manifest check failed: file does not match generated source definitions", file=sys.stderr)
        expected_ids = [row["dataset_id"] for row in rows]
        current_ids = [row["dataset_id"] for row in current_rows]
        print(f"expected dataset ids: {expected_ids}", file=sys.stderr)
        print(f"current dataset ids:  {current_ids}", file=sys.stderr)
        return 1

    print(f"manifest check passed for {len(rows)} datasets")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build or validate the raw data manifest")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that data/raw/manifest.csv matches the script definitions without rewriting it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows()

    if args.check:
        return check_manifest(rows)

    write_manifest(rows)
    print(f"wrote manifest with {len(rows)} datasets to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
