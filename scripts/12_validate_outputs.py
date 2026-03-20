#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import numpy as np

from _spatial_common import OUTPUTS_DIR, validate_geojson, validate_parquet


GRID_PATH = OUTPUTS_DIR / "grid.geojson"
SPECIES_PATH = OUTPUTS_DIR / "species_layers.parquet"
THREATS_PATH = OUTPUTS_DIR / "threat_layers.parquet"
COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
TENSOR_PATH = OUTPUTS_DIR / "risk_tensor.npz"
HEATMAPS_PATH = OUTPUTS_DIR / "risk_heatmaps.png"
INTERACTIVE_MAP_PATH = OUTPUTS_DIR / "interactive_map.html"


def validate_join_consistency(
    grid: gpd.GeoDataFrame,
    species: gpd.GeoDataFrame,
    threats: gpd.GeoDataFrame,
    composite: gpd.GeoDataFrame,
    tensor: np.lib.npyio.NpzFile,
) -> None:
    grid_ids = list(grid["cell_id"])
    species_ids = list(species["cell_id"])
    threat_ids = list(threats["cell_id"])
    composite_ids = list(composite["cell_id"])
    tensor_ids = list(tensor["cell_ids"])
    if not (grid_ids == species_ids == threat_ids == composite_ids == tensor_ids):
        raise ValueError("cell_id ordering is inconsistent across outputs")


def main() -> int:
    parser = argparse.ArgumentParser(description="validate final tensor and visualization outputs")
    parser.parse_args()

    grid = validate_geojson(
        GRID_PATH,
        ["cell_id", "grid_size_m", "metric_crs", "grid_version", "cell_area_m2", "row_index", "col_index"],
        "analysis grid",
    )
    species = validate_parquet(
        SPECIES_PATH,
        ["cell_id", "elephant_density_norm", "rhino_support_norm", "lion_support_norm", "herbivore_support_norm", "geometry"],
        "species layers",
    )
    threats = validate_parquet(
        THREATS_PATH,
        ["cell_id", "poaching_threat_norm", "wildfire_threat_norm", "tourism_pressure_norm", "tourism_threat_norm", "geometry"],
        "threat layers",
    )
    composite = validate_geojson(
        COMPOSITE_PATH,
        ["cell_id", "poaching_risk_norm", "wildfire_risk_norm", "tourism_risk_norm", "composite_risk_norm"],
        "composite risk layer",
    )
    if not TENSOR_PATH.is_file():
        raise FileNotFoundError(f"missing output: {TENSOR_PATH}")
    tensor = np.load(TENSOR_PATH, allow_pickle=False)
    validate_join_consistency(grid, species, threats, composite, tensor)
    if "metadata" not in tensor:
        raise ValueError("risk tensor is missing metadata")
    json.loads(str(tensor["metadata"]))
    if not HEATMAPS_PATH.is_file() or HEATMAPS_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty output: {HEATMAPS_PATH}")
    if not INTERACTIVE_MAP_PATH.is_file() or INTERACTIVE_MAP_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty output: {INTERACTIVE_MAP_PATH}")
    html = INTERACTIVE_MAP_PATH.read_text(encoding="utf-8")
    if "leaflet" not in html.lower():
        raise ValueError("interactive map does not look renderable")
    print("final output validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
