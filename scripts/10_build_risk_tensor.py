#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import geopandas as gpd
import numpy as np

from _spatial_common import OUTPUTS_DIR, validate_geojson, validate_parquet, write_geojson


GRID_PATH = OUTPUTS_DIR / "grid.geojson"
SPECIES_PATH = OUTPUTS_DIR / "species_layers.parquet"
THREATS_PATH = OUTPUTS_DIR / "threat_layers.parquet"
TENSOR_PATH = OUTPUTS_DIR / "risk_tensor.npz"
COMPOSITE_PATH = OUTPUTS_DIR / "composite_risk.geojson"
SPECIES_NAMES = np.array(["elephant", "rhino", "lion", "herbivore"], dtype="U16")
THREAT_NAMES = np.array(["poaching", "wildfire", "tourism"], dtype="U16")


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    grid = validate_geojson(
        GRID_PATH,
        [
            "cell_id",
            "grid_size_m",
            "metric_crs",
            "grid_version",
            "cell_area_m2",
            "centroid_x_m",
            "centroid_y_m",
            "row_index",
            "col_index",
        ],
        "analysis grid",
    )
    species = validate_parquet(
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
    )
    threats = validate_parquet(
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
    )
    return grid, species, threats


def build_tensor(species: gpd.GeoDataFrame, threats: gpd.GeoDataFrame) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    merged = species.merge(
        threats.drop(columns="geometry"),
        on="cell_id",
        how="inner",
        suffixes=("", "_threat"),
    )
    species_matrix = np.column_stack(
        [
            merged["elephant_density_norm"].to_numpy(),
            merged["rhino_support_norm"].to_numpy(),
            merged["lion_support_norm"].to_numpy(),
            merged["herbivore_support_norm"].to_numpy(),
        ]
    )
    threat_matrix = np.column_stack(
        [
            merged["poaching_threat_norm"].to_numpy(),
            merged["wildfire_threat_norm"].to_numpy(),
            merged["tourism_pressure_norm"].to_numpy(),
        ]
    )
    tensor = species_matrix[:, :, None] * threat_matrix[:, None, :]

    merged["composite_risk_norm"] = tensor.mean(axis=(1, 2))
    merged["poaching_risk_norm"] = tensor[:, :, 0].mean(axis=1)
    merged["wildfire_risk_norm"] = tensor[:, :, 1].mean(axis=1)
    merged["tourism_risk_norm"] = tensor[:, :, 2].mean(axis=1)
    merged["tourism_interaction_norm"] = merged["tourism_threat_norm"]
    return tensor, merged


def write_tensor(tensor: np.ndarray, merged: gpd.GeoDataFrame) -> None:
    metadata = {
        "species_norm_columns": {
            "elephant": "elephant_density_norm",
            "rhino": "rhino_support_norm",
            "lion": "lion_support_norm",
            "herbivore": "herbivore_support_norm",
        },
        "threat_norm_columns": {
            "poaching": "poaching_threat_norm",
            "wildfire": "wildfire_threat_norm",
            "tourism": "tourism_pressure_norm",
        },
        "tensor_definition": "tensor[cell, species, threat] = species_presence_norm * threat_exposure_norm",
    }
    np.savez(
        TENSOR_PATH,
        tensor=tensor,
        cell_ids=merged["cell_id"].to_numpy(dtype="U64"),
        species=SPECIES_NAMES,
        threats=THREAT_NAMES,
        metadata=np.array(json.dumps(metadata), dtype="U2048"),
    )

    composite = merged[
        [
            "cell_id",
            "grid_size_m",
            "metric_crs",
            "grid_version",
            "cell_area_m2",
            "row_index",
            "col_index",
            "elephant_density_norm",
            "rhino_support_norm",
            "lion_support_norm",
            "herbivore_support_norm",
            "poaching_threat_norm",
            "wildfire_threat_norm",
            "tourism_pressure_norm",
            "tourism_interaction_norm",
            "poaching_risk_norm",
            "wildfire_risk_norm",
            "tourism_risk_norm",
            "composite_risk_norm",
            "geometry",
        ]
    ].copy()
    write_geojson(composite, COMPOSITE_PATH)


def check_outputs() -> None:
    if not TENSOR_PATH.is_file():
        raise FileNotFoundError(f"missing output: {TENSOR_PATH}")
    loaded = np.load(TENSOR_PATH, allow_pickle=False)
    tensor = loaded["tensor"]
    if tensor.ndim != 3:
        raise ValueError(f"risk tensor must be 3D, found shape {tensor.shape}")
    if tensor.shape[1] != len(SPECIES_NAMES) or tensor.shape[2] != len(THREAT_NAMES):
        raise ValueError(f"unexpected tensor shape {tensor.shape}")
    composite = validate_geojson(
        COMPOSITE_PATH,
        [
            "cell_id",
            "elephant_density_norm",
            "rhino_support_norm",
            "lion_support_norm",
            "herbivore_support_norm",
            "poaching_threat_norm",
            "wildfire_threat_norm",
            "tourism_pressure_norm",
            "tourism_interaction_norm",
            "poaching_risk_norm",
            "wildfire_risk_norm",
            "tourism_risk_norm",
            "composite_risk_norm",
        ],
        "composite risk layer",
    )
    if len(composite) != tensor.shape[0]:
        raise ValueError("composite layer row count does not match tensor cell count")
    print(f"risk tensor check passed for shape {tensor.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="build the analysis-ready species-by-threat risk tensor")
    parser.add_argument("--check", action="store_true", help="validate existing tensor and composite outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    _, species, threats = load_inputs()
    tensor, merged = build_tensor(species, threats)
    write_tensor(tensor, merged)
    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
