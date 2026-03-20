from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WGS84 = "EPSG:4326"


def read_layer(path: Path | str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    return gdf.to_crs(WGS84)


def read_parquet_points(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    return gdf.to_crs(WGS84)


def normalize_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    normalized = gdf.copy()
    normalized["geometry"] = normalized.geometry.map(make_valid)
    normalized = normalized[~normalized.geometry.is_empty & normalized.geometry.notna()].copy()
    return normalized


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_geometries(gdf).to_file(path, driver="GeoJSON")


def ensure_columns(frame: pd.DataFrame, required_columns: list[str], label: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required columns: {joined}")


def validate_geojson(path: Path, required_columns: list[str], label: str) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing output: {path}")
    gdf = read_layer(path)
    ensure_columns(gdf, required_columns, label)
    if gdf.empty:
        raise ValueError(f"{label} is empty")
    if gdf.crs.to_string() != WGS84:
        raise ValueError(f"{label} must be stored in {WGS84}, found {gdf.crs}")
    if not gdf.geometry.is_valid.all():
        raise ValueError(f"{label} contains invalid geometries")
    return gdf


def validate_parquet(path: Path, required_columns: list[str], label: str) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing output: {path}")
    gdf = read_parquet_points(path)
    ensure_columns(gdf, required_columns, label)
    if gdf.empty:
        raise ValueError(f"{label} is empty")
    if gdf.crs.to_string() != WGS84:
        raise ValueError(f"{label} must be stored in {WGS84}, found {gdf.crs}")
    if not gdf.geometry.is_valid.all():
        raise ValueError(f"{label} contains invalid geometries")
    return gdf
