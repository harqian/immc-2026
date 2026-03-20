#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shapely.geometry import Point

from _spatial_common import PROCESSED_DIR, RAW_DIR, WGS84


BOUNDARY_PATH = PROCESSED_DIR / "etosha_boundary.geojson"
LION_IMAGE_PATH = RAW_DIR / "carnivore_reference/orc_detection_1.webp"
RHINO_IMAGE_PATH = RAW_DIR / "rhino_reference/rhino_pages-14.png"
LION_OUTPUT_PATH = RAW_DIR / "carnivore_reference/lion_detections_digitized.csv"
RHINO_OUTPUT_PATH = RAW_DIR / "rhino_reference/rhino_detections_digitized.csv"
LION_GCP_PATH = RAW_DIR / "carnivore_reference/lion_georeference_gcps.csv"
RHINO_GCP_PATH = RAW_DIR / "rhino_reference/rhino_georeference_gcps.csv"
METRIC_CRS = "EPSG:32733"
LION_CROP = (0, 140, 595, 1680)
RHINO_CROP = (980, 100, 1500, 1180)


def load_boundary() -> Polygon:
    boundary = gpd.read_file(BOUNDARY_PATH)
    if boundary.empty:
        raise ValueError(f"processed boundary is empty: {BOUNDARY_PATH}")
    return boundary.to_crs(WGS84).geometry.iloc[0]


def crop_rgb(path, crop: tuple[int, int, int, int]) -> np.ndarray:
    image = np.array(Image.open(path).convert("RGB"))
    y0, x0, y1, x1 = crop
    return image[y0:y1, x0:x1]


def read_gcps(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["target_label", "pixel_x", "pixel_y", "longitude", "latitude", "notes"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{path} is missing GCP columns: {joined}")
    if len(frame) < 6:
        raise ValueError(f"{path} needs at least 6 GCPs for a quadratic warp")
    return frame


def quadratic_terms(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return np.c_[np.ones(len(points)), x, y, x * x, x * y, y * y]


def fit_quadratic_transform(gcp_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    source = gcp_frame[["pixel_x", "pixel_y"]].to_numpy(dtype=float)
    target = gcp_frame[["longitude", "latitude"]].to_numpy(dtype=float)
    design = quadratic_terms(source)
    lon_params, *_ = np.linalg.lstsq(design, target[:, 0], rcond=None)
    lat_params, *_ = np.linalg.lstsq(design, target[:, 1], rcond=None)
    predicted = np.c_[design @ lon_params, design @ lat_params]
    rmse = float(np.sqrt(((predicted - target) ** 2).sum(axis=1).mean()))
    return lon_params, lat_params, rmse


def apply_quadratic_transform(lon_params: np.ndarray, lat_params: np.ndarray, points: np.ndarray) -> np.ndarray:
    design = quadratic_terms(np.asarray(points, dtype=float))
    return np.c_[design @ lon_params, design @ lat_params]


def detect_lion_markers(image: np.ndarray) -> np.ndarray:
    green_mask = (
        (image[:, :, 1] > 120)
        & (image[:, :, 1] > image[:, :, 0] * 1.15)
        & (image[:, :, 1] > image[:, :, 2] * 1.15)
    )
    opened = ndi.binary_opening(green_mask, iterations=1)
    labels, _ = ndi.label(opened)
    detections: list[tuple[float, float]] = []

    for label_id, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        component = labels[slices] == label_id
        area = int(component.sum())
        if area < 15 or area > 400:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        fill_ratio = area / float(height * width)
        if fill_ratio < 0.18 or fill_ratio > 0.75:
            continue
        ys, xs = np.where(component)
        detections.append((slices[1].start + xs.mean(), slices[0].start + ys.mean()))

    return np.array(detections)


def detect_rhino_markers(image: np.ndarray) -> np.ndarray:
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    brown_mask = (
        (red > 95)
        & (red < 190)
        & (green > 80)
        & (green < 170)
        & (blue < 120)
        & (red > blue * 1.15)
    )
    opened = ndi.binary_opening(brown_mask, iterations=1)
    labels, _ = ndi.label(opened)
    detections: list[tuple[float, float]] = []

    for label_id, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        component = labels[slices] == label_id
        area = int(component.sum())
        if area < 4 or area > 150:
            continue
        ys, xs = np.where(component)
        detections.append((slices[1].start + xs.mean(), slices[0].start + ys.mean()))

    return np.array(detections)


def build_digitized_frame(
    species: str,
    image_path,
    gcp_path,
    crop: tuple[int, int, int, int],
    detector,
    boundary,
) -> pd.DataFrame:
    cropped = crop_rgb(image_path, crop)
    gcp_frame = read_gcps(gcp_path)
    lon_params, lat_params, georeference_rmse_deg = fit_quadratic_transform(gcp_frame)
    detection_pixels = detector(cropped)
    if len(detection_pixels) == 0:
        raise ValueError(f"no {species} detections found in {image_path}")

    georeferenced = apply_quadratic_transform(lon_params, lat_params, detection_pixels)
    boundary_buffer = gpd.GeoSeries([boundary], crs=WGS84).to_crs(METRIC_CRS).buffer(10000).to_crs(WGS84).iloc[0]

    frame = pd.DataFrame(
        {
            "detection_id": [f"{species}_digitized_{index + 1:03d}" for index in range(len(georeferenced))],
            "species": species,
            "source_image": image_path.name,
            "pixel_x": np.round(detection_pixels[:, 0], 2),
            "pixel_y": np.round(detection_pixels[:, 1], 2),
            "longitude": np.round(georeferenced[:, 0], 6),
            "latitude": np.round(georeferenced[:, 1], 6),
            "within_boundary_buffer": [boundary_buffer.contains(Point(xy)) for xy in georeferenced],
            "georeference_method": "quadratic_gcp_warp_from_public_map_features",
            "georeference_rmse_deg": np.round(georeference_rmse_deg, 6),
            "control_point_count": len(gcp_frame),
            "notes": "approximate map-digitized detections from a public figure; not survey-grade coordinates",
        }
    )
    return frame


def write_digitized_csv(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_digitized_csv(path, species: str, minimum_rows: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing digitized reference csv: {path}")
    frame = pd.read_csv(path)
    required = [
        "detection_id",
        "species",
        "source_image",
        "pixel_x",
        "pixel_y",
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
    if len(frame) < minimum_rows:
        raise ValueError(f"{path} has too few rows for {species}: {len(frame)} < {minimum_rows}")
    if set(frame["species"]) != {species}:
        raise ValueError(f"{path} contains unexpected species labels: {sorted(set(frame['species']))}")
    if frame[["longitude", "latitude"]].isna().any().any():
        raise ValueError(f"{path} contains null coordinates")


def main() -> int:
    parser = argparse.ArgumentParser(description="digitize lion and rhino detections from image-only reference maps")
    parser.add_argument("--check", action="store_true", help="validate existing digitized raw csv outputs")
    args = parser.parse_args()

    if args.check:
        validate_digitized_csv(LION_OUTPUT_PATH, "lion", 20)
        validate_digitized_csv(RHINO_OUTPUT_PATH, "rhino", 100)
        print("reference-map digitization check passed")
        return 0

    boundary = load_boundary()
    lion_frame = build_digitized_frame(
        species="lion",
        image_path=LION_IMAGE_PATH,
        gcp_path=LION_GCP_PATH,
        crop=LION_CROP,
        detector=detect_lion_markers,
        boundary=boundary,
    )
    rhino_frame = build_digitized_frame(
        species="rhino",
        image_path=RHINO_IMAGE_PATH,
        gcp_path=RHINO_GCP_PATH,
        crop=RHINO_CROP,
        detector=detect_rhino_markers,
        boundary=boundary,
    )
    write_digitized_csv(lion_frame, LION_OUTPUT_PATH)
    write_digitized_csv(rhino_frame, RHINO_OUTPUT_PATH)
    validate_digitized_csv(LION_OUTPUT_PATH, "lion", 20)
    validate_digitized_csv(RHINO_OUTPUT_PATH, "rhino", 100)
    print(f"wrote {len(lion_frame)} lion detections to {LION_OUTPUT_PATH}")
    print(f"wrote {len(rhino_frame)} rhino detections to {RHINO_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
