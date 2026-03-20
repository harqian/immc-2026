#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_DIRECTORIES = [
    "data/raw",
    "data/raw/etosha_boundary",
    "data/raw/roads",
    "data/raw/waterholes",
    "data/raw/gates",
    "data/raw/camps",
    "data/raw/elephants",
    "data/raw/rhino_reference",
    "data/raw/carnivore_reference",
    "data/raw/wildfires",
    "data/processed",
    "outputs",
    "scripts",
]
REQUIRED_FILES = [
    "requirements.txt",
    "data/raw/README.md",
    "data/raw/manifest.csv",
    "scripts/01_build_manifest.py",
]
REQUIRED_MODULES = [
    "contextily",
    "folium",
    "geopandas",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "pyarrow",
    "pyogrio",
    "pyproj",
    "rasterio",
    "requests",
    "scipy",
    "shapely",
]


def validate_paths() -> list[str]:
    errors: list[str] = []
    for relative_path in REQUIRED_DIRECTORIES:
        path = PROJECT_ROOT / relative_path
        if not path.is_dir():
            errors.append(f"missing directory: {relative_path}")
    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            errors.append(f"missing file: {relative_path}")
    return errors


def validate_imports() -> list[str]:
    errors: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - import failure path is runtime-only
            errors.append(f"failed to import {module_name}: {exc}")
    return errors


def main() -> int:
    errors = []
    errors.extend(validate_paths())
    errors.extend(validate_imports())

    if errors:
        print("environment validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("environment validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
