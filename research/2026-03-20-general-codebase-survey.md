---
date: 2026-03-21T03:27:15Z
researcher: codex
git_commit: 379a49a8286c673a09db21d22bd4b937384b9bff
branch: master
topic: "General codebase survey"
tags: [research, codebase, geospatial-pipeline, optimization]
status: complete
---

# Research: General codebase survey

**Date**: 2026-03-21T03:27:15Z
**Git Commit**: `379a49a8286c673a09db21d22bd4b937384b9bff`
**Branch**: `master`

## Research Question

Provide a general, scoped survey of the codebase without going deep into every implementation detail.

## Summary

This repository is a staged geospatial analysis pipeline for Etosha risk mapping, followed by an optimization-input preparation layer. The current repo contract is documented in `README.md`, which describes raw inputs under `data/raw/`, normalized layers under `data/processed/`, reusable outputs under `outputs/`, and a manifest-driven provenance model for raw artifacts (`README.md:1-25`).

The code is organized primarily as numbered scripts under `scripts/`. Scripts `00` through `12` build and validate the spatial analysis pipeline: environment validation, raw manifest generation, normalized base layers, a 5 km analysis grid, per-cell spatial features, wildlife support layers, threat layers, a species-by-threat risk tensor, and visualization outputs (`scripts/00_validate_environment.py:9-49`, `scripts/06_build_grid.py:14-20`, `scripts/10_build_risk_tensor.py:13-24`, `scripts/11_visualize.py:16-27`, `scripts/12_validate_outputs.py:13-19`).

Scripts `13` through `18` extend the repository into surveillance optimization preparation. They derive candidate surveillance sites, terrain and operability proxy factors, intervention candidates, and validate the config plus upstream input contract. The optimization entry point currently validates inputs and configuration only; it does not solve an optimization model yet (`scripts/13_build_surveillance_candidate_sites.py:27-50`, `scripts/14_build_terrain_costs.py:18-32`, `scripts/15_build_surveillance_matrices.py:18-25`, `scripts/16_optimize_surveillance.py:23-28`).

## Detailed Findings

### Repository Shape

- Top-level directories are `data/`, `outputs/`, `plans/`, `scripts/`, and `snapshots/`, with processed geospatial artifacts already present in `data/processed/` and `outputs/`.
- `data/raw/README.md` defines the raw-data contract, expected dataset directories, and the manifest schema used to track provenance for source artifacts and manually digitized derivatives (`data/raw/README.md:1-53`).
- `scripts/_spatial_common.py` provides the shared path constants, CRS convention (`EPSG:4326`), geometry normalization, and GeoJSON/Parquet validation helpers used throughout the pipeline (`scripts/_spatial_common.py:10-15`, `scripts/_spatial_common.py:31-40`, `scripts/_spatial_common.py:50-75`).

### Data Contract And Provenance

- `scripts/00_validate_environment.py` checks for required directories, required files, and importable dependencies before the rest of the pipeline runs (`scripts/00_validate_environment.py:9-49`, `scripts/00_validate_environment.py:52-87`).
- `scripts/01_build_manifest.py` defines the raw data manifest in code. `SOURCE_DEFINITIONS` enumerates the datasets, their source URLs, acquisition methods, local paths, and notes for the current MVP inputs (`scripts/01_build_manifest.py:12-23`, `scripts/01_build_manifest.py:23-189`).
- The manifest script can either write `data/raw/manifest.csv` or validate that the existing file matches the in-code definitions via `--check` (`scripts/01_build_manifest.py:215-260`).

### Spatial Pipeline

- Boundary normalization starts with OSM-derived raw Etosha boundary and pan GeoJSON exports, unions them, clips the pan to the boundary, and writes `data/processed/etosha_boundary.geojson` plus `data/processed/pan_polygon.geojson` (`scripts/02_normalize_boundary.py:11-15`, `scripts/02_normalize_boundary.py:17-45`).
- Infrastructure normalization reads the Geofabrik Namibia roads shapefile zip plus curated/raw CSVs for waterholes, camps, and gates, clips/selects them against the processed boundary, and writes normalized GeoJSON layers for roads, tourist roads, waterholes, camps, and gates (`scripts/03_normalize_infrastructure.py:13-24`, `scripts/03_normalize_infrastructure.py:31-43`, `scripts/03_normalize_infrastructure.py:73-103`).
- `scripts/04_digitize_reference_maps.py` georeferences public lion and rhino figure images using quadratic GCP fits, detects markers in cropped images, and writes raw digitized detection CSVs back into `data/raw/` (`scripts/04_digitize_reference_maps.py:15-25`, `scripts/04_digitize_reference_maps.py:58-71`, `scripts/04_digitize_reference_maps.py:131-170`, `scripts/04_digitize_reference_maps.py:214-236`).
- `scripts/04_normalize_wildlife.py` converts raw elephant GBIF points into a processed Parquet layer and turns digitized lion/rhino detections into buffered support-area GeoJSON layers, clipping them against the boundary and pan polygon (`scripts/04_normalize_wildlife.py:11-20`, `scripts/04_normalize_wildlife.py:52-56`, `scripts/04_normalize_wildlife.py:85-116`).
- `scripts/05_normalize_wildfires.py` reads the EONET wildfire JSON feed export, extracts point observations, filters them to the processed boundary, and writes `data/processed/wildfire_history.geojson` (`scripts/05_normalize_wildfires.py:13-16`, `scripts/05_normalize_wildfires.py:22-43`).

### Grid, Features, And Risk Surfaces

- `scripts/06_build_grid.py` builds a 5 km clipped grid in UTM zone 33S (`EPSG:32733`), stores polygon cells and centroid points separately, and identifies each cell with a versioned `cell_id` (`scripts/06_build_grid.py:15-20`, `scripts/06_build_grid.py:26-79`).
- `scripts/07_build_features.py` computes per-cell distances to boundary, roads, tourist roads, gates, camps, waterholes, and the pan; pan overlap area/ratio; wildfire event counts; and a terrain-class label. It writes `outputs/grid_features.parquet` (`scripts/07_build_features.py:12-23`, `scripts/07_build_features.py:100-167`).
- `scripts/08_build_species_layers.py` joins the grid to those features and derives four per-cell wildlife layers: elephant density from KDE over GBIF points, rhino support from buffered rhino reference areas plus waterhole distance, lion support from lion zones, and herbivore support from waterhole, pan, fire, and terrain signals (`scripts/08_build_species_layers.py:13-20`, `scripts/08_build_species_layers.py:97-137`, `scripts/08_build_species_layers.py:140-180`).
- `scripts/09_build_threat_layers.py` combines the feature and species layers into normalized poaching, wildfire, and tourism measures. The component values and formula strings are stored alongside the outputs in `outputs/threat_layers.parquet` (`scripts/09_build_threat_layers.py:12-29`, `scripts/09_build_threat_layers.py:86-127`, `scripts/09_build_threat_layers.py:130-158`).
- `scripts/10_build_risk_tensor.py` multiplies species-presence columns by threat-exposure columns to create a 3D tensor with dimensions `[cell, species, threat]`, saves it to `outputs/risk_tensor.npz`, and writes a composite risk GeoJSON with per-cell risk summaries (`scripts/10_build_risk_tensor.py:16-24`, `scripts/10_build_risk_tensor.py:70-106`, `scripts/10_build_risk_tensor.py:109-159`).
- `scripts/11_visualize.py` renders both static PNG heatmaps and a Folium interactive HTML map from the composite layer and support layers, producing `outputs/risk_heatmaps.png` and `outputs/interactive_map.html` (`scripts/11_visualize.py:16-27`, `scripts/11_visualize.py:74-114`, `scripts/11_visualize.py:126-197`).
- `scripts/12_validate_outputs.py` validates that the main analysis outputs exist, share aligned `cell_id` ordering, include tensor metadata, and have non-empty visualization files (`scripts/12_validate_outputs.py:22-35`, `scripts/12_validate_outputs.py:42-76`).

### Optimization Preparation Layer

- `_optimization_common.py` centralizes YAML config loading and validation for asset types, daily availability, optimization scenarios, and the phase 1 input contract against `grid_features.parquet` and `composite_risk.geojson` (`scripts/_optimization_common.py:21-27`, `scripts/_optimization_common.py:28-109`, `scripts/_optimization_common.py:306-372`).
- `data/configs/asset_types.yaml` defines four asset types: `person`, `car`, `drone`, and `camera`, with unit cost, coverage radius, speed, eligible site kinds, terrain modifier metadata, unit caps, and camera-specific bundle/suppression fields (`data/configs/asset_types.yaml:1-74`).
- `data/configs/daily_asset_availability.yaml` defines per-scenario budget and daily capacity fields, currently for `etosha_placeholder_baseline` (`data/configs/daily_asset_availability.yaml:1-10`).
- `data/configs/optimization_scenarios.yaml` defines the baseline scenario’s active asset types, alpha values, candidate-site settings, protection-benefit weights, human-operability settings, and artificial waterhole intervention parameters (`data/configs/optimization_scenarios.yaml:1-42`).
- `scripts/13_build_surveillance_candidate_sites.py` builds candidate surveillance sites by combining existing camps, gates, waterholes, and top high-risk grid cells, deduplicating by distance, attaching nearest-cell context, and assigning support flags for each asset type. The output is `data/processed/surveillance_candidate_sites.geojson` (`scripts/13_build_surveillance_candidate_sites.py:27-50`, `scripts/13_build_surveillance_candidate_sites.py:108-118`, `scripts/13_build_surveillance_candidate_sites.py:154-181`, `scripts/13_build_surveillance_candidate_sites.py:184-212`, `scripts/13_build_surveillance_candidate_sites.py:241-350`).
- `scripts/14_build_terrain_costs.py` merges grid features with composite risk and scenario settings to derive habitat class, terrain roughness, mobility factors, camera visibility, protection benefit, and human operability penalty per grid cell, then writes `data/processed/terrain_cost_surface.parquet` (`scripts/14_build_terrain_costs.py:18-32`, `scripts/14_build_terrain_costs.py:84-208`).
- `scripts/15_build_surveillance_matrices.py` currently builds `data/processed/waterhole_interventions.geojson` by selecting remote candidate cells for artificial waterhole interventions using the candidate-site and terrain-cost outputs plus scenario settings (`scripts/15_build_surveillance_matrices.py:18-25`, `scripts/15_build_surveillance_matrices.py:81-152`).
- `scripts/16_optimize_surveillance.py` is the current optimization entry point. It validates the config bundle and phase 1 upstream input contract, and raises `NotImplementedError` unless called with `--validate-only` (`scripts/16_optimize_surveillance.py:9-28`).
- `scripts/18_validate_optimization_outputs.py` is a narrower validation wrapper over the same config bundle and phase 1 input contract (`scripts/18_validate_optimization_outputs.py:9-21`).

## Code References

- `README.md:1-25` - top-level repository contract and bootstrap commands.
- `data/raw/README.md:1-53` - raw-data layout and manifest contract.
- `scripts/_spatial_common.py:10-15` - shared root/path constants and CRS convention.
- `scripts/01_build_manifest.py:23-189` - in-code dataset manifest definitions.
- `scripts/06_build_grid.py:26-79` - clipped 5 km grid and centroid generation.
- `scripts/07_build_features.py:130-167` - per-cell distance, overlap, fire-count, and terrain feature derivation.
- `scripts/08_build_species_layers.py:165-180` - normalized wildlife support layer assembly.
- `scripts/09_build_threat_layers.py:138-158` - threat-layer composition and formula persistence.
- `scripts/10_build_risk_tensor.py:92-159` - tensor creation and composite risk output writing.
- `scripts/11_visualize.py:126-197` - Folium interactive map rendering.
- `scripts/13_build_surveillance_candidate_sites.py:337-350` - candidate-site assembly flow.
- `scripts/14_build_terrain_costs.py:119-176` - protection-benefit and operability derivation.
- `scripts/_optimization_common.py:338-372` - optimization input-contract validation and bundle summary.

## Architecture Documentation

The current architecture is file-oriented and stage-based. Each numbered script consumes artifacts from earlier stages, writes named outputs into `data/processed/` or `outputs/`, and exposes a `--check` mode that validates the current output contract rather than rebuilding it.

The spatial side of the repository works from raw source artifacts toward derived gridded analysis products. It starts with normalized park geometry and infrastructure layers, overlays those onto a fixed 5 km grid, derives per-cell features, then uses those features to compute species support, threat measures, and a species-by-threat tensor. Visualization is downstream from the composite risk surface.

The optimization side is currently an input-preparation and validation layer. It depends on the composite/grid outputs plus YAML scenario/config files, then derives surveillance candidate sites, terrain-operability factors, and waterhole intervention candidates. The repository currently validates these inputs but does not yet include a solve implementation in the `16_optimize_surveillance.py` entry point.

## Open Questions

- None for this scoped survey. This document is intended as a current-state map of the existing repository structure and runtime flow.
