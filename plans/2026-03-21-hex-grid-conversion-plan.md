# Hex Grid Conversion Implementation Plan

## Overview

replace the current square Etosha analysis grid with a regular hexagonal tessellation whose cell area is approximately equal to the current 25 km² square cells, then propagate that change through every downstream stage that consumes grid geometry or grid metadata so all derived outputs remain internally consistent.

## Current State Analysis

the current pipeline defines the analysis lattice in one place and then treats its outputs as a stable contract everywhere else.

- [`scripts/06_build_grid.py#L26`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py#L26) through [`scripts/06_build_grid.py#L79`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py#L79) generate clipped square cells with `shapely.geometry.box(...)`, emit `grid.geojson` and `grid_centroids.geojson`, and encode square-lattice identity through `row_index`, `col_index`, and `cell_id`.
- [`scripts/07_build_features.py#L26`](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L26) through [`scripts/07_build_features.py#L78`](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L78) require those grid columns as input and derive distances, overlaps, wildfire counts, and terrain classes from the grid polygons and centroids.
- [`scripts/10_build_risk_tensor.py#L27`](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L27) through [`scripts/10_build_risk_tensor.py#L67`](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L67) still validate the inherited grid metadata and write it back into `composite_risk.geojson`.
- [`scripts/12_validate_outputs.py#L36`](/Users/hq/code/immc-risk-intervention/scripts/12_validate_outputs.py#L36) through [`scripts/12_validate_outputs.py#L70`](/Users/hq/code/immc-risk-intervention/scripts/12_validate_outputs.py#L70) assume the final analysis outputs all align on the same `cell_id` ordering.
- [`scripts/13_build_surveillance_candidate_sites.py#L66`](/Users/hq/code/immc-risk-intervention/scripts/13_build_surveillance_candidate_sites.py#L66) through [`scripts/13_build_surveillance_candidate_sites.py#L105`](/Users/hq/code/immc-risk-intervention/scripts/13_build_surveillance_candidate_sites.py#L105) and [`scripts/15_build_surveillance_matrices.py#L47`](/Users/hq/code/immc-risk-intervention/scripts/15_build_surveillance_matrices.py#L47) through [`scripts/15_build_surveillance_matrices.py#L89`](/Users/hq/code/immc-risk-intervention/scripts/15_build_surveillance_matrices.py#L89) depend on `grid_centroids.geojson` and grid-derived terrain/cell outputs for optimization prep.
- [`scripts/11_visualize.py#L74`](/Users/hq/code/immc-risk-intervention/scripts/11_visualize.py#L74) through [`scripts/11_visualize.py#L156`](/Users/hq/code/immc-risk-intervention/scripts/11_visualize.py#L156) and [`scripts/17_visualize_optimization.py#L90`](/Users/hq/code/immc-risk-intervention/scripts/17_visualize_optimization.py#L90) through [`scripts/17_visualize_optimization.py#L174`](/Users/hq/code/immc-risk-intervention/scripts/17_visualize_optimization.py#L174) are geometry-driven, so they will render hexagons automatically once the upstream polygon layers change.

the current square-specific leakage is the main implementation constraint:

- `box(...)` hardcodes square geometry.
- `GRID_SIZE_M = 5000` currently implies square edge length, not a generic cell-scale parameter.
- `row_index` and `col_index` are required in multiple contracts even though most downstream calculations only need `cell_id`, centroid coordinates, area, and geometry.

## Desired End State

the pipeline produces a single, consistent set of hexagonal grid-derived artifacts:

- `outputs/grid.geojson` contains clipped regular hexagon cells covering the Etosha boundary.
- `outputs/grid_centroids.geojson` contains matching centroids for those hex cells.
- downstream derived outputs (`grid_features.parquet`, `species_layers.parquet`, `threat_layers.parquet`, `risk_tensor.npz`, `composite_risk.geojson`, optimization prep artifacts, and visualization outputs) all rebuild successfully from the hex grid without referencing stale square-specific assumptions.
- the visual outputs show hexagonal cells rather than square cells.

the chosen geometric invariant is approximate area preservation:

- target area per hex cell should stay close to the current `25_000_000 m²` represented by `GRID_SIZE_M = 5000`.
- exact cell count parity with the current square grid is not required.

### Key Discoveries

- [`scripts/06_build_grid.py#L41`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py#L41) is the only place where square polygons are materially created.
- [`scripts/07_build_features.py#L115`](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L115) through [`scripts/07_build_features.py#L167`](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L115) operate on geometry and centroids generically, so they should survive a polygon-shape change once input contract requirements are updated.
- [`scripts/10_build_risk_tensor.py#L135`](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L135) through [`scripts/10_build_risk_tensor.py#L159`](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L135) still forward square-grid metadata into `composite_risk.geojson`, so output schema cleanup is required.
- visualization code does not need bespoke hex rendering logic because it already plots polygon geometries directly.

## What We're NOT Doing

- we are not preserving exact square-grid row and column semantics for analytical meaning.
- we are not trying to maintain one-to-one comparability with prior square-grid cell IDs or prior cell counts.
- we are not changing risk formulas, optimization heuristics, or visualization styling beyond what the geometry change naturally causes.
- we are not introducing configurable multiple tessellation types unless that becomes necessary during implementation.
- we are not backfilling migration adapters for old square-based output files; the plan assumes full regeneration of derived artifacts.

## Implementation Approach

make the tessellation change at the source and then reduce downstream contracts to the metadata that actually matters. specifically:

1. replace square cell generation with regular hexagon generation in metric CRS.
2. rename or reinterpret square-specific grid constants and metadata so the new schema is honest.
3. update every loader and validator that hardcodes square-specific required columns.
4. rebuild all dependent outputs from the new grid artifacts in pipeline order.
5. verify both contract integrity and visible geometry changes.

the main schema decision is to stop forcing downstream stages to require `row_index` and `col_index`. those columns describe the old square lattice more strongly than the rest of the pipeline needs. if implementation convenience requires retaining lattice indices for the hex generator, they should be treated as optional generator metadata rather than a required cross-stage contract.

## Phase 1: Replace Grid Generation With Hexagons

### Overview

update the source-of-truth grid builder so it emits clipped regular hexagon polygons and matching centroids with approximately preserved area.

### Changes Required

#### 1. hex geometry generation
**File**: [`scripts/06_build_grid.py`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py)
**Changes**:

- replace `box(...)`-based square creation with a helper that constructs regular hexagon polygons in `EPSG:32733`.
- derive a hex side length from the current target cell area:

```python
target_area_m2 = GRID_TARGET_AREA_M2
hex_side_length_m = math.sqrt((2.0 * target_area_m2) / (3.0 * math.sqrt(3.0)))
```

- choose one orientation consistently, preferably flat-top hexagons for simpler row stepping, and compute horizontal/vertical spacing from that orientation.
- iterate candidate hex centers across the boundary bounding box, clip each hexagon to the Etosha boundary, discard empty intersections, and compute centroids from the clipped geometry.
- update grid versioning to reflect the geometry change, for example `etosha_hex_25km2_v1`.

#### 2. grid metadata cleanup
**File**: [`scripts/06_build_grid.py`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py)
**Changes**:

- replace `GRID_SIZE_M` with a geometry-agnostic parameter such as `GRID_TARGET_AREA_M2`.
- add explicit hex metadata columns such as `cell_target_area_m2` and `hex_side_length_m`.
- stop requiring `row_index` and `col_index` as stable output contract fields.
- generate `cell_id` values that do not imply square cells. acceptable patterns:

```python
cell_id = f"{GRID_VERSION}_{cell_index:04d}"
```

or, if lattice indexing is useful internally:

```python
cell_id = f"{GRID_VERSION}_r{row_index:03d}_q{col_index:03d}"
```

the preferred approach is sequential `cell_index` IDs because they avoid encoding obsolete square semantics into durable identifiers.

#### 3. output checks
**File**: [`scripts/06_build_grid.py`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py)
**Changes**:

- update `check_outputs()` to validate the new required columns.
- add sanity checks for:
  - non-null `cell_id`
  - equal row counts between polygons and centroids
  - aligned `cell_id` ordering
  - positive `cell_area_m2`
  - non-empty geometries

### Success Criteria

#### Automated Verification
- [x] `act && python3 scripts/06_build_grid.py`
- [x] `act && python3 scripts/06_build_grid.py --check`
- [x] `outputs/grid.geojson` and `outputs/grid_centroids.geojson` are regenerated and non-empty

#### Manual Verification
- [ ] inspect [`outputs/grid.geojson`](/Users/hq/code/immc-risk-intervention/outputs/grid.geojson) or a quick rendered preview and confirm interior cells are visibly hexagonal rather than rectangular
- [ ] inspect [`outputs/grid_centroids.geojson`](/Users/hq/code/immc-risk-intervention/outputs/grid_centroids.geojson) and confirm centroid count matches polygon count

**Implementation Note**: after completing this phase and the automated verification above passes, pause and confirm the grid artifact shape visually before proceeding.

---

## Phase 2: Update Downstream Analysis Contracts

### Overview

remove square-grid assumptions from the analysis pipeline so all feature, threat, and risk outputs can be regenerated from the hex grid without schema breakage.

### Changes Required

#### 1. feature pipeline input contract
**File**: [`scripts/07_build_features.py`](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py)
**Changes**:

- update `load_inputs()` so the required grid and centroid columns reflect the new hex schema.
- update `features = grid_metric[...]` selection to keep only the grid metadata that still exists.
- keep geometry-based calculations unchanged unless a schema dependency breaks them.
- update `check_outputs()` to validate the new column set.

expected retained columns:

```python
[
    "cell_id",
    "metric_crs",
    "grid_version",
    "cell_area_m2",
    "centroid_x_m",
    "centroid_y_m",
    "geometry",
]
```

expected new retained columns if added in phase 1:

```python
[
    "cell_target_area_m2",
    "hex_side_length_m",
]
```

#### 2. tensor and composite output schema
**File**: [`scripts/10_build_risk_tensor.py`](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py)
**Changes**:

- update grid validation to stop requiring dropped square-specific columns.
- update the `composite = merged[[...]]` selection so the output schema remains honest and matches the new upstream grid metadata.
- preserve `cell_id`, `cell_area_m2`, `grid_version`, and risk columns.

#### 3. final output validation
**File**: [`scripts/12_validate_outputs.py`](/Users/hq/code/immc-risk-intervention/scripts/12_validate_outputs.py)
**Changes**:

- update required grid columns in the final validation wrapper.
- keep `cell_id` ordering validation unchanged.

#### 4. any other analysis-stage validators/loaders
**Files**:
- [`scripts/08_build_species_layers.py`](/Users/hq/code/immc-risk-intervention/scripts/08_build_species_layers.py)
- [`scripts/09_build_threat_layers.py`](/Users/hq/code/immc-risk-intervention/scripts/09_build_threat_layers.py)

**Changes**:

- verify whether either file hardcodes square-era grid columns.
- update required column lists if they still mention `grid_size_m`, `row_index`, or `col_index`.

### Success Criteria

#### Automated Verification
- [x] `act && python3 scripts/07_build_features.py`
- [x] `act && python3 scripts/08_build_species_layers.py`
- [x] `act && python3 scripts/09_build_threat_layers.py`
- [x] `act && python3 scripts/10_build_risk_tensor.py`
- [x] `act && python3 scripts/12_validate_outputs.py`

#### Manual Verification
- [ ] inspect [`outputs/composite_risk.geojson`](/Users/hq/code/immc-risk-intervention/outputs/composite_risk.geojson) and confirm risk polygons are hex-shaped
- [ ] confirm there are no obviously stale square-era metadata fields left in the regenerated composite output unless they have been intentionally redefined

**Implementation Note**: after completing this phase and all automated verification passes, pause for manual confirmation that the core analysis outputs are coherent before proceeding.

---

## Phase 3: Update Optimization Preparation Contracts

### Overview

make the optimization-prep layer consume the regenerated hex grid artifacts without depending on square-only metadata.

### Changes Required

#### 1. candidate site generation
**File**: [`scripts/13_build_surveillance_candidate_sites.py`](/Users/hq/code/immc-risk-intervention/scripts/13_build_surveillance_candidate_sites.py)
**Changes**:

- update `load_inputs()` to validate the new centroid schema instead of requiring square-era columns.
- keep `make_high_risk_site_records()` logic intact because it ranks by `composite_risk_norm` and uses centroid geometry, both of which remain valid with hex cells.
- verify any nearest-cell or merged output fields still derive correctly from regenerated centroids.

#### 2. surveillance matrices / intervention prep
**File**: [`scripts/15_build_surveillance_matrices.py`](/Users/hq/code/immc-risk-intervention/scripts/15_build_surveillance_matrices.py)
**Changes**:

- update centroid validation to the new schema.
- keep cell-level distance and priority logic intact unless regenerated cell counts or spacing expose hidden assumptions.

#### 3. optimization validation wrappers
**Files**:
- [`scripts/16_optimize_surveillance.py`](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py)
- [`scripts/18_validate_optimization_outputs.py`](/Users/hq/code/immc-risk-intervention/scripts/18_validate_optimization_outputs.py)
- [`scripts/_optimization_common.py`](/Users/hq/code/immc-risk-intervention/scripts/_optimization_common.py)

**Changes**:

- verify whether shared optimization input-contract validation still expects dropped grid columns.
- update those required field lists if necessary.

### Success Criteria

#### Automated Verification
- [x] `act && python3 scripts/13_build_surveillance_candidate_sites.py --scenario-id etosha_placeholder_baseline`
- [x] `act && python3 scripts/14_build_terrain_costs.py --scenario-id etosha_placeholder_baseline`
- [x] `act && python3 scripts/15_build_surveillance_matrices.py --scenario-id etosha_placeholder_baseline`
- [x] `act && python3 scripts/16_optimize_surveillance.py --validate-only`
- [x] `act && python3 scripts/18_validate_optimization_outputs.py`

#### Manual Verification
- [ ] inspect regenerated candidate and intervention outputs to confirm they reference valid hex-grid-derived cell IDs
- [ ] spot-check that high-risk candidate sites still sit on centroid points within the new hex tessellation

**Implementation Note**: after completing this phase and all automated verification passes, pause for manual review of optimization prep outputs before moving to visualization refresh.

---

## Phase 4: Refresh Visualization Outputs And Confirm Visible Hexagons

### Overview

rebuild visualization artifacts from regenerated geometry and confirm the rendered maps now show hexagonal cells consistently.

### Changes Required

#### 1. analysis visualization refresh
**File**: [`scripts/11_visualize.py`](/Users/hq/code/immc-risk-intervention/scripts/11_visualize.py)
**Changes**:

- no logic changes are expected unless a required-column list or tooltip field still references removed square-era metadata.
- rerender outputs from the new upstream artifacts.

#### 2. optimization visualization refresh
**File**: [`scripts/17_visualize_optimization.py`](/Users/hq/code/immc-risk-intervention/scripts/17_visualize_optimization.py)
**Changes**:

- no geometry-specific changes are expected because polygon rendering is already generic.
- rerender outputs after optimization-prep regeneration.

### Success Criteria

#### Automated Verification
- [ ] `act && python3 scripts/11_visualize.py`
- [ ] `act && python3 scripts/17_visualize_optimization.py`
- [ ] both visualization scripts pass their built-in output checks

#### Manual Verification
- [ ] open [`outputs/risk_heatmaps.png`](/Users/hq/code/immc-risk-intervention/outputs/risk_heatmaps.png) and confirm the filled cells are visibly hexagonal
- [ ] open [`outputs/interactive_map.html`](/Users/hq/code/immc-risk-intervention/outputs/interactive_map.html) and confirm the interactive composite layer renders hexagons
- [ ] open [`outputs/optimization_diagnostics.png`](/Users/hq/code/immc-risk-intervention/outputs/optimization_diagnostics.png) and [`outputs/optimization_map.html`](/Users/hq/code/immc-risk-intervention/outputs/optimization_map.html) and confirm optimization cells also render as hexagons wherever cell polygons are displayed

**Implementation Note**: after this phase, the shape change should be human-visible in every map artifact that renders grid-derived polygons.

---

## Testing Strategy

### Unit Tests

- add a focused helper-level test or one-off verification script for hexagon generation math if there is already a testing location for script helpers
- verify that generated hex polygons have six sides before clipping for interior cells
- verify that hex side length yields cell area within an acceptable tolerance of the 25 km² target

### Integration Tests

- rebuild the pipeline from `scripts/06_build_grid.py` through `scripts/12_validate_outputs.py`
- rebuild the optimization-prep pipeline from `scripts/13_build_surveillance_candidate_sites.py` through `scripts/18_validate_optimization_outputs.py`
- confirm `cell_id` ordering stays aligned across regenerated outputs

### Manual Testing Steps

1. run `act && python3 scripts/06_build_grid.py` and inspect [`outputs/grid.geojson`](/Users/hq/code/immc-risk-intervention/outputs/grid.geojson) for visible hex cells
2. run the full analysis rebuild through `scripts/12_validate_outputs.py`
3. open [`outputs/risk_heatmaps.png`](/Users/hq/code/immc-risk-intervention/outputs/risk_heatmaps.png) and verify the heatmap cells are hexagonal
4. open [`outputs/interactive_map.html`](/Users/hq/code/immc-risk-intervention/outputs/interactive_map.html) and verify the interactive polygon layer is hexagonal
5. run the optimization-prep rebuild and open [`outputs/optimization_map.html`](/Users/hq/code/immc-risk-intervention/outputs/optimization_map.html) to confirm downstream consistency

## Performance Considerations

- hex generation will likely produce a different number of cells than the current square grid; downstream runtime may move accordingly.
- clipping many hexagons can be slightly more expensive than clipping squares because each candidate polygon has more vertices.
- the current pipeline is artifact-driven rather than latency-sensitive, so correctness and consistency matter more than micro-optimizing generation speed.

## Migration Notes

- do not trust existing files in `outputs/` or `data/processed/` after the grid schema changes.
- regenerate all artifacts that depend directly or indirectly on `grid.geojson` or `grid_centroids.geojson`.
- update any research notes or handoff docs that describe the analysis lattice as “5 km square” after the implementation lands.

## References

- original request: “change all of the grids into hexagons” with approximate preservation acceptable
- square grid source: [`scripts/06_build_grid.py#L26`](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py#L26)
- downstream feature contract: [`scripts/07_build_features.py#L26`](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L26)
- composite output schema: [`scripts/10_build_risk_tensor.py#L135`](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L135)
- final analysis validation: [`scripts/12_validate_outputs.py#L36`](/Users/hq/code/immc-risk-intervention/scripts/12_validate_outputs.py#L36)
- optimization input consumer: [`scripts/13_build_surveillance_candidate_sites.py#L66`](/Users/hq/code/immc-risk-intervention/scripts/13_build_surveillance_candidate_sites.py#L66)
- optimization cell consumer: [`scripts/15_build_surveillance_matrices.py#L47`](/Users/hq/code/immc-risk-intervention/scripts/15_build_surveillance_matrices.py#L47)
- analysis visualization: [`scripts/11_visualize.py#L74`](/Users/hq/code/immc-risk-intervention/scripts/11_visualize.py#L74)
- optimization visualization: [`scripts/17_visualize_optimization.py#L90`](/Users/hq/code/immc-risk-intervention/scripts/17_visualize_optimization.py#L90)
