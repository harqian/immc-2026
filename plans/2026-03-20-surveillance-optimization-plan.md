# Etosha Surveillance Optimization Layer Implementation Plan

## Overview

Build a new downstream optimization layer that consumes the existing Etosha grid and composite risk outputs, then recommends daily deployment of surveillance assets across candidate sites. The optimizer will allocate people, cars, drones, and fixed cameras under a scalar budget and per-asset count caps to optimize two objectives: risk-weighted coverage and risk-weighted minimum response time.

This plan intentionally leaves external data-source selection open because the real datasets will be provided later. The goal here is to define the optimization work now, not to build a reusable source-ingestion platform.

## Current State Analysis

The repo already contains a complete geospatial risk-surface pipeline, but it stops at descriptive analytics:

- [scripts/06_build_grid.py](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py#L17) builds a metric 5 km grid over Etosha and assigns stable `cell_id` values.
- [scripts/07_build_features.py](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L100) computes reusable per-cell distance features, including distances to roads, tourist roads, gates, camps, waterholes, and the park boundary.
- [scripts/08_build_species_layers.py](/Users/hq/code/immc-risk-intervention/scripts/08_build_species_layers.py#L140) derives species support layers aligned to the grid.
- [scripts/09_build_threat_layers.py](/Users/hq/code/immc-risk-intervention/scripts/09_build_threat_layers.py#L86) creates threat surfaces, but “surveillance” is only represented indirectly as distance from camps.
- [scripts/10_build_risk_tensor.py](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L70) produces the analysis-ready tensor and `composite_risk_norm`, which is the correct weight input for optimization.

There is no existing representation of:

- deployable surveillance assets
- candidate surveillance sites
- daily asset availability
- costs or budgets
- response-time matrices
- integer programming or linear programming models
- Pareto frontier analysis for deployment tradeoffs

The repo is therefore well prepared for optimization inputs, but it does not yet contain any optimization machinery.

## Desired End State

After this plan is implemented, the repo should be able to:

- generate or load a finite set of candidate surveillance sites
- read the later-provided operational and terrain datasets needed by the model
- compute per-cell asset-specific coverage feasibility and response-time matrices
- solve a daily placement optimization under budget and per-asset caps
- report optimal or near-optimal deployment plans for people, cars, drones, and cameras
- expose both objective values:
  - risk-weighted coverage
  - risk-weighted minimum response time
- produce a small Pareto frontier by sweeping coverage constraints and minimizing response time
- export machine-readable outputs plus maps and tables for inspection

The plan is complete when a fresh checkout can run the optimization pipeline against the later-provided real inputs and produce:

- a deployment recommendation file
- per-cell achieved coverage and response outputs
- a frontier summary over several tradeoff settings
- static and interactive visual diagnostics

### Key Discoveries

- The optimizer should use the existing `composite_risk_norm` output rather than invent a new risk score, because the user chose composite-only optimization and the repo already exports it in [outputs/composite_risk.geojson](/Users/hq/code/immc-risk-intervention/outputs/composite_risk.geojson).
- `composite_risk_norm` should remain the base risk source throughout the optimization; it should not be replaced by `protection_benefit`.
- The existing grid and distance features already eliminate most geospatial preprocessing work for v1, especially in [scripts/07_build_features.py](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L130).
- A daily placement problem with finite candidate sites is best modeled as a mixed-integer facility-location / maximum-coverage formulation rather than vehicle routing.
- Cameras and mobile assets should not be collapsed into one generic asset type: cameras primarily affect coverage, while people, cars, and drones affect both coverage and response time.
- Camera range is far smaller than the grid resolution, so cameras should not be modeled as ordinary cell-scale sensing assets. At v1, cameras should be treated as waterhole-specific risk-suppression assets.
- The optimization should use a positive per-cell `protection_benefit` proxy to represent where better protection does the most good for wildlife.
- Human performance should be reduced by a separate `human_operability_penalty` that depends on animal abundance, distance from camp, and terrain roughness.
- `protection_benefit` is a coverage/protection objective weight, not a replacement risk map.
- The park boundary should remain the fence/access proxy in v1; the plan should not wait for a separate fence dataset.
- Artificial waterholes should be treated as proactive interventions with both financial cost and an infrastructure/tourism side effect, not just as passive background features.

## What We're NOT Doing

- We are not building real-time dispatch, within-day rerouting, or a persistent simulation of moving agents.
- We are not solving a continuous siting problem over arbitrary coordinates; all deployment happens on a finite candidate-site set.
- We are not coupling this phase to any specific external source links for terrain, telemetry, or infrastructure refreshes.
- We are not building a web app or operational control dashboard.
- We are not assuming sensitive wildlife telemetry will be available publicly.
- We are not implementing stochastic uncertainty modeling, adversarial behavior modeling, or patrol scheduling in this phase.
- We are not building road-network routing from first principles if a simpler travel-time proxy is sufficient for the MVP.
- We are not using a separate fence dataset in v1; the boundary remains the explicit proxy for perimeter access and management edge effects.

## Implementation Approach

Use a staged downstream optimization pipeline that sits on top of the existing geospatial outputs:

1. define the minimum required model inputs we will expect once the real data arrives
2. generate finite candidate sites from existing processed layers and risk outputs
3. build protection-benefit, human-operability, and proactive-waterhole inputs
4. precompute asset-specific coverage and response matrices from sites to cells
5. solve a mixed-integer daily placement problem
6. trace a small Pareto frontier with an epsilon-constraint formulation
7. export recommendation artifacts and visual diagnostics

### Optimization Formulation

The default model should be a mixed-integer program implemented in `pyomo`, with HiGHS as the default open-source solver.

The formulation should use:

- integer site-allocation variables for each asset type
- binary or bounded assignment variables linking sites to covered cells
- binary or bounded variables for the selected fastest responder per cell

### Camera Modeling Rule

Because the grid is 5 km and the expected camera sensing radius is only on the order of tens of meters, cameras should not be modeled as normal spatial coverage assets in the same way as people, cars, or drones.

Instead, the optimization should treat cameras as a waterhole-lockdown mechanism:

- cameras may only be deployed at eligible waterhole sites unless later data says otherwise
- cameras do not materially improve the response-time objective
- cameras contribute little or nothing to cell-scale coverage at 5 km resolution
- cameras increase achieved protection in cells influenced by the protected waterhole

The v1 assumption should be discrete and explicit:

- deploying `5` cameras at a waterhole means that waterhole is treated as effectively locked down
- a locked-down waterhole increases local achieved protection in nearby cells by a configured gain factor

The chosen v1 implementation is the pragmatic approximation:

```text
protected_gain[c] = base_protected_gain[c] + camera_gain_factor * camera_lockdown_influence[c,w]
```

Where:

- `base_protected_gain[c]` is the baseline protection effect without camera lockdown
- `camera_lockdown_influence[c,w]` is the influence of waterhole `w` on cell `c`
- `camera_gain_factor` is a configured constant for a fully locked-down waterhole

If partial camera deployment is allowed, then the suppression can scale up to the full effect at `5` cameras. If partial deployment is not useful operationally, then the model should require cameras at waterholes in bundles of `5`.

This is intentionally a pragmatic v1 approximation. It improves achieved protection near locked-down waterholes directly, rather than modifying the underlying composite risk map.

The camera effect must still remain local:

- only cells inside the configured waterhole influence zone may receive camera-based protection gain
- distant or unrelated cells must not be affected
- the gain term must stay bounded by explicit configured limits

The model should optimize two objectives:

1. maximize protection-benefit-weighted coverage
2. minimize composite-risk-weighted minimum response time

### Fire Response Urgency Rule

The optimization must preserve the two-objective structure:

1. protection-benefit-weighted coverage
2. composite-risk-weighted response

Fire urgency should not replace the second objective. Instead, it should make the response objective much harsher for cells where delayed arrival is especially dangerous because fire can spread rapidly.

The chosen v1 form is a thresholded exponential penalty inside the response objective:

```text
fire_delay_penalty[c] = wildfire_risk[c] * (exp(beta_fire * max(0, t[c] - tau_fire_min)) - 1)
```

Where:

- `wildfire_risk[c]` is the wildfire-related risk for cell `c`
- `t[c]` is realized response time for cell `c`
- `tau_fire_min` is the response-time threshold after which fire delay becomes much more dangerous
- `beta_fire` controls how sharply the penalty ramps after the threshold

This means:

- small response-time differences below the threshold are not over-penalized
- once response exceeds the threshold, the optimizer should strongly prefer faster fire access
- high-wildfire-risk cells become disproportionately important when response is slow

Because exact exponentials are inconvenient inside a mixed-integer solver, v1 should implement this with a piecewise-linear approximation of the thresholded exponential curve.

The recommended solve pattern is epsilon-constraint rather than a single weighted sum:

- solve once for maximum achievable risk-weighted coverage
- enforce `coverage >= alpha * coverage_max` for a small set of `alpha` values
- under each `alpha`, minimize risk-weighted response time

This produces a defensible tradeoff frontier instead of hiding the tradeoff in arbitrary objective weights.

## Repository Layout

The new optimization layer should add files like:

```text
.
├── data/
│   ├── processed/
│   │   ├── surveillance_candidate_sites.geojson
│   │   ├── terrain_cost_surface.parquet
│   │   ├── waterhole_interventions.geojson
│   │   ├── coverage_matrix.parquet
│   │   └── response_time_matrix.parquet
│   └── configs/
│       ├── asset_types.yaml
│       ├── daily_asset_availability.yaml
│       └── optimization_scenarios.yaml
├── outputs/
│   ├── optimization_frontier.csv
│   ├── optimization_solution.geojson
│   ├── optimization_cells.parquet
│   ├── optimization_summary.json
│   ├── optimization_frontier.png
│   └── optimization_map.html
├── scripts/
│   ├── 13_build_surveillance_candidate_sites.py
│   ├── 14_build_terrain_costs.py
│   ├── 15_build_surveillance_matrices.py
│   ├── 16_optimize_surveillance.py
│   ├── 17_visualize_optimization.py
│   └── 18_validate_optimization_outputs.py
└── plans/
    └── 2026-03-20-surveillance-optimization-plan.md
```

## Expected Inputs

These are the minimum model inputs we will need once the real datasets are provided. They are not meant to define a reusable platform contract; they are just the concrete files the Etosha optimizer will expect.

### 1. Candidate Sites

**File**: `data/processed/surveillance_candidate_sites.geojson`

Required fields:

- `site_id`
- `site_kind`
- `source`
- `candidate_rank`
- `supports_people`
- `supports_cars`
- `supports_drones`
- `supports_cameras`
- `base_cost_fixed`
- `waterhole_influence_radius_m`
- `geometry`

Expected site kinds for v1:

- `waterhole`
- `camp`
- `gate`
- `high_risk_cell`
- `manual_override`

For waterhole sites, the candidate-site artifact should also preserve enough metadata to support camera suppression logic.

### 2. Asset Parameters

**File**: `data/configs/asset_types.yaml`

Required fields per asset type:

- `asset_type`
- `unit_cost`
- `coverage_radius_m`
- `response_speed_kmh`
- `site_eligibility`
- `terrain_modifier_profile`
- `max_units_per_site`
- `counts_toward_budget`
- `camera_bundle_size`
- `risk_suppression_factor`

Expected v1 asset types:

- `person`
- `car`
- `drone`
- `camera`

### 3. Daily Asset Availability

**File**: `data/configs/daily_asset_availability.yaml`

Required fields:

- `scenario_id`
- `budget_total`
- `max_people`
- `max_cars`
- `max_drones`
- `max_cameras`
- `tau_fire_min`
- `beta_fire`
- `lambda_fire`

### 4. Terrain Cost Surface

**File**: `data/processed/terrain_cost_surface.parquet`

Required fields:

- `cell_id`
- `terrain_class`
- `habitat_class`
- `slope_mean_deg`
- `foot_speed_factor`
- `car_speed_factor`
- `drone_speed_factor`
- `camera_visibility_factor`
- `source`

This input should also preserve or join the inputs needed to derive:

- `protection_benefit`
- `human_operability_penalty`

### 5. Proactive Waterhole Interventions

**File**: `data/processed/waterhole_interventions.geojson`

Required fields:

- `intervention_site_id`
- `kind`
- `capital_cost`
- `tourism_cost`
- `expected_density_dispersion_benefit`
- `geometry`

Expected v1 kinds:

- `artificial_waterhole`

### 6. Coverage Matrix

**File**: `data/processed/coverage_matrix.parquet`

Required fields:

- `site_id`
- `cell_id`
- `asset_type`
- `is_coverable`
- `effective_coverage_score`

### 7. Response Time Matrix

**File**: `data/processed/response_time_matrix.parquet`

Required fields:

- `site_id`
- `cell_id`
- `asset_type`
- `response_time_min`
- `travel_mode`
- `source`

## Phase 1: Minimal Inputs And Dependency Setup

### Overview

Add the minimum files, parameters, and dependencies needed for optimization work before the real datasets are wired in.

### Changes Required

#### 1. Add optimization dependencies
**Files**: `requirements.txt`, optionally a dedicated optimization requirements block in `README.md`

**Changes**:

- add `pyomo`
- add a HiGHS-compatible interface such as `highspy`
- keep geospatial and plotting dependencies aligned with the existing pipeline

#### 2. Add temporary parameter files
**Files**: `data/configs/asset_types.yaml`, `data/configs/daily_asset_availability.yaml`, `data/configs/optimization_scenarios.yaml`

**Changes**:

- define only the required keys the optimizer needs
- include one example Etosha scenario using placeholder values where real values are not known yet
- avoid “safe defaults”; missing parameters should fail loudly
- include explicit parameters controlling the `protection_benefit` proxy
- include explicit parameters controlling how `human_operability_penalty` is built from animal abundance, camp distance, and terrain roughness
- include explicit costs and benefits for artificial waterhole interventions

#### 3. Add validation utilities
**Files**: `scripts/16_optimize_surveillance.py`, `scripts/18_validate_optimization_outputs.py`, or a small shared helper

**Changes**:

- validate required files and columns
- validate that all asset types referenced in scenarios exist in the asset config
- validate that count caps and budget are non-negative and explicit
- validate that any artificial waterhole intervention has both monetary and tourism-cost terms defined explicitly

### Success Criteria

#### Automated Verification
- [x] dependency install succeeds in the project venv
- [x] config files parse and validate
- [x] missing required config keys fail loudly

#### Manual Verification
- [ ] the temporary parameter files are clear enough to be replaced by the real data later
- [ ] there are no implicit fallback values for cost, radius, speed, or asset limits

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 2: Candidate-Site Generation

### Overview

Create a finite deployment search space for the optimizer. Candidate sites should be generated primarily from waterholes, plus existing camps, gates, and optionally high-risk cells, until the real siting data is available.

### Changes Required

#### 1. Generate candidate sites
**File**: `scripts/13_build_surveillance_candidate_sites.py`

**Changes**:

- read [data/processed/waterholes.geojson](/Users/hq/code/immc-risk-intervention/data/processed/waterholes.geojson)
- read [data/processed/camps.geojson](/Users/hq/code/immc-risk-intervention/data/processed/camps.geojson)
- read [data/processed/gates.geojson](/Users/hq/code/immc-risk-intervention/data/processed/gates.geojson)
- read [outputs/composite_risk.geojson](/Users/hq/code/immc-risk-intervention/outputs/composite_risk.geojson)
- create site records for:
  - all waterholes
  - all camps
  - all gates
  - top `N` high-risk grid centroids
- deduplicate nearby sites within a configurable merge distance
- assign per-site asset eligibility flags

#### 2. Encode candidate ranking
**File**: `scripts/13_build_surveillance_candidate_sites.py`

**Changes**:

- preserve source type in `site_kind`
- assign `candidate_rank` so scenarios can restrict to top-ranked sites if needed

### Success Criteria

#### Automated Verification
- [x] `surveillance_candidate_sites.geojson` is written successfully
- [x] every site has a unique `site_id`
- [x] every site has at least one allowed asset type
- [x] every site lies within or plausibly adjacent to the park boundary

#### Manual Verification
- [ ] inspect the site map and confirm most candidates cluster around operationally plausible locations
- [ ] confirm high-risk synthetic sites are not overwhelming waterholes/camps/gates

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 3: Terrain And Mobility Factors

### Overview

Build a terrain and mobility layer so different asset types can have different effective coverage and response characteristics once the real terrain inputs are available.

### Changes Required

#### 1. Build terrain cost surface
**File**: `scripts/14_build_terrain_costs.py`

**Changes**:

- join the existing grid to the later-provided habitat and elevation inputs
- derive per-cell slope and habitat classes
- translate those into movement and sensing modifiers by asset type
- preserve or derive a waterhole influence field needed for camera suppression
- join species-abundance or wildlife-density information needed to derive a positive `protection_benefit` term
- join or derive the inputs needed for `human_operability_penalty`, including:
  - animal abundance or dangerous-wildlife concentration
  - distance from camp
  - terrain roughness or slope

#### 2. Define asset-specific terrain profiles
**File**: `data/configs/asset_types.yaml`

**Changes**:

- allow different terrain penalties for foot, car, drone, and camera visibility
- explicitly represent that drones may perform relatively better in steep or rough terrain
- represent `protection_benefit` as a positive wildlife-facing proxy, for example species-priority-weighted wildlife presence combined with threat exposure or intervention leverage
- represent `human_operability_penalty` as an explicit penalty on human movement, sensing, or both

### Success Criteria

#### Automated Verification
- [x] every grid cell has exactly one terrain-cost record
- [x] all speed and visibility modifiers are numeric and finite
- [x] no modifier is silently missing
- [x] `protection_benefit` is present and finite for every grid cell
- [x] `human_operability_penalty` is present and finite for every grid cell

#### Manual Verification
- [ ] steep or rough terrain plausibly penalizes people/cars more than drones
- [ ] open areas plausibly improve camera visibility relative to dense habitat
- [ ] cells with high wildlife abundance, large camp remoteness, or rough terrain visibly reduce human effectiveness more than drone effectiveness

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 4: Proactive Waterhole Interventions

### Overview

Model artificial waterholes as proactive interventions that can change wildlife distribution at a cost. These interventions should not be treated as free infrastructure.

### Changes Required

#### 1. Build intervention candidates
**File**: `scripts/15_build_surveillance_matrices.py` or a dedicated helper if the logic becomes large

**Changes**:

- define the candidate artificial-waterhole intervention sites
- assign both:
  - direct monetary cost
  - tourism/infrastructure cost
- assign an expected local wildlife-dispersion benefit

#### 2. Define the intervention effect
**File**: `scripts/15_build_surveillance_matrices.py`

**Changes**:

- model artificial waterholes as changing the local wildlife-distribution pattern in a way that improves the protection objective
- represent this as a change in the relevant `protection_benefit` field or its local spatial concentration
- make the sign convention explicit:
  - lower concentration is the intended benefit
  - increased tourism/infrastructure presence is the intended cost

### Success Criteria

#### Automated Verification
- [x] every intervention candidate has explicit monetary and tourism-cost terms
- [x] every intervention candidate has an explicit density-dispersion benefit term
- [x] intervention effects do not create negative density or invalid risk values

#### Manual Verification
- [ ] selected artificial waterholes improve local protection benefit structure rather than simply deleting wildlife value
- [ ] the tourism/infrastructure penalty is visible in the scenario accounting

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 5: Coverage And Response Matrix Precomputation

### Overview

Precompute the geometry-heavy inputs the solver needs so the optimization stage remains mostly tabular and deterministic.

### Changes Required

#### 1. Build coverage matrix
**File**: `scripts/15_build_surveillance_matrices.py`

**Changes**:

- compute whether each asset at each site can cover each cell
- define `effective_coverage_score` using:
  - risk-cell inclusion within sensor radius
  - terrain visibility factors
  - optional attenuation by distance inside the sensing radius

Coverage should differ by asset type:

- `camera`: do not treat as ordinary grid-scale coverage; instead compute waterhole-specific suppression influence
- `camera`: do not treat as ordinary grid-scale coverage; instead compute waterhole-specific protection gain influence
- `drone`: mobile sensing with radius and altitude-tolerant mobility
- `person`: short-range local sensing
- `car`: road-biased or road-assisted sensing and mobility

Human coverage logic should be penalized by `human_operability_penalty`.

The same script should compute a camera protection-gain influence table for waterhole sites, for example:

- `site_id`
- `cell_id`
- `asset_type = camera`
- `protection_gain_influence`

This table should quantify how strongly a locked-down waterhole improves protection in each nearby cell.

#### 2. Build response-time matrix
**File**: `scripts/15_build_surveillance_matrices.py`

**Changes**:

- compute response time from each site to each cell for each mobile asset type
- define v1 response time as “travel time from assigned base/site to cell”
- use the minimum across deployed assets in the optimization stage
- set camera response times to `null` or exclude cameras from the response-time objective entirely
- prepare any additional breakpoints needed for the thresholded fire-delay penalty approximation

Travel-time logic for v1 should be explicit and simple:

- `person`: Euclidean distance adjusted by `foot_speed_factor`
- `drone`: Euclidean distance adjusted by `drone_speed_factor`
- `car`: Euclidean or road-biased distance adjusted by `car_speed_factor`

Human response logic should also include `human_operability_penalty` where appropriate.

If road-network routing is not yet implemented, the response-time matrix should record that the car estimate is a proxy, not a routed guarantee.

### Success Criteria

#### Automated Verification
- [x] coverage matrix row count matches `sites x cells x asset_types`
- [x] response-time matrix row count matches all mobile asset/site/cell combinations
- [x] all response times are positive and finite where defined
- [x] cameras are excluded cleanly from response-time calculations
- [x] camera protection-gain influence is present for waterhole sites and absent elsewhere unless explicitly allowed
- [x] fire-delay penalty inputs are present and consistent with the chosen thresholded approximation

#### Manual Verification
- [ ] nearby cells have lower response times than distant cells
- [ ] cars are generally faster than people for distant cells under the same terrain
- [ ] drones retain good performance in terrain where cars or people degrade
- [ ] locking down a waterhole visibly improves local achieved protection without affecting unrelated distant areas
- [ ] fire-prone cells become much more expensive to leave with slow response after the threshold

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 6: Optimization Model

### Overview

Implement the daily surveillance placement model as a mixed-integer program with risk-weighted coverage maximization and risk-weighted response-time minimization.

### Changes Required

#### 1. Build the core model
**File**: `scripts/16_optimize_surveillance.py`

**Changes**:

- load candidate sites, terrain costs, matrices, asset parameters, availability caps, composite risk, and the derived `protection_benefit` / `human_operability_penalty` terms
- load any artificial-waterhole intervention candidates and their cost/benefit terms
- define integer deployment variables:
  - `x[s,a]`: count of asset type `a` placed at site `s`
- define coverage variables:
  - `y[c]`: whether cell `c` receives meaningful protection from at least one deployed sensing or monitoring asset
- define response variables:
  - `z[c,s,a]`: whether deployed mobile asset type `a` at site `s` is the selected responder for cell `c`
  - `t[c]`: realized minimum response time for cell `c`
- define fire-penalty variables:
  - piecewise-linear auxiliary variables needed to approximate the thresholded exponential fire-delay term
- define camera-lockdown variables:
  - `k[w]`: number of cameras deployed at waterhole site `w`
  - `l[w]`: whether waterhole site `w` is fully locked down
- define proactive intervention variables:
  - `u[h]`: whether artificial waterhole intervention `h` is selected

#### 2. Encode constraints
**File**: `scripts/16_optimize_surveillance.py`

**Changes**:

- budget constraint:
  - total deployed asset cost must be `<= budget_total`
- per-asset count caps:
  - deployed people `<= max_people`
  - deployed cars `<= max_cars`
  - deployed drones `<= max_drones`
  - deployed cameras `<= max_cameras`
- site eligibility:
  - asset deployments must respect site support flags
- site capacity:
  - each site must honor `max_units_per_site`
- camera bundle logic:
  - if v1 uses bundle deployment, `l[w] = 1` only when `k[w] >= 5`
  - if fractional bundles are disallowed, camera deployment at waterholes must occur in multiples tied to the lockdown rule
- coverage feasibility:
  - a cell can only be marked covered if at least one deployed asset can cover it
- response feasibility:
  - a cell can only select a responder from a deployed mobile asset/site pair
- risk suppression feasibility:
  - only waterhole sites with sufficient cameras may trigger camera-based protection gain
- intervention feasibility:
  - only selected artificial waterholes may change the local `protection_benefit` structure
- fire-delay approximation feasibility:
  - the auxiliary fire penalty variables must correctly represent the thresholded exponential penalty as a piecewise-linear approximation

#### 3. Encode the objectives
**File**: `scripts/16_optimize_surveillance.py`

**Changes**:

- coverage objective:

```text
maximize sum_c protection_benefit_after_interventions[c] * y[c]
```

- response objective:

```text
minimize sum_c composite_risk[c] * t[c]
       + lambda_fire * sum_c wildfire_risk[c] * (exp(beta_fire * max(0, t[c] - tau_fire_min)) - 1)
       + tourism_penalty_term
```

Where:

- `protection_benefit_after_interventions[c]` is the post-camera, post-intervention benefit proxy after any locked-down waterholes and selected artificial-waterhole interventions are applied
- `composite_risk[c]` remains the base risk source for the response objective
- `human_operability_penalty[c]` is applied in the human-specific coverage and response feasibility terms, not folded into the positive wildlife-benefit term

In implementation, the exponential term should be replaced by a piecewise-linear approximation with explicit breakpoints.

#### 4. Implement epsilon-constraint frontier generation
**File**: `scripts/16_optimize_surveillance.py`

**Changes**:

- solve once for maximum achievable coverage
- for a configured set of `alpha` values, enforce:

```text
sum_c protection_benefit_after_interventions[c] * y[c] >= alpha * protection_max
```

- under each `alpha`, minimize weighted response time
- write each solve to a frontier table

### Success Criteria

#### Automated Verification
- [x] the model builds successfully for the temporary example scenario
- [x] the solver returns a feasible solution for at least one temporary scenario
- [x] the budget and count-cap constraints are satisfied in the emitted solution
- [x] the frontier table contains monotone coverage targets and valid response values
- [x] camera deployments only create protection-gain effects at eligible waterholes
- [x] the fire-delay penalty increases sharply only after `tau_fire_min`
- [x] artificial-waterhole interventions apply both their intended wildlife-facing benefit and their tourism/infrastructure cost

#### Manual Verification
- [ ] increasing budget or asset counts improves at least one objective in a sensible way
- [ ] camera-heavy solutions improve coverage more than response time
- [ ] mobile-heavy solutions improve response time more than fixed-camera-only deployments
- [ ] allocating `5` cameras to a high-value waterhole improves nearby achieved protection in a visible and explainable way
- [ ] scenarios with severe fire urgency shift deployment toward faster fire response access
- [ ] artificial-waterhole selection improves protection benefit in targeted areas but is not chosen everywhere because its costs are explicit

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 7: Outputs And Visualization

### Overview

Export optimization outputs in forms that are easy to audit spatially and numerically.

### Changes Required

#### 1. Export recommendation artifacts
**File**: `scripts/16_optimize_surveillance.py`

**Changes**:

- write per-site deployment counts
- write per-cell achieved coverage and realized response time
- write per-cell `composite_risk`
- write per-cell `protection_benefit`
- write per-cell `human_operability_penalty`
- write per-cell wildfire urgency penalty contribution
- write selected artificial-waterhole interventions and their cost/benefit accounting
- write scenario-level summary metrics and frontier results

#### 2. Visualize optimization outputs
**File**: `scripts/17_visualize_optimization.py`

**Changes**:

- static plot of chosen deployment sites over the composite risk map
- static plot of baseline protection benefit vs post-camera/post-intervention protection benefit
- static plot of protection benefit before vs after artificial-waterhole interventions
- static frontier chart of coverage vs response time
- static plot of fire-delay penalty vs response time for the chosen scenario parameters
- interactive map with:
  - candidate sites
  - selected sites
  - asset-type symbology
  - optional coverage shading
  - optional response-time choropleth
  - optional camera suppression shading around locked-down waterholes
  - optional wildfire urgency overlay
  - optional artificial-waterhole intervention overlay

### Success Criteria

#### Automated Verification
- [ ] all expected output files are written and non-empty
- [ ] visualization scripts run without error
- [ ] frontier chart and map files exist for the example scenario

#### Manual Verification
- [ ] selected sites appear operationally plausible
- [ ] top-risk zones are visibly better covered in the chosen solution
- [ ] cells with poor response times are spatially understandable rather than random artifacts

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Phase 8: Validation And Scenario Analysis

### Overview

Add explicit validation so the optimization outputs are trustworthy and interpretable.

### Changes Required

#### 1. Validate optimization outputs
**File**: `scripts/18_validate_optimization_outputs.py`

**Changes**:

- verify every selected site exists in the candidate-site file
- verify no asset counts exceed caps or site capacities
- verify total cost does not exceed budget
- verify every covered cell is actually coverable by at least one selected asset
- verify every response time is attained by at least one selected mobile asset
- verify every camera-driven protection gain is traceable to a locked-down eligible waterhole
- verify fire-delay penalties are zero or small below `tau_fire_min` and ramp sharply above it
- verify artificial-waterhole benefits and tourism costs are both included in the reported objective/accounting

#### 2. Add scenario sweeps
**Files**: `data/configs/optimization_scenarios.yaml`, `scripts/16_optimize_surveillance.py`

**Changes**:

- support small budget sweeps
- support toggling asset types on and off
- support top-site truncation or scenario-specific site subsets

### Success Criteria

#### Automated Verification
- [ ] validation script passes on the example scenario outputs
- [ ] at least one budget sweep produces multiple feasible frontier points
- [ ] disabling an asset type changes the solution in a reproducible way

#### Manual Verification
- [ ] the solution behavior under budget changes matches intuition
- [ ] removing drones, cars, or cameras creates understandable tradeoff shifts

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation before proceeding.

---

## Testing Strategy

### Unit Tests

- schema validation for all config files
- candidate-site deduplication logic
- terrain modifier application by asset type
- coverage feasibility calculation for a small toy geometry
- response-time calculation for a small toy geometry
- budget and cap constraint construction
- epsilon-constraint frontier generation logic
- thresholded fire-delay penalty approximation against the target exponential curve
- `protection_benefit` construction under different species-weight choices
- `human_operability_penalty` under high animal-abundance, high camp-distance, and rough-terrain cells

### Integration Tests

- end-to-end run from existing composite risk outputs to optimization outputs on one temporary scenario
- regression test ensuring all emitted `site_id` and `cell_id` joins are consistent
- scenario comparison test where increasing budget weakly improves achievable coverage
- scenario comparison test where locking down a waterhole increases achieved protection in its influence zone
- scenario comparison test where lowering `tau_fire_min` or increasing `beta_fire` makes slow-fire-access solutions less favorable
- scenario comparison test where enabling artificial waterholes changes protection-benefit patterns and deployment choices

### Manual Testing Steps

1. Run the existing spatial pipeline to confirm optimization inputs are current.
2. Generate candidate sites and inspect them on a map.
3. Build terrain and surveillance matrices and spot-check values for several cells.
4. Inspect `protection_benefit` and `human_operability_penalty` on a few representative cells.
5. Solve one baseline scenario and inspect the chosen sites, asset counts, locked-down waterholes, and any artificial-waterhole interventions.
6. Confirm that adding `5` cameras to a waterhole changes nearby achieved protection as expected.
7. Confirm that fire-risk cells incur sharply larger penalties when response exceeds `tau_fire_min`.
8. Run a small frontier sweep and confirm the coverage/response tradeoff is monotone and sensible.

## Performance Considerations

- Precompute coverage and response matrices once per source refresh so the solver only works on tabular inputs.
- Candidate-site count must stay bounded; if high-risk centroid generation explodes site count, the MIP will degrade quickly.
- If the exact MIP becomes too slow, the first fallback should be tightening candidate-site generation, not switching immediately to a weaker heuristic.
- A linear-programming relaxation should be available for debugging and bound estimation, but the production recommendation should use integer decisions.

## Migration Notes

- This work should not modify the existing risk-surface scripts unless a small metadata addition is required.
- The optimization stage should consume the existing outputs as read-only upstream dependencies.
- New processed and output artifacts should use new filenames so they do not collide with the current risk-map deliverables.
- When the real datasets arrive, the temporary parameter files and any placeholder terrain wiring should be replaced directly rather than abstracted behind a generic ingestion layer.

## References

- Existing grid generation: [scripts/06_build_grid.py](/Users/hq/code/immc-risk-intervention/scripts/06_build_grid.py#L26)
- Existing per-cell features: [scripts/07_build_features.py](/Users/hq/code/immc-risk-intervention/scripts/07_build_features.py#L130)
- Existing species layers: [scripts/08_build_species_layers.py](/Users/hq/code/immc-risk-intervention/scripts/08_build_species_layers.py#L165)
- Existing threat layers: [scripts/09_build_threat_layers.py](/Users/hq/code/immc-risk-intervention/scripts/09_build_threat_layers.py#L138)
- Existing risk tensor and composite score: [scripts/10_build_risk_tensor.py](/Users/hq/code/immc-risk-intervention/scripts/10_build_risk_tensor.py#L92)
- Existing MVP plan: [plans/2026-03-19-etosha-actual-data-mvp-plan.md](/Users/hq/code/immc-risk-intervention/plans/2026-03-19-etosha-actual-data-mvp-plan.md#L1)
