# Impact Modeling Implementation Plan

## Overview

Implement the impact modeling component: a coupled ODE system that simulates wildlife population dynamics under wildfire, poaching, and tourism pressures over 5 years. The simulation runs twice — once as a **baseline** (no optimization intervention) and once as an **optimized** scenario (using outputs from the optimization component) — to quantify the effect of the proposed resource allocation strategy.

## Current State Analysis

- Risk heat maps produce per-cell normalized threat and species layers (1007 hex cells)
- Optimization produces per-cell coverage, response times, fire delay penalties, and intervention selections
- The pipeline is at script 19 (sensitivity). Impact will be scripts **20** (simulation) and **21** (visualization)
- No impact/simulation code currently exists
- The paper specifies a full ODE system with 3 subsystems (wildfire, poaching, tourism) coupled through shared resources and carrying capacity

### Key Discoveries:
- Optimization cells parquet (`outputs/optimization_cells.parquet`) contains all spatial metrics needed to derive aggregate intervention parameters: `response_time_min`, `covered`, `wildfire_risk_norm`, `composite_risk`, `protection_benefit_effective`
- Optimization summary JSON has `selected_interventions` (3 artificial waterholes at α=0.95)
- Mean response time in high-fire cells is 19.1 min vs 53.3 min overall — optimization already prioritizes fire-prone areas
- Coverage fraction is 0.297 (29.7% of cells have a real responder)
- The codebase uses try/except import pattern for `_spatial_common` and `_optimization_common`
- All scripts follow numbered naming convention with `check_outputs()` validation

## Desired End State

Two new scripts:
1. `scripts/20_impact_simulation.py` — runs the ODE system for baseline and optimized scenarios, saves results
2. `scripts/21_visualize_impact.py` — generates publication-quality figures and a `.tex` fragment

Output files:
- `outputs/impact_simulation.csv` — time series of all state variables for both scenarios
- `outputs/impact_summary.json` — key metrics (final populations, cumulative deaths, % improvement)
- `outputs/impact_population_trajectories.png` — 4 species × 2 scenarios
- `outputs/impact_threat_dynamics.png` — fire intensity, poaching effort, tourism over time
- `outputs/impact_cumulative_deaths.png` — stacked bar or area chart by threat × species
- `outputs/impact_carrying_capacity.png` — resource dynamics and K_s(t)
- `outputs/impact.tex` — LaTeX fragment for the paper

### How to verify:
- Run `act && python3 scripts/20_impact_simulation.py` — should complete without error and produce CSV + JSON
- Run `act && python3 scripts/21_visualize_impact.py` — should produce 4 PNGs + tex
- Inspect `outputs/impact_population_trajectories.png` — populations should be plausible (no negative values, no explosions, rhino most vulnerable, zebra most resilient)
- Optimized scenario should show measurably better outcomes than baseline across all species
- Fire intensity should show seasonal peaks during dry season (around day 180, 545, 910, 1275, 1640)

## What We're NOT Doing

- Per-cell spatial simulation (this is a park-wide aggregate ODE model)
- Stochastic/Monte Carlo runs (deterministic only)
- Additional species beyond the 4 keystone (zebra, lion, elephant, black rhino)
- Climate change projections or multi-decade horizons
- Sensitivity analysis on impact parameters (could be future work)
- Interactive maps for impact (static plots only)

## Implementation Approach

The simulation is a system of coupled ODEs solved with `scipy.integrate.solve_ivp`. State vector:

```
[R_water, R_forage, R_space, F, P, N_zebra, N_lion, N_elephant, N_rhino]
```

9 state variables. Two resources would undersell the model; three (water, forage, space) is the sweet spot for Liebig's law to be meaningful. The optimization linkage works through two channels:

1. **Fire suppression u(t)**: In baseline, u(t) = 0 (no active suppression). In optimized, u(t) is derived from the optimization's fire response capability — specifically, the coverage-weighted inverse response time in high-fire cells. This makes `c(t) = c_0 + c_1 * u(t)` larger in the optimized scenario, suppressing fire faster.

2. **Enforcement Λ**: In baseline, enforcement is just Λ_0 (minimal). In optimized, Λ = Λ_0 + Λ_opt where Λ_opt is proportional to coverage fraction and inversely related to mean response time. More coverage + faster response = higher poaching detection risk.

3. **Tourism modulation**: The 3 artificial waterhole interventions disperse wildlife away from tourist corridors, effectively reducing the tourism-wildlife interaction coefficient β_s slightly in the optimized scenario.

---

## Phase 1: Impact Simulation Config and Parameters

### Overview
Create the YAML config and establish all parameter values needed for the ODE system.

### Changes Required:

#### 1. Impact config file
**File**: `data/configs/impact_parameters.yaml`

```yaml
simulation:
  t_start: 0
  t_end: 1825        # 5 years in days
  dt_max: 1.0        # max step size for solver

species:
  zebra:
    N0: 13000
    r_s: null          # governed by LV, not logistic
    mu_fire_base: 0.003
    alpha_poach_base: 0.0003
    w_poach: 0.08
    beta_tourism: 0.0003   # scaled down from paper's 0.3 by 1/1000
    resource_requirements:
      water: 0.005
      forage: 0.006
      space: 0.004
  lion:
    N0: 500
    r_s: null          # governed by LV
    mu_fire_base: 0.01
    alpha_poach_base: 0.006
    w_poach: 0.7
    beta_tourism: 0.0006
    resource_requirements:
      water: 0.008
      forage: 0.002
      space: 0.012
  elephant:
    N0: 2500
    r_s: 0.0003        # slow reproduction
    mu_fire_base: 0.015
    alpha_poach_base: 0.008
    w_poach: 1.0
    beta_tourism: 0.0007
    resource_requirements:
      water: 0.015
      forage: 0.012
      space: 0.008
  black_rhino:
    N0: 70
    r_s: 0.0004
    mu_fire_base: 0.02
    alpha_poach_base: 0.012
    w_poach: 1.5
    beta_tourism: 0.0008
    resource_requirements:
      water: 0.010
      forage: 0.008
      space: 0.010

lotka_volterra:
  lambda_1: 0.002       # zebra intrinsic growth (daily)
  lambda_2: 0.000005    # lion conversion efficiency
  gamma: 0.000008       # predation rate
  delta: 0.0008         # lion natural death rate (daily)
  zeta: 0.000002        # lion intraspecific competition

resources:
  water:
    R0: 1000.0
    r_h: 0.02           # recovery rate
    lambda_fire: 0.3    # fire damage rate
    lambda_tour: 0.05   # tourism damage rate
  forage:
    R0: 1000.0
    r_h: 0.03           # grassland recovers faster
    lambda_fire: 0.5    # fire heavily damages forage
    lambda_tour: 0.02
  space:
    R0: 1000.0
    r_h: 0.01           # habitat structure slow to recover
    lambda_fire: 0.1    # fire has less effect on space
    lambda_tour: 0.08   # infrastructure permanently reduces space

wildfire:
  gamma_0: 0.6
  d_0: 0.2
  d_1: 0.8
  t_dry: 180
  c_0: 0.1
  c_1: 0.5
  seasonal_mortality_amplitude: 0.3

poaching:
  gamma_p: 0.001        # effort adjustment rate
  r_p: 0.00005          # revenue coefficient (scaled)
  omega: 0.08           # opportunity cost
  sigma_p: 0.5          # risk sensitivity
  lambda_0: 0.1         # baseline enforcement
  lambda_tourism: 0.3   # enforcement increase per tourism unit
  t_wet: 0              # peak wet season timing (Jan)
  seasonal_poach_amplitude: 0.3
  P0: 0.5               # initial poaching effort

tourism:
  w_v: 0.5
  w_w: 0.3
  w_p: 0.2
  V_max: 1000
  W_max: 500
  # driving functions (seasonal with growth)
  V_base: 400           # base daily visitors
  V_amplitude: 200      # seasonal swing
  V_growth_rate: 0.0001 # slow annual growth
  t_peak_tourism: 200   # peak tourism slightly after dry season start
  W_base: 200
  W_amplitude: 100
  P_infra: 0.15         # permanent infrastructure level (normalized)

optimization_linkage:
  # these are computed from optimization outputs at runtime
  # but we store the mapping coefficients here
  fire_suppression_scale: 1.0      # how much coverage translates to u(t)
  enforcement_scale: 0.5           # how much coverage translates to Λ_opt
  tourism_beta_reduction: 0.05     # fractional reduction in β_s from interventions
```

### Success Criteria:

#### Automated Verification:
- [x] YAML loads without error: `python3 -c "import yaml; yaml.safe_load(open('data/configs/impact_parameters.yaml'))"`
- [x] All required keys present (checked by script)

#### Manual Verification:
- [ ] Parameter values are ecologically plausible (cross-check with paper tables)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 2: Core ODE Simulation Engine

### Overview
Build `scripts/20_impact_simulation.py` containing the ODE system, scenario runner, and output generation.

### Changes Required:

#### 1. Main simulation script
**File**: `scripts/20_impact_simulation.py`

**Architecture:**

```
load_config()           → reads impact_parameters.yaml
load_optimization()     → reads optimization outputs, computes linkage params
build_ode_system()      → returns the RHS function for solve_ivp
run_scenario()          → runs one scenario (baseline or optimized)
compare_scenarios()     → runs both, computes diffs
save_outputs()          → writes CSV + JSON
check_outputs()         → validates output files
main()                  → orchestrates everything
```

**Key implementation details:**

1. **State vector** (9 components):
   ```python
   # indices into state vector
   I_WATER, I_FORAGE, I_SPACE = 0, 1, 2
   I_FIRE = 3
   I_POACH = 4
   I_ZEBRA, I_LION, I_ELEPHANT, I_RHINO = 5, 6, 7, 8
   ```

2. **ODE RHS function** — a closure that captures all parameters:
   ```python
   def make_rhs(params, scenario_type):
       def rhs(t, y):
           R_w, R_f, R_sp, F, P, N_z, N_l, N_e, N_r = y
           # clamp to non-negative
           ...
           # compute dryness, suppression, tourism disturbance
           ...
           # resource ODEs
           ...
           # carrying capacities via Liebig's law
           ...
           # threat mortalities
           ...
           # population ODEs (LV for zebra/lion, logistic for others)
           ...
           return [dR_w, dR_f, dR_sp, dF, dP, dN_z, dN_l, dN_e, dN_r]
       return rhs
   ```

3. **Optimization linkage computation** — done once at load time:
   ```python
   def compute_optimization_linkage(cells_df, summary, config):
       # fire suppression: u_opt based on coverage and response time in fire-prone cells
       fire_cells = cells_df[cells_df['wildfire_risk_norm'] > cells_df['wildfire_risk_norm'].quantile(0.5)]
       covered_fire = fire_cells[fire_cells['covered'] == True]
       if len(covered_fire) > 0:
           # inverse response time, weighted by fire risk
           weights = covered_fire['wildfire_risk_norm']
           mean_inv_response = (weights / covered_fire['response_time_min'].clip(lower=1)).sum() / weights.sum()
           u_opt = config['fire_suppression_scale'] * (len(covered_fire) / len(fire_cells)) * min(1.0, mean_inv_response * 10)
       else:
           u_opt = 0.0

       # enforcement: proportional to overall coverage quality
       coverage_frac = cells_df['covered'].mean()
       mean_response = cells_df.loc[cells_df['covered'], 'response_time_min'].mean()
       lambda_opt = config['enforcement_scale'] * coverage_frac * (225.0 / max(mean_response, 1.0))

       # tourism reduction from interventions
       n_interventions = summary['chosen_solution']['selected_interventions']
       beta_reduction = config['tourism_beta_reduction'] * n_interventions

       return u_opt, lambda_opt, beta_reduction
   ```

   For the **baseline** scenario: `u(t) = 0`, `Λ_opt = 0`, `beta_reduction = 0`
   For the **optimized** scenario: uses computed values above

4. **Solver call**:
   ```python
   from scipy.integrate import solve_ivp
   sol = solve_ivp(rhs, [t_start, t_end], y0, method='RK45',
                   max_step=dt_max, dense_output=True)
   ```

5. **Post-processing**: evaluate solution on daily grid, compute cumulative deaths by integrating mortality terms, build output DataFrame.

6. **Output CSV columns**:
   ```
   day, scenario,
   R_water, R_forage, R_space,
   F, P, T,
   N_zebra, N_lion, N_elephant, N_rhino,
   K_zebra, K_lion, K_elephant, K_rhino,
   deaths_fire_zebra, deaths_fire_lion, deaths_fire_elephant, deaths_fire_rhino,
   deaths_poach_zebra, deaths_poach_lion, deaths_poach_elephant, deaths_poach_rhino,
   deaths_tour_zebra, deaths_tour_lion, deaths_tour_elephant, deaths_tour_rhino,
   cumulative_deaths_zebra, cumulative_deaths_lion, cumulative_deaths_elephant, cumulative_deaths_rhino
   ```

7. **Output JSON** (summary):
   ```json
   {
     "scenarios": {
       "baseline": {
         "final_populations": {"zebra": ..., "lion": ..., "elephant": ..., "rhino": ...},
         "cumulative_deaths": {"zebra": ..., ...},
         "deaths_by_threat": {"wildfire": ..., "poaching": ..., "tourism": ...},
         "min_populations": {"zebra": ..., ...}
       },
       "optimized": { ... }
     },
     "improvement": {
       "population_change_pct": {"zebra": ..., ...},
       "deaths_averted": {"zebra": ..., ...},
       "deaths_averted_pct": {"zebra": ..., ...}
     },
     "optimization_linkage": {
       "u_opt": ...,
       "lambda_opt": ...,
       "beta_reduction": ...
     }
   }
   ```

### Success Criteria:

#### Automated Verification:
- [x] Script runs without error: `act && python3 scripts/20_impact_simulation.py`
- [x] `outputs/impact_simulation.csv` exists and has expected columns
- [x] `outputs/impact_summary.json` exists and is valid JSON
- [x] No negative populations in CSV
- [x] Optimized scenario has strictly fewer cumulative deaths than baseline for all 4 species
- [x] Fire intensity F(t) stays in [0, 1]
- [x] Poaching effort P(t) stays non-negative

#### Manual Verification:
- [ ] Final populations are ecologically plausible (no species at 0 unless truly driven to extinction, no unrealistic growth)
- [ ] Seasonal fire peaks visible in the data
- [ ] Rhino is the most vulnerable species; zebra the most resilient

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 3: Visualization

### Overview
Build `scripts/21_visualize_impact.py` to produce publication-quality figures matching the style of existing optimization diagnostic plots.

### Changes Required:

#### 1. Visualization script
**File**: `scripts/21_visualize_impact.py`

**Four figures:**

1. **Population Trajectories** (`impact_population_trajectories.png`)
   - 2×2 grid, one subplot per species
   - Each subplot: baseline (dashed red) vs optimized (solid blue) population over time
   - Y-axis: population count; X-axis: time in days (with year markers)
   - Shaded region between curves shows improvement
   - Title per subplot: species name + final population numbers

2. **Threat Dynamics** (`impact_threat_dynamics.png`)
   - 3×1 vertical stack: fire intensity F(t), poaching effort P(t), tourism disturbance T(t)
   - Each panel: baseline vs optimized
   - Dry season peaks should be clearly visible in fire panel
   - Annotation arrows on key features (e.g., "dry season peak")

3. **Cumulative Deaths** (`impact_cumulative_deaths.png`)
   - Grouped bar chart: 4 species × 3 threats × 2 scenarios
   - Or: 2 panels (baseline, optimized), each with stacked bars per species
   - Clear legend, numeric labels on bars

4. **Resource & Carrying Capacity** (`impact_carrying_capacity.png`)
   - Top row: 3 resource levels R_m(t) (water, forage, space) baseline vs optimized
   - Bottom row: 4 carrying capacities K_s(t) baseline vs optimized
   - Shows how resource degradation flows through to population limits

**Style**: Match existing figures — `plt.style.use('seaborn-v0_8-whitegrid')` or similar, consistent font sizes, `fig.savefig(path, dpi=200, bbox_inches='tight')`.

### Success Criteria:

#### Automated Verification:
- [ ] Script runs without error: `act && python3 scripts/21_visualize_impact.py`
- [ ] All 4 PNG files exist and are non-empty
- [ ] No matplotlib warnings about missing data

#### Manual Verification:
- [ ] Population trajectories show clear separation between baseline and optimized
- [ ] Fire intensity shows ~5 seasonal peaks (one per year)
- [ ] Rhino trajectory is the most concerning; zebra is the most stable
- [ ] Figures are publication quality (clear labels, no overlapping text, legible at A4 paper size)

**Implementation Note**: After completing this phase and all automated verification passes, pause here for manual confirmation from the human before proceeding to the next phase.

---

## Phase 4: LaTeX Output and Integration

### Overview
Generate a `.tex` fragment and ensure the impact section integrates with the existing paper.

### Changes Required:

#### 1. LaTeX generation in visualization script
**File**: `scripts/21_visualize_impact.py` (extend with tex generation)

The `impact.tex` file should contain:
- A results subsection with key numbers from the summary JSON
- Figure includes for all 4 plots with captions
- A summary table comparing baseline vs optimized final populations and cumulative deaths

#### 2. Validation script
**File**: `scripts/22_validate_impact_outputs.py`

Following the pattern of `scripts/12_validate_outputs.py` and `scripts/18_validate_optimization_outputs.py`:
- Check all output files exist
- Validate CSV schema
- Validate JSON schema
- Check PNG files are non-empty
- Check ecological constraints (no negative populations, F in [0,1], etc.)

### Success Criteria:

#### Automated Verification:
- [ ] `act && python3 scripts/22_validate_impact_outputs.py` passes
- [ ] `outputs/impact.tex` compiles (no LaTeX errors when included)

#### Manual Verification:
- [ ] LaTeX output reads well in the paper context
- [ ] Numbers in tex match the JSON summary

**Implementation Note**: After completing this phase, the impact component is complete. The full pipeline is: risk (08-12) → optimization (13-19) → impact (20-22).

---

## Parameter Calibration Notes

### Optimization → Impact Linkage (detailed derivation)

The optimization outputs are spatial (per-cell), but the impact model is aggregate (park-wide). The bridge:

**Fire suppression `u_opt`:**
- Take cells with above-median wildfire risk (the cells that matter for fire)
- Of those, compute what fraction are covered by the optimization
- Weight by inverse response time (faster response = more effective suppression)
- Scale result to [0, 1] range
- In the ODE: `c(t) = c_0 + c_1 * u_opt` (constant in optimized scenario)
- Expected value: ~0.6–0.8 given that high-fire cells have mean response time of 19.1 min

**Enforcement `Λ_opt`:**
- Coverage fraction (0.297) × response quality (225 / mean_response_time)
- Scaled by enforcement_scale coefficient
- In the ODE: replaces Λ_0 with Λ_0 + Λ_opt in poaching dynamics
- This raises the cost term for poachers, reducing equilibrium poaching effort

**Tourism β reduction:**
- Each artificial waterhole intervention disperses wildlife away from tourist corridors
- 3 interventions × 0.05 reduction factor = 15% reduction in tourism mortality coefficients
- Applied as multiplicative: `β_s_opt = β_s * (1 - beta_reduction)`

### LV Parameter Rationale

- `λ_1 = 0.002/day` → zebra doubling time ~350 days without predation, realistic for large herbivores
- `γ = 0.000008` → at 500 lions and 13000 zebra, predation removes ~52 zebra/day, roughly 1.5% of the herd per month
- `λ_2 = 0.000005` → lion growth from prey; at equilibrium, birth rate balances death
- `δ = 0.0008/day` → lion natural lifespan ~3.4 years (wild lions live 10-14 years but this includes all mortality)
- `ζ = 0.000002` → mild intraspecific competition; becomes significant at ~400+ lions

### Tourism β_s Scaling

The paper lists β values of 0.3–0.8, but those are interaction rates, not daily mortality rates. We scale by ~1/1000 to get daily mortality coefficients (0.0003–0.0008). At T(t) ≈ 0.4 and N_zebra = 13000, this gives ~1.6 zebra deaths/day from tourism — plausible for a large park.

### Resource Normalization

All three resources start at R_m^0 = 1000 (arbitrary units). Per-capita requirements r_{m,s} are chosen so that initial K_s(t) ≈ 1.5 × N_s(0), giving populations room to grow or shrink without immediately hitting the capacity ceiling. For example:
- Zebra water: r = 0.005, so K_zebra_water = 1000/0.005 = 200,000 (not binding)
- Zebra forage: r = 0.006, so K_zebra_forage = 1000/0.006 ≈ 166,667 (not binding)
- Zebra space: r = 0.004, so K_zebra_space = 1000/0.004 = 250,000 (not binding)
- For rhino: water r = 0.010 → K = 100,000; forage r = 0.008 → K = 125,000; space r = 0.010 → K = 100,000

The carrying capacities are very high initially because resources are at baseline. As fire and tourism degrade resources, K_s drops and becomes the operative constraint. This is intentional — the model shows how cumulative resource degradation eventually limits populations even when direct mortality is moderate.

---

## Testing Strategy

### Unit-level checks (built into script 20):
- State vector never goes negative (clamped in RHS)
- F(t) ∈ [0, 1] (clamped)
- P(t) ≥ 0 (clamped)
- Resources R_m ≥ 0 (clamped)
- Solver converges (check sol.success)

### Integration checks (script 22):
- CSV and JSON exist with correct schemas
- Populations are plausible (within 2 orders of magnitude of initial)
- Optimized scenario strictly dominates baseline (more survivors, fewer deaths)
- Seasonal patterns present in fire intensity

### Manual testing:
- Visual inspection of all 4 figures
- Cross-check summary numbers against paper narrative

## References

- Optimization cells: `outputs/optimization_cells.parquet` (33 columns, 1007 rows)
- Optimization summary: `outputs/optimization_summary.json`
- Composite risk: `outputs/composite_risk.geojson`
- Paper equations: Impact Modeling section (provided in conversation)
- Existing script patterns: `scripts/19_sensitivity_analysis.py`, `scripts/17_visualize_optimization.py`
