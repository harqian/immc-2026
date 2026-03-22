---
date: 2026-03-21T17:27:43-0700
git_commit: 38e80812b3c43d0c370404d730d13b664150283a
branch: master
topic: "Etosha Optimization Assumptions Implementation Strategy"
tags: [implementation, strategy, optimization, etosha, surveillance]
status: in_progress
---

# Handoff: Etosha optimization assumptions and solver state

## Task(s)
- Implemented the approved plan from [plans/2026-03-20-surveillance-optimization-plan.md](/Users/hq/code/immc-risk-intervention/plans/2026-03-20-surveillance-optimization-plan.md) through Phase 7 and committed those phases already.
- Completed and committed a follow-up change to bound fire-delay penalties and simplify the temporary ranger/vehicle proxy in commit `38e8081` (`Bound fire delay penalties and simplify ranger vehicle proxy`).
- Current work in progress:
  - researching more realistic Etosha staffing/assets
  - changing the optimization config/model so existing park assets are free and only additional units cost money
  - trying to use `100` people as the on-shift human count per user request
  - trying to regenerate optimization outputs under those new assumptions
- Current blocker:
  - the latest uncommitted baseline-assets configuration increases the feasible set, but the full frontier solve still does not finish successfully, so outputs remain stale from the earlier committed run.

## Critical References
- [plans/2026-03-20-surveillance-optimization-plan.md](/Users/hq/code/immc-risk-intervention/plans/2026-03-20-surveillance-optimization-plan.md)
- [data/configs/daily_asset_availability.yaml](/Users/hq/code/immc-risk-intervention/data/configs/daily_asset_availability.yaml)
- [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py)

## Recent changes
- Committed earlier:
  - bounded fire-delay curve in [scripts/15_build_surveillance_matrices.py](/Users/hq/code/immc-risk-intervention/scripts/15_build_surveillance_matrices.py) and [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py)
  - regenerated/committed breakpoint + optimization artifacts in commit `38e8081`
- Uncommitted current changes:
  - added included baseline asset config keys in [scripts/_optimization_common.py](/Users/hq/code/immc-risk-intervention/scripts/_optimization_common.py):40-53 and validation in [scripts/_optimization_common.py](/Users/hq/code/immc-risk-intervention/scripts/_optimization_common.py):210-245
  - set current scenario availability to `max_people: 100`, `included_people: 100`, `max_cars: 295`, and explicit included counts for other assets in [data/configs/daily_asset_availability.yaml](/Users/hq/code/immc-risk-intervention/data/configs/daily_asset_availability.yaml):1-14
  - restored `person` as an active asset in [data/configs/optimization_scenarios.yaml](/Users/hq/code/immc-risk-intervention/data/configs/optimization_scenarios.yaml):4-8
  - added baseline-free asset costing via `INCLUDED_ASSET_KEYS`, `model.excess_assets`, and `model.excess_asset_balance` in [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py):52-57, [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py):388-445
  - kept the earlier fix to skip empty asset-cap sets in [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py):458-463
  - updated budget accounting in solution extraction to charge only units above baseline in [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py):635-651

## Learnings
- The original fire-delay exponential was the main numeric blowup. The bounded sigmoid-style breakpoint table flattened the maximum penalty from `1.085e7` to `220.406`, and response objectives dropped from billions to ~`1e5`.
- The “red near dispatch node” issue was mostly visual or due unserved dummy cells, not random fire penalty noise. In successful runs, the highest-penalty cells were still `dummy / none / 225 min`; the curve change only prevented them from overwhelming the scale.
- The optimization model assumed all asset types were always active. When `person` was removed from `active_asset_types`, Pyomo crashed on a trivial boolean constraint in the asset cap loop. That is fixed by skipping empty `applicable` sets in [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py):458-463.
- Research results:
  - [2026_IMMC_Problem.pdf](/Users/hq/Downloads/2026_IMMC_Problem.pdf) explicitly states `295 personnel` stationed in Etosha, but that is all ministry staff, not necessarily frontline rangers.
  - official/current material supports vehicle, foot, and aerial patrols in Etosha, but I did **not** find a defensible Etosha-only count for drones or fixed security cameras.
  - user asked to use `100` people as an on-shift count; this is a modeling assumption from the user, not sourced directly.
- The baseline-free asset-cost implementation validates, but the full frontier solve with `100` included people currently still does not write outputs. Last observed run:
  - `loaded optimization data ... 7015 response arcs`
  - `coverage max = 115.225`
  - cleared `alpha=1.00`
  - advanced to `alpha=0.95`
  - tmux session exited without refreshed outputs; latest visible artifacts are still from Mar 20 21:55.
- There is a likely gap between current solve strategy and richer baseline capacity assumptions. The next agent should assume the main bottleneck is frontier-solve robustness, not config validation.

## Artifacts
- Approved implementation plan: [plans/2026-03-20-surveillance-optimization-plan.md](/Users/hq/code/immc-risk-intervention/plans/2026-03-20-surveillance-optimization-plan.md)
- Last successful committed checkpoint:
  - commit `38e8081`
  - [data/processed/fire_delay_breakpoints.parquet](/Users/hq/code/immc-risk-intervention/data/processed/fire_delay_breakpoints.parquet)
  - [outputs/optimization_frontier.csv](/Users/hq/code/immc-risk-intervention/outputs/optimization_frontier.csv)
  - [outputs/optimization_cells.parquet](/Users/hq/code/immc-risk-intervention/outputs/optimization_cells.parquet)
  - [outputs/optimization_summary.json](/Users/hq/code/immc-risk-intervention/outputs/optimization_summary.json)
  - [outputs/optimization_frontier.png](/Users/hq/code/immc-risk-intervention/outputs/optimization_frontier.png)
  - [outputs/optimization_diagnostics.png](/Users/hq/code/immc-risk-intervention/outputs/optimization_diagnostics.png)
  - [outputs/optimization_map.html](/Users/hq/code/immc-risk-intervention/outputs/optimization_map.html)
- Current source/config files to read before resuming:
  - [data/configs/daily_asset_availability.yaml](/Users/hq/code/immc-risk-intervention/data/configs/daily_asset_availability.yaml)
  - [data/configs/optimization_scenarios.yaml](/Users/hq/code/immc-risk-intervention/data/configs/optimization_scenarios.yaml)
  - [scripts/_optimization_common.py](/Users/hq/code/immc-risk-intervention/scripts/_optimization_common.py)
  - [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py)
  - [scripts/15_build_surveillance_matrices.py](/Users/hq/code/immc-risk-intervention/scripts/15_build_surveillance_matrices.py)
- Research/source docs referenced in-session:
  - [2026_IMMC_Problem.pdf](/Users/hq/Downloads/2026_IMMC_Problem.pdf)
  - MEFT State of the Parks report: `https://www.meft.gov.na/files/files/State%20of%20the%20Parks%20Report.pdf`
  - MEFT fire management strategy: `https://www.meft.gov.na/files/downloads/66c_Fire%20Management_Strategy%20Final%20Version.pdf`
  - NWR live waterhole camera note: `https://www.nwr.com.na/nwr-launches-live-waterhole-camera-in-etosha/`

## Action Items & Next Steps
- Decide how to handle the missing defensible counts for drones/cameras:
  - leave `included_drones` / `included_cameras` at `0`
  - or introduce explicit placeholder baselines and document them clearly
- Reproduce the current solve failure with the uncommitted baseline-free config and capture the exact termination path:
  - outputs stayed stale after the tmux run even though `alpha=1.00` appeared to clear and `alpha=0.95` started
  - likely need to run [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py) directly or instrument it to distinguish solver timeout/no incumbent from downstream failure
- Likely next engineering move:
  - make frontier solving robust to hard points by either:
    - handling `NoFeasibleSolutionError` / no-incumbent cases more gracefully
    - skipping or approximating `alpha=1.00` with the coverage-max solution
    - or otherwise changing solve strategy so outputs can be regenerated under the richer baseline-capacity setup
- Once outputs regenerate successfully:
  - rerun [scripts/17_visualize_optimization.py](/Users/hq/code/immc-risk-intervention/scripts/17_visualize_optimization.py)
  - inspect whether selected asset usage increases materially under `100` free people
  - commit only relevant tracked files; ignore unrelated `.DS_Store`, `research/`, `outputs/risk_diagnostics/`, `scripts/visualize_risk_components.py`, and pycache

## Other Notes
- Current dirty worktree at handoff time:
  - modified: [data/configs/daily_asset_availability.yaml](/Users/hq/code/immc-risk-intervention/data/configs/daily_asset_availability.yaml)
  - modified: [scripts/16_optimize_surveillance.py](/Users/hq/code/immc-risk-intervention/scripts/16_optimize_surveillance.py)
  - modified: [scripts/_optimization_common.py](/Users/hq/code/immc-risk-intervention/scripts/_optimization_common.py)
  - unrelated dirty/untracked files to ignore:
    - `.DS_Store`
    - `outputs/risk_diagnostics/`
    - `research/`
    - `scripts/__pycache__/_optimization_common.cpython-314.pyc`
    - `scripts/visualize_risk_components.py`
- Earlier completed commits before `38e8081`:
  - `6f168f8` phase 1
  - `499d452` phase 2
  - `7b89e53` phase 3
  - `379a49a` phase 4
  - `08b89cc` phase 5
  - `268f552` phase 6
  - `b6c07a3` phase 7
- The user has been iterating quickly on semantics. Most recent durable preference signals:
  - wants more realistic Etosha-scale assumptions
  - wants existing/current park assets to be free and only extra units to cost money
  - wants `100` people interpreted as roughly on-shift / present at a given time
  - is skeptical that `3` drones and `20` cameras are realistic, but no solid count has been found yet
