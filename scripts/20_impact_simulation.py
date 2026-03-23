#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import solve_ivp

try:
    from _spatial_common import OUTPUTS_DIR
except ModuleNotFoundError:
    from scripts._spatial_common import OUTPUTS_DIR

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data/configs/impact_parameters.yaml"
OPTIM_CELLS_PATH = OUTPUTS_DIR / "optimization_cells.parquet"
OPTIM_SUMMARY_PATH = OUTPUTS_DIR / "optimization_summary.json"

SIMULATION_CSV_PATH = OUTPUTS_DIR / "impact_simulation.csv"
SUMMARY_JSON_PATH = OUTPUTS_DIR / "impact_summary.json"

# state vector indices
I_WATER, I_FORAGE, I_SPACE = 0, 1, 2
I_FIRE = 3
I_POACH = 4
I_ZEBRA, I_LION, I_ELEPHANT, I_RHINO = 5, 6, 7, 8
N_STATES = 9

SPECIES_NAMES = ["zebra", "lion", "elephant", "black_rhino"]
SPECIES_INDICES = [I_ZEBRA, I_LION, I_ELEPHANT, I_RHINO]
RESOURCE_NAMES = ["water", "forage", "space"]
RESOURCE_INDICES = [I_WATER, I_FORAGE, I_SPACE]


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_optimization_outputs() -> tuple[pd.DataFrame, dict]:
    cells = pd.read_parquet(OPTIM_CELLS_PATH)
    summary = json.loads(OPTIM_SUMMARY_PATH.read_text(encoding="utf-8"))
    return cells, summary


def compute_optimization_linkage(cells: pd.DataFrame, summary: dict, linkage_cfg: dict) -> tuple[float, float, float]:
    fire_cells = cells[cells["wildfire_risk_norm"] > cells["wildfire_risk_norm"].quantile(0.5)]
    covered_fire = fire_cells[fire_cells["covered"] == True]
    if len(covered_fire) > 0:
        weights = covered_fire["wildfire_risk_norm"]
        mean_inv_response = (weights / covered_fire["response_time_min"].clip(lower=1)).sum() / weights.sum()
        u_opt = linkage_cfg["fire_suppression_scale"] * (len(covered_fire) / len(fire_cells)) * min(1.0, mean_inv_response * 10)
    else:
        u_opt = 0.0

    covered_mask = cells["covered"] == True
    coverage_frac = covered_mask.mean()
    mean_response = cells.loc[covered_mask, "response_time_min"].mean() if covered_mask.any() else 225.0
    lambda_opt = linkage_cfg["enforcement_scale"] * coverage_frac * (225.0 / max(mean_response, 1.0))

    n_interventions = summary["chosen_solution"]["selected_interventions"]
    beta_reduction = linkage_cfg["tourism_beta_reduction"] * n_interventions

    return float(u_opt), float(lambda_opt), float(beta_reduction)


def build_rainfall_interpolator(cfg_rain: dict):
    """build a smooth daily rainfall fraction from monthly mm totals.
    returns a function rain_frac(t) in [dry_floor, 1]."""
    monthly = np.array(cfg_rain["monthly_mm"], dtype=float)
    max_mm = monthly.max()
    dry_floor = cfg_rain["dry_floor"]
    # mid-day of each month (Jan=15, Feb=46, ...)
    mid_days = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    # normalize to [0, 1] then scale to [dry_floor, 1]
    fracs = dry_floor + (1 - dry_floor) * monthly / max_mm
    # extend periodically for smooth interpolation
    mid_ext = np.concatenate([mid_days - 365, mid_days, mid_days + 365])
    frac_ext = np.concatenate([fracs, fracs, fracs])

    def rain_frac(t: float) -> float:
        day_in_year = t % 365.0
        return float(np.interp(day_in_year, mid_days, fracs))
    return rain_frac


def compute_tourism(t: float, cfg_tour: dict) -> float:
    V = cfg_tour["V_base"] + cfg_tour["V_amplitude"] * np.cos(2 * np.pi * (t - cfg_tour["t_peak_tourism"]) / 365.0)
    V *= (1 + cfg_tour["V_growth_rate"] * t)
    W = cfg_tour["W_base"] + cfg_tour["W_amplitude"] * np.cos(2 * np.pi * (t - cfg_tour["t_peak_tourism"]) / 365.0)

    T = (cfg_tour["w_v"] * min(V, cfg_tour["V_max"]) / cfg_tour["V_max"]
         + cfg_tour["w_w"] * min(W, cfg_tour["W_max"]) / cfg_tour["W_max"]
         + cfg_tour["w_p"] * cfg_tour["P_infra"])
    return float(np.clip(T, 0, 1))


def make_rhs(cfg: dict, u_opt: float, lambda_opt: float, beta_reduction: float):
    sp_cfg = cfg["species"]
    lv = cfg["lotka_volterra"]
    res_cfg = cfg["resources"]
    wf = cfg["wildfire"]
    po = cfg["poaching"]
    tour_cfg = cfg["tourism"]
    rain_fn = build_rainfall_interpolator(cfg["rainfall"])
    forage_lag = cfg["rainfall"]["forage_lag_days"]

    # pre-extract species params in order
    species_keys = SPECIES_NAMES
    N0s = [sp_cfg[s]["N0"] for s in species_keys]
    mu_fires = [sp_cfg[s]["mu_fire_base"] for s in species_keys]
    alpha_poachs = [sp_cfg[s]["alpha_poach_base"] for s in species_keys]
    w_poachs = [sp_cfg[s]["w_poach"] for s in species_keys]
    beta_tours = [sp_cfg[s]["beta_tourism"] * (1 - beta_reduction) for s in species_keys]
    r_s_vals = [sp_cfg[s]["r_s"] for s in species_keys]

    # resource requirements matrix: [species][resource]
    req = [[sp_cfg[s]["resource_requirements"][r] for r in RESOURCE_NAMES] for s in species_keys]

    R0s = [res_cfg[r]["R0"] for r in RESOURCE_NAMES]
    r_hs = [res_cfg[r]["r_h"] for r in RESOURCE_NAMES]
    lam_fires = [res_cfg[r]["lambda_fire"] for r in RESOURCE_NAMES]
    lam_tours = [res_cfg[r]["lambda_tour"] for r in RESOURCE_NAMES]

    lambda_1 = lv["lambda_1"]
    lambda_2 = lv["lambda_2"]
    gamma_lv = lv["gamma"]
    delta_lv = lv["delta"]
    zeta_lv = lv["zeta"]

    gamma_0 = wf["gamma_0"]
    d_0 = wf["d_0"]
    d_1 = wf["d_1"]
    t_dry = wf["t_dry"]
    c_0 = wf["c_0"]
    c_1 = wf["c_1"]
    seasonal_mort_amp = wf["seasonal_mortality_amplitude"]

    gamma_p = po["gamma_p"]
    r_p = po["r_p"]
    omega_p = po["omega"]
    sigma_p = po["sigma_p"]
    lambda_0_po = po["lambda_0"]
    lambda_tourism_po = po["lambda_tourism"]
    t_wet = po["t_wet"]
    seasonal_poach_amp = po["seasonal_poach_amplitude"]

    def rhs(t, y):
        R_w, R_f, R_sp, F, P, N_z, N_l, N_e, N_r = y
        # clamp
        R_w = max(R_w, 0.0)
        R_f = max(R_f, 0.0)
        R_sp = max(R_sp, 0.0)
        F = np.clip(F, 0.0, 1.0)
        P = max(P, 0.0)
        N_z = max(N_z, 0.0)
        N_l = max(N_l, 0.0)
        N_e = max(N_e, 0.0)
        N_r = max(N_r, 0.0)
        pops = [N_z, N_l, N_e, N_r]

        # --- dryness & fire ---
        dryness = d_0 + d_1 * 0.5 * (1 + np.cos(2 * np.pi * (t - t_dry) / 365.0))
        containment = c_0 + c_1 * u_opt
        dF = gamma_0 * dryness * (1 - F) - containment * F

        # --- tourism disturbance ---
        T_val = compute_tourism(t, tour_cfg)

        # --- enforcement ---
        enforcement = lambda_0_po + lambda_opt + lambda_tourism_po * T_val

        # --- poaching dynamics ---
        # seasonal modulation: poaching peaks in wet season (easier movement)
        poach_season = 1 + seasonal_poach_amp * np.cos(2 * np.pi * (t - t_wet) / 365.0)
        total_value = sum(w_poachs[i] * pops[i] for i in range(4))
        revenue = r_p * total_value * poach_season
        cost = omega_p + sigma_p * enforcement
        dP = gamma_p * P * (revenue - cost)

        # --- resource dynamics (recovery modulated by rainfall) ---
        rain_water = rain_fn(t)
        rain_forage = rain_fn(t - forage_lag)  # vegetation lags rainfall
        rain_mods = [rain_water, rain_forage, 1.0]  # space not rainfall-dependent
        Rs = [R_w, R_f, R_sp]
        dRs = [0.0, 0.0, 0.0]
        for m in range(3):
            R_m = max(Rs[m], 0)
            recovery = r_hs[m] * rain_mods[m] * (R0s[m] - R_m)
            fire_damage = lam_fires[m] * F * R_m
            tour_damage = lam_tours[m] * T_val * R_m
            dRs[m] = recovery - fire_damage - tour_damage

        # --- carrying capacities via Liebig's law ---
        Ks = []
        for i in range(4):
            K_candidates = []
            for m in range(3):
                if req[i][m] > 0:
                    K_candidates.append(max(Rs[m], 0) / req[i][m])
            Ks.append(min(K_candidates) if K_candidates else 1e9)

        # --- fire mortality (seasonal peaks) ---
        fire_season = 1 + seasonal_mort_amp * np.cos(2 * np.pi * (t - t_dry) / 365.0)
        fire_morts = [mu_fires[i] * F * fire_season * pops[i] for i in range(4)]

        # --- poaching mortality ---
        poach_morts = [alpha_poachs[i] * P * w_poachs[i] * pops[i] for i in range(4)]

        # --- tourism mortality ---
        tour_morts = [beta_tours[i] * T_val * pops[i] for i in range(4)]

        # --- population ODEs ---
        # zebra: LV prey
        K_z = Ks[0]
        growth_z = lambda_1 * N_z * (1 - N_z / max(K_z, 1)) - gamma_lv * N_z * N_l
        dN_z = growth_z - fire_morts[0] - poach_morts[0] - tour_morts[0]

        # lion: LV predator
        K_l = Ks[1]
        growth_l = lambda_2 * gamma_lv * N_z * N_l - delta_lv * N_l - zeta_lv * N_l * N_l
        dN_l = growth_l - fire_morts[1] - poach_morts[1] - tour_morts[1]

        # elephant: logistic
        K_e = Ks[2]
        r_e = r_s_vals[2] if r_s_vals[2] else 0.0003
        growth_e = r_e * N_e * (1 - N_e / max(K_e, 1))
        dN_e = growth_e - fire_morts[2] - poach_morts[2] - tour_morts[2]

        # rhino: logistic
        K_r = Ks[3]
        r_r = r_s_vals[3] if r_s_vals[3] else 0.0004
        growth_r = r_r * N_r * (1 - N_r / max(K_r, 1))
        dN_r = growth_r - fire_morts[3] - poach_morts[3] - tour_morts[3]

        return [dRs[0], dRs[1], dRs[2], dF, dP, dN_z, dN_l, dN_e, dN_r]

    return rhs


def build_initial_state(cfg: dict) -> np.ndarray:
    y0 = np.zeros(N_STATES)
    for m, rname in enumerate(RESOURCE_NAMES):
        y0[RESOURCE_INDICES[m]] = cfg["resources"][rname]["R0"]
    y0[I_FIRE] = 0.1
    y0[I_POACH] = cfg["poaching"]["P0"]
    for i, sname in enumerate(SPECIES_NAMES):
        y0[SPECIES_INDICES[i]] = cfg["species"][sname]["N0"]
    return y0


def run_scenario(cfg: dict, u_opt: float, lambda_opt: float, beta_reduction: float, label: str) -> pd.DataFrame:
    sim = cfg["simulation"]
    rhs = make_rhs(cfg, u_opt, lambda_opt, beta_reduction)
    y0 = build_initial_state(cfg)

    sol = solve_ivp(rhs, [sim["t_start"], sim["t_end"]], y0,
                    method="RK45", max_step=sim["dt_max"], dense_output=True)
    if not sol.success:
        raise RuntimeError(f"solver failed for {label}: {sol.message}")

    days = np.arange(sim["t_start"], sim["t_end"] + 1, dtype=float)
    Y = sol.sol(days)  # shape (9, n_days)

    # clamp post-hoc
    Y = np.maximum(Y, 0)
    Y[I_FIRE] = np.clip(Y[I_FIRE], 0, 1)

    # compute derived quantities on daily grid
    rows = []
    sp_cfg = cfg["species"]
    wf = cfg["wildfire"]
    po = cfg["poaching"]
    tour_cfg = cfg["tourism"]
    res_cfg = cfg["resources"]

    cum_deaths = {s: 0.0 for s in SPECIES_NAMES}

    for di, t in enumerate(days):
        state = Y[:, di]
        R_w, R_f, R_sp, F, P = state[0], state[1], state[2], state[3], state[4]
        pops = {s: state[SPECIES_INDICES[i]] for i, s in enumerate(SPECIES_NAMES)}

        # carrying capacities
        Rs = [R_w, R_f, R_sp]
        Ks = {}
        for i, s in enumerate(SPECIES_NAMES):
            K_candidates = []
            for m, rname in enumerate(RESOURCE_NAMES):
                r_req = sp_cfg[s]["resource_requirements"][rname]
                if r_req > 0:
                    K_candidates.append(max(Rs[m], 0) / r_req)
            Ks[s] = min(K_candidates) if K_candidates else 1e9

        T_val = compute_tourism(t, tour_cfg)

        # daily mortality rates
        fire_season = 1 + wf["seasonal_mortality_amplitude"] * np.cos(2 * np.pi * (t - wf["t_dry"]) / 365.0)
        poach_season = 1 + po["seasonal_poach_amplitude"] * np.cos(2 * np.pi * (t - po["t_wet"]) / 365.0)

        row = {"day": int(t), "scenario": label,
               "R_water": R_w, "R_forage": R_f, "R_space": R_sp,
               "F": F, "P": P, "T": T_val}

        for i, s in enumerate(SPECIES_NAMES):
            N_s = pops[s]
            beta_s = sp_cfg[s]["beta_tourism"] * (1 - beta_reduction)

            d_fire = sp_cfg[s]["mu_fire_base"] * F * fire_season * N_s
            d_poach = sp_cfg[s]["alpha_poach_base"] * P * sp_cfg[s]["w_poach"] * N_s
            d_tour = beta_s * T_val * N_s

            cum_deaths[s] += d_fire + d_poach + d_tour

            row[f"N_{s}"] = N_s
            row[f"K_{s}"] = Ks[s]
            row[f"deaths_fire_{s}"] = d_fire
            row[f"deaths_poach_{s}"] = d_poach
            row[f"deaths_tour_{s}"] = d_tour
            row[f"cumulative_deaths_{s}"] = cum_deaths[s]

        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame,
                  u_opt: float, lambda_opt: float, beta_reduction: float) -> dict:
    summary = {"scenarios": {}, "improvement": {}, "optimization_linkage": {}}

    for label, df in [("baseline", baseline_df), ("optimized", optimized_df)]:
        last = df.iloc[-1]
        scenario_data = {
            "final_populations": {},
            "cumulative_deaths": {},
            "deaths_by_threat": {"wildfire": 0.0, "poaching": 0.0, "tourism": 0.0},
            "min_populations": {},
        }
        for s in SPECIES_NAMES:
            scenario_data["final_populations"][s] = float(last[f"N_{s}"])
            scenario_data["cumulative_deaths"][s] = float(last[f"cumulative_deaths_{s}"])
            scenario_data["min_populations"][s] = float(df[f"N_{s}"].min())
            scenario_data["deaths_by_threat"]["wildfire"] += float(df[f"deaths_fire_{s}"].sum())
            scenario_data["deaths_by_threat"]["poaching"] += float(df[f"deaths_poach_{s}"].sum())
            scenario_data["deaths_by_threat"]["tourism"] += float(df[f"deaths_tour_{s}"].sum())
        summary["scenarios"][label] = scenario_data

    bl = summary["scenarios"]["baseline"]
    opt = summary["scenarios"]["optimized"]
    improvement = {"population_change_pct": {}, "deaths_averted": {}, "deaths_averted_pct": {}}
    for s in SPECIES_NAMES:
        bl_pop = bl["final_populations"][s]
        opt_pop = opt["final_populations"][s]
        improvement["population_change_pct"][s] = 100.0 * (opt_pop - bl_pop) / max(bl_pop, 1) if bl_pop > 0 else 0.0
        bl_deaths = bl["cumulative_deaths"][s]
        opt_deaths = opt["cumulative_deaths"][s]
        improvement["deaths_averted"][s] = bl_deaths - opt_deaths
        improvement["deaths_averted_pct"][s] = 100.0 * (bl_deaths - opt_deaths) / max(bl_deaths, 1) if bl_deaths > 0 else 0.0
    summary["improvement"] = improvement

    summary["optimization_linkage"] = {
        "u_opt": u_opt,
        "lambda_opt": lambda_opt,
        "beta_reduction": beta_reduction,
    }
    return summary


def check_outputs() -> None:
    if not SIMULATION_CSV_PATH.is_file():
        raise FileNotFoundError(f"missing: {SIMULATION_CSV_PATH}")
    if not SUMMARY_JSON_PATH.is_file():
        raise FileNotFoundError(f"missing: {SUMMARY_JSON_PATH}")

    df = pd.read_csv(SIMULATION_CSV_PATH)
    if df.empty:
        raise ValueError("simulation CSV is empty")

    for s in SPECIES_NAMES:
        col = f"N_{s}"
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
        if (df[col] < 0).any():
            raise ValueError(f"negative population in {col}")

    if (df["F"] < -1e-9).any() or (df["F"] > 1 + 1e-9).any():
        raise ValueError("fire intensity F out of [0,1]")
    if (df["P"] < -1e-9).any():
        raise ValueError("poaching effort P is negative")

    summary = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
    bl = summary["scenarios"]["baseline"]
    opt = summary["scenarios"]["optimized"]
    for s in SPECIES_NAMES:
        if opt["cumulative_deaths"][s] >= bl["cumulative_deaths"][s]:
            raise ValueError(f"optimized scenario has more deaths than baseline for {s}")

    print("impact simulation output check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="run coupled ODE impact simulation")
    parser.add_argument("--check", action="store_true", help="validate existing outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    cfg = load_config()
    cells, summary = load_optimization_outputs()
    u_opt, lambda_opt, beta_reduction = compute_optimization_linkage(
        cells, summary, cfg["optimization_linkage"]
    )

    print(f"optimization linkage: u_opt={u_opt:.4f}, lambda_opt={lambda_opt:.4f}, beta_reduction={beta_reduction:.4f}")

    print("running baseline scenario...")
    baseline_df = run_scenario(cfg, 0.0, 0.0, 0.0, "baseline")

    print("running optimized scenario...")
    optimized_df = run_scenario(cfg, u_opt, lambda_opt, beta_reduction, "optimized")

    combined = pd.concat([baseline_df, optimized_df], ignore_index=True)
    combined.to_csv(SIMULATION_CSV_PATH, index=False)

    result_summary = build_summary(baseline_df, optimized_df, u_opt, lambda_opt, beta_reduction)
    SUMMARY_JSON_PATH.write_text(json.dumps(result_summary, indent=2), encoding="utf-8")

    for s in SPECIES_NAMES:
        bl_final = result_summary["scenarios"]["baseline"]["final_populations"][s]
        opt_final = result_summary["scenarios"]["optimized"]["final_populations"][s]
        pct = result_summary["improvement"]["population_change_pct"][s]
        print(f"  {s}: {bl_final:.0f} -> {opt_final:.0f} ({pct:+.1f}%)")

    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
