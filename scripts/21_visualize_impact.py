#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from _spatial_common import OUTPUTS_DIR
except ModuleNotFoundError:
    from scripts._spatial_common import OUTPUTS_DIR

SIMULATION_CSV_PATH = OUTPUTS_DIR / "impact_simulation.csv"
SUMMARY_JSON_PATH = OUTPUTS_DIR / "impact_summary.json"

POPULATION_PNG = OUTPUTS_DIR / "impact_population_trajectories.png"
THREAT_PNG = OUTPUTS_DIR / "impact_threat_dynamics.png"
DEATHS_PNG = OUTPUTS_DIR / "impact_cumulative_deaths.png"
CAPACITY_PNG = OUTPUTS_DIR / "impact_carrying_capacity.png"
TEX_PATH = OUTPUTS_DIR / "impact.tex"

SPECIES = ["zebra", "lion", "elephant", "black_rhino"]
SPECIES_LABELS = {"zebra": "Plains Zebra", "lion": "Lion",
                  "elephant": "Elephant", "black_rhino": "Black Rhino"}
SPECIES_COLORS = {"zebra": "#2a6f97", "lion": "#c44536",
                  "elephant": "#4c956c", "black_rhino": "#6a4c93"}

BASELINE_STYLE = dict(color="#c44536", linestyle="--", alpha=0.85, linewidth=1.4)
OPTIMIZED_STYLE = dict(color="#2a6f97", linestyle="-", alpha=0.95, linewidth=1.8)
YEAR_TICKS = [0, 365, 730, 1095, 1460, 1825]
YEAR_LABELS = ["0", "1", "2", "3", "4", "5"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = pd.read_csv(SIMULATION_CSV_PATH)
    bl = df[df["scenario"] == "baseline"].reset_index(drop=True)
    opt = df[df["scenario"] == "optimized"].reset_index(drop=True)
    summary = json.loads(SUMMARY_JSON_PATH.read_text(encoding="utf-8"))
    return bl, opt, summary


def render_population_trajectories(bl: pd.DataFrame, opt: pd.DataFrame, summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, sp in enumerate(SPECIES):
        ax = axes[i]
        col = f"N_{sp}"
        days = bl["day"]

        ax.plot(days, bl[col], label="baseline", **BASELINE_STYLE)
        ax.plot(days, opt[col], label="optimized", **OPTIMIZED_STYLE)
        ax.fill_between(days, bl[col], opt[col], alpha=0.12, color="#2a6f97")

        bl_final = summary["scenarios"]["baseline"]["final_populations"][sp]
        opt_final = summary["scenarios"]["optimized"]["final_populations"][sp]
        pct = summary["improvement"]["population_change_pct"][sp]
        ax.set_title(f"{SPECIES_LABELS[sp]}:  {bl_final:,.0f} → {opt_final:,.0f}  ({pct:+.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("time (years)")
        ax.set_ylabel("population")
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels(YEAR_LABELS)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)

    fig.suptitle("Population Trajectories: Baseline vs Optimized", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(POPULATION_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_threat_dynamics(bl: pd.DataFrame, opt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    days = bl["day"]
    panels = [
        ("F", "wildfire intensity $F(t)$", [0, 1]),
        ("P", "poaching effort $P(t)$", None),
        ("T", "tourism disturbance $T(t)$", [0, 1]),
    ]

    for ax, (col, ylabel, ylim) in zip(axes, panels):
        ax.plot(days, bl[col], label="baseline", **BASELINE_STYLE)
        ax.plot(days, opt[col], label="optimized", **OPTIMIZED_STYLE)
        ax.fill_between(days, bl[col], opt[col], alpha=0.10, color="#2a6f97")
        ax.set_ylabel(ylabel, fontsize=11)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels(YEAR_LABELS)

    # annotate dry season peaks on fire panel
    fire_bl = bl["F"].values
    for year in range(5):
        peak_day = 180 + year * 365
        if peak_day < len(fire_bl):
            peak_val = fire_bl[peak_day]
            axes[0].annotate("dry season", xy=(peak_day, peak_val),
                             xytext=(peak_day + 40, peak_val + 0.06),
                             fontsize=7, alpha=0.6,
                             arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))

    axes[2].set_xlabel("time (years)")
    fig.suptitle("Threat Dynamics: Baseline vs Optimized", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(THREAT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_cumulative_deaths(bl: pd.DataFrame, opt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    threats = ["fire", "poach", "tour"]
    threat_labels = {"fire": "Wildfire", "poach": "Poaching", "tour": "Tourism"}
    threat_colors = {"fire": "#e76f51", "poach": "#264653", "tour": "#e9c46a"}

    for ax, (scenario_label, df) in zip(axes, [("Baseline", bl), ("Optimized", opt)]):
        x = np.arange(len(SPECIES))
        width = 0.25
        for j, threat in enumerate(threats):
            vals = [df[f"deaths_{threat}_{sp}"].sum() for sp in SPECIES]
            bars = ax.bar(x + j * width, vals, width, label=threat_labels[threat],
                          color=threat_colors[threat], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 10:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(scenario_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([SPECIES_LABELS[s] for s in SPECIES], fontsize=9)
        ax.set_ylabel("cumulative deaths (5 years)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Cumulative Deaths by Threat and Species", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DEATHS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_carrying_capacity(bl: pd.DataFrame, opt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    days = bl["day"]

    # top row: resources
    resources = [("R_water", "Water"), ("R_forage", "Forage"), ("R_space", "Space")]
    for j, (col, label) in enumerate(resources):
        ax = axes[0, j]
        ax.plot(days, bl[col], label="baseline", **BASELINE_STYLE)
        ax.plot(days, opt[col], label="optimized", **OPTIMIZED_STYLE)
        ax.fill_between(days, bl[col], opt[col], alpha=0.10, color="#2a6f97")
        ax.set_title(f"$R_\\mathrm{{{label.lower()}}}(t)$", fontsize=11)
        ax.set_ylabel("resource units")
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels(YEAR_LABELS)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # hide unused top-right cell
    axes[0, 3].set_visible(False)

    # bottom row: carrying capacities
    for j, sp in enumerate(SPECIES):
        ax = axes[1, j]
        col = f"K_{sp}"
        ax.plot(days, bl[col], label="baseline $K$", **BASELINE_STYLE)
        ax.plot(days, opt[col], label="optimized $K$", **OPTIMIZED_STYLE)
        # overlay population
        pop_col = f"N_{sp}"
        ax.plot(days, bl[pop_col], label="baseline $N$",
                color="#c44536", linestyle=":", alpha=0.5, linewidth=1.0)
        ax.plot(days, opt[pop_col], label="optimized $N$",
                color="#2a6f97", linestyle=":", alpha=0.5, linewidth=1.0)
        ax.set_title(f"{SPECIES_LABELS[sp]} $K(t)$", fontsize=11)
        ax.set_ylabel("individuals")
        ax.set_xticks(YEAR_TICKS)
        ax.set_xticklabels(YEAR_LABELS)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle("Resource Dynamics and Carrying Capacity", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CAPACITY_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_tex(summary: dict) -> None:
    bl = summary["scenarios"]["baseline"]
    opt = summary["scenarios"]["optimized"]
    imp = summary["improvement"]
    linkage = summary["optimization_linkage"]

    species_display = {"zebra": "Plains zebra", "lion": "Lion",
                       "elephant": "Elephant", "black_rhino": "Black rhino"}

    # build table rows
    pop_rows = []
    death_rows = []
    for sp in SPECIES:
        label = species_display[sp]
        bl_pop = bl["final_populations"][sp]
        opt_pop = opt["final_populations"][sp]
        pct = imp["population_change_pct"][sp]
        pop_rows.append(f"    {label} & {bl_pop:,.0f} & {opt_pop:,.0f} & {pct:+.1f}\\% \\\\")

        bl_d = bl["cumulative_deaths"][sp]
        opt_d = opt["cumulative_deaths"][sp]
        averted = imp["deaths_averted"][sp]
        averted_pct = imp["deaths_averted_pct"][sp]
        death_rows.append(f"    {label} & {bl_d:,.0f} & {opt_d:,.0f} & {averted:,.0f} & {averted_pct:.1f}\\% \\\\")

    tex = f"""\\subsection{{Impact Simulation Results}}

The coupled ODE system was integrated over a five-year horizon ($t \\in [0, 1825]$~days)
using an explicit Runge--Kutta solver (RK45, $\\Delta t_{{\\max}} = 1$~day) for two scenarios:
a \\emph{{baseline}} with no active intervention ($u(t)=0$, $\\Lambda_{{\\text{{opt}}}}=0$, $\\Delta\\beta=0$)
and an \\emph{{optimized}} scenario whose parameters are derived from the surveillance
optimization output (\\S\\ref{{sec:optimization}}): fire suppression
$u_{{\\text{{opt}}}} = {linkage["u_opt"]:.3f}$,
enforcement boost $\\Lambda_{{\\text{{opt}}}} = {linkage["lambda_opt"]:.3f}$,
and tourism mortality reduction $\\Delta\\beta = {linkage["beta_reduction"]:.0%}$.

Table~\\ref{{tab:impact-populations}} compares final populations after five years.
Every species benefits from the optimized resource allocation, with lions showing
the largest relative improvement ($+{imp["population_change_pct"]["lion"]:.1f}\\%$)
and black rhino recovering above their initial count in the optimized scenario.

\\begin{{table}}[H]
\\centering
\\caption{{Final populations after five years under baseline and optimized scenarios.}}
\\label{{tab:impact-populations}}
\\begin{{tabular}}{{lrrr}}
\\hline
\\textbf{{Species}} & \\textbf{{Baseline}} & \\textbf{{Optimized}} & \\textbf{{Change}} \\\\
\\hline
{chr(10).join(pop_rows)}
\\hline
\\end{{tabular}}
\\end{{table}}

Table~\\ref{{tab:impact-deaths}} summarises cumulative mortality averted by the intervention.

\\begin{{table}}[H]
\\centering
\\caption{{Cumulative deaths over five years and deaths averted by the optimized strategy.}}
\\label{{tab:impact-deaths}}
\\begin{{tabular}}{{lrrrr}}
\\hline
\\textbf{{Species}} & \\textbf{{Baseline}} & \\textbf{{Optimized}} & \\textbf{{Averted}} & \\textbf{{Averted \\%}} \\\\
\\hline
{chr(10).join(death_rows)}
\\hline
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{impact_population_trajectories.png}}
\\caption{{Population trajectories for four keystone species under baseline (dashed red)
and optimized (solid blue) scenarios.  Shaded regions indicate the population gain
attributable to the surveillance and intervention strategy.}}
\\label{{fig:impact-pop}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{impact_threat_dynamics.png}}
\\caption{{Threat dynamics over the five-year simulation.  Wildfire intensity $F(t)$ shows
seasonal peaks during the dry season (day~180 of each year); the optimized scenario
suppresses fire more effectively.  Poaching effort $P(t)$ declines under increased
enforcement.  Tourism disturbance $T(t)$ is identical between scenarios (exogenous driver)
but its mortality impact is reduced by the waterhole dispersion interventions.}}
\\label{{fig:impact-threats}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{impact_cumulative_deaths.png}}
\\caption{{Cumulative deaths over five years disaggregated by threat type.  Wildfire is the
dominant source of mortality for all species; the optimized scenario reduces fire-related
deaths through faster suppression response.}}
\\label{{fig:impact-deaths}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{impact_carrying_capacity.png}}
\\caption{{Resource dynamics (top) and resulting carrying capacities (bottom).
Fire and tourism degrade water, forage, and space; carrying capacity $K_s(t)$
is set by the most limiting resource via Liebig's law.  Dotted lines show
actual populations $N_s(t)$ for comparison.}}
\\label{{fig:impact-capacity}}
\\end{{figure}}
"""
    TEX_PATH.write_text(tex, encoding="utf-8")


def check_outputs() -> None:
    for path in [POPULATION_PNG, THREAT_PNG, DEATHS_PNG, CAPACITY_PNG]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty: {path}")
    if not TEX_PATH.is_file() or TEX_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty: {TEX_PATH}")
    print("impact visualization output check passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="visualize impact simulation results")
    parser.add_argument("--check", action="store_true", help="validate existing outputs")
    args = parser.parse_args()

    if args.check:
        check_outputs()
        return 0

    bl, opt, summary = load_data()

    print("rendering population trajectories...")
    render_population_trajectories(bl, opt, summary)

    print("rendering threat dynamics...")
    render_threat_dynamics(bl, opt)

    print("rendering cumulative deaths...")
    render_cumulative_deaths(bl, opt)

    print("rendering resource and carrying capacity...")
    render_carrying_capacity(bl, opt)

    print("generating LaTeX fragment...")
    generate_tex(summary)

    check_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
