"""
Step 2 of Topic 2: how sure are we about each tau, and do the five turning
ages sit closer together than random chance would produce?

Two separate questions, two separate methods:

1. Confidence per biomarker: bootstrap. Resample people with replacement,
   re-run find_best_tau on the resampled data, do this 500 times. The
   spread of those 500 estimates tells us how much the turning age would
   plausibly move if we'd surveyed a slightly different 5,000 people.

2. Do the five real point estimates cluster: a Monte Carlo test against a
   uniform-random null. If turning ages were scattered anywhere in 25-75
   with no coordination, how often would five random draws land as close
   together as our five actual estimates did? That's the p-value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from breakpoint import find_best_tau, TAU_GRID

DATA_PATH = "../data/processed/nhanes_2017_2018_biomarkers.csv"
FIG_PATH = "../figures/turning_points_unweighted.png"

# Only the five biomarkers Topic 1 found to be genuinely curved.
CURVED = {
    "LBXSAL":   "Albumin",
    "LBXSCR":   "Creatinine",
    "LBXHGB":   "Hemoglobin",
    "LBXHSCRP": "hs-CRP",
    "LBXGLU":   "Fasting glucose",
}

N_BOOT = 500
N_SIM = 5000
SEED = 42


def bootstrap_ci(age, y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(age)
    taus = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        taus[b] = find_best_tau(age[idx], y[idx])
    lo, hi = np.percentile(taus, [2.5, 97.5])
    return lo, hi, taus


def clustering_test(observed_taus, tau_grid=TAU_GRID, n_sim=N_SIM, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    observed_spread = float(np.std(observed_taus))
    null_spreads = np.array([
        np.std(rng.choice(tau_grid, size=len(observed_taus), replace=True))
        for _ in range(n_sim)
    ])
    p_value = float(np.mean(null_spreads <= observed_spread))
    return observed_spread, null_spreads, p_value


def main():
    df = pd.read_csv(DATA_PATH)
    rows = []

    for col, label in CURVED.items():
        sub = df[["RIDAGEYR", col]].dropna()
        age, y = sub["RIDAGEYR"].values, sub[col].values

        tau_hat = find_best_tau(age, y)
        lo, hi, _ = bootstrap_ci(age, y)

        rows.append({"biomarker": label, "n": len(sub), "tau": tau_hat,
                     "ci_low": lo, "ci_high": hi})
        print(f"{label:18s} n={len(sub):5d}  turning age = {tau_hat}  "
              f"(95% CI: {lo:.0f}-{hi:.0f})")

    res = pd.DataFrame(rows)
    observed_spread, null_spreads, p_value = clustering_test(res["tau"].values)

    print(f"\nObserved spread (SD) of the five turning ages: {observed_spread:.2f} years")
    print(f"Under a uniform-random null (no coordination): "
          f"P(spread this small or smaller) = {p_value:.3f}")

    # --- plot: where the five turning points actually land ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.errorbar(res["tau"], range(len(res)),
                xerr=[res["tau"] - res["ci_low"], res["ci_high"] - res["tau"]],
                fmt="o", color="crimson", capsize=4)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["biomarker"])
    ax.set_xlabel("Estimated turning age (95% bootstrap CI)")
    ax.set_xlim(20, 80)
    ax.axvline(res["tau"].median(), color="gray", linestyle=":", linewidth=1)
    ax.set_title("Where each biomarker bends")

    ax2 = axes[1]
    ax2.hist(null_spreads, bins=30, color="lightgray", edgecolor="white")
    ax2.axvline(observed_spread, color="crimson", linewidth=2, label="observed")
    ax2.set_xlabel("Spread (SD, years) of 5 turning ages")
    ax2.set_title(f"Null distribution (no coordination)\np = {p_value:.3f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=130)
    print(f"\nSaved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
