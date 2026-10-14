"""
Topic 2, weighted, on the corrected biomarker set from the weighted Topic 1
pass: creatinine, hemoglobin, white blood cells, hs-CRP, fasting glucose.
Albumin is out (wasn't really curved once weighted); white blood cells is
in (was hiding a real curve the unweighted fit missed).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from weighted_lib import find_best_tau_weighted, cluster_bootstrap, TAU_GRID

DATA_PATH = "../data/processed/nhanes_2017_2018_biomarkers.csv"
FIG_PATH = "../figures/turning_points_weighted.png"

CURVED = {
    "LBXSCR":   ("Creatinine", "WTMEC2YR"),
    "LBXHGB":   ("Hemoglobin", "WTMEC2YR"),
    "LBXWBCSI": ("White blood cells", "WTMEC2YR"),
    "LBXHSCRP": ("hs-CRP", "WTMEC2YR"),
    "LBXGLU":   ("Fasting glucose", "WTSAF2YR"),
}

N_SIM = 5000
SEED = 42


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

    for col, (label, weight_col) in CURVED.items():
        sub = df[["RIDAGEYR", col, weight_col, "SDMVSTRA", "SDMVPSU"]].dropna()
        age, y, w = sub["RIDAGEYR"].values, sub[col].values, sub[weight_col].values
        strata, psu = sub["SDMVSTRA"].values, sub["SDMVPSU"].values

        tau_hat = find_best_tau_weighted(age, y, w)
        boot = cluster_bootstrap(find_best_tau_weighted, age, y, w, strata, psu,
                                  n_boot=200, seed=42)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        rows.append({"biomarker": label, "n": len(sub), "tau": tau_hat,
                     "ci_low": lo, "ci_high": hi})
        print(f"{label:20s} n={len(sub):5d}  turning age = {tau_hat}  "
              f"(95% CI: {lo:.0f}-{hi:.0f})")

    res = pd.DataFrame(rows)
    observed_spread, null_spreads, p_value = clustering_test(res["tau"].values)

    print(f"\nObserved spread (SD): {observed_spread:.2f} years")
    print(f"Clustering test p-value: {p_value:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.errorbar(res["tau"], range(len(res)),
                xerr=[res["tau"] - res["ci_low"], res["ci_high"] - res["tau"]],
                fmt="o", color="crimson", capsize=4)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(res["biomarker"])
    ax.set_xlabel("Estimated turning age (95% cluster-bootstrap CI)")
    ax.set_xlim(20, 80)
    ax.axvline(res["tau"].median(), color="gray", linestyle=":", linewidth=1)
    ax.set_title("Where each biomarker bends (weighted)")

    ax2 = axes[1]
    ax2.hist(null_spreads, bins=30, color="lightgray", edgecolor="white")
    ax2.axvline(observed_spread, color="crimson", linewidth=2, label="observed")
    ax2.set_xlabel("Spread (SD, years) of 5 turning ages")
    ax2.set_title(f"Null distribution (no coordination)\np = {p_value:.3f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=130)
    print(f"\nSaved figure: {FIG_PATH}")
    res.to_csv("../data/processed/topic2_weighted.csv", index=False)


if __name__ == "__main__":
    main()
