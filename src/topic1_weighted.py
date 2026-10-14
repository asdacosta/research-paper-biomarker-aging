"""
Topic 1, weighted. Same question as the first pass -- straight line or
curve -- but now: (a) each person counts according to their NHANES weight,
using WTMEC2YR for the five exam-collected biomarkers and WTSAF2YR for
fasting glucose specifically, since it comes from a smaller fasting
sub-sample with its own weight; and (b) "significant curvature" is decided
by a cluster bootstrap 95% CI on the quadratic coefficient excluding zero,
not by AIC -- AIC's theoretical justification gets murky once weights are
involved, and a bootstrap CI is more defensible here.
"""

import numpy as np
import pandas as pd
from weighted_lib import wls_quadratic, cluster_bootstrap

DATA_PATH = "../data/processed/nhanes_2017_2018_biomarkers.csv"

BIOMARKERS = {
    "LBXSAL":   ("Albumin (g/dL)", "WTMEC2YR"),
    "LBXSCR":   ("Creatinine (mg/dL)", "WTMEC2YR"),
    "LBXHGB":   ("Hemoglobin (g/dL)", "WTMEC2YR"),
    "LBXWBCSI": ("White blood cells (1000 cells/uL)", "WTMEC2YR"),
    "LBXHSCRP": ("hs-CRP (mg/L)", "WTMEC2YR"),
    "LBXGLU":   ("Fasting glucose (mg/dL)", "WTSAF2YR"),
}


def quad_coef(age, y, w):
    return wls_quadratic(age, y, w).params[2]


def main():
    df = pd.read_csv(DATA_PATH)
    rows = []

    for col, (label, weight_col) in BIOMARKERS.items():
        sub = df[["RIDAGEYR", col, weight_col, "SDMVSTRA", "SDMVPSU"]].dropna()
        age = sub["RIDAGEYR"].values
        y = sub[col].values
        w = sub[weight_col].values
        strata = sub["SDMVSTRA"].values
        psu = sub["SDMVPSU"].values

        model = wls_quadratic(age, y, w)
        point_coef = model.params[2]

        boot = cluster_bootstrap(quad_coef, age, y, w, strata, psu, n_boot=200, seed=42)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        curved = not (lo <= 0 <= hi)

        rows.append({
            "biomarker": label, "n": len(sub), "weight_used": weight_col,
            "quad_coef": point_coef, "ci_low": lo, "ci_high": hi, "curved": curved,
        })
        print(f"{label:35s} n={len(sub):5d}  quad coef={point_coef:9.5f}  "
              f"95% CI=[{lo:9.5f}, {hi:9.5f}]  curved={curved}")

    pd.DataFrame(rows).to_csv(
        "../data/processed/topic1_weighted.csv", index=False)


if __name__ == "__main__":
    main()
