"""
Step 2 of Topic 1: for each biomarker, does age predict it better as a
straight line, or does letting the line bend (a quadratic term) actually
earn its keep?

Method, kept deliberately simple: fit y = a + b*age (linear), then
y = a + b*age + c*age^2 (quadratic), on the SAME people (rows where that
biomarker isn't missing). Compare with AIC (lower is better) and check
whether the age^2 coefficient is statistically distinguishable from zero.
A quadratic isn't the final word on nonlinearity -- it's the simplest
possible way to ask "does curvature help at all" before reaching for
anything fancier.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = "../data/processed/nhanes_2017_2018_biomarkers.csv"
FIG_PATH = "../figures/age_curves_weighted.png"

BIOMARKERS = {
    "LBXSAL":   ("Albumin (g/dL)", "WTMEC2YR"),
    "LBXSCR":   ("Creatinine (mg/dL)", "WTMEC2YR"),
    "LBXHGB":   ("Hemoglobin (g/dL)", "WTMEC2YR"),
    "LBXWBCSI": ("White blood cells (1000 cells/uL)", "WTMEC2YR"),
    "LBXHSCRP": ("hs-CRP (mg/L)", "WTMEC2YR"),
    "LBXGLU":   ("Fasting glucose (mg/dL)", "WTSAF2YR"),
}


def fit_both(df, col, weight_col):
    """Fit linear and quadratic WLS for one biomarker vs age, using its NHANES weight."""
    sub = df[["RIDAGEYR", col, weight_col]].dropna()
    age = sub["RIDAGEYR"].values
    y = sub[col].values
    w = sub[weight_col].values

    X_lin = sm.add_constant(age)
    lin = sm.WLS(y, X_lin, weights=w).fit()

    X_quad = sm.add_constant(np.column_stack([age, age**2]))
    quad = sm.WLS(y, X_quad, weights=w).fit()

    return sub, lin, quad, w


def main():
    df = pd.read_csv(DATA_PATH)
    results = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (col, (label, weight_col)) in enumerate(BIOMARKERS.items()):
        sub, lin, quad, w = fit_both(df, col, weight_col)
        age = sub["RIDAGEYR"].values
        y = sub[col].values

        results.append({
            "biomarker": label,
            "n": len(sub),
            "quad_coef": quad.params[2],
            "quad_pvalue": quad.pvalues[2],
        })

        # --- plot ---
        ax = axes[i]
        ax.scatter(age, y, alpha=0.08, s=8, color="steelblue")

        age_bins = np.arange(18, 82, 2)
        bin_means = []
        for a in age_bins[:-1]:
            mask = (age >= a) & (age < a + 2)
            bin_means.append(np.average(y[mask], weights=w[mask]) if mask.sum() else np.nan)
        ax.plot(age_bins[:-1] + 1, bin_means, "o", color="black", markersize=3, label="2-yr weighted age bin mean")

        age_grid = np.linspace(18, 80, 100)
        lin_pred = lin.predict(sm.add_constant(age_grid))
        quad_pred = quad.predict(sm.add_constant(np.column_stack([age_grid, age_grid**2])))
        ax.plot(age_grid, lin_pred, "--", color="orange", label="linear fit")
        ax.plot(age_grid, quad_pred, "-", color="crimson", label="quadratic fit")

        # Zoom the picture to where most people actually are (0.5th-99.5th
        # percentile). The regression above already used every point --
        # this line only changes what's visible, not what was fit.
        lo, hi = np.percentile(y, [0.5, 99.5])
        pad = (hi - lo) * 0.08
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Age")
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=130)
    print(f"Saved figure: {FIG_PATH}\n")

    res_df = pd.DataFrame(results)
    pd.set_option("display.width", 120)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
