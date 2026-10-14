"""
Topic 3: build the smallest honest dynamical system that could describe how
hs-CRP and creatinine move together as age increases -- and be upfront about
what this kind of model can and can't claim from cross-sectional data.

The model:
    d(CRP)/da   = b1 + g1*CRP + d1*Creat
    d(Creat)/da = b2 + g2*CRP + d2*Creat

We don't have anyone's CRP and creatinine measured twice, years apart --
NHANES is one measurement per person. So "d/da" here can only come from the
POPULATION-level trend: fit each biomarker as a smooth curve against age,
take the derivative of that curve. That's a real limitation, not a hidden
one: this system describes how the population average moves, not how any
individual's body does. A true test of individual-level coupling needs
longitudinal data -- the same person, sampled repeatedly -- which is exactly
what a follow-up study (HRS, for instance) would add.

Given Topic 2's own finding -- essentially zero partial correlation between
these two biomarkers once age is accounted for -- expect the cross-coupling
terms (g1's weight on Creat, and g2's weight on CRP) to be small. If they
come out large anyway, that's a sign the fit is absorbing shared-age
structure, not real interaction, and the writeup needs to say so plainly.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

DATA_PATH = "../data/processed/nhanes_2017_2018_biomarkers.csv"
FIG_PATH = "../figures/toy_dynamics.png"

AGE_GRID = np.arange(18, 81)


def fit_quadratic(age, y):
    X = sm.add_constant(np.column_stack([age, age**2]))
    return sm.OLS(y, X).fit()


def curve_and_derivative(model, age_grid):
    b0, b1, b2 = model.params
    level = b0 + b1 * age_grid + b2 * age_grid**2
    deriv = b1 + 2 * b2 * age_grid
    return level, deriv


def main():
    df = pd.read_csv(DATA_PATH)
    sub = df[["RIDAGEYR", "LBXHSCRP", "LBXSCR"]].dropna()
    age, crp, creat = sub["RIDAGEYR"].values, sub["LBXHSCRP"].values, sub["LBXSCR"].values

    crp_model = fit_quadratic(age, crp)
    creat_model = fit_quadratic(age, creat)

    crp_level, crp_deriv = curve_and_derivative(crp_model, AGE_GRID)
    creat_level, creat_deriv = curve_and_derivative(creat_model, AGE_GRID)

    # --- fit the coupled linear system on the trajectory grid ---
    predictors = sm.add_constant(np.column_stack([crp_level, creat_level]))

    sys_crp = sm.OLS(crp_deriv, predictors).fit()
    sys_creat = sm.OLS(creat_deriv, predictors).fit()

    print("d(CRP)/da   ~ const + CRP + Creatinine")
    print(f"  coefficients: b1={sys_crp.params[0]:.4f}  g1(on CRP)={sys_crp.params[1]:.4f}  "
          f"d1(on Creat)={sys_crp.params[2]:.4f}   R2={sys_crp.rsquared:.4f}")
    print("d(Creat)/da ~ const + CRP + Creatinine")
    print(f"  coefficients: b2={sys_creat.params[0]:.4f}  g2(on CRP)={sys_creat.params[1]:.4f}  "
          f"d2(on Creat)={sys_creat.params[2]:.4f}   R2={sys_creat.rsquared:.4f}")

    # Jacobian: since the model is linear, this matrix is the same everywhere.
    J = np.array([
        [sys_crp.params[1],   sys_crp.params[2]],
        [sys_creat.params[1], sys_creat.params[2]],
    ])
    eigvals, eigvecs = np.linalg.eig(J)

    print(f"\nJacobian:\n{J}")
    print(f"\nEigenvalues: {eigvals}")
    for ev in eigvals:
        kind = "real" if np.isreal(ev) else "complex"
        sign = "stable (decays)" if ev.real < 0 else "unstable (grows)"
        print(f"  {ev:.4f}  -- {kind}, {sign}")

    # --- plot: phase portrait + vector field ---
    fig, ax = plt.subplots(figsize=(7, 6))

    crp_range = np.linspace(crp_level.min() * 0.8, crp_level.max() * 1.2, 15)
    creat_range = np.linspace(creat_level.min() * 0.8, creat_level.max() * 1.2, 15)
    CG, RG = np.meshgrid(crp_range, creat_range)
    dCRP = sys_crp.params[0] + sys_crp.params[1] * CG + sys_crp.params[2] * RG
    dCreat = sys_creat.params[0] + sys_creat.params[1] * CG + sys_creat.params[2] * RG
    ax.quiver(CG, RG, dCRP, dCreat, color="lightgray", alpha=0.7, width=0.003)

    sc = ax.scatter(crp_level, creat_level, c=AGE_GRID, cmap="viridis", s=25, zorder=3)
    ax.plot(crp_level, creat_level, color="black", linewidth=0.8, zorder=2)
    plt.colorbar(sc, label="Age")

    ax.set_xlabel("hs-CRP (population mean, mg/L)")
    ax.set_ylabel("Creatinine (population mean, mg/dL)")
    ax.set_title("Population trajectory through (CRP, Creatinine) space, ages 18-80\n"
                  "gray arrows: the fitted system's implied direction of movement")

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=130)
    print(f"\nSaved figure: {FIG_PATH}")


if __name__ == "__main__":
    main()
