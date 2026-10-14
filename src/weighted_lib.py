"""
Same three questions as before -- does it curve, where does it bend, does a
coupled model make sense -- but every fit now uses NHANES's own weights, and
every confidence interval now respects the actual survey design instead of
pretending we have a simple random sample.

Why this matters concretely: NHANES oversamples some groups on purpose (to
get enough older adults, for instance), so an unweighted average over-counts
whoever was oversampled. The weight corrects for that. Separately, NHANES
doesn't survey random scattered individuals -- it surveys people clustered
into a smaller number of physical locations (primary sampling units, PSUs).
People in the same PSU are more alike than two random strangers, which means
a naive standard error understates real uncertainty. The fix used here:
resample whole PSUs, not individuals, so that clustering shows up in the
resampled variation the same way it would in a genuinely different sample.

NHANES 2017-2018 has exactly 2 PSUs in each of 15 strata -- confirmed
directly from the data, not assumed. That specific structure is what makes
"resample 2 PSUs with replacement, within each stratum" a sensible bootstrap:
each replicate independently decides, per stratum, which PSU (or which one
twice) represents that stratum this time around.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

TAU_GRID = np.arange(25, 76)


# ---------- weighted point estimates ----------

def wls_quadratic(age, y, w):
    X = sm.add_constant(np.column_stack([age, age**2]))
    return sm.WLS(y, X, weights=w).fit()


def weighted_hinge_rss(age, y, w, tau):
    design = np.column_stack([age, np.clip(age - tau, 0, None)])
    X = sm.add_constant(design)
    model = sm.WLS(y, X, weights=w).fit()
    resid = model.resid
    return float(np.sum(w * resid**2))


def find_best_tau_weighted(age, y, w, tau_grid=TAU_GRID):
    rss = [weighted_hinge_rss(age, y, w, t) for t in tau_grid]
    return int(tau_grid[int(np.argmin(rss))])


# ---------- cluster bootstrap (PSU within stratum, with replacement) ----------

def cluster_bootstrap_indices(strata, psu, rng):
    """
    One bootstrap replicate's row indices. For each stratum, draw 2 PSU
    slots with replacement from that stratum's 2 real PSUs, and take every
    row belonging to whichever PSU was drawn into each slot.
    """
    idx_parts = []
    df_design = pd.DataFrame({"stratum": strata, "psu": psu, "row": np.arange(len(strata))})
    for stratum_id, group in df_design.groupby("stratum"):
        psus_here = group["psu"].unique()
        assert len(psus_here) == 2, f"expected 2 PSUs, stratum {stratum_id} has {len(psus_here)}"
        draws = rng.choice(psus_here, size=2, replace=True)
        for d in draws:
            idx_parts.append(group.loc[group["psu"] == d, "row"].values)
    return np.concatenate(idx_parts)


def cluster_bootstrap(fit_fn, age, y, w, strata, psu, n_boot=200, seed=42):
    """
    fit_fn(age, y, w) -> scalar estimate. Returns array of n_boot replicate
    estimates, resampling PSUs within strata each time.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = cluster_bootstrap_indices(strata, psu, rng)
        out[b] = fit_fn(age[idx], y[idx], w[idx])
    return out
