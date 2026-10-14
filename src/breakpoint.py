"""
Step 1 of Topic 2: for each of the five curved biomarkers, where's the bend?

Method, kept as simple as the question allows: a "hinge" model.
y = a + b*age + c*max(0, age - tau)
Below tau, the slope is b. Above tau, the slope is b + c. The two segments
meet exactly at tau -- no jump, just a change in steepness. We don't know
tau in advance, so we try every plausible age as a candidate, fit that
version, and keep whichever tau gives the smallest residual error. That's
the whole idea: it's a straight line that's allowed to change its mind
once, and we search for where it changes its mind most convincingly.
"""

import numpy as np
import statsmodels.api as sm

TAU_GRID = np.arange(25, 76)  # candidate turning ages; margin at both ends
                                # so every fit has real data on both sides


def hinge_design(age, tau):
    return np.column_stack([age, np.clip(age - tau, 0, None)])


def fit_hinge(age, y, tau):
    X = sm.add_constant(hinge_design(age, tau))
    return sm.OLS(y, X).fit()


def find_best_tau(age, y, tau_grid=TAU_GRID):
    """Try every candidate tau, return the one with lowest residual error."""
    best_tau, best_rss = None, np.inf
    for tau in tau_grid:
        model = fit_hinge(age, y, tau)
        rss = float(np.sum(model.resid ** 2))
        if rss < best_rss:
            best_rss, best_tau = rss, tau
    return best_tau
