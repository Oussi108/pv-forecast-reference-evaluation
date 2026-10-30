"""X02 (v44 extension B, revised): a fitted optimal convex reference.

The released D01 audit already scored a dual-axis plane-of-array smart persistence on the 2023 keys, and it was worse
than the GHI forms (Array 1B 6 h daylight MAE 0.163-0.179 vs 0.095-0.108), so a physical clear-sky reference is not
pursued. Instead the reference is made as hard as the simple forecasts allow: for each lead and each target hour of day,
non-negative weights summing to one are fitted on 2022 daylight rows to the candidates already in the prepared arrays
(persistence, previous day, hour climatology, smart persistence), by least absolute error. 2023 and 2024 labels are
never used. The fitted reference is then used like any other reference in the v44 contrasts.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

CANDIDATES = ('persistence', 'previous_day', 'climatology', 'smart')
MIN_ROWS = 50


def l1_simplex_weights(X, y):
    """argmin_w sum|y - Xw| s.t. w >= 0, sum w = 1 (linear program)."""
    n, k = X.shape
    # variables: w (k), u (n) with u >= |y - Xw|
    c = np.concatenate([np.zeros(k), np.ones(n)])
    eye = sparse.identity(n, format='csr')
    A_ub = sparse.vstack([sparse.hstack([sparse.csr_matrix(X), -eye]), sparse.hstack([sparse.csr_matrix(-X), -eye])]).tocsr()
    b_ub = np.concatenate([y, -y])
    A_eq = np.concatenate([np.ones(k), np.zeros(n)])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (k + n), method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[:k]


def fit_weights(references, y, target_hour, fit_mask, max_rows=4000, seed=42):
    """Weights per target hour (24 x 4). Hours with too few rows use the weights fitted on all fit rows."""
    rng = np.random.default_rng(seed)

    def solve(rows):
        if len(rows) > max_rows:
            rows = np.sort(rng.choice(rows, max_rows, replace=False))
        return l1_simplex_weights(references[rows].astype(float), y[rows].astype(float))

    overall = solve(np.flatnonzero(fit_mask))
    weights = np.tile(overall, (24, 1))
    for h in range(24):
        rows = np.flatnonzero(fit_mask & (target_hour == h))
        if len(rows) >= MIN_ROWS:
            weights[h] = solve(rows)
    return weights


def predict(references, target_hour, weights):
    return np.clip(np.einsum('ij,ij->i', references, weights[target_hour]), 0, 1.2)
