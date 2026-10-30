"""X03 (v44 extension C): drift-triggered model updating with a Page-Hinkley monitor.

The deployed model's daylight absolute errors are aggregated per target day. A day's mean error enters the monitor only
after the day has ended (all of its labels are then observed). The monitor (Page-Hinkley, increase direction) works on
errors standardized by the mean and standard deviation of the frozen fit's daily daylight errors over 2021:

    m_T = sum_{t<=T} (z_t - zbar_T - delta),   M_T = min_{t<=T} m_t,   alarm when m_T - M_T > lambda

On an alarm at the end of day d the model is refitted on all rows whose TARGET lies in the 365 days before 00:00 of day
d+1, and is used for every row ISSUED from that instant on; the monitor restarts. Refits are at least MIN_GAP_DAYS
apart. delta = 0.5 is fixed; lambda is chosen from LAMBDAS (per family, on seed 11) by the lowest 2022 daylight MAE,
with the same procedure started on 1 Jan 2022 from the frozen 2021 fit (no 2023 or 2024 label is used). The fitting itself is supplied by the
caller (the u01 LightGBM / neural fitting functions), so this module contains only the policy.
"""
import numpy as np
import pandas as pd

DELTA = 0.5
LAMBDAS = (2.0, 4.0, 8.0, 16.0)
MIN_GAP_DAYS = 30
WINDOW_DAYS = 365
DAY = 24 * 3600 * 10**9


class PageHinkley:
    def __init__(self, lam, delta=DELTA):
        self.lam, self.delta = lam, delta
        self.reset()

    def reset(self):
        self.n, self.mean, self.m, self.min_m = 0, 0.0, 0.0, 0.0

    def update(self, z):
        self.n += 1
        self.mean += (z - self.mean) / self.n
        self.m += z - self.mean - self.delta
        self.min_m = min(self.min_m, self.m)
        return self.m - self.min_m > self.lam


def run_policy(issue_ns, target_ns, y, daylight, fit_predict, initial_pred, start_ns, end_ns, lam, scale):
    """Simulate the drift-triggered policy for rows issued in [start_ns, end_ns).

    fit_predict(train_mask, predict_mask) -> predictions for predict_mask rows (fit only on train_mask rows).
    initial_pred: predictions of the model deployed at start_ns for all rows (NaN allowed outside the period).
    scale: (mean, sd) of the frozen fit's daily daylight MAE over 2021.
    Returns (predictions for all rows, list of refit instants in ns).
    """
    pred = np.asarray(initial_pred, float).copy()
    period = (issue_ns >= start_ns) & (issue_ns < end_ns)
    first_day = start_ns - start_ns % DAY
    days = range(first_day, end_ns, DAY)
    monitor = PageHinkley(lam)
    refits, last_refit = [], -np.inf
    for day_start in days:
        day_end = day_start + DAY
        rows = period & daylight & (target_ns >= day_start) & (target_ns < day_end) & (issue_ns >= (refits[-1] if refits else start_ns))
        if not rows.any():
            continue
        z = (np.mean(np.abs(pred[rows] - y[rows])) - scale[0]) / scale[1]
        if monitor.update(z) and (day_end - last_refit) >= MIN_GAP_DAYS * DAY and day_end < end_ns:
            train = (target_ns < day_end) & (target_ns >= day_end - WINDOW_DAYS * DAY)
            forward = period & (issue_ns >= day_end)
            # No label at or after the refit instant is used, and the new model only serves later issues.
            assert target_ns[train].max() < day_end <= issue_ns[forward].min()
            pred[forward] = fit_predict(train, forward)
            refits.append(day_end)
            last_refit = day_end
            monitor.reset()
    return pred, refits


def select_lambda(issue_ns, target_ns, y, daylight, fit_predict, frozen_pred, scale, year=2022):
    start = pd.Timestamp(f'{year}-01-01').value
    end = pd.Timestamp(f'{year + 1}-01-01').value
    scored = (issue_ns >= start) & (issue_ns < end) & daylight & (target_ns < end)
    results = {}
    for lam in LAMBDAS:
        pred, refits = run_policy(issue_ns, target_ns, y, daylight, fit_predict, frozen_pred, start, end, lam, scale)
        results[lam] = dict(mae=float(np.mean(np.abs(pred[scored] - y[scored]))), refits=len(refits))
    best = min(LAMBDAS, key=lambda lam: (results[lam]['mae'], lam))
    return best, results
