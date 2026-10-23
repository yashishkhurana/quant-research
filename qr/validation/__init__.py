"""Validation layer (week 3). Stubs with the intended signatures so strategies can call them now.

- deflated_sharpe: Bailey & López de Prado (2014). Probability the observed Sharpe beats zero
  after adjusting for the number of trials, skew, kurtosis and sample length.
- walk_forward: rolling fit/test windows; returns out-of-sample returns stitched together.
- param_stability: grid of metrics over a 2-D parameter space, for the heatmaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def deflated_sharpe(returns: pd.Series, n_trials: int, bars_per_year: float,
                    var_trials: float | None = None, sr_benchmark: float = 0.0) -> dict:
    """var_trials = variance of the per-bar Sharpe across the strategies you tried.
    Default 1/T is the zero-skill i.i.d. approximation; pass the real number from your grid."""
    r = returns.dropna()
    T = len(r)
    sr = r.mean() / r.std(ddof=1)                          # per-bar Sharpe
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    var_sr = var_trials if var_trials is not None else 1.0 / T
    # expected max Sharpe of n_trials zero-skill trials (Euler–Mascheroni approximation)
    if n_trials <= 1:                                      # single trial: plain PSR against sr_benchmark
        e_max = 0.0
    else:
        e_max = np.sqrt(var_sr) * ((1 - 0.5772) * stats.norm.ppf(1 - 1 / n_trials) + 0.5772 * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    sr0 = max(sr_benchmark / np.sqrt(bars_per_year), e_max)
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return {"sharpe_ann": float(sr * np.sqrt(bars_per_year)), "sr0_ann": float(sr0 * np.sqrt(bars_per_year)),
            "psr": float(stats.norm.cdf(z)), "n_trials": n_trials, "T": T}


def walk_forward(fit_fn, predict_fn, data: pd.DataFrame, train_bars: int, test_bars: int) -> pd.DataFrame:
    """fit_fn(train_df) -> params; predict_fn(params, test_df) -> weights DataFrame."""
    out = []
    start = 0
    while start + train_bars + test_bars <= len(data):
        train = data.iloc[start:start + train_bars]
        test = data.iloc[start + train_bars:start + train_bars + test_bars]
        params = fit_fn(train)
        out.append(predict_fn(params, test))
        start += test_bars
    return pd.concat(out) if out else pd.DataFrame()


def param_stability(run_fn, grid_a: list, grid_b: list, metric: str = "sharpe") -> pd.DataFrame:
    """run_fn(a, b) -> Result. Returns a DataFrame indexed by a, columns b, of the metric."""
    tbl = pd.DataFrame(index=grid_a, columns=grid_b, dtype=float)
    for a in grid_a:
        for b in grid_b:
            tbl.loc[a, b] = run_fn(a, b).metrics.get(metric, np.nan)
    return tbl
