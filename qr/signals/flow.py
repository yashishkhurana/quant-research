"""Order-flow features from our own L2 snapshots and trades, and a forward-return study.

Features (all computed per snapshot, then resampled to a bar clock):
- ofi:   order-flow imbalance at the top of book, Cont, Kukanov & Stoikov (2014):
         e_t = 1[b_t >= b_{t-1}] q^b_t - 1[b_t <= b_{t-1}] q^b_{t-1} - 1[a_t <= a_{t-1}] q^a_t + 1[a_t >= a_{t-1}] q^a_{t-1}
         summed over the bar. Positive = net buying pressure at the touch.
- depth_imb: (sum bid size - sum ask size) / (sum bid + sum ask) over N levels, bar mean.
- microprice_dev: (microprice - mid) / mid in bps, bar mean. Microprice = (a*qb + b*qa)/(qa+qb).
- tfi:   trade-flow imbalance, (buy volume - sell volume) / total volume over the bar.
- spread_bps: bar mean quoted spread.

Study: for each feature, standardise with a trailing z-score (no look-ahead), regress forward
log mid return over horizons h in {1, 5, 15, 30} bars, report slope, t-stat (Newey-West),
R^2, and the decile spread of forward returns. Then a one-feature strategy: sign of the
z-score when |z| > threshold, hold h bars, with costs, through the engine.

This is a signal study on days of data, not a backtest on years. The note must say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _top(book: pd.DataFrame):
    return book["bid_px_0"], book["bid_sz_0"], book["ask_px_0"], book["ask_sz_0"]


def ofi_events(book: pd.DataFrame) -> pd.Series:
    b, qb, a, qa = _top(book)
    b1, qb1, a1, qa1 = b.shift(1), qb.shift(1), a.shift(1), qa.shift(1)
    e = (np.where(b >= b1, qb, 0) - np.where(b <= b1, qb1, 0)
         - np.where(a <= a1, qa, 0) + np.where(a >= a1, qa1, 0))
    return pd.Series(e, index=book.index).fillna(0.0)


def depth_imbalance(book: pd.DataFrame, levels: int = 5) -> pd.Series:
    bid = sum(book[f"bid_sz_{i}"].fillna(0) for i in range(levels))
    ask = sum(book[f"ask_sz_{i}"].fillna(0) for i in range(levels))
    return (bid - ask) / (bid + ask).replace(0, np.nan)


def microprice_dev_bps(book: pd.DataFrame) -> pd.Series:
    b, qb, a, qa = _top(book)
    mid = (a + b) / 2
    micro = (a * qb + b * qa) / (qa + qb).replace(0, np.nan)
    return (micro - mid) / mid * 1e4


def spread_bps(book: pd.DataFrame) -> pd.Series:
    b, _, a, _ = _top(book)
    return (a - b) / ((a + b) / 2) * 1e4


def bar_features(book: pd.DataFrame, trades: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Resample snapshot-level features to a bar clock. Bars with no snapshot are dropped."""
    mid = ((book["ask_px_0"] + book["bid_px_0"]) / 2)
    feats = pd.DataFrame({
        "ofi": ofi_events(book).resample(rule).sum(),
        "depth_imb": depth_imbalance(book).resample(rule).mean(),
        "micro_dev": microprice_dev_bps(book).resample(rule).mean(),
        "spread_bps": spread_bps(book).resample(rule).mean(),
        "mid": mid.resample(rule).last(),
        "n_snap": mid.resample(rule).count(),
    })
    if trades is not None and not trades.empty:
        signed = trades["qty"] * trades["side"]
        vol = trades["qty"].resample(rule).sum()
        feats["tfi"] = (signed.resample(rule).sum() / vol.replace(0, np.nan))
        feats["volume"] = vol
    return feats[feats["n_snap"] > 0]


def zscore(x: pd.Series, window: int = 120) -> pd.Series:
    """Trailing z-score; uses only past bars."""
    m = x.rolling(window, min_periods=window // 2).mean()
    s = x.rolling(window, min_periods=window // 2).std(ddof=1)
    return (x - m) / s.replace(0, np.nan)


def forward_returns(mid: pd.Series, horizons=(1, 5, 15, 30)) -> pd.DataFrame:
    lm = np.log(mid)
    return pd.DataFrame({h: lm.shift(-h) - lm for h in horizons})


def newey_west_t(x: np.ndarray, y: np.ndarray, lags: int) -> tuple[float, float, float]:
    """OLS y = a + b x; returns (b, t_NW, r2). HAC with Bartlett kernel, `lags` lags."""
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    n = len(y)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    t = beta[1] / np.sqrt(V[1, 1])
    r2 = 1 - u.var() / y.var()
    return float(beta[1]), float(t), float(r2)


def signal_study(feats: pd.DataFrame, features=("ofi", "depth_imb", "micro_dev", "tfi"),
                 horizons=(1, 5, 15, 30), z_window: int = 120) -> pd.DataFrame:
    fr = forward_returns(feats["mid"], horizons) * 1e4            # bps
    rows = []
    for f in features:
        if f not in feats:
            continue
        z = zscore(feats[f], z_window)
        for h in horizons:
            d = pd.concat([z.rename("z"), fr[h].rename("y")], axis=1).dropna()
            if len(d) < 200:
                continue
            b, t, r2 = newey_west_t(d["z"].values, d["y"].values, lags=h)
            dec = pd.qcut(d["z"], 10, labels=False, duplicates="drop")
            spread = d["y"].groupby(dec).mean()
            rows.append({"feature": f, "horizon": h, "n": len(d), "slope_bps_per_sd": b,
                         "t_nw": t, "r2": r2, "top_minus_bottom_decile_bps": float(spread.iloc[-1] - spread.iloc[0]),
                         "corr": float(d["z"].corr(d["y"]))})
    return pd.DataFrame(rows)


def flow_strategy_weights(feats: pd.DataFrame, feature: str, horizon: int, threshold: float = 1.0,
                          z_window: int = 120) -> pd.Series:
    """Enter sign(z) when |z| > threshold, hold `horizon` bars (no re-entry while held). Weight in
    {-1, 0, +1}; the engine prices it on `mid` and charges turnover."""
    z = zscore(feats[feature], z_window).values
    w = np.zeros(len(z))
    i = 0
    while i < len(z):
        if not np.isnan(z[i]) and abs(z[i]) > threshold:
            w[i:i + horizon] = np.sign(z[i])
            i += horizon
        else:
            i += 1
    return pd.Series(w, index=feats.index)
