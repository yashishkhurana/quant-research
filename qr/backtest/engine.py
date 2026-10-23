"""Vectorised backtester for weight-based strategies on perps.

Conventions
-----------
- `prices`: DataFrame, index = bar timestamps (UTC), columns = assets. NaN = not tradeable
  (not yet listed, or delisted). A position in an asset whose price turns NaN is force-closed
  at the last valid price with full cost; that is how survivorship is handled, not by dropping it.
- `weights`: target portfolio weights as fraction of equity, same shape as prices. The weight
  at row t is decided with information up to and including bar t and EARNS the return from t to
  t+1. Lag your signals before passing them in; the engine will not do it for you, and that is
  deliberate so the look-ahead is visible in your code.
- `funding`: optional DataFrame of funding rates (fraction, e.g. 0.0001) at settlement timestamps.
  A long perp PAYS positive funding; short receives. Aligned to bars by forward-filling the
  settlement onto the next bar.
- Costs are charged on turnover: |w_t - w_{t-1} * (1+r_t)/(1+r_p,t)| summed across assets,
  times (fee_bps + slippage_bps). Slippage is a flat model; the microstructure project replaces it.

Returns a `Result` with the per-bar return series, equity curve, turnover, cost drag and a
metrics dict. Bars per year is inferred from the index spacing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Result:
    returns: pd.Series            # net per-bar portfolio return
    gross_returns: pd.Series      # before costs and funding
    equity: pd.Series
    turnover: pd.Series           # sum |dw| per bar
    costs: pd.Series              # cost drag per bar (positive number)
    funding_pnl: pd.Series        # funding received (+) / paid (-)
    weights: pd.DataFrame         # realised weights after force-closes
    bars_per_year: float
    metrics: dict = field(default_factory=dict)


def infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 365.0
    dt = pd.Series(index).diff().median()
    return pd.Timedelta(days=365) / dt


def _metrics(r: pd.Series, equity: pd.Series, turnover: pd.Series, bpy: float) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {}
    mu, sd = r.mean(), r.std(ddof=1)
    downside = r[r < 0].std(ddof=1)
    dd = equity / equity.cummax() - 1
    ann_ret = (1 + r).prod() ** (bpy / len(r)) - 1
    return {
        "bars": int(len(r)),
        "ann_return": float(ann_ret),
        "ann_vol": float(sd * np.sqrt(bpy)),
        "sharpe": float(mu / sd * np.sqrt(bpy)) if sd > 0 else float("nan"),
        "sortino": float(mu / downside * np.sqrt(bpy)) if downside and downside > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "calmar": float(ann_ret / -dd.min()) if dd.min() < 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover_per_bar": float(turnover.mean()),
        "ann_turnover": float(turnover.mean() * bpy),
        "final_equity": float(equity.iloc[-1]),
    }


def run(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    funding: pd.DataFrame | None = None,
    max_gross: float | None = None,
    rebalance_band: float = 0.0,
) -> Result:
    """`rebalance_band`: no-trade band as a fraction of the target weight. With 0 the book is
    rebalanced to the exact target every bar (charging price drift every bar). With e.g. 0.25 an
    asset is only traded when its realised weight is more than 25% of target away from target,
    or when the target itself changes sign / goes to zero. Realised weights then drift between
    trades, and the book that earns the next bar's return is the realised one, not the target.
    """
    prices = prices.sort_index()
    weights = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    tradeable = prices.notna()
    weights = weights.where(tradeable, 0.0)              # untradeable -> forced flat

    if max_gross is not None:
        gross = weights.abs().sum(axis=1)
        scale = (max_gross / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(scale, axis=0)

    rets = prices.pct_change(fill_method=None).fillna(0.0).values
    W = weights.values
    n_bars, n_assets = W.shape

    sett = np.zeros_like(W)
    if funding is not None and not funding.empty:
        # A settlement at time s is charged to the position held into the first bar >= s.
        f = funding.reindex(columns=prices.columns).fillna(0.0)
        pos = prices.index.searchsorted(f.index, side="left")
        for i, row in zip(pos, f.values):
            if i < n_bars:
                sett[i] += row

    held = np.zeros(n_assets)                            # realised weights held into bar t
    gross_r = np.zeros(n_bars); turnover = np.zeros(n_bars); fund = np.zeros(n_bars)
    realised = np.zeros_like(W)
    rate = (fee_bps + slippage_bps) / 1e4
    for t in range(n_bars):
        r = rets[t]
        g = float(held @ r)                              # portfolio return over bar t
        gross_r[t] = g
        fund[t] = -float(held @ sett[t])                 # long pays positive funding
        drifted = held * (1 + r) / (1 + g) if g > -1 else np.zeros(n_assets)
        target = W[t]
        if rebalance_band > 0:
            gap = np.abs(target - drifted)
            must = (target == 0) | (np.sign(target) != np.sign(drifted)) | (gap > rebalance_band * np.abs(target))
            new = np.where(must, target, drifted)
        else:
            new = target
        turnover[t] = float(np.abs(new - drifted).sum())
        held = new
        realised[t] = new

    cost = turnover * rate
    net = gross_r - cost + fund
    net[0] = 0.0
    idx = prices.index
    net_r = pd.Series(net, index=idx); equity = (1 + net_r).cumprod()
    bpy = infer_bars_per_year(idx)
    res = Result(net_r, pd.Series(gross_r, index=idx), equity, pd.Series(turnover, index=idx),
                 pd.Series(cost, index=idx), pd.Series(fund, index=idx),
                 pd.DataFrame(realised, index=idx, columns=prices.columns), bpy)
    res.metrics = _metrics(net_r, equity, res.turnover, bpy)
    res.metrics["cost_drag_ann"] = float(cost.mean() * bpy)
    res.metrics["funding_ann"] = float(fund.mean() * bpy)
    return res


def summary(res: Result) -> pd.Series:
    return pd.Series(res.metrics).round(4)
