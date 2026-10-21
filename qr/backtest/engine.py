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
) -> Result:
    prices = prices.sort_index()
    weights = weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    # Untradeable -> zero weight; the force-close cost is picked up by turnover below.
    tradeable = prices.notna()
    weights = weights.where(tradeable, 0.0)

    if max_gross is not None:
        gross = weights.abs().sum(axis=1)
        scale = (max_gross / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(scale, axis=0)

    rets = prices.pct_change(fill_method=None)           # r_t = p_t / p_{t-1} - 1
    rets_filled = rets.fillna(0.0)

    w_prev = weights.shift(1).fillna(0.0)                 # weight held during bar t
    gross_r = (w_prev * rets_filled).sum(axis=1)          # portfolio return over bar t

    # Drift: weights held through bar t grow with asset returns before rebalance at t.
    drifted = w_prev * (1 + rets_filled)
    drifted = drifted.div(1 + gross_r, axis=0).fillna(0.0)
    dw = (weights - drifted).abs()
    turnover = dw.sum(axis=1)
    cost = turnover * (fee_bps + slippage_bps) / 1e4

    fund_pnl = pd.Series(0.0, index=prices.index)
    if funding is not None and not funding.empty:
        # A settlement at time s is charged to the position held into the first bar >= s.
        sett = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        pos = prices.index.searchsorted(funding.index, side="left")
        for i, (ts, row) in zip(pos, funding.iterrows()):
            if i < len(prices.index):
                sett.iloc[i] += row.reindex(prices.columns).fillna(0.0).values
        fund_pnl = -(w_prev * sett).sum(axis=1)          # long pays positive funding

    net_r = gross_r - cost + fund_pnl
    net_r.iloc[0] = 0.0
    equity = (1 + net_r).cumprod()
    bpy = infer_bars_per_year(prices.index)

    res = Result(net_r, gross_r, equity, turnover, cost, fund_pnl, weights, bpy)
    res.metrics = _metrics(net_r, equity, turnover, bpy)
    res.metrics["cost_drag_ann"] = float(cost.mean() * bpy)
    res.metrics["funding_ann"] = float(fund_pnl.mean() * bpy)
    return res


def summary(res: Result) -> pd.Series:
    return pd.Series(res.metrics).round(4)
