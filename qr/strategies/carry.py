"""Funding-rate carry on Binance USDT-M perps, delta-neutral: long spot, short perp.

Mechanism. A perp whose funding rate is positive pays longs -> shorts. Short the perp, hold the
same notional in spot, collect funding every 8h. Price exposure nets out except for basis moves
(spot vs perp), which the engine captures because both legs are priced separately.

Signal. At each bar, trailing mean funding over `lookback` settlements, annualised. Enter an
asset when annualised carry > `entry_apr`, exit when < `exit_apr` (hysteresis cuts churn).
Capital is split equally across qualifying assets, capped at `max_assets`, each leg at most
`max_weight` of equity.

Risk layer.
- `max_gross` caps total gross exposure (both legs count).
- Drawdown kill-switch: if equity is more than `kill_dd` below its peak, flatten everything
  and stay flat for `cooldown_bars` bars.
- Liquidation distance: the short-perp leg at leverage L liquidates after a ~1/L move; with
  `max_weight` per leg and gross capped the implied leverage stays well under 2x, so this is
  reported rather than enforced.

Costs. Rebalances pay fee+slippage on both legs through the engine's turnover; funding is
charged by the engine on the perp columns only. The engine lags nothing: `weights(t)` uses
funding settled at or before t and earns t -> t+1.

Expectation. Carry on majors is thin after 2023 (often < 10% APR); it is the alts that pay,
and they are the ones that delist, so the universe must include delisted contracts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SETTLEMENTS_PER_YEAR = 3 * 365


@dataclass
class CarryParams:
    lookback: int = 9            # settlements (9 = 3 days)
    entry_apr: float = 0.10
    exit_apr: float = 0.03
    max_assets: int = 10
    max_weight: float = 0.15     # per leg, fraction of equity
    max_gross: float = 2.0
    kill_dd: float = 0.08
    cooldown_bars: int = 24 * 3  # bars (hours) flat after a kill


def annualised_carry(funding: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing mean funding per settlement, annualised. Index = settlement timestamps."""
    return funding.rolling(lookback, min_periods=max(2, lookback // 2)).mean() * SETTLEMENTS_PER_YEAR


def build_weights(prices: pd.DataFrame, funding: pd.DataFrame, p: CarryParams) -> pd.DataFrame:
    """prices: columns like 'BTC-PERP' and 'BTC-SPOT' on a common bar index.
    funding: columns = base symbols ('BTC'), index = settlement times.
    Returns weights on the same columns as prices."""
    bases = sorted({c.split("-")[0] for c in prices.columns if c.endswith("-PERP")})
    bases = [b for b in bases if f"{b}-SPOT" in prices.columns and b in funding.columns]

    carry = annualised_carry(funding[bases], p.lookback)
    carry = carry.reindex(prices.index, method="ffill")          # last settled value, no look-ahead

    tradeable = prices[[f"{b}-PERP" for b in bases]].notna().values & prices[[f"{b}-SPOT" for b in bases]].notna().values
    carry = carry.where(tradeable, np.nan)

    # hysteresis state machine, vectorised per column
    on = pd.DataFrame(False, index=prices.index, columns=bases)
    state = np.zeros(len(bases), dtype=bool)
    cv = carry.values
    out = np.zeros_like(cv, dtype=bool)
    for i in range(len(prices.index)):
        row = cv[i]
        state = np.where(np.isnan(row), False, np.where(state, row > p.exit_apr, row > p.entry_apr))
        out[i] = state
    on.loc[:, :] = out

    # rank by carry among 'on' assets; keep top max_assets
    ranked = carry.where(on).rank(axis=1, ascending=False)
    selected = ranked.le(p.max_assets) & on
    n = selected.sum(axis=1).replace(0, np.nan)
    per_asset = (1.0 / n).clip(upper=p.max_weight)
    leg = selected.astype(float).mul(per_asset, axis=0).fillna(0.0)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for b in bases:
        w[f"{b}-SPOT"] = leg[b]
        w[f"{b}-PERP"] = -leg[b]

    gross = w.abs().sum(axis=1)
    scale = (p.max_gross / gross).clip(upper=1.0).fillna(1.0)
    return w.mul(scale, axis=0)


def kill_mask(equity: pd.Series, p: CarryParams, prior: np.ndarray | None = None) -> np.ndarray:
    """True on bars that must be flat: every drawdown breach plus `cooldown_bars` after it.
    Breaches are detected on the equity curve that already includes earlier kills, so the
    mask is grown monotonically (union with `prior`) and the loop in run_with_risk converges."""
    dd = equity / equity.cummax() - 1
    breach = (dd < -p.kill_dd).values
    mask = np.zeros(len(equity), dtype=bool) if prior is None else prior.copy()
    for i in np.where(breach)[0]:
        mask[i + 1:i + 1 + p.cooldown_bars] = True       # flat from the NEXT bar: the breach bar is already realised
    return mask


def run_with_risk(prices: pd.DataFrame, funding_rates: pd.DataFrame, p: CarryParams,
                  fee_bps: float = 4.0, slippage_bps: float = 3.0, max_iter: int = 8):
    """Full pipeline: weights -> engine -> kill-switch -> engine, until the flat mask stops growing.
    funding_rates columns = base symbols; mapped onto the perp columns for the engine."""
    from qr.backtest.engine import run

    f_perp = funding_rates.rename(columns={b: f"{b}-PERP" for b in funding_rates.columns})
    w = build_weights(prices, funding_rates, p)
    kw = dict(fee_bps=fee_bps, slippage_bps=slippage_bps, funding=f_perp, max_gross=p.max_gross)
    res = run(prices, w, **kw)
    mask = None
    for _ in range(max_iter):
        new = kill_mask(res.equity, p, mask)
        if mask is not None and (new == mask).all():
            break
        mask = new
        res = run(prices, w * (~mask)[:, None], **kw)
    res.metrics["kill_bars"] = int(mask.sum()) if mask is not None else 0
    return res


def implied_leverage(res) -> pd.Series:
    """Gross exposure of the perp leg relative to equity: the number a risk desk asks for."""
    perp_cols = [c for c in res.weights.columns if c.endswith("-PERP")]
    return res.weights[perp_cols].abs().sum(axis=1)
