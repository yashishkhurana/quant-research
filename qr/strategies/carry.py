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
    universe_n: int = 40         # tradeable set = top N by trailing dollar volume, point in time
    volume_window: int = 24 * 30 # bars of trailing volume used for that rank
    hedged: bool = True          # False = short perp only, no spot leg (the "unhedged carry" people ran)
    universe_exit_n: int = 60    # a coin leaves the tradeable set only when it drops below this rank
    rebalance_band: float = 0.25 # engine no-trade band, fraction of target weight


def point_in_time_universe(dollar_volume: pd.DataFrame, n: int, window: int,
                           exit_n: int | None = None) -> pd.DataFrame:
    """Boolean mask, True where an asset is in the tradeable set by trailing `window`-bar dollar
    volume as of that bar. Uses only past volume, so a coin enters when it becomes liquid and
    leaves when it stops trading. With `exit_n` > n the membership has hysteresis: enter at rank
    <= n, leave only at rank > exit_n, so coins at the boundary do not flip in and out hourly."""
    tv = dollar_volume.fillna(0.0).rolling(window, min_periods=window // 4).sum()
    rank = tv.where(tv > 0).rank(axis=1, ascending=False)
    if exit_n is None or exit_n <= n:
        return rank.le(n)
    enter = rank.le(n).values; stay = rank.le(exit_n).values
    out = np.zeros_like(enter); state = np.zeros(enter.shape[1], dtype=bool)
    for i in range(len(enter)):
        state = np.where(state, stay[i], enter[i])
        out[i] = state
    return pd.DataFrame(out, index=rank.index, columns=rank.columns)


def annualised_carry(funding: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing mean funding per fixed 8h window, annualised. `funding` is the raw per-settlement
    matrix (index = settlement times); it is first summed onto 8h UTC windows so that coins on
    4h or 1h settlement are comparable with 8h coins and `lookback` means 8h windows for all.
    Index = window ends (00:00, 08:00, 16:00 UTC); the value at T is known at T."""
    from qr.data.funding import per_window, WINDOWS_PER_YEAR
    f8 = per_window(funding)
    return f8.rolling(lookback, min_periods=max(2, lookback // 2)).mean() * WINDOWS_PER_YEAR


def build_weights(prices: pd.DataFrame, funding: pd.DataFrame, p: CarryParams,
                  dollar_volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """prices: columns like 'BTC-PERP' and 'BTC-SPOT' on a common bar index.
    funding: columns = base symbols ('BTC'), index = settlement times.
    dollar_volume: optional, columns = base symbols, perp quote volume per bar; when given the
    tradeable set at each bar is the top `p.universe_n` by trailing volume (point in time).
    Returns weights on the same columns as prices."""
    bases = sorted({c.split("-")[0] for c in prices.columns if c.endswith("-PERP")})
    bases = [b for b in bases if b in funding.columns and (not p.hedged or f"{b}-SPOT" in prices.columns)]

    carry = annualised_carry(funding[bases], p.lookback)
    window_ends = carry.index                                     # 8h window ends: the decision clock
    carry = carry.reindex(prices.index, method="ffill")          # last settled value, no look-ahead

    # priced: both legs have a price (a missing price is a delisting and forces an exit on that bar).
    # tradeable: priced AND inside the point-in-time volume universe. Universe membership is only
    # consulted when a selection is decided; a coin drifting out of the top-N between decisions is
    # not force-closed, which is what made the hourly rank flips into round trips.
    priced = prices[[f"{b}-PERP" for b in bases]].notna().values.copy()
    if p.hedged:                                                  # the hedge needs a spot price too
        priced &= prices[[f"{b}-SPOT" for b in bases]].notna().values
    tradeable = priced.copy()
    if dollar_volume is not None:
        in_univ = point_in_time_universe(dollar_volume.reindex(columns=bases).reindex(prices.index),
                                         p.universe_n, p.volume_window, p.universe_exit_n)
        tradeable &= in_univ.fillna(False).values
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

    # rank by carry among 'on' assets; keep top max_assets. The selection is only re-decided on
    # bars where a settlement has just landed (the signal cannot change in between); on other bars
    # the previous selection is carried forward, minus anything that stopped being tradeable.
    ranked = carry.where(on).rank(axis=1, ascending=False)
    fresh = ranked.le(p.max_assets) & on
    sett_bar = pd.Series(False, index=prices.index)
    sett_bar.iloc[np.unique(np.clip(prices.index.searchsorted(window_ends, side="left"), 0, len(prices.index) - 1))] = True
    sel = np.zeros_like(fresh.values); prev = np.zeros(fresh.shape[1], dtype=bool)
    fv, px_v, sb = fresh.values, priced, sett_bar.values
    for i in range(len(sel)):
        prev = fv[i] if sb[i] else (prev & px_v[i])              # between decisions: only delistings exit
        sel[i] = prev
    selected = pd.DataFrame(sel, index=fresh.index, columns=fresh.columns)
    n = selected.sum(axis=1).replace(0, np.nan)
    per_asset = (1.0 / n).clip(upper=p.max_weight)
    leg = selected.astype(float).mul(per_asset, axis=0).fillna(0.0)

    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for b in bases:
        w[f"{b}-PERP"] = -leg[b]
        if p.hedged:
            w[f"{b}-SPOT"] = leg[b]

    gross = w.abs().sum(axis=1)
    scale = (p.max_gross / gross).clip(upper=1.0).fillna(1.0)
    return w.mul(scale, axis=0)


def kill_mask(returns: pd.Series, p: CarryParams, active: np.ndarray | None = None) -> np.ndarray:
    """True on bars that must be flat. Walks the bar returns once: track a peak, and when equity
    falls more than `kill_dd` below it, go flat for `cooldown_bars`; on re-entry the peak is reset
    to current equity so the drawdown condition is measured from the re-entry, not from an
    all-time high the flat book can never climb back to. `active` (optional) marks bars where the
    book had a position; breaches are only counted on those bars."""
    r = returns.fillna(0.0).values
    n = len(r); mask = np.zeros(n, dtype=bool)
    eq = 1.0; peak = 1.0; i = 0
    while i < n:
        eq *= (1 + r[i])
        peak = max(peak, eq)
        trading = True if active is None else bool(active[i])
        if trading and eq / peak - 1 < -p.kill_dd:
            end = min(n, i + 1 + p.cooldown_bars)
            mask[i + 1:end] = True
            i = end
            peak = eq                                      # reference reset at re-entry
            continue
        i += 1
    return mask


def run_with_risk(prices: pd.DataFrame, funding_rates: pd.DataFrame, p: CarryParams,
                  fee_bps: float = 4.0, slippage_bps: float = 3.0, max_iter: int = 8,
                  dollar_volume: pd.DataFrame | None = None):
    """Full pipeline: weights -> engine -> kill-switch -> engine, until the flat mask stops growing.
    funding_rates columns = base symbols; mapped onto the perp columns for the engine."""
    from qr.backtest.engine import run

    f_perp = funding_rates.rename(columns={b: f"{b}-PERP" for b in funding_rates.columns})
    w = build_weights(prices, funding_rates, p, dollar_volume)
    kw = dict(fee_bps=fee_bps, slippage_bps=slippage_bps, funding=f_perp, max_gross=p.max_gross,
              rebalance_band=p.rebalance_band)
    res = run(prices, w, **kw)
    mask = np.zeros(len(w), dtype=bool)
    for _ in range(max_iter):
        active = (res.weights.abs().sum(axis=1) > 0).values
        new = kill_mask(res.returns, p, active)
        if (new == mask).all():
            break
        mask = new
        res = run(prices, w * (~mask)[:, None], **kw)
    res.metrics["kill_bars"] = int(mask.sum())
    res.metrics["kill_events"] = int((mask[1:] & ~mask[:-1]).sum())
    return res


def implied_leverage(res) -> pd.Series:
    """Gross exposure of the perp leg relative to equity: the number a risk desk asks for."""
    perp_cols = [c for c in res.weights.columns if c.endswith("-PERP")]
    return res.weights[perp_cols].abs().sum(axis=1)
