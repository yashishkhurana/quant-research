"""Capacity of a weight-based book: how much capital it can deploy, and what impact costs it.

There is no open-interest history for Binance perps (the API serves 30 days), so capacity is
defined on traded volume, which exists for every contract back to 2023, delisted ones included.

Definitions
-----------
ADV_i(t)      trailing `window`-day mean of daily dollar volume, completed days only (shift 1).
cap_i(t)      the largest notional the book will hold in name i: `participation` x ADV_i(t).
              For a hedged pair (spot long, perp short) both legs carry the same notional, so
              the pair's cap is participation x min(ADV_spot, ADV_perp). Participation is the
              share of one day's volume the desk is willing to be when building or unwinding.
C(t)          static capacity of the target book: the largest equity B such that B*|w_i(t)| <= cap_i(t)
              for every held name, i.e. min_i cap_i / |w_i|. Reported as a distribution over
              active bars.
Capped weights at book size B: w_i^B = sign(w_i) * min(|w_i|, cap_i / B). The shortfall is
              undeployed capital and earns nothing.
Impact        square-root law (Almgren et al. 2005; the form every desk uses): a trade of
              notional Q in a name with daily volatility sigma and daily volume ADV costs
              Y * sigma * sqrt(Q / ADV) of its notional, Y ~ 1. Charged on every trade the
              engine makes, per leg, on top of the flat fee + slippage already in the engine.
              This is a one-way cost; it does not depend on how fast the trade is worked, which
              is the usual simplification and errs on the cheap side for trades done inside an hour.

All inputs are matrices on the bar index (hourly) with the engine's column names ('BTC-PERP',
'BTC-SPOT').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_DAY = 24


def adv(dollar_volume: pd.DataFrame, window_days: int = 30, bars_per_day: int = BARS_PER_DAY) -> pd.DataFrame:
    """Trailing mean daily dollar volume, completed bars only: the value at bar t uses bars
    < t. NaN until a quarter of the window has data, and where the contract has no volume.
    Mean hourly volume x 24, not window sum / window_days: a coin listed 10 days ago has 10 days
    of volume, and dividing by 30 would understate its ADV threefold."""
    w = window_days * bars_per_day
    daily = dollar_volume.rolling(w, min_periods=w // 4).mean().shift(1) * bars_per_day
    return daily.where(dollar_volume.notna().rolling(w, min_periods=1).max().shift(1) > 0)


def daily_vol(prices: pd.DataFrame, window_days: int = 30, bars_per_day: int = BARS_PER_DAY) -> pd.DataFrame:
    """Trailing daily volatility from bar log returns, completed bars only."""
    w = window_days * bars_per_day
    r = np.log(prices).diff()
    return (r.rolling(w, min_periods=w // 4).std(ddof=1) * np.sqrt(bars_per_day)).shift(1)


def pair_cap(adv_perp: pd.DataFrame, adv_spot: pd.DataFrame | None, participation: float,
             columns: pd.Index) -> pd.DataFrame:
    """Notional cap per engine column. adv_* have base-symbol columns. For a hedged book the cap
    for both legs of a base is participation x min(perp ADV, spot ADV); for a perp-only column
    (no spot) it is participation x perp ADV. Columns with no ADV (either leg, e.g. a listing
    younger than adv's minimum history) get NaN: unknown, not zero. cap_weights treats NaN as 0
    (do not size what cannot be measured); static_capacity leaves it out of the min."""
    out = pd.DataFrame(np.nan, index=adv_perp.index, columns=columns)
    for c in columns:
        base, leg = c.rsplit("-", 1)
        if base not in adv_perp.columns:
            continue
        a = adv_perp[base]
        if adv_spot is not None and base in adv_spot.columns and f"{base}-SPOT" in columns:
            a = np.minimum(a, adv_spot[base])
        out[c] = participation * a
    return out


def static_capacity(weights: pd.DataFrame, cap: pd.DataFrame) -> pd.Series:
    """min_i cap_i / |w_i| over held names with a known cap, per bar. NaN on bars with no position
    or where no held name has a cap. See uncapped_share for how often a held name had none."""
    w = weights.abs()
    ratio = cap.reindex_like(w).where(w > 0) / w.where(w > 0)
    return ratio.min(axis=1, skipna=True)


def uncapped_held(weights: pd.DataFrame, cap: pd.DataFrame) -> pd.Series:
    """Per bar: True if some held name has no cap (NaN). NaN on bars with no position."""
    held = weights.abs() > 0
    return (held & cap.reindex_like(weights).isna()).any(axis=1).where(held.any(axis=1))


def cap_weights(weights: pd.DataFrame, cap: pd.DataFrame, book_size: float) -> pd.DataFrame:
    """Clip |w_i| at cap_i / book_size, keeping the sign. Both legs of a hedged pair share a cap
    so they stay equal and the pair stays delta-neutral."""
    limit = (cap / book_size).reindex(index=weights.index, columns=weights.columns).fillna(0.0)
    return np.sign(weights) * np.minimum(weights.abs(), limit)


def impact_drag(trades: pd.DataFrame, sigma: pd.DataFrame, adv_cols: pd.DataFrame,
                book_size: float, y: float = 1.0) -> pd.Series:
    """Per-bar impact cost as a fraction of equity.
    trades: per-column |dw| actually traded (fraction of equity), from the engine.
    sigma: daily volatility per column. adv_cols: daily dollar volume per column (perp ADV for
    -PERP, spot ADV for -SPOT). Cost_i = |dw_i| * Y * sigma_i * sqrt(B |dw_i| / ADV_i)."""
    q = trades.abs() * book_size
    frac = (q / adv_cols.reindex(index=trades.index, columns=trades.columns)).clip(lower=0.0)
    per_name = trades.abs() * y * sigma.reindex(index=trades.index, columns=trades.columns) * np.sqrt(frac)
    return per_name.fillna(0.0).sum(axis=1)


def adv_by_column(adv_perp: pd.DataFrame, adv_spot: pd.DataFrame | None, columns: pd.Index) -> pd.DataFrame:
    """Each engine column's own ADV (perp columns -> perp ADV, spot columns -> spot ADV)."""
    out = pd.DataFrame(np.nan, index=adv_perp.index, columns=columns)
    for c in columns:
        base, leg = c.rsplit("-", 1)
        src = adv_perp if leg == "PERP" else adv_spot
        if src is not None and base in src.columns:
            out[c] = src[base]
    return out


def deployed_share(capped: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Gross exposure actually held over gross exposure wanted, per bar; NaN when flat."""
    g_t = target.abs().sum(axis=1)
    return (capped.abs().sum(axis=1) / g_t.where(g_t > 0))
