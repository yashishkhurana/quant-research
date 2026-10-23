"""Risk layer for multi-venue books: things a backtest of a hedged pair leaves out.

vol_kill_mask      drawdown stop scaled to the book's own volatility. A fixed 8% stop built for a
                   hedged book fired 177 times on a 72%-vol unhedged one and became the strategy
                   (notes/01_carry.md, 1b); the trigger here is k x the book's trailing daily vol x
                   sqrt(horizon), so it means the same thing for a 3%-vol and a 70%-vol book.
venue_gross_cap    hard cap on gross exposure per venue (exchange / withdrawal risk), applied by
                   scaling whole pairs so both legs stay equal.
liquidation_mask   each venue margins its own leg. At leverage L a leg is liquidated when its
                   adverse move since entry exceeds 1/L - maintenance; the pair is then closed
                   (both legs) until the next decision re-enters. Assumes no margin is moved
                   between venues during an episode, which is the conservative case.
adl_exposure       a proxy, not a queue simulation: share of a leg's held hours in which it was
                   in profit by more than `profit_pct` during an hour when the coin moved more than
                   `shock_pct` (the conditions under which Binance/Bybit ADL ranks a position high).
venue_basis        venue-vs-venue price divergence per coin (BN vs BY closes, both with full
                   history): daily std and worst day, in bps. HL's own premium series is in
                   qr.strategies.xvenue.basis_risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_kill_mask(returns: pd.Series, k: float = 3.0, horizon_days: float = 5.0, vol_window_days: int = 30,
                  cooldown_bars: int = 72, floor: float = 0.01, active: np.ndarray | None = None,
                  bars_per_day: int = 24) -> np.ndarray:
    """True on bars where the book must be flat. Threshold at bar t = max(floor, k * sigma_daily(t)
    * sqrt(horizon_days)), sigma from returns strictly before t (no look-ahead). Drawdown is measured
    from the peak since the last re-entry; after a trigger the book is flat for `cooldown_bars`."""
    r = returns.fillna(0.0).values
    sig = (returns.rolling(vol_window_days * bars_per_day, min_periods=bars_per_day * 7).std(ddof=1)
           * np.sqrt(bars_per_day)).shift(1).bfill().fillna(0.0).values
    thr = np.maximum(floor, k * sig * np.sqrt(horizon_days))
    act = np.ones(len(r), dtype=bool) if active is None else active
    mask = np.zeros(len(r), dtype=bool)
    eq, peak, flat_until = 1.0, 1.0, -1
    for t in range(len(r)):
        if t <= flat_until:
            mask[t] = True
            continue
        if t - 1 == flat_until:                          # re-entry: reset the reference
            eq, peak = 1.0, 1.0
        if act[t]:
            eq *= 1 + r[t]
            peak = max(peak, eq)
            if eq / peak - 1 < -thr[t]:
                flat_until = t + cooldown_bars
    return mask


def venue_of(col: str) -> str:
    return col.rsplit("-", 1)[1]


def coin_of(col: str) -> str:
    return col.rsplit("-", 1)[0]


def venue_gross_cap(weights: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Scale pairs so no venue carries more than `cap` gross (fraction of equity). A coin's two legs
    are scaled together by the tightest of its two venues' factors, so pairs stay neutral."""
    w = weights.copy()
    venues = sorted({venue_of(c) for c in w.columns})
    coins = sorted({coin_of(c) for c in w.columns})
    for _ in range(3):                                   # scaling one venue can relieve another
        g = {v: w[[c for c in w.columns if venue_of(c) == v]].abs().sum(axis=1) for v in venues}
        f = {v: (cap / g[v]).clip(upper=1.0).fillna(1.0) for v in venues}
        for coin in coins:
            legs = [c for c in w.columns if coin_of(c) == coin]
            held = w[legs].ne(0)
            fac = pd.concat([f[venue_of(c)].where(held[c], 1.0) for c in legs], axis=1).min(axis=1)
            w[legs] = w[legs].mul(fac, axis=0)
    return w


def liquidation_mask(prices: pd.DataFrame, weights: pd.DataFrame, leverage: float = 3.0,
                     maintenance: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (weights with liquidated pairs closed, events). An episode is a run of bars where a
    leg's target weight keeps its sign; entry price = the leg's price on the episode's first bar.
    On the first bar where the adverse move exceeds 1/leverage - maintenance, the coin's legs go
    to zero for the rest of that episode."""
    thr = 1.0 / leverage - maintenance
    w = weights.copy()
    events = []
    px = prices.reindex(index=w.index, columns=w.columns)
    for c in w.columns:
        s = np.sign(w[c].values)
        p = px[c].values
        start = None
        for t in range(len(s)):
            if s[t] == 0 or np.isnan(p[t]):
                start = None
                continue
            if start is None or s[t] != s[t - 1]:
                start = t
            move = p[t] / p[start] - 1
            adverse = move if s[t] < 0 else -move
            if adverse > thr:
                end = t
                while end < len(s) and s[end] == s[t]:
                    end += 1
                legs = [x for x in w.columns if coin_of(x) == coin_of(c)]
                w.iloc[t:end, [w.columns.get_loc(x) for x in legs]] = 0.0
                events.append({"ts": w.index[t], "leg": c, "adverse_move": adverse, "bars_closed": end - t})
                s = np.sign(w[c].values)                 # the rest of this episode is now flat
                start = None
    return w, pd.DataFrame(events)


def adl_exposure(prices: pd.DataFrame, weights: pd.DataFrame, shock_pct: float = 0.10,
                 profit_pct: float = 0.10) -> pd.DataFrame:
    """Per leg: held hours, hours in a shock (|1h move| > shock_pct) while the leg was in profit by
    more than profit_pct since entry, and that count as a share of held hours."""
    px = prices.reindex(index=weights.index, columns=weights.columns)
    r1 = px.pct_change(fill_method=None)
    rows = []
    for c in weights.columns:
        s = np.sign(weights[c].values); p = px[c].values; rr = r1[c].values
        held = shocks = 0; entry = None
        for t in range(len(s)):
            if s[t] == 0 or np.isnan(p[t]):
                entry = None; continue
            if entry is None or s[t] != s[t - 1]:
                entry = p[t]
            held += 1
            pnl = (p[t] / entry - 1) * s[t]
            if abs(rr[t]) > shock_pct and pnl > profit_pct:
                shocks += 1
        if held:
            rows.append({"leg": c, "held_hours": held, "adl_risk_hours": shocks, "share": shocks / held})
    return pd.DataFrame(rows)


def venue_basis(close_a: pd.DataFrame, close_b: pd.DataFrame, mult_a: dict, mult_b: dict,
                bars_per_day: int = 24) -> pd.DataFrame:
    """log(a/b) per coin on multiplier-adjusted closes, in bps: mean, daily std of its change, worst
    one-day change. A long-a / short-b pair gains or loses exactly this change on top of funding."""
    rows = []
    for c in close_a.columns.intersection(close_b.columns):
        a = close_a[c] / mult_a.get(c, 1.0); b = close_b[c] / mult_b.get(c, 1.0)
        lb = np.log(a / b).dropna()
        if len(lb) < bars_per_day * 7:
            continue
        d = lb.diff(bars_per_day).dropna()
        rows.append({"coin": c, "hours": len(lb), "mean_bps": lb.mean() * 1e4,
                     "daily_std_bps": d.std() * 1e4, "worst_day_bps": d.abs().max() * 1e4})
    return pd.DataFrame(rows).set_index("coin") if rows else pd.DataFrame()
