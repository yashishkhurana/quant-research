"""Multi-venue funding book: for each coin, long the perp on the venue that pays least funding and
short it on the venue that pays most, across Binance (BN), Hyperliquid (HL) and Bybit (BY).

Signal. For every venue pair (a, b) and coin, the funding spread s = f_b - f_a per fixed 8h UTC
window (each venue's settlements summed onto the same windows; qr.data.funding.per_window), and
its trailing mean over `lookback` windows.

Entry rule (rule='net', the default). Enter only where the trade is expected to pay for itself:
    expected_net_bps = |trailing s| (bps per window) x E[hold] - round_trip_bps(a, b) - margin_bps > 0
E[hold] is the expected REMAINING hold from entry, point in time: if the coin's spread spell is k
windows old at the decision, E[L - k | L > k] over spells that ENDED in the trailing
`hold_window_days`, pooled across coins and pairs (expected_remaining). A spell is a run of windows
where |trailing s| annualised is above `spell_apr` with one sign. Before enough spells have ended
there is no estimate and no entry. round_trip_bps = 2 x (fee + slippage) on each venue, from
qr.data.venues.TAKER_FEE_BPS. rule='raw' is the old rule (|trailing s| > entry_apr) for comparison.

Exit. A held pair is kept while its trailing spread keeps the sign it was entered with and stays
above `exit_apr`; it exits on a sign flip, below exit_apr, or a missing price. Selection changes
only at 8h window ends. One pair per coin (the one with the highest expected net), top
`max_assets` coins, equal weight per leg up to `max_weight`, gross capped at `max_gross`, and no
venue above `venue_cap` gross (qr.risk.venue_gross_cap).

Columns '<COIN>-<VENUE>'. Prices: BN and BY legs at their own venue's close; HL legs at the BN close
(HL serves only ~7 months of candles; if a coin has no BN close, the BY close). So BN-BY pairs carry
real venue basis P&L; pairs with an HL leg carry none, and HL basis is reported from its premium.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from qr.data.funding import WINDOWS_PER_YEAR, per_window
from qr.data.venues import TAKER_FEE_BPS
from qr.risk import venue_gross_cap


@dataclass
class MVParams:
    rule: str = "net"            # 'net' (cost-aware) or 'raw'
    lookback: int = 9            # 8h windows for the trailing spread
    margin_bps: float = 5.0      # required expected net, bps of leg notional
    spell_apr: float = 0.10      # |spread| that counts as a spell, for E[hold]
    hold_window_days: int = 180  # spells ending in this trailing window estimate E[hold]
    min_spells: int = 30         # no estimate (and no 'net' entry) with fewer ended spells
    entry_apr: float = 0.10      # 'raw' rule only
    exit_apr: float = 0.03
    max_assets: int = 10
    max_weight: float = 0.15
    max_gross: float = 2.0
    venue_cap: float = 1.0       # gross per venue, fraction of equity
    slippage_bps: float = 3.0


def round_trip_bps(a: str, b: str, slip: float) -> float:
    return 2 * (TAKER_FEE_BPS[a] + slip) + 2 * (TAKER_FEE_BPS[b] + slip)


def pair_spreads(funding: dict[str, pd.DataFrame], lookback: int) -> dict[tuple[str, str], pd.DataFrame]:
    """{(a, b): trailing mean of f_b - f_a per 8h window} on coins listed on both venues."""
    f8 = {v: per_window(f) for v, f in funding.items()}
    out = {}
    for a, b in combinations(sorted(f8), 2):
        coins = f8[a].columns.intersection(f8[b].columns)
        if len(coins) == 0:
            continue
        idx = f8[a].index.union(f8[b].index)
        s = f8[b][coins].reindex(idx) - f8[a][coins].reindex(idx)
        out[(a, b)] = s.rolling(lookback, min_periods=min(lookback, max(2, lookback // 2))).mean()
    return out


def spell_ends(trail: pd.DataFrame, spell_apr: float) -> pd.DataFrame:
    """One row per completed spell: end (window end where it stopped) and length in windows."""
    rows = []
    thr = spell_apr / WINDOWS_PER_YEAR
    for c in trail.columns:
        v = trail[c].values; idx = trail.index
        state = np.where(np.isnan(v), 0, np.where(v > thr, 1, np.where(v < -thr, -1, 0)))
        run, cur = 0, 0
        for i, st in enumerate(state):
            if st != 0 and st == cur:
                run += 1
            else:
                if cur != 0 and run > 0:
                    rows.append({"end": idx[i], "length": run})
                cur, run = st, (1 if st != 0 else 0)
    return pd.DataFrame(rows)


def expected_hold(spreads: dict, window_ends: pd.DatetimeIndex, p: MVParams) -> pd.Series:
    """Point-in-time E[hold] in windows at each window end, pooled over pairs and coins."""
    ends = pd.concat([spell_ends(t, p.spell_apr) for t in spreads.values()], ignore_index=True)
    if ends.empty:
        return pd.Series(np.nan, index=window_ends)
    ends = ends.sort_values("end")
    t_end = ends["end"].values; ln = ends["length"].values.astype(float)
    cs = np.concatenate([[0.0], np.cumsum(ln)])
    out = np.full(len(window_ends), np.nan)
    w = pd.Timedelta(days=p.hold_window_days).to_timedelta64()
    for i, t in enumerate(window_ends.values):
        hi = np.searchsorted(t_end, t, side="right")          # spells that ended at or before t
        lo = np.searchsorted(t_end, t - w, side="right")
        n = hi - lo
        if n >= p.min_spells:
            out[i] = (cs[hi] - cs[lo]) / n
    return pd.Series(out, index=window_ends)


def spell_ages(trail: pd.DataFrame, spell_apr: float) -> pd.DataFrame:
    """Windows into the current spell (same sign, |trailing| above spell_apr), at each window end;
    0 when not in a spell. Uses data up to and including that window only."""
    thr = spell_apr / WINDOWS_PER_YEAR
    v = trail.values
    st = np.where(np.isnan(v), 0, np.where(v > thr, 1, np.where(v < -thr, -1, 0)))
    age = np.zeros(st.shape, dtype=float)
    for i in range(len(st)):
        if i == 0:
            age[i] = (st[i] != 0)
        else:
            age[i] = np.where(st[i] == 0, 0, np.where(st[i] == st[i - 1], age[i - 1] + 1, 1))
    return pd.DataFrame(age, index=trail.index, columns=trail.columns)


def expected_remaining(spreads: dict, window_ends: pd.DatetimeIndex, p: MVParams, kmax: int = 120) -> np.ndarray:
    """E[L - k | L > k] in windows, for ages k = 0..kmax, at each window end: the mean remaining
    length of spells that ENDED in the trailing `hold_window_days` and lasted longer than k, pooled
    over pairs and coins. NaN where fewer than `min_spells` such spells exist. Shape (len, kmax+1).
    Conditioning on age is what makes this the remaining hold from the moment of entry; the
    unconditional mean spell length overstates it for any entry after a spell's first window."""
    ends = pd.concat([spell_ends(t, p.spell_apr) for t in spreads.values()], ignore_index=True)
    out = np.full((len(window_ends), kmax + 1), np.nan)
    if ends.empty:
        return out
    ends = ends.sort_values("end")
    t_end = ends["end"].values; ln = ends["length"].values.astype(int)
    w = pd.Timedelta(days=p.hold_window_days).to_timedelta64()
    ks = np.arange(kmax + 1)
    for i, t in enumerate(window_ends.values):
        hi = np.searchsorted(t_end, t, side="right"); lo = np.searchsorted(t_end, t - w, side="right")
        L = np.sort(ln[lo:hi])
        if len(L) == 0:
            continue
        cnt = len(L) - np.searchsorted(L, ks, side="right")             # spells with L > k
        suf = np.concatenate([np.cumsum(L[::-1])[::-1], [0]])            # sum of L over the tail
        tail_sum = suf[np.searchsorted(L, ks, side="right")]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(cnt >= p.min_spells, tail_sum / cnt - ks, np.nan)
    return out


def build_weights(prices: pd.DataFrame, funding: dict[str, pd.DataFrame], p: MVParams) -> tuple[pd.DataFrame, dict]:
    """prices: engine price matrix with '<COIN>-<VENUE>' columns (NaN = not tradeable there).
    Returns (weights, diagnostics)."""
    spreads = pair_spreads(funding, p.lookback)
    wends = sorted(set().union(*[s.index for s in spreads.values()]))
    wends = pd.DatetimeIndex(wends)
    ehold = expected_hold(spreads, wends, p) if p.rule == "net" else pd.Series(np.nan, index=wends)
    bar = prices.index
    wends = wends[(wends >= bar[0]) & (wends <= bar[-1])]
    dpos = np.unique(np.clip(bar.searchsorted(wends, side="left"), 0, len(bar) - 1))
    dts = bar[dpos]                                               # decision bars
    dec = np.zeros(len(bar), dtype=bool); dec[dpos] = True

    coins = sorted({c.rsplit("-", 1)[0] for c in prices.columns})
    cidx = {c: i for i, c in enumerate(coins)}
    pairs = list(spreads)
    priced = prices.notna()
    # per pair, at decision bars only: trailing spread and score (expected net bps, or |apr|)
    score = np.full((len(pairs), len(dts), len(coins)), np.nan)
    trail = np.full_like(score, np.nan)
    eh = ehold.reindex(dts, method="ffill").values
    erem = None
    if p.rule == "net":
        er = expected_remaining(spreads, wends, p)                        # (window ends, ages)
        loc = np.clip(wends.searchsorted(dts, side="right") - 1, 0, len(wends) - 1)
        erem = er[loc]                                                   # last window end at/before each decision bar
    for k, (a, b) in enumerate(pairs):
        s = spreads[(a, b)]
        cs = [c for c in s.columns if c in cidx and f"{c}-{a}" in prices.columns and f"{c}-{b}" in prices.columns]
        if not cs:
            continue
        sv = s[cs].reindex(dts, method="ffill").values
        ok = priced[[f"{c}-{a}" for c in cs]].loc[dts].values & priced[[f"{c}-{b}" for c in cs]].loc[dts].values
        sv = np.where(ok, sv, np.nan)
        cols = [cidx[c] for c in cs]
        trail[k][:, cols] = sv
        if p.rule == "net":
            age = spell_ages(s[cs], p.spell_apr).reindex(dts, method="ffill").fillna(0).values.astype(int)
            age = np.clip(age, 0, erem.shape[1] - 1)
            hold = np.take_along_axis(erem, age, axis=1)                  # E[remaining | this spell's age]
            hold = np.where(age > 0, hold, np.nan)                       # not in a spell: no entry
            score[k][:, cols] = np.abs(sv) * 1e4 * hold - round_trip_bps(a, b, p.slippage_bps) - p.margin_bps
        else:
            apr = np.abs(sv) * WINDOWS_PER_YEAR
            score[k][:, cols] = np.where(apr > p.entry_apr, apr, np.nan)

    exit_thr = p.exit_apr / WINDOWS_PER_YEAR
    held_pair = np.full(len(coins), -1); held_sign = np.zeros(len(coins))
    D_pair = np.full((len(dts), len(coins)), -1); D_sign = np.zeros((len(dts), len(coins)))
    for t in range(len(dts)):
        for j in np.where(held_pair >= 0)[0]:
            v = trail[held_pair[j], t, j]
            if np.isnan(v) or np.sign(v) != held_sign[j] or abs(v) < exit_thr:
                held_pair[j] = -1; held_sign[j] = 0
        sc = np.where(np.isnan(score[:, t, :]), -np.inf, score[:, t, :])
        best, bestk = sc.max(axis=0), sc.argmax(axis=0)
        cand = [j for j in range(len(coins)) if held_pair[j] < 0 and best[j] > 0]
        n_free = p.max_assets - int((held_pair >= 0).sum())
        for j in sorted(cand, key=lambda j: -best[j])[:max(0, n_free)]:
            held_pair[j] = bestk[j]; held_sign[j] = np.sign(trail[bestk[j], t, j])
        D_pair[t] = held_pair; D_sign[t] = held_sign

    # carry decisions forward to every bar; a missing price on either leg exits until the next decision
    seg = np.cumsum(dec) - 1                                      # index of the last decision at/before each bar
    live = seg >= 0
    pair_of = np.full((len(bar), len(coins)), -1); sign_of = np.zeros((len(bar), len(coins)))
    pair_of[live] = D_pair[seg[live]]; sign_of[live] = D_sign[seg[live]]
    for k, (a, b) in enumerate(pairs):
        for j, c in enumerate(coins):
            m = pair_of[:, j] == k
            if not m.any():
                continue
            ok = priced[f"{c}-{a}"].values & priced[f"{c}-{b}"].values
            bad = pd.Series(m & ~ok).groupby(seg).cummax().values   # once broken, stays broken in the segment
            pair_of[bad, j] = -1; sign_of[bad, j] = 0
    n = (pair_of >= 0).sum(axis=1).astype(float)
    leg = np.where(pair_of >= 0, np.minimum(1.0 / np.where(n > 0, n, np.nan), p.max_weight)[:, None], 0.0)
    leg = np.nan_to_num(leg)

    W = np.zeros((len(bar), len(prices.columns))); col = {c: i for i, c in enumerate(prices.columns)}
    for j, c in enumerate(coins):
        for k, (a, b) in enumerate(pairs):
            m = pair_of[:, j] == k
            if not m.any():
                continue
            # spread = f_b - f_a > 0: b pays more -> short b, long a
            W[m, col[f"{c}-{b}"]] = -sign_of[m, j] * leg[m, j]
            W[m, col[f"{c}-{a}"]] = sign_of[m, j] * leg[m, j]
    w = pd.DataFrame(W, index=bar, columns=prices.columns)
    g = w.abs().sum(axis=1)
    w = w.mul((p.max_gross / g).clip(upper=1.0).fillna(1.0), axis=0)
    w = venue_gross_cap(w, p.venue_cap)
    diag = {"expected_hold": ehold, "pairs": pairs, "decision_bars": dec}
    return w, diag


def engine_inputs(venues: dict, coins: list[str] | None = None, bn: str = "BN") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engine prices and funding with '<COIN>-<VENUE>' columns. `venues`: {name: qr.data.venues.Venue}."""
    base_idx = venues[bn].close.index
    px, fd = {}, {}
    bn_close = venues[bn].close
    by_close = venues["BY"].close if "BY" in venues and venues["BY"].close is not None else None
    for v, d in venues.items():
        for c in d.funding.columns:
            if coins is not None and c not in coins:
                continue
            if v == "HL":
                src = bn_close[c] if c in bn_close.columns else (by_close[c] if by_close is not None and c in by_close.columns else None)
                if src is not None:                     # HL tradeable only while its funding exists
                    live = d.funding[c].dropna()
                    if live.empty:
                        continue
                    src = src.where((src.index >= live.index[0]) & (src.index <= live.index[-1] + pd.Timedelta(hours=1)))
            else:
                src = d.close[c] if d.close is not None and c in d.close.columns else None
            if src is None:
                continue
            if c in d.delisted:
                src = src.where(src.index < d.delisted[c])
            px[f"{c}-{v}"] = src.reindex(base_idx)
            fd[f"{c}-{v}"] = d.funding[c]
    prices = pd.DataFrame(px).sort_index(axis=1)
    funding = pd.DataFrame(fd).sort_index()
    return prices, funding


def fees(columns: pd.Index, slip: float) -> tuple[pd.Series, pd.Series]:
    f = pd.Series({c: TAKER_FEE_BPS[c.rsplit("-", 1)[1]] for c in columns})
    return f, pd.Series(slip, index=columns)


def run_with_risk(prices, funding_cols, w, p: MVParams, stop: str = "vol", max_iter: int = 8,
                  kill_dd: float = 0.08, cooldown_bars: int = 72):
    """Engine + drawdown stop. stop='vol' (qr.risk.vol_kill_mask), 'fixed' (kill_dd from peak), 'none'."""
    from qr.backtest.engine import run
    from qr.risk import vol_kill_mask
    from qr.strategies.carry import CarryParams, kill_mask
    fee, slip = fees(prices.columns, p.slippage_bps)
    kw = dict(fee_bps=fee, slippage_bps=slip, funding=funding_cols, max_gross=p.max_gross, rebalance_band=0.25)
    res = run(prices, w, **kw)
    if stop == "none":
        return res
    mask = np.zeros(len(w), dtype=bool)
    cp = CarryParams(kill_dd=kill_dd, cooldown_bars=cooldown_bars)
    for _ in range(max_iter):
        active = (res.weights.abs().sum(axis=1) > 0).values
        new = vol_kill_mask(res.returns, cooldown_bars=cooldown_bars, active=active) if stop == "vol" \
            else kill_mask(res.returns, cp, active)
        if (new == mask).all():
            break
        mask = new
        res = run(prices, w * (~mask)[:, None], **kw)
    res.metrics["kill_bars"] = int(mask.sum()); res.metrics["kill_events"] = int((mask[1:] & ~mask[:-1]).sum())
    return res
