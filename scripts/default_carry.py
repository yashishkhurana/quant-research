"""Default-rate carry: hold the most liquid hedgeable coins long spot / short perp and collect
whatever funding they pay (mostly Binance's default 0.01% per 8h), with almost no trading.

Universe at the first bar of each month: top N hedgeable coins by trailing 30-day perp dollar
volume (volume lagged one bar, so the rank uses completed bars only), among coins with both a
perp and a spot price at that bar. Each coin: +1/N spot, -1/N perp (gross 2.0x). Weights drift
between month starts and are reset there; a coin losing either price mid-month exits both legs.

Funding is attributed per settlement against that settlement's own default (0.01% per 8h scaled
to the interval in force: 0.005% at 4h, 0.00125% at 1h): at default, above, below.

    python scripts/default_carry.py [--n 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qr.backtest.engine import run
from qr.data.funding import has_spot, interval_hours, load_funding
from qr.data.prices import load_kline_matrix

START, END = "2023-01-01", "2026-10-23"
FEE, SLIP = 4.0, 3.0
OUT = Path("reports/default_carry")
PERIODS = {"2023-24": ("2023-01-01", "2024-12-31 23:00"), "2025-26": ("2025-01-01", END),
           "full": (START, END)}


def build(n: int):
    u = [l.strip() for l in open("data/universe.txt") if l.strip()]
    H = has_spot(u)
    perp = load_kline_matrix(H, "PERP", "close", START, END)
    spot = load_kline_matrix(H, "SPOT", "close", START, END)
    dvol = load_kline_matrix(H, "PERP", "quote_vol", START, END)
    bases = sorted(set(perp.columns) & set(spot.columns))
    idx = perp.index.union(spot.index)
    perp, spot, dvol = (x.reindex(index=idx, columns=bases) for x in (perp, spot, dvol))

    both = perp.notna() & spot.notna()
    tv = dvol.fillna(0.0).rolling(24 * 30, min_periods=24 * 7).sum().shift(1)
    per = idx.tz_localize(None).to_period("M")
    month_start = pd.Series(np.r_[True, per[1:] != per[:-1]], index=idx)

    sel = pd.DataFrame(False, index=idx, columns=bases)
    for t in idx[month_start.values]:
        r = tv.loc[t].where(both.loc[t]).dropna()
        sel.loc[t, r.nlargest(n).index] = True
    # hold the month-start selection through the month; exit both legs if either price goes
    keep = np.broadcast_to(month_start.values[:, None], sel.shape)
    sel = sel.astype(float).where(keep).ffill().fillna(0.0).astype(bool) & both

    w = pd.DataFrame(0.0, index=idx, columns=[f"{b}-{m}" for b in bases for m in ("SPOT", "PERP")])
    for b in bases:
        w[f"{b}-SPOT"] = sel[b].astype(float) / n
        w[f"{b}-PERP"] = -sel[b].astype(float) / n     # not -sel[b]: on bools that is logical NOT
    prices = pd.concat({f"{b}-SPOT": spot[b] for b in bases} | {f"{b}-PERP": perp[b] for b in bases}, axis=1)[w.columns]

    raw = load_funding(bases, START, END)
    return prices, w, raw, month_start, sel


def attribute_funding(res, raw: pd.DataFrame, prices_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Funding PnL per bar split by whether each settlement was at, above or below its default."""
    iv = interval_hours(raw).bfill()                  # first settlement of a coin has no previous one
    default = 0.0001 * iv / 8
    cats = {"at_default": (raw - default).abs() < 1e-10, "above": raw > default + 1e-10, "below": raw < default - 1e-10}
    held_perp = res.weights.filter(like="-PERP").shift(1).fillna(0.0)
    held_perp.columns = [c.replace("-PERP", "") for c in held_perp.columns]
    pos = prices_index.searchsorted(raw.index, side="left")
    out = {}
    for name, m in cats.items():
        r = raw.where(m, 0.0).fillna(0.0)
        sett = pd.DataFrame(0.0, index=prices_index, columns=raw.columns)
        ok = pos < len(prices_index)
        sett_vals = np.zeros((len(prices_index), raw.shape[1]))
        np.add.at(sett_vals, pos[ok], r.values[ok])
        sett.loc[:, :] = sett_vals
        out[name] = -(held_perp.reindex(columns=raw.columns).fillna(0.0) * sett).sum(axis=1)
    return pd.DataFrame(out)


def stats(res, fa: pd.DataFrame, a: str, b: str) -> dict:
    r = res.returns.loc[a:b]; bpy = res.bars_per_year
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    ann = lambda x: float(x.loc[a:b].mean() * bpy)
    return {"ann_return": float(eq.iloc[-1] ** (bpy / len(r)) - 1), "ann_vol": float(r.std() * np.sqrt(bpy)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(bpy)), "max_dd": float(dd),
            "price_basis_ann": ann(res.gross_returns), "funding_ann": ann(res.funding_pnl),
            "funding_at_default_ann": ann(fa["at_default"]), "funding_above_ann": ann(fa["above"]),
            "funding_below_ann": ann(fa["below"]), "cost_ann": -ann(res.costs),
            "turnover_ann": ann(res.turnover)}


def main(a):
    prices, w, raw, month_start, sel = build(a.n)
    f_perp = raw.rename(columns=lambda b: f"{b}-PERP")
    res = run(prices, w, fee_bps=FEE, slippage_bps=SLIP, funding=f_perp, rebalance_bars=month_start)
    fa = attribute_funding(res, raw, prices.index)
    gap = float((fa.sum(axis=1) - res.funding_pnl).abs().sum())
    assert gap < 1e-9, f"attribution does not add up: {gap}"

    table = pd.DataFrame({p: stats(res, fa, *ab) for p, ab in PERIODS.items()})
    held = sel.sum(axis=1)
    print(f"top {a.n} by trailing 30d perp $ volume, monthly; coins held: median {held.median():.0f}, min {held.min()}")
    print(f"distinct coins ever held: {int(sel.any().sum())}\n")
    print(table.round(4).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"summary_n{a.n}.csv")
    json.dump({"n": a.n, "coins_ever_held": sorted(sel.columns[sel.any()].tolist())}, open(OUT / f"universe_n{a.n}.json", "w"))
    from qr.backtest.report import tearsheet
    tearsheet(res, f"Default-rate carry, top {a.n} liquid, monthly", out=OUT / f"tearsheet_n{a.n}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    main(ap.parse_args())
