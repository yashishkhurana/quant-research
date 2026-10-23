"""Multi-venue funding book (Binance, Hyperliquid, Bybit): data on one clock, cost-aware entry,
per-venue capacity, and a risk layer. Writes reports/mvenue/*; a hand-written note goes in notes/.

    python -m qr.data.hl_history funding --start 2024-01-01          # data (see each module)
    python -m qr.data.hl_history candles                             # HL volume, last ~7 months
    python -m qr.data.bybit_history funding|klines|oi --start 2023-01-01
    python scripts/run_mvenue.py --start 2024-01-01

Sections of the output, in order:
  1 data      coins per venue and per pair after name + price matching; delistings per venue
  2 spreads   funding spread per venue pair by year; spell lengths (what E[hold] is estimated from)
  3 backtest  raw rule vs cost-aware rule, fixed 8% stop vs vol-scaled stop; 12-cell grid on the
              cost-aware rule with the best cell deflated for 12 trials
  4 capacity  per-venue caps (2% of each venue's ADV; 1% of OI where Bybit has history), static
              capacity of the book, capped runs with square-root impact from $1m to $100m
  5 risk      liquidations per venue at 3x, ADL exposure proxy, BN-BY basis, HL premium basis,
              gross per venue
In-sample throughout; no walk-forward yet.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qr.backtest.report import tearsheet
from qr.capacity import adv, cap_pairs, daily_vol, impact_drag, static_capacity, uncapped_held, venue_leg_caps
from qr.data.funding import WINDOWS_PER_YEAR, per_window
from qr.data.venues import load_binance, load_bybit, load_hyperliquid, price_matched
from qr.risk import adl_exposure, liquidation_mask, venue_basis
from qr.strategies.mvenue import MVParams, build_weights, engine_inputs, pair_spreads, run_with_risk, spell_ends
from qr.strategies.xvenue import basis_risk
from qr.validation import deflated_sharpe

OUT = Path("reports/mvenue")


FULL = ("2023-01-01", "2026-10-23")          # matrices are cached for this range; runs slice it


def load(start: str, end: str, min_adv: float):
    bn, by, hl = load_binance(*FULL), load_bybit(*FULL), load_hyperliquid(*FULL)
    for v in (bn, by, hl):
        v.funding = v.funding.loc[start:end]
    # Bybit: keep a coin only if its price agrees with Binance (when both list it)
    shared = set(bn.close.columns) & set(by.close.columns)
    ok = set(price_matched(bn, by))
    drop_by = sorted(shared - ok)
    by.funding = by.funding.drop(columns=[c for c in drop_by if c in by.funding.columns])
    # Hyperliquid: the same hourly check against Binance (else Bybit) over the hours HL candles cover
    # (~7 months). Coins without HL candles (delisted before that) cannot be checked and are kept.
    drop_hl, hl_unchecked = [], []
    if hl.close is not None:
        ok_hl = set(price_matched(bn, hl)) | set(price_matched(by, hl))
        for c in hl.funding.columns:
            if c not in hl.close.columns or hl.close[c].notna().sum() < 24 * 7:
                hl_unchecked.append(c)
            elif (c in bn.close.columns or c in by.close.columns) and c not in ok_hl:
                drop_hl.append(c)
    hl.funding = hl.funding.drop(columns=[c for c in drop_hl if c in hl.funding.columns])
    venues = {"BN": bn, "HL": hl, "BY": by}
    prices, funding_cols = engine_inputs(venues)
    prices = prices.loc[start:end]
    # liquidity universe (point in time): the coin's BN ADV, else BY ADV, >= min_adv. Used for
    # selection only; the engine sees every price (a forced exit still trades at a price).
    adv_bn, adv_by = adv(bn.dvol).reindex(prices.index), adv(by.dvol).reindex(prices.index)
    liq = adv_bn.combine_first(adv_by)
    sel_prices = prices.copy()
    for c in sel_prices.columns:
        coin = c.rsplit("-", 1)[0]
        if coin in liq.columns:
            sel_prices[c] = sel_prices[c].where(liq[coin] >= min_adv)
        else:
            sel_prices[c] = np.nan
    info = {"coins": {v: int(d.funding.shape[1]) for v, d in venues.items()},
            "delisted": {v: len(d.delisted) for v, d in venues.items()},
            "bybit_dropped_price_mismatch": drop_by, "hl_dropped_price_mismatch": drop_hl,
            "hl_unchecked_no_candles": len(hl_unchecked),
            "engine_columns": int(prices.shape[1])}
    return venues, prices, sel_prices, funding_cols, info


def spread_tables(venues, lookback: int):
    f8 = {v: per_window(d.funding) for v, d in venues.items()}
    rows, spells = [], []
    for (a, b), trail in pair_spreads({v: d.funding for v, d in venues.items()}, lookback).items():
        coins = f8[a].columns.intersection(f8[b].columns)
        s = (f8[b][coins] - f8[a][coins]) * 1e4
        x = s.stack().dropna()
        for yr, g in x.groupby(x.index.get_level_values(0).year):
            rows.append({"pair": f"{a}-{b}", "year": yr, "coins": int(g.index.get_level_values(1).nunique()),
                         "coin_windows": len(g), "median_bps": g.median(), "mean_abs_bps": g.abs().mean(),
                         "share_abs_gt_1bp": (g.abs() > 1).mean(), "share_abs_gt_2bp": (g.abs() > 2).mean(),
                         "share_abs_gt_5bp": (g.abs() > 5).mean(), "share_b_higher": (g > 0).mean()})
        e = spell_ends(trail, 0.10)
        if len(e):
            e["pair"] = f"{a}-{b}"; spells.append(e)
    sp = pd.concat(spells) if spells else pd.DataFrame(columns=["end", "length", "pair"])
    sp_tab = sp.groupby("pair")["length"].describe(percentiles=[0.5, 0.9]) if len(sp) else pd.DataFrame()
    return pd.DataFrame(rows), sp_tab


def summarize(name, res):
    m = res.metrics
    return {"run": name, "ann_return": m["ann_return"], "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
            "funding_ann": m["funding_ann"], "costs_ann": m["cost_drag_ann"], "turnover_ann": m["ann_turnover"],
            "kill_events": m.get("kill_events", 0)}


def main(a):
    global OUT
    OUT = Path(a.out) if a.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    venues, prices, sel, fcols, info = load(a.start, a.end, a.min_adv)
    coin_ok = sel.notna().T.groupby(lambda c: c.rsplit("-", 1)[0]).any().T
    info["min_adv"] = a.min_adv
    info["coins_passing_floor_per_bar"] = coin_ok.sum(axis=1).describe(percentiles=[0.1, 0.5, 0.9]).round(1).to_dict()
    print(json.dumps({k: v for k, v in info.items() if "dropped" not in k}, indent=1))
    funding = {v: d.funding for v, d in venues.items()}

    st, sp = spread_tables(venues, 9)
    st.to_csv(OUT / "spreads_by_year.csv", index=False); sp.to_csv(OUT / "spread_spells.csv")

    # --- backtests
    runs, rets = [], {}
    base = None
    for rule in ("raw", "net"):
        p = MVParams(rule=rule)
        w, diag = build_weights(sel, funding, p)
        if rule == "net":
            base = (w, diag)
            diag["expected_hold"].to_csv(OUT / "expected_hold.csv")
        for stop in ("fixed", "vol"):
            r = run_with_risk(prices, fcols, w, p, stop=stop)
            runs.append(summarize(f"{rule} rule, {stop} stop", r)); rets[(rule, stop)] = r
    w_net, diag = base
    p0 = MVParams()
    liq_ev = pd.DataFrame()
    for L in a.leverage:
        w_liq, ev = liquidation_mask(prices, w_net, leverage=L)
        r_liq = run_with_risk(prices, fcols, w_liq, p0, stop="vol")
        runs.append(summarize(f"net rule, vol stop, liquidations at {L:g}x", r_liq))
        ev["leverage"] = L; liq_ev = pd.concat([liq_ev, ev], ignore_index=True)
    pd.DataFrame(runs).to_csv(OUT / "backtests.csv", index=False)
    tearsheet(rets[("net", "vol")], "Multi-venue funding, cost-aware entry, vol-scaled stop", out=OUT / "tearsheet.png")

    # grid on the cost-aware rule
    grid, trials = [], []
    cells = [(9, 5.0), (21, 10.0)] if a.quick else itertools.product([3, 9, 21], [0.0, 5.0, 10.0, 20.0])
    for lb, mg in cells:
        p = MVParams(lookback=lb, margin_bps=mg)
        w, _ = build_weights(sel, funding, p)
        r = run_with_risk(prices, fcols, w, p, stop="vol", max_iter=4)
        grid.append({"lookback": lb, "margin_bps": mg, "sharpe": r.metrics["sharpe"], "ann_return": r.metrics["ann_return"],
                     "turnover_ann": r.metrics["ann_turnover"]}); trials.append(r.returns)
    g = pd.DataFrame(grid); g.to_csv(OUT / "grid.csv", index=False)
    ib = int(g["sharpe"].idxmax()); bpy = rets[("net", "vol")].bars_per_year
    best = {"cell": g.iloc[ib][["lookback", "margin_bps"]].to_dict(), "sharpe": float(g["sharpe"].iloc[ib]),
            "psr_vs_zero": deflated_sharpe(trials[ib], 1, bpy, var_trials=0.0)["psr"],
            "dsr_psr": deflated_sharpe(trials[ib], len(trials), bpy)["psr"],
            "dsr_sr0_ann": deflated_sharpe(trials[ib], len(trials), bpy)["sr0_ann"],
            "cells_positive": int((g["sharpe"] > 0).sum())}

    # --- capacity
    adv_v = {v: adv(d.dvol).reindex(prices.index) for v, d in venues.items() if d.dvol is not None}
    oi_v = {v: d.oi_usd.reindex(prices.index) for v, d in venues.items() if d.oi_usd is not None}
    sigma = daily_vol(prices)
    adv_cols = pd.DataFrame({c: adv_v[c.rsplit("-", 1)[1]][c.rsplit("-", 1)[0]]
                             if c.rsplit("-", 1)[1] in adv_v and c.rsplit("-", 1)[0] in adv_v[c.rsplit("-", 1)[1]].columns
                             else np.nan for c in prices.columns}, index=prices.index)
    cap = venue_leg_caps(prices.columns, adv_v, oi_v, a.participation, a.oi_share)
    C = static_capacity(w_net, cap); U = uncapped_held(w_net, cap)
    cap_rows = []
    for yr in sorted(set(C.dropna().index.year)):
        c = C[C.index.year == yr].dropna(); u = U[U.index.year == yr].dropna().astype(float)
        cap_rows.append({"year": yr, "p10_usd": c.quantile(.1), "median_usd": c.median(), "p90_usd": c.quantile(.9),
                         "share_bars_unsizable_leg": u.mean()})
    pd.DataFrame(cap_rows).to_csv(OUT / "capacity_static.csv", index=False)
    curve = []
    years = (prices.index[-1] - prices.index[0]) / pd.Timedelta(days=365)
    for B in a.book:
        wb = cap_pairs(w_net, cap, B)
        r = run_with_risk(prices, fcols, wb, p0, stop="vol", max_iter=4)
        drag = impact_drag(r.trades, sigma, adv_cols, B, a.y)
        net = r.returns - drag; eq = (1 + net).cumprod()
        dep = wb.abs().sum(axis=1) / w_net.abs().sum(axis=1).where(w_net.abs().sum(axis=1) > 0)
        curve.append({"book_usd": B, "ann_return_before_impact": r.metrics["ann_return"],
                      "impact_drag_ann": float(drag.mean() * r.bars_per_year),
                      "ann_return_after_impact": float(eq.iloc[-1] ** (1 / years) - 1),
                      "pnl_usd_per_year": float(B * (eq.iloc[-1] - 1) / years), "deployed_share": float(dep.mean())})
    pd.DataFrame(curve).to_csv(OUT / "capacity_curve.csv", index=False)

    # --- risk
    liq_ev.to_csv(OUT / "liquidations.csv", index=False)
    adl = adl_exposure(prices, w_net); adl.to_csv(OUT / "adl_exposure.csv", index=False)
    bb = venue_basis(venues["BN"].close, venues["BY"].close, venues["BN"].multiplier, venues["BY"].multiplier)
    held_coins = sorted({c.rsplit("-", 1)[0] for c in w_net.columns[(w_net != 0).any()]})
    bb.loc[bb.index.intersection(held_coins)].to_csv(OUT / "basis_bn_by_held.csv")
    from qr.data.store import read_range
    prem = {}
    for c in held_coins:
        raw = venues["HL"].raw_names.get(c)
        if raw:
            d = read_range("raw", "hyperliquid", "funding", f"{raw}-PERP", a.start, a.end)
            if not d.empty and "premium" in d:
                prem[c] = d["premium"][~d.index.round("min").duplicated()]
    if prem:
        basis_risk(pd.DataFrame(prem)).to_csv(OUT / "basis_hl_premium_held.csv")
    vg = {v: w_net[[c for c in w_net.columns if c.endswith("-" + v)]].abs().sum(axis=1) for v in ("BN", "HL", "BY")}
    vshare = pd.DataFrame(vg); vshare.describe().to_csv(OUT / "venue_gross.csv")

    summary = {"info": info, "backtests": runs, "best_cell": best,
               "liquidation_events": liq_ev.groupby("leverage").size().to_dict() if len(liq_ev) else {},
               "adl_mean_share": float(adl["share"].mean()) if len(adl) else None,
               "share_of_bars_with_a_leg_on": {v: float((g > 0).mean()) for v, g in vg.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(pd.DataFrame(runs).to_string(index=False))
    print(g.to_string(index=False)); print(json.dumps(best, indent=1, default=str))
    print(pd.DataFrame(cap_rows).to_string(index=False)); print(pd.DataFrame(curve).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01"); ap.add_argument("--end", default="2026-10-23")
    ap.add_argument("--min-adv", type=float, default=5e6, help="point-in-time liquidity floor, USD/day")
    ap.add_argument("--participation", type=float, default=0.02)
    ap.add_argument("--oi-share", type=float, default=0.01)
    ap.add_argument("--leverage", type=float, nargs="+", default=[3.0, 2.0])
    ap.add_argument("--out", default=None, help="output directory (default reports/mvenue)")
    ap.add_argument("--book", type=float, nargs="+", default=[1e6, 3e6, 1e7, 2.5e7, 5e7, 1e8])
    ap.add_argument("--y", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="2-cell grid (smoke test)")
    main(ap.parse_args())
