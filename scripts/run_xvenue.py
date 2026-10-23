"""Cross-venue funding carry, Binance vs Hyperliquid: spread statistics, backtest, basis risk,
capacity snapshot. Writes reports/xvenue/*. notes/02_xvenue.md is hand-written.

    python -m qr.data.hl_history funding --start 2024-01-01      # once, ~1.5h
    python -m qr.data.hl_history snapshot                          # today's OI / volume
    python scripts/fetch_oi_snapshot.py                            # Binance OI today (if not done)
    python scripts/run_xvenue.py --start 2024-01-01

HL coin names map to Binance bases: 'kPEPE' -> '1000PEPE' etc. (HL's k-prefix is the same
1000x contract). Anything else matches by name. Coins on one venue only are dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qr.backtest.report import tearsheet
from qr.data.funding import WINDOWS_PER_YEAR, load_funding, per_window
from qr.data.prices import load_kline_matrix
from qr.data.store import ROOT, read_range
from qr.strategies.xvenue import XVenueParams, basis_risk, build_weights, run_with_risk, spread_apr

OUT = Path("reports/xvenue")
OUT.mkdir(parents=True, exist_ok=True)


def hl_to_bn(name: str) -> str:
    return "1000" + name[1:] if name.startswith("k") and name[1:].isupper() else name


def load_hl_funding(start: str, end: str, cache: bool = True):
    """Rate and premium matrices, columns = BINANCE base names, index = HL hourly settlement times."""
    path = ROOT / "processed" / "hyperliquid" / f"funding_matrix_{start}_{end}.parquet"
    ppath = path.with_name(path.name.replace("funding", "premium"))
    if cache and path.exists() and ppath.exists():
        return pd.read_parquet(path), pd.read_parquet(ppath)
    d = ROOT / "raw" / "hyperliquid" / "funding"
    rates, prems = {}, {}
    for sym_dir in sorted(d.glob("*-PERP")):
        coin = sym_dir.name[:-5]
        f = read_range("raw", "hyperliquid", "funding", sym_dir.name, start, end)
        if f.empty:
            continue
        f.index = f.index.round("min")
        f = f[~f.index.duplicated(keep="last")]
        rates[hl_to_bn(coin)] = f["rate"]; prems[hl_to_bn(coin)] = f["premium"]
    rate = pd.DataFrame(rates).sort_index(); prem = pd.DataFrame(prems).sort_index()
    rate.index.name = prem.index.name = "ts"
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True); rate.to_parquet(path); prem.to_parquet(ppath)
    return rate, prem


def spread_stats(f_bn: pd.DataFrame, f_hl: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    coins = sorted(set(f_bn.columns) & set(f_hl.columns))
    s8 = (per_window(f_hl[coins]) - per_window(f_bn[coins])) * 1e4        # bps per 8h window
    long_ = s8.stack().rename("spread_bps").reset_index(); long_.columns = ["ts", "coin", "spread_bps"]
    long_["year"] = long_["ts"].dt.year
    rows = []
    for yr, d in long_.groupby("year"):
        x = d["spread_bps"]
        rows.append({"year": yr, "coin_windows": len(d), "coins": d["coin"].nunique(),
                     "mean_bps": x.mean(), "median_bps": x.median(), "p10": x.quantile(0.1), "p90": x.quantile(0.9),
                     "share_hl_higher": (x > 0).mean(), "share_abs_gt_1bp": (x.abs() > 1).mean(),
                     "share_abs_gt_2bp": (x.abs() > 2).mean(), "share_abs_gt_5bp": (x.abs() > 5).mean()})
    by_year = pd.DataFrame(rows)
    # persistence: spells of |trailing spread APR| above entry, in windows
    apr = spread_apr(f_bn, f_hl, lookback)
    on = apr.abs() > 0.10
    spells = []
    for c in on.columns:
        v = on[c].fillna(False).values
        run = 0
        for b in v:
            if b: run += 1
            elif run: spells.append(run); run = 0
        if run: spells.append(run)
    sp = pd.Series(spells)
    persist = pd.DataFrame([{"spells": len(sp), "median_windows": sp.median(), "p75_windows": sp.quantile(0.75),
                             "share_ge_6_windows_2d": (sp >= 6).mean(), "share_ge_21_windows_1w": (sp >= 21).mean()}])
    return by_year, persist


def capacity_snapshot(weights: pd.DataFrame, bn_snap: pd.DataFrame | None, hl_snap: pd.DataFrame | None,
                      days: int = 7) -> pd.DataFrame:
    """For names held in the last `days`: book size at which the position is 1% of OI and 2% of 24h
    volume on each venue; binding = the smallest. HL snapshot columns: oi_usd, day_ntl_vlm (index HL name);
    Binance: oi_usd, adv_perp_usd_30d (index base)."""
    recent = weights.iloc[-days * 24:]
    hl_cols = [c for c in recent.columns if c.endswith("-HL")]
    held = (recent[hl_cols].abs() > 0).mean()
    rows = []
    for col, share in held[held > 0].items():
        base = col[:-3]; w = recent[col].abs()[recent[col].abs() > 0].mean()
        r = {"name": base, "held_share_7d": share, "leg_weight": w}
        if hl_snap is not None:
            hl_name = next((n for n in hl_snap.index if hl_to_bn(n) == base), None)
            if hl_name is not None:
                r["hl_oi_usd"] = hl_snap.loc[hl_name, "oi_usd"]; r["hl_vol24_usd"] = hl_snap.loc[hl_name, "day_ntl_vlm"]
                r["book_at_1pct_hl_oi"] = 0.01 * r["hl_oi_usd"] / w; r["book_at_2pct_hl_vol"] = 0.02 * r["hl_vol24_usd"] / w
        if bn_snap is not None and base in bn_snap.index:
            r["bn_oi_usd"] = bn_snap.loc[base, "oi_usd"]; r["book_at_1pct_bn_oi"] = 0.01 * r["bn_oi_usd"] / w
            if "adv_perp_usd_30d" in bn_snap:
                r["book_at_2pct_bn_adv"] = 0.02 * bn_snap.loc[base, "adv_perp_usd_30d"] / w
        caps = [v for k, v in r.items() if k.startswith("book_at_")]
        r["binding_book_usd"] = min(caps) if caps else np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("binding_book_usd")


def main(a):
    bases = [l.strip() for l in open("data/universe.txt") if l.strip()]
    px = load_kline_matrix(bases, "PERP", "close", a.start, a.end)
    dvol = load_kline_matrix(bases, "PERP", "quote_vol", a.start, a.end)
    f_bn = load_funding([b for b in bases if b in px.columns], a.start, a.end)
    f_hl, prem = load_hl_funding(a.start, a.end)
    coins = sorted(set(f_bn.columns) & set(f_hl.columns) & set(px.columns))
    print(f"{len(coins)} coins on both venues, {px.index[0]} -> {px.index[-1]}")
    f_bn, f_hl, prem = f_bn[coins], f_hl[coins], prem[coins]

    by_year, persist = spread_stats(f_bn, f_hl, a.lookback)
    by_year.to_csv(OUT / "spread_by_year.csv", index=False); persist.to_csv(OUT / "spread_persistence.csv", index=False)
    print(by_year.round(3).to_string(index=False)); print(persist.round(3).to_string(index=False))

    base = XVenueParams(lookback=a.lookback)
    res = run_with_risk(px, f_bn, f_hl, base, dollar_volume=dvol)
    tearsheet(res, "Cross-venue funding, Binance vs Hyperliquid, base params", out=OUT / "tearsheet.png")
    m = dict(res.metrics)
    m["avg_pairs"] = float((res.weights.filter(like="-HL").abs() > 0).sum(axis=1).mean())
    m["coins_on_both_venues"] = len(coins)
    print(pd.Series(m).round(4).to_string())

    grid = []
    for lb in (3, 9, 21):
        for entry in (0.05, 0.10, 0.20, 0.40):
            r = run_with_risk(px, f_bn, f_hl, XVenueParams(lookback=lb, entry_apr=entry, exit_apr=entry * 0.3), dollar_volume=dvol)
            grid.append({"lookback": lb, "entry_apr": entry, "sharpe": r.metrics["sharpe"], "ann_return": r.metrics["ann_return"],
                         "ann_turnover": r.metrics["ann_turnover"], "funding_ann": r.metrics["funding_ann"],
                         "cost_drag_ann": r.metrics["cost_drag_ann"], "max_dd": r.metrics["max_drawdown"]})
    grid = pd.DataFrame(grid); grid.to_csv(OUT / "grid.csv", index=False); print(grid.round(3).to_string(index=False))
    m["grid_sharpe_min"], m["grid_sharpe_max"], m["grid_n_trials"] = float(grid.sharpe.min()), float(grid.sharpe.max()), len(grid)

    br = basis_risk(prem); br.to_csv(OUT / "basis_risk.csv")
    held = [c[:-3] for c in res.weights.columns if c.endswith("-HL") and (res.weights[c].abs() > 0).any()]
    print("basis risk, held coins:"); print(br.loc[[c for c in held if c in br.index]].round(1).describe().to_string())

    bn_files = sorted(glob.glob("data/processed/binance/oi_snapshot_*.csv")); hl_files = sorted(glob.glob("data/processed/hyperliquid/snapshot_*.csv"))
    bn_snap = pd.read_csv(bn_files[-1], index_col=0) if bn_files else None
    hl_snap = pd.read_csv(hl_files[-1], index_col=0) if hl_files else None
    cap = capacity_snapshot(build_weights(px, f_bn, f_hl, base, dvol), bn_snap, hl_snap)
    cap.to_csv(OUT / "capacity_snapshot.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}"); print(cap.to_string(index=False))

    (OUT / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    md = ["# Cross-venue funding (autogenerated by scripts/run_xvenue.py)", "",
          f"{len(coins)} coins on both venues, {a.start} to {a.end}. Both legs priced off the Binance perp; "
          "basis risk reported separately from HL's premium series.", "",
          "## Spread by year (bps per 8h window, HL minus Binance)", by_year.round(3).to_string(index=False), "",
          "## Persistence of |spread APR| > 10%", persist.round(3).to_string(index=False), "",
          "## Base params", pd.Series(m).round(4).to_string(), "",
          "## Grid", grid.round(3).to_string(index=False), "",
          "## Capacity snapshot, names held in the last 7 days", cap.to_string(index=False), ""]
    (OUT / "note_autogen.md").write_text("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01"); ap.add_argument("--end", default="2026-10-23")
    ap.add_argument("--lookback", type=int, default=9)
    main(ap.parse_args())
