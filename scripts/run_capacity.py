"""Capacity of the funding-carry book on Binance: how much capital it can deploy at a
participation cap, and what square-root impact does to the return as the book grows.

    python scripts/run_capacity.py                      # hedged + unhedged, base params
    python scripts/run_capacity.py --book 1e6 1e7 2.5e7 1e8 --participation 0.01 0.02 0.05

Reads the processed matrices (qr.data.prices / qr.data.funding caches). Writes
reports/carry/capacity_curve.csv, capacity_static.csv, capacity_curve.png, capacity_autogen.md.
The method is in qr/capacity.py; notes/01_carry.md is hand-written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qr.capacity import (adv, adv_by_column, cap_weights, daily_vol, deployed_share, impact_drag,
                         pair_cap, static_capacity, uncapped_held)
from qr.data.funding import has_spot, load_funding
from qr.data.prices import load_kline_matrix
from qr.strategies.carry import CarryParams, build_weights, run_with_risk

OUT = Path("reports/carry")
OUT.mkdir(parents=True, exist_ok=True)


def load(start: str, end: str):
    bases = [l.strip() for l in open("data/universe.txt") if l.strip()]
    perp = load_kline_matrix(bases, "PERP", "close", start, end)
    bases = [b for b in bases if b in perp.columns]
    spot = load_kline_matrix(has_spot(bases), "SPOT", "close", start, end)
    dv_perp = load_kline_matrix(bases, "PERP", "quote_vol", start, end)
    dv_spot = load_kline_matrix(list(spot.columns), "SPOT", "quote_vol", start, end)
    funding = load_funding(bases, start, end)
    idx = perp.index
    prices = pd.concat([perp.add_suffix("-PERP"), spot.reindex(idx).add_suffix("-SPOT")], axis=1).sort_index()
    return prices, funding[[b for b in bases if b in funding.columns]], dv_perp.reindex(idx), dv_spot.reindex(idx)


def sweep(prices, funding, dv_perp, dv_spot, p: CarryParams, books, parts, y: float,
          fee_bps: float = 4.0, slippage_bps: float = 3.0):
    a_perp, a_spot = adv(dv_perp), adv(dv_spot)
    sigma = daily_vol(prices)
    adv_cols = adv_by_column(a_perp, a_spot, prices.columns)
    w = build_weights(prices, funding, p, dv_perp)
    years = (prices.index[-1] - prices.index[0]) / pd.Timedelta(days=365)
    base = run_with_risk(prices, funding, p, fee_bps, slippage_bps, dollar_volume=dv_perp, weights=w) if books else None
    rows, static = [], {}
    for part in parts:
        cap = pair_cap(a_perp, a_spot if p.hedged else None, part, prices.columns)
        static[part] = (static_capacity(w, cap), uncapped_held(w, cap))
        for B in books:
            wb = cap_weights(w, cap, B)
            res = run_with_risk(prices, funding, p, fee_bps, slippage_bps, dollar_volume=dv_perp, weights=wb)
            drag = impact_drag(res.trades, sigma, adv_cols, B, y)
            net = res.returns - drag
            eq = (1 + net).cumprod()
            dep = deployed_share(wb, w)                   # what the cap took away, before the kill switch
            sd = net.std(ddof=1)
            rows.append({
                "hedged": p.hedged, "participation": part, "book_usd": B,
                "ann_return_fees_only": res.metrics["ann_return"],
                "impact_drag_ann": float(drag.mean() * res.bars_per_year),
                "ann_return_after_impact": float(eq.iloc[-1] ** (1 / years) - 1),
                "sharpe_after_impact": float(net.mean() / sd * np.sqrt(res.bars_per_year)) if sd > 0 else np.nan,
                "max_dd_after_impact": float((eq / eq.cummax() - 1).min()),
                "deployed_share_mean": float(dep.mean()),
                "deployed_share_p10": float(dep.quantile(0.10)),
                "pnl_usd_per_year": float(B * (eq.iloc[-1] - 1) / years),
                "avg_names": float((res.weights.filter(like="-PERP").abs() > 0).sum(axis=1)[dep.notna()].mean()),
                "kill_events": res.metrics.get("kill_events", 0),
            })
    return base, pd.DataFrame(rows), static


def static_table(static: dict[float, tuple[pd.Series, pd.Series]]) -> pd.DataFrame:
    """Distribution of static capacity over active bars, by year. Names without a cap (listings
    younger than adv's minimum history) are left out of the min; share_bars_uncapped_name says how
    often the book held one, i.e. how often the figure ignores a name that could not be sized."""
    rows = []
    for part, (C, U) in static.items():
        u = U.dropna().astype(float)
        c = C.reindex(u.index)
        for yr in list(sorted(set(u.index.year))) + ["all"]:
            s = c if yr == "all" else c[c.index.year == yr]
            uu = u if yr == "all" else u[u.index.year == yr]
            s = s.dropna()
            rows.append({"participation": part, "year": yr, "active_bars": int(len(uu)),
                         "p10_usd": s.quantile(0.10), "median_usd": s.median(), "p90_usd": s.quantile(0.90),
                         "share_bars_uncapped_name": float(uu.mean())})
    return pd.DataFrame(rows)


def plot(curve: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (h, part), d in curve.groupby(["hedged", "participation"]):
        lab = f"{'hedged' if h else 'unhedged'}, {part:.0%} of ADV"
        ls = "-" if h else "--"
        axes[0].plot(d["book_usd"], d["pnl_usd_per_year"] / 1e6, ls, marker="o", ms=3, label=lab)
        axes[1].plot(d["book_usd"], d["deployed_share_mean"], ls, marker="o", ms=3, label=lab)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("book size, USD"); ax.grid(alpha=0.3)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("net P&L per year, USD m (fees, funding, impact)")
    axes[1].set_ylabel("share of wanted gross actually held")
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=120)


def autogen(curve: pd.DataFrame, static: pd.DataFrame, base_metrics: dict, y: float) -> str:
    L = ["# Capacity (autogenerated by scripts/run_capacity.py)", "",
         f"Impact: square-root law, Y = {y}. Caps: participation x trailing-30d min(perp ADV, spot ADV) "
         "for the hedged pair, perp ADV for the unhedged book. Undeployed capital earns zero.", ""]
    L.append("## Static capacity of the target book, min_i cap_i/|w_i| over active bars")
    L.append(static.to_string(index=False, formatters={"participation": "{:.0%}".format,
             "share_bars_uncapped_name": "{:.1%}".format, **{c: "{:,.0f}".format for c in ("p10_usd", "median_usd", "p90_usd")}}))
    L.append("")
    L.append("## Capacity curve")
    cols = ["hedged", "participation", "book_usd", "ann_return_fees_only", "impact_drag_ann",
            "ann_return_after_impact", "sharpe_after_impact", "deployed_share_mean", "pnl_usd_per_year", "avg_names"]
    L.append(curve[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return "\n".join(L) + "\n"


def main(a):
    prices, funding, dv_perp, dv_spot = load(a.start, a.end)
    print(f"{len(funding.columns)} perps, {dv_spot.shape[1]} with spot, {len(prices)} bars")
    curves, statics, base = [], [], {}
    for hedged in ([True, False] if a.hedged == "both" else [a.hedged == "true"]):
        p = CarryParams(universe_n=a.universe_n, hedged=hedged)
        b, curve, st = sweep(prices, funding, dv_perp, dv_spot, p, [] if a.static_only else a.book, a.participation, a.y)
        t = static_table(st); t.insert(0, "hedged", hedged); statics.append(t)
        if a.static_only:
            continue
        base[hedged] = b.metrics
        curves.append(curve)
        print(curve.to_string(index=False))
    static = pd.concat(statics, ignore_index=True)
    if a.static_only:                                   # the curve and its files are left as they were
        static.to_csv(OUT / "capacity_static.csv", index=False)
        print(static.to_string(index=False)); return
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(OUT / "capacity_curve.csv", index=False)
    static.to_csv(OUT / "capacity_static.csv", index=False)
    plot(curve, OUT / "capacity_curve.png")
    (OUT / "capacity_autogen.md").write_text(autogen(curve, static, base, a.y))
    (OUT / "capacity_base_metrics.json").write_text(json.dumps({str(k): v for k, v in base.items()}, indent=2, default=float))
    print(static.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default="2026-10-23")
    ap.add_argument("--universe-n", type=int, default=40)
    ap.add_argument("--hedged", default="both", choices=["both", "true", "false"])
    ap.add_argument("--book", type=float, nargs="+", default=[1e6, 3e6, 1e7, 2.5e7, 5e7, 1e8])
    ap.add_argument("--participation", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    ap.add_argument("--y", type=float, default=1.0, help="impact coefficient")
    ap.add_argument("--static-only", action="store_true", help="recompute capacity_static.csv only (no backtests)")
    main(ap.parse_args())
