"""Open-interest check on the carry book's current holdings: for every name the target book held in
the last `--days` days, the book size at which that position is 1% of the contract's open interest
against the book size at which it is 2% of volume. Hedged and unhedged.

Binance serves no open-interest history, so the default is today's snapshot (scripts/fetch_oi_snapshot.py)
against recent target weights (before the kill switch). `--history` uses the vendor's OI history
(qr.data.chd, from 2025-06-28, the ~110 coins the studies held most) over every 8h window instead.

    python scripts/oi_check.py                     # last 7 days of holdings, latest snapshot
    python scripts/oi_check.py --history           # every 8h window since 2025-07-01

Writes reports/carry/oi_check_{hedged,unhedged}.csv, or with --history oi_history_{hedged,unhedged}.csv
(one row per window) and oi_history_summary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_capacity import load  # noqa: E402

from qr.strategies.carry import CarryParams, build_weights  # noqa: E402

OUT = Path("reports/carry")


def check(w: pd.DataFrame, oi: pd.DataFrame, days: int) -> pd.DataFrame:
    """w: target weights; returns one row per perp held in the last `days` days."""
    w = w.filter(like="-PERP").abs()
    w = w.loc[w.index[-1] - pd.Timedelta(days=days):]
    held = w.columns[(w > 0).any()]
    df = pd.DataFrame({"share_held": (w[held] > 0).mean(), "w_when_held": w[held].where(w[held] > 0).mean()})
    df.index = [c[:-5] for c in df.index]
    df = df.join(oi[["oi_usd", "adv_perp_usd_30d", "adv_spot_usd_30d"]])
    df["vol_over_oi"] = df.adv_perp_usd_30d / df.oi_usd
    df["book_at_1pct_oi"] = 0.01 * df.oi_usd / df.w_when_held
    df["book_at_2pct_adv"] = 0.02 * df[["adv_perp_usd_30d", "adv_spot_usd_30d"]].min(axis=1) / df.w_when_held
    df["oi_binds"] = df.book_at_1pct_oi < df.book_at_2pct_adv
    return df.sort_values("book_at_1pct_oi")


def history(w: pd.DataFrame, oi: pd.DataFrame, a_perp: pd.DataFrame, a_spot: pd.DataFrame | None,
            since: str) -> pd.DataFrame:
    """One row per 8h window (bars at 00/08/16 UTC) from `since`: the book size at which the first
    held name reaches 2% of volume, the same with each name also capped at 1% of its perp OI, which
    name binds, and how much of the held exposure has a known OI at all."""
    w = w.filter(like="-PERP").abs().loc[since:]
    w = w[w.index.hour % 8 == 0]
    w.columns = [c[:-5] for c in w.columns]
    vol = a_perp.reindex(index=w.index, columns=w.columns)
    if a_spot is not None:
        vol = np.fmin(vol, a_spot.reindex(index=w.index, columns=w.columns))
    cap_v = 0.02 * vol
    o = 0.01 * oi.reindex(index=w.index, columns=w.columns)
    cap_o = np.fmin(cap_v, o).where(cap_v.notna())
    held = w > 0
    rows = []
    for t in w.index[held.any(axis=1)]:
        h = held.loc[t]; n = w.loc[t, h]
        bv, bo = (cap_v.loc[t, h] / n), (cap_o.loc[t, h] / n)
        rows.append({"ts": t, "names": int(h.sum()), "book_volume_only": bv.min(), "book_with_oi": bo.min(),
                     "binding_name": bo.idxmin() if bo.notna().any() else None,
                     "oi_binds": bool(bo.min() < bv.min()),
                     "exposure_share_with_oi": float(n[o.loc[t, h].notna()].sum() / n.sum())})
    return pd.DataFrame(rows).set_index("ts")


def main_history(a):
    from qr.capacity import adv
    from qr.data.venues import _matrix, _names_on_disk
    prices, funding, dv_perp, dv_spot = load(a.start, a.end)
    a_perp, a_spot = adv(dv_perp), adv(dv_spot)
    have = set(_names_on_disk("chd_binance_futures", "open_interest"))
    oi = _matrix("chd_binance_futures", "open_interest", {b: b for b in funding.columns if b in have}, "oi_usd", a.start, a.end)
    oi = oi.reindex(prices.index, method="ffill", limit=8)
    summ = []
    for hedged in (True, False):
        w = build_weights(prices, funding, CarryParams(universe_n=a.universe_n, hedged=hedged), dv_perp)
        df = history(w, oi, a_perp, a_spot if hedged else None, a.since)
        name = "hedged" if hedged else "unhedged"
        df.to_csv(OUT / f"oi_history_{name}.csv")
        full = df[df.exposure_share_with_oi >= 0.999]
        for label, d in (("all windows", df), ("windows with OI for every held name", full)):
            summ.append({"book": name, "windows": label, "n": len(d),
                         "median_book_volume_only": d.book_volume_only.median(), "median_book_with_oi": d.book_with_oi.median(),
                         "share_oi_binds": d.oi_binds.mean(), "median_exposure_share_with_oi": d.exposure_share_with_oi.median()})
        print(f"\n== {name}: {len(df)} windows since {a.since}; top binding names:")
        print(df.binding_name.value_counts().head(8).to_string())
    s = pd.DataFrame(summ); s.to_csv(OUT / "oi_history_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False, float_format=lambda v: f"{v:,.3g}"))


def main(a):
    if a.history:
        return main_history(a)
    OUT.mkdir(parents=True, exist_ok=True)
    snap = sorted(Path("data/processed/binance").glob("oi_snapshot_*.csv"))[-1]
    oi = pd.read_csv(snap, index_col=0)
    prices, funding, dv_perp, _ = load(a.start, a.end)
    pd.set_option("display.width", 200)
    for hedged in (True, False):
        w = build_weights(prices, funding, CarryParams(universe_n=a.universe_n, hedged=hedged), dv_perp)
        df = check(w, oi, a.days)
        name = "hedged" if hedged else "unhedged"
        df.to_csv(OUT / f"oi_check_{name}.csv")
        print(f"\n== {name}: {len(df)} names held in the {a.days} days to {w.index[-1]:%Y-%m-%d}, "
              f"against {snap.name}; OI binds for {int(df.oi_binds.sum())} of {len(df)}")
        print(df.to_string(float_format=lambda v: f"{v:,.3g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default="2026-10-23")
    ap.add_argument("--universe-n", type=int, default=40)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--history", action="store_true", help="vendor OI history over every 8h window")
    ap.add_argument("--since", default="2025-07-01")
    main(ap.parse_args())
