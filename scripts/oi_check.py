"""Open-interest check on the carry book's current holdings: for every name the target book held in
the last `--days` days, the book size at which that position is 1% of the contract's open interest
against the book size at which it is 2% of volume. Hedged and unhedged.

Binance serves no open-interest history, so this is today's snapshot (scripts/fetch_oi_snapshot.py)
against recent target weights (before the kill switch); it cannot be run over the backtest.

    python scripts/oi_check.py                     # last 7 days of holdings, latest snapshot

Writes reports/carry/oi_check_hedged.csv and oi_check_unhedged.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main(a):
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
    main(ap.parse_args())
