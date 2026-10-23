"""Is below-default funding concentrated in the most liquid coins, and more so in 2025-26?

Panel: one row per (hedgeable coin, 8h window) where the coin is listed and has volume history.
Volume bucket is point in time: rank by trailing 30-day perp dollar volume (completed bars only)
among the coins listed in that window, 1 = most liquid.

Columns
    ts            window end (UTC)
    coin          base symbol
    period        "2023-24" or "2025-26"
    f8            funding paid in the window (fraction per 8h; default is 0.0001)
    excess_bps    (f8 - default) in bps per 8h
    vol_rank      1 = highest trailing 30d volume in that window
    bucket        "1-20", "21-50", "51-100", "101+"
    below, at, above   bool: f8 below / at / above the default

    python scripts/funding_by_volume.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qr.data.funding import has_spot, load_funding, per_window
from qr.data.prices import load_kline_matrix

START, END = "2023-01-01", "2026-10-23"
DEFAULT, EPS = 0.0001, 1e-10
BUCKETS = [0, 20, 50, 100, np.inf]
LABELS = ["1-20", "21-50", "51-100", "101+"]


def load_panel() -> pd.DataFrame:
    u = [l.strip() for l in open("data/universe.txt") if l.strip()]
    H = has_spot(u)
    f8 = per_window(load_funding(H, START, END)[H])
    dvol = load_kline_matrix(H, "PERP", "quote_vol", START, END).reindex(columns=f8.columns)
    tv = dvol.fillna(0.0).rolling(24 * 30, min_periods=24 * 7).sum().shift(1)   # completed bars only
    tv = tv.reindex(f8.index)                                                     # value at each window end
    rank = tv.where(f8.notna() & (tv > 0)).rank(axis=1, ascending=False, method="first")

    panel = pd.DataFrame({"f8": f8.stack(), "vol_rank": rank.stack()}).dropna().reset_index()
    panel.columns = ["ts", "coin", "f8", "vol_rank"]
    panel["period"] = np.where(panel["ts"] < pd.Timestamp("2025-01-01", tz="UTC"), "2023-24", "2025-26")
    panel["bucket"] = pd.cut(panel["vol_rank"], BUCKETS, labels=LABELS)
    panel["excess_bps"] = (panel["f8"] - DEFAULT) * 1e4
    panel["below"] = panel["f8"] < DEFAULT - EPS
    panel["at"] = (panel["f8"] - DEFAULT).abs() <= EPS
    panel["above"] = panel["f8"] > DEFAULT + EPS
    return panel


def main(min_windows: int = 1) -> None:
    panel = load_panel()
    print(f"{len(panel):,} coin-windows, {panel['coin'].nunique()} coins, "
          f"{panel['ts'].min():%Y-%m-%d} -> {panel['ts'].max():%Y-%m-%d}")
    print("\nRows per period and bucket (sanity check, not the answer):")
    print(panel.groupby(["period", "bucket"], observed=False).size().unstack().to_string())

    # ---- comparison: share of windows below default, per coin, by bucket and period ----------
    # A coin moves between buckets over time; its share in a (period, bucket) cell uses only the
    # windows it spent there. Cell value = mean over coins; CI = bootstrap over coins.
    per_coin = (panel.groupby(["period", "bucket", "coin"], observed=True)["below"]
                .agg(share="mean", windows="size").reset_index())
    per_coin = per_coin[per_coin["windows"] >= min_windows]       # drop coins that only passed through
    rng = np.random.default_rng(0)
    rows = []
    for (per, b), d in per_coin.groupby(["period", "bucket"], observed=True):
        x = d["share"].values
        boot = rng.choice(x, size=(2000, len(x)), replace=True).mean(axis=1)
        rows.append({"period": per, "bucket": b, "coins": len(x), "mean_share_below": x.mean(),
                     "median_share_below": np.median(x),
                     "ci95_lo": np.percentile(boot, 2.5), "ci95_hi": np.percentile(boot, 97.5)})
    print(f"\nShare of windows below default, averaged per coin, coins with >= {min_windows} windows "
          f"in the cell (bootstrap 95% CI over coins):")
    print(pd.DataFrame(rows).set_index(["period", "bucket"]).round(3).to_string())

    # ---- did the top bucket's odds of being below default rise more than the tail's? ----------
    # ratio = [odds(1-20, 25-26) / odds(1-20, 23-24)] / [odds(101+, 25-26) / odds(101+, 23-24)],
    # odds of the per-coin mean share; bootstrap resamples coins independently within each cell.
    odds = lambda p: p / (1 - p)
    cell = {(per, b): d["share"].values for (per, b), d in per_coin.groupby(["period", "bucket"], observed=True)}
    keys = [("2023-24", "1-20"), ("2025-26", "1-20"), ("2023-24", "101+"), ("2025-26", "101+")]
    point = {k: cell[k].mean() for k in keys}
    ratio = lambda m: (odds(m[keys[1]]) / odds(m[keys[0]])) / (odds(m[keys[3]]) / odds(m[keys[2]]))
    n_boot = 10_000
    draws = {k: rng.choice(cell[k], size=(n_boot, len(cell[k])), replace=True).mean(axis=1) for k in keys}
    boot = ratio(draws)
    print(f"\nOdds change 2023-24 -> 2025-26: top 1-20 x{odds(point[keys[1]]) / odds(point[keys[0]]):.2f}, "
          f"tail 101+ x{odds(point[keys[3]]) / odds(point[keys[2]]):.2f}")
    print(f"Ratio of odds changes (top / tail): {ratio(point):.2f}, bootstrap 95% CI "
          f"{np.percentile(boot, 2.5):.2f} to {np.percentile(boot, 97.5):.2f}, "
          f"share of draws above 1: {(boot > 1).mean():.3f}  ({n_boot:,} draws over coins)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-windows", type=int, default=1, help="min windows a coin spends in a (period, bucket) cell")
    main(ap.parse_args().min_windows)
