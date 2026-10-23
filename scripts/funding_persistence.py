"""How long does a coin's funding stay strictly above Binance's default, and does a spell pay
for the round trip of trading it?

Universe: coins with a Binance spot leg (hedgeable). Clock: fixed 8h windows (qr.data.funding).
Spell: consecutive windows with funding > DEFAULT per 8h; a NaN window (unlisted) ends it.
Trade per spell: enter at the end of the first window above default (that is when it is known),
earn funding from the next window on, exit at the end of the first window back at or below
default. Realised funding is compared with ROUND_TRIP (4 legs x 7 bps).

    python scripts/funding_persistence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qr.data.funding import WINDOWS_PER_YEAR, has_spot, load_funding, per_window

DEFAULT = 0.0001          # 0.01% per 8h
EPS = 1e-9
ROUND_TRIP = 4 * 7 / 1e4  # enter + exit, spot and perp, 4 bps fee + 3 bps slippage
HORIZONS = [1, 2, 3, 6, 9, 21, 42, 90, 153]   # windows; 153 = 51 days, the swap break-even
ENTRY_BUCKETS = [0.1095, 0.2, 0.5, 1.0, np.inf]  # APR of the first window above default


def spells(f8: pd.DataFrame) -> pd.DataFrame:
    """One row per spell: coin, start, length (windows), entry APR, realised funding of the trade."""
    rows = []
    above = (f8 > DEFAULT + EPS)
    for c in f8.columns:
        x = f8[c].values; a = above[c].values; n = len(x)
        i = 0
        while i < n:
            if not a[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and a[j + 1]:
                j += 1
            # spell = windows i..j. Trade: enter at end of i, earn windows i+1 .. exit window e,
            # where e = j+1 (first window back at/below default), if it exists and is listed.
            e = j + 1
            earned = np.nansum(x[i + 1:e + 1]) if e < n else np.nansum(x[i + 1:])
            rows.append({"coin": c, "start": f8.index[i], "length": j - i + 1,
                         "entry_apr": x[i] * WINDOWS_PER_YEAR, "earned": earned,
                         "censored": e >= n or np.isnan(x[e]) if e < n else True})
            i = j + 1
    return pd.DataFrame(rows)


def main() -> None:
    u = [l.strip() for l in open("data/universe.txt") if l.strip()]
    raw = load_funding(u, "2023-01-01", "2026-10-23")
    f8 = per_window(raw[has_spot(list(raw.columns))])
    s = spells(f8)
    print(f"{f8.shape[1]} hedgeable coins, {len(f8)} windows, {len(s)} spells above default "
          f"({s['censored'].sum()} cut off by delisting or the sample end)\n")

    L = s["length"]
    print("Spell length (windows of 8h):")
    print(f"  median {L.median():.0f}  mean {L.mean():.1f}  p75 {L.quantile(.75):.0f}  p90 {L.quantile(.9):.0f}  "
          f"p99 {L.quantile(.99):.0f}  max {L.max()}")
    surv = pd.Series({k: (L >= k).mean() for k in HORIZONS}, name="P(length >= k)")
    surv.index = [f"{k}w ({k / 3:.0f}d)" if k >= 3 else f"{k}w" for k in HORIZONS]
    print("\nSurvival, share of spells still above default after k windows:")
    print(surv.round(3).to_string())

    print(f"\nTrade per spell (enter when first above default, exit when back at default), "
          f"vs {ROUND_TRIP * 1e4:.0f} bps round trip:")
    print(f"  share of spells whose funding covers the round trip: {(s['earned'] > ROUND_TRIP).mean():.3f}")
    print(f"  mean / median funding earned (bps): {s['earned'].mean() * 1e4:.1f} / {s['earned'].median() * 1e4:.1f}")
    print(f"  mean net of round trip (bps): {(s['earned'] - ROUND_TRIP).mean() * 1e4:.1f}")

    s["bucket"] = pd.cut(s["entry_apr"], ENTRY_BUCKETS, right=False,
                         labels=["11-20%", "20-50%", "50-100%", ">100%"])
    g = s.groupby("bucket", observed=False).agg(
        spells=("length", "size"), median_len=("length", "median"), mean_len=("length", "mean"),
        covers_cost=("earned", lambda e: (e > ROUND_TRIP).mean()),
        mean_earned_bps=("earned", lambda e: e.mean() * 1e4),
        mean_net_bps=("earned", lambda e: (e - ROUND_TRIP).mean() * 1e4))
    print("\nBy APR of the first window above default:")
    print(g.round(2).to_string())

    # Same buckets, split by period, with a bootstrap 95% CI on the mean net (spell earnings are skewed)
    s["period"] = np.where(s["start"] < pd.Timestamp("2025-01-01", tz="UTC"), "2023-24", "2025-26")
    rng = np.random.default_rng(0)

    def ci(net: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
        if len(net) < 2:
            return (np.nan, np.nan)
        means = rng.choice(net, size=(n_boot, len(net)), replace=True).mean(axis=1)
        return tuple(np.percentile(means, [2.5, 97.5]))

    rows = []
    for (per, b), d in s.groupby(["period", "bucket"], observed=False):
        net = (d["earned"].values - ROUND_TRIP) * 1e4
        lo, hi = ci(net)
        rows.append({"period": per, "bucket": b, "spells": len(d),
                     "covers_cost": (d["earned"] > ROUND_TRIP).mean() if len(d) else np.nan,
                     "median_earned_bps": d["earned"].median() * 1e4 if len(d) else np.nan,
                     "mean_net_bps": net.mean() if len(d) else np.nan, "ci95_lo": lo, "ci95_hi": hi})
    print("\nBy period and entry APR, mean net of round trip with bootstrap 95% CI (bps):")
    print(pd.DataFrame(rows).set_index(["period", "bucket"]).round(2).to_string())


if __name__ == "__main__":
    main()
