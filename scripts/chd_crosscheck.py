"""Cross-check our exchange pulls (Binance, Bybit, Hyperliquid) against an independent vendor
(CryptoHFTData) over a window: funding rates, hourly prices, and (where we have it) open interest.

    export CHD_API_KEY=...
    python scripts/chd_crosscheck.py --start 2026-10-01 --end 2026-10-14 \
        --bases BTC ETH SOL XRP DOGE SUI PEPE ZEC LSK MARSCOIN

What is compared, per venue and coin
  funding   our settled rate at time T  vs  the vendor's last pre-settlement funding_rate in the
            hour ending at T (the predicted rate converges to the settled one by T). Diff in bps.
  price     our hourly close for the bar opening at t (= price at t+1h)  vs  the vendor's last mark
            price in the hour [t, t+1h). Mark is not last trade, so a few bps of gap is normal.
  oi        Bybit only (we hold daily OI): our oi_usd at 00:00  vs  vendor's last oi_usd in the
            23:00 hour of the previous day.

Writes reports/data_quality/chd_crosscheck_<venue>.csv (per coin) and a summary to stdout.
A coin that disagrees on funding by more than 1 bp on more than 1% of settlements, or whose
price gap has a p99 above 50 bps, is flagged; those are the ones to look at by hand.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from qr.data.chd import chd_symbol, fetch_file, hourly_summary
from qr.data.venues import load_binance, load_bybit, load_hyperliquid

OUT = Path("reports/data_quality")
OUT.mkdir(parents=True, exist_ok=True)
EXCH = {"BN": "binance_futures", "BY": "bybit", "HL": "hyperliquid_futures"}
LOAD = {"BN": load_binance, "BY": load_bybit, "HL": load_hyperliquid}
FULL = ("2023-01-01", "2026-10-23")


def vendor_hours(exchange: str, symbol: str, dtype: str, start: str, end: str, workers: int = 8) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D", tz="UTC")
    jobs = [(d, h) for d in days for h in range(24)]

    def one(j):
        d, h = j
        return hourly_summary(fetch_file(exchange, d.strftime("%Y-%m-%d"), h, symbol, dtype), dtype, d + pd.Timedelta(hours=h))

    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(one, jobs) if r is not None]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ts").sort_index()          # index = hour END


def compare(venue: str, base: str, v, start: str, end: str) -> dict:
    raw = v.raw_names.get(base, base)
    sym = chd_symbol(EXCH[venue], raw)
    mark = vendor_hours(EXCH[venue], sym, "mark_price", start, end)
    out = {"venue": venue, "base": base, "vendor_symbol": sym, "vendor_hours": int(len(mark))}
    if mark.empty or base not in v.funding.columns:
        out["note"] = "no vendor data" if mark.empty else "not in our funding matrix"
        return out
    # funding: our settlement at T vs vendor's last funding_rate in the hour ending at T
    ours = v.funding[base].loc[start:end].dropna()
    vend = mark["funding_rate"].reindex(ours.index)
    d = ((ours - vend) * 1e4).dropna()
    out.update({"funding_n": int(len(d)), "funding_mean_abs_bps": float(d.abs().mean()) if len(d) else np.nan,
                "funding_max_abs_bps": float(d.abs().max()) if len(d) else np.nan,
                "funding_share_gt_1bp": float((d.abs() > 1).mean()) if len(d) else np.nan,
                "funding_missing_vendor": int(vend.isna().sum())})
    # price: our close at bar-open t (price at t+1h) vs vendor's last mark in [t, t+1h) (indexed t+1h)
    if v.close is not None and base in v.close.columns:
        c = v.close[base].loc[start:end].dropna()           # contract units, as the vendor's symbol
        c.index = c.index + pd.Timedelta(hours=1)
        vm = mark["mark"].reindex(c.index)
        g = ((c / vm - 1) * 1e4).dropna()
        out.update({"price_n": int(len(g)), "price_median_abs_bps": float(g.abs().median()) if len(g) else np.nan,
                    "price_p99_abs_bps": float(g.abs().quantile(0.99)) if len(g) else np.nan,
                    "price_max_abs_bps": float(g.abs().max()) if len(g) else np.nan})
    # open interest, where we hold it (Bybit daily)
    if v.oi_usd is not None and base in v.oi_usd.columns:
        oi = vendor_hours(EXCH[venue], sym, "open_interest", start, end)
        if not oi.empty:
            o = v.oi_usd[base].loc[start:end].dropna()
            o = o[o.index.hour == 0]
            vo = oi["oi_usd"].reindex(o.index)
            r = (o / vo - 1).dropna()
            out.update({"oi_n": int(len(r)), "oi_median_abs_pct": float(r.abs().median() * 100) if len(r) else np.nan,
                        "oi_max_abs_pct": float(r.abs().max() * 100) if len(r) else np.nan})
    out["flag"] = bool((out.get("funding_share_gt_1bp", 0) or 0) > 0.01 or (out.get("price_p99_abs_bps", 0) or 0) > 50)
    return out


def main(a):
    for venue in a.venues:
        v = LOAD[venue](*FULL)                              # the cached matrices; a new range rebuilds all coins
        rows = [compare(venue, b, v, a.start, a.end) for b in a.bases]
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"chd_crosscheck_{venue}.csv", index=False)
        pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
        print(f"\n== {venue} vs CryptoHFTData, {a.start} to {a.end} ==")
        print(df.to_string(index=False))
        ok = df[df.get("flag", pd.Series(False, index=df.index)) == False]
        print(f"{len(ok)}/{len(df)} coins agree within tolerance; flagged: {list(df[df.flag == True].base) if 'flag' in df else []}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-01"); ap.add_argument("--end", default="2026-10-14")
    ap.add_argument("--venues", nargs="*", default=["BN", "BY", "HL"])
    ap.add_argument("--bases", nargs="*", default=["BTC", "ETH", "SOL", "XRP", "DOGE", "SUI", "PEPE", "ZEC", "LSK", "MARSCOIN"])
    main(ap.parse_args())
