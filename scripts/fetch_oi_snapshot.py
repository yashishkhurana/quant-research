"""Current open interest for every listed USDT-M perp, as a sanity bound on capacity.

Binance serves no OI history beyond 30 days, so this is a snapshot, dated, and used only to
check the volume-based caps against the size of the market being shorted. One call per symbol.

    python scripts/fetch_oi_snapshot.py

Writes data/processed/binance/oi_snapshot_<date>.csv with
    base, oi_base, mark_px, oi_usd, adv_perp_usd_30d, adv_spot_usd_30d
and prints the top 40 by OI with the 1% / 2% of OI and of ADV a book could hold per name.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qr.capacity import adv
from qr.data.history import _get, list_symbols
from qr.data.prices import load_kline_matrix

START, END = "2023-01-01", "2026-10-23"


def main():
    syms = list_symbols(include_delisted=False)
    marks = {d["symbol"]: float(d["markPrice"]) for d in _get("/fapi/v1/premiumIndex")}
    rows = []
    for s, base in zip(syms["symbol"], syms["base"]):
        oi = _get("/fapi/v1/openInterest", ok_missing=True, symbol=s)
        if oi is None or s not in marks:
            continue
        rows.append({"base": base, "oi_base": float(oi["openInterest"]), "mark_px": marks[s],
                     "oi_usd": float(oi["openInterest"]) * marks[s]})
    df = pd.DataFrame(rows).set_index("base")
    bases = list(df.index)
    a_perp = adv(load_kline_matrix(bases, "PERP", "quote_vol", START, END)).iloc[-1]
    a_spot = adv(load_kline_matrix(bases, "SPOT", "quote_vol", START, END)).iloc[-1]
    df["adv_perp_usd_30d"] = a_perp.reindex(df.index)
    df["adv_spot_usd_30d"] = a_spot.reindex(df.index)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path("data/processed/binance") / f"oi_snapshot_{day}.csv"
    df.sort_values("oi_usd", ascending=False).to_csv(out)
    top = df.sort_values("oi_usd", ascending=False).head(40).copy()
    top["1pct_oi"] = top["oi_usd"] * 0.01
    top["2pct_min_adv"] = 0.02 * top[["adv_perp_usd_30d", "adv_spot_usd_30d"]].min(axis=1)
    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print(top[["oi_usd", "adv_perp_usd_30d", "adv_spot_usd_30d", "1pct_oi", "2pct_min_adv"]])
    print(f"\n{len(df)} contracts, total OI ${df['oi_usd'].sum() / 1e9:,.1f}bn, "
          f"top-40 OI ${top['oi_usd'].sum() / 1e9:,.1f}bn -> {out}")


if __name__ == "__main__":
    main()
