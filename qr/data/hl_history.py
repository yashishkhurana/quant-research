"""Hyperliquid perps history: funding (hourly, full history), hourly candles (last ~5,000 only),
and a point-in-time snapshot of open interest and 24h notional volume per coin.

Same store layout as Binance: data/raw/hyperliquid/<dataset>/<COIN>-PERP/<day>.parquet, so
qr.data.store.read_range works unchanged. The HL funding rate is an HOURLY rate (fraction);
qr.data.funding.per_window sums it onto 8h windows, which makes it comparable with Binance.

    python -m qr.data.hl_history funding --start 2024-01-01            # all coins listed on HL
    python -m qr.data.hl_history candles                                # last 5000 1h bars per coin
    python -m qr.data.hl_history snapshot                               # OI + 24h volume today

Rate limits: the info endpoint allows ~1200 weight/min; fundingHistory costs 20, so about one
call per second. A funding pull from 2024 for 150 coins is ~5k calls, roughly 1.5 hours.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from qr.data.store import ROOT, append_rows

INFO = "https://api.hyperliquid.xyz/info"
_S = requests.Session()
CANDLE_LIMIT = 5000


def _post(body: dict, pause: float = 1.0):
    err = None
    for attempt in range(6):
        try:
            r = _S.post(INFO, json=body, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = e; time.sleep(2 ** attempt); continue
        if r.status_code == 429 or r.status_code >= 500:
            err = requests.HTTPError(f"{r.status_code} on {body.get('type')}"); time.sleep(2 ** attempt); continue
        r.raise_for_status()
        time.sleep(pause)
        return r.json()
    raise err


def meta_and_ctxs() -> pd.DataFrame:
    """One row per listed perp: name, max_leverage, funding (hourly), oi (base units), mark_px,
    oracle_px, oi_usd, day_ntl_vlm (24h notional volume, USD)."""
    meta, ctxs = _post({"type": "metaAndAssetCtxs"}, pause=0.2)
    rows = []
    for u, c in zip(meta["universe"], ctxs):
        if u.get("isDelisted"):
            continue
        mark = float(c["markPx"]); oi = float(c["openInterest"])
        rows.append({"name": u["name"], "max_leverage": u.get("maxLeverage"), "funding": float(c["funding"]),
                     "oi": oi, "mark_px": mark, "oracle_px": float(c["oraclePx"]), "oi_usd": oi * mark,
                     "day_ntl_vlm": float(c["dayNtlVlm"]), "premium": float(c["premium"] or 0.0)})
    return pd.DataFrame(rows).set_index("name")


def all_names(include_delisted: bool = True) -> list[str]:
    """Every perp in HL's meta universe. Delisted coins stay in the list (flagged isDelisted) and
    their funding history is still served, so a backtest universe built from this list is not
    restricted to today's survivors."""
    meta = _post({"type": "meta"}, pause=0.2)
    return [u["name"] for u in meta["universe"] if include_delisted or not u.get("isDelisted")]


def funding_history(coin: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Hourly funding for one coin: columns rate (hourly fraction), premium. Paginates 500 at a time."""
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
    out = []
    while t0 < t1:
        rows = _post({"type": "fundingHistory", "coin": coin, "startTime": t0, "endTime": t1})
        if not rows:
            break
        out += rows
        t0 = int(rows[-1]["time"]) + 1
        if len(rows) < 500:
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df = df.set_index("ts")[["rate", "premium"]]
    return df[~df.index.duplicated(keep="last")].sort_index()


def candles(coin: str, interval: str = "1h", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """OHLCV candles. HL serves at most the last 5,000 per request and does not page further back,
    so for 1h this is ~7 months. quote_vol = base volume x close (HL reports base volume)."""
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else int(time.time() * 1000)
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else t1 - CANDLE_LIMIT * 3_600_000
    rows = _post({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": t0, "endTime": t1}},
                 pause=0.3)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
    out = pd.DataFrame({"open": df["o"].astype(float), "high": df["h"].astype(float), "low": df["l"].astype(float),
                        "close": df["c"].astype(float), "volume": df["v"].astype(float), "n_trades": df["n"].astype(int)})
    out.index = df["ts"].values; out.index.name = "ts"
    out["quote_vol"] = out["volume"] * out["close"]
    return out[~out.index.duplicated(keep="last")].sort_index()


def download(dataset: str, coins: list[str], start: str = "2024-01-01", end: str | None = None) -> None:
    for c in coins:
        try:
            df = funding_history(c, start, end) if dataset == "funding" else candles(c, "1h")
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"{c}: skipped ({e})", flush=True); continue
        n = append_rows(df, "raw", "hyperliquid", "funding" if dataset == "funding" else "klines_1h", f"{c}-PERP")
        print(f"{c} {dataset}: {len(df)} rows -> {len(n)} day files", flush=True)


def snapshot() -> Path:
    df = meta_and_ctxs()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = ROOT / "processed" / "hyperliquid" / f"snapshot_{day}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("oi_usd", ascending=False).to_csv(out)
    print(f"{len(df)} coins, total OI ${df['oi_usd'].sum() / 1e9:,.2f}bn, 24h volume "
          f"${df['day_ntl_vlm'].sum() / 1e9:,.2f}bn -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["funding", "candles", "snapshot"])
    ap.add_argument("--coins", nargs="*", help="default: every coin listed on HL")
    ap.add_argument("--start", default="2024-01-01"); ap.add_argument("--end", default=None)
    a = ap.parse_args()
    if a.dataset == "snapshot":
        snapshot()
    else:
        coins = a.coins or all_names(include_delisted=(a.dataset == "funding"))
        print(f"{len(coins)} coins")
        download(a.dataset, coins, a.start, a.end)
