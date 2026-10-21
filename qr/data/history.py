"""Historical loaders for Binance USDT-M perps: 1-minute klines, funding, open interest.

Public REST, no key. Everything lands in the store under raw/binance/<dataset>/<SYM>-PERP.
Survivorship: `list_symbols(include_delisted=True)` returns delisted contracts too, so a
universe built from it does not silently drop the tokens that went to zero.

Usage:
    python -m qr.data.history klines --symbols BTC ETH --start 2024-01-01 --end 2024-01-31
    python -m qr.data.history funding --symbols BTC --start 2023-01-01
    python -m qr.data.history universe --top 50
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from qr.data.store import append_rows

BASE = "https://fapi.binance.com"
_S = requests.Session()


def _get(path: str, **params):
    for attempt in range(5):
        r = _S.get(BASE + path, params=params, timeout=20)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def list_symbols(include_delisted: bool = True) -> pd.DataFrame:
    info = _get("/fapi/v1/exchangeInfo")
    rows = []
    for s in info["symbols"]:
        if s["quoteAsset"] != "USDT" or s["contractType"] != "PERPETUAL":
            continue
        if not include_delisted and s["status"] != "TRADING":
            continue
        rows.append({
            "symbol": s["symbol"], "base": s["baseAsset"], "status": s["status"],
            "onboard": pd.to_datetime(s["onboardDate"], unit="ms", utc=True),
            "delivery": pd.to_datetime(s["deliveryDate"], unit="ms", utc=True) if s.get("deliveryDate") else pd.NaT,
        })
    return pd.DataFrame(rows)


def top_by_volume(n: int = 50) -> list[str]:
    t = _get("/fapi/v1/ticker/24hr")
    df = pd.DataFrame(t)
    df = df[df["symbol"].str.endswith("USDT")]
    df["quoteVolume"] = df["quoteVolume"].astype(float)
    return df.sort_values("quoteVolume", ascending=False)["symbol"].head(n).str.replace("USDT", "").tolist()


def klines_1m(base: str, start: str, end: str) -> pd.DataFrame:
    """1-minute klines, paged 1500 at a time. Returns ts-indexed OHLCV + trades + taker buy vol."""
    sym = base.upper() + "USDT"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out = []
    while t0 < t1:
        rows = _get("/fapi/v1/klines", symbol=sym, interval="1m", startTime=t0, endTime=t1, limit=1500)
        if not rows:
            break
        out += rows
        t0 = rows[-1][0] + 60_000
        time.sleep(0.05)
    if not out:
        return pd.DataFrame()
    cols = ["ts", "open", "high", "low", "close", "volume", "close_ts", "quote_vol",
            "n_trades", "taker_buy_vol", "taker_buy_quote", "_"]
    df = pd.DataFrame(out, columns=cols).drop(columns=["close_ts", "_"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").astype(float)
    df["n_trades"] = df["n_trades"].astype(int)
    return df


def funding(base: str, start: str, end: str | None = None) -> pd.DataFrame:
    sym = base.upper() + "USDT"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end or pd.Timestamp.utcnow(), tz="UTC").timestamp() * 1000) if end else None
    out = []
    while True:
        params = dict(symbol=sym, startTime=t0, limit=1000)
        if t1:
            params["endTime"] = t1
        rows = _get("/fapi/v1/fundingRate", **params)
        if not rows:
            break
        out += rows
        if len(rows) < 1000:
            break
        t0 = rows[-1]["fundingTime"] + 1
        time.sleep(0.05)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    df["mark"] = df.get("markPrice", pd.Series(dtype=float)).astype(float)
    return df.set_index("ts")[["rate", "mark"]]


def open_interest_5m(base: str, start: str, end: str) -> pd.DataFrame:
    """OI history at 5m; Binance only serves ~30 days back on this endpoint."""
    sym = base.upper() + "USDT"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out = []
    while t0 < t1:
        rows = _get("/futures/data/openInterestHist", symbol=sym, period="5m", startTime=t0, endTime=t1, limit=500)
        if not rows:
            break
        out += rows
        t0 = rows[-1]["timestamp"] + 300_000
        time.sleep(0.05)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["oi"] = df["sumOpenInterest"].astype(float)
    df["oi_usd"] = df["sumOpenInterestValue"].astype(float)
    return df.set_index("ts")[["oi", "oi_usd"]]


def download(dataset: str, bases: list[str], start: str, end: str) -> None:
    fn = {"klines": klines_1m, "funding": funding, "oi": open_interest_5m}[dataset]
    ds_name = {"klines": "klines_1m", "funding": "funding", "oi": "oi"}[dataset]
    for b in bases:
        df = fn(b, start, end)
        n = append_rows(df, "raw", "binance", ds_name, f"{b.upper()}-PERP")
        print(f"{b}: {len(df)} rows -> {len(n)} day files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["klines", "funding", "oi", "universe"])
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args()
    if a.dataset == "universe":
        print("\n".join(top_by_volume(a.top)))
    else:
        download(a.dataset, a.symbols, a.start, a.end)
