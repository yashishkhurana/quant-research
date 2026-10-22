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


def _get(path: str, base: str = BASE, ok_missing: bool = False, **params):
    """GET with retries on rate limits, 5xx and dropped/timed-out connections.
    ok_missing: return None on 400/404 (e.g. no spot market for this base)."""
    err = None
    for attempt in range(6):
        try:
            r = _S.get(base + path, params=params, timeout=20)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = e; time.sleep(2 ** attempt)
            continue
        if ok_missing and r.status_code in (400, 404):
            return None
        if r.status_code == 429 or r.status_code >= 500:
            err = requests.HTTPError(f"{r.status_code} on {path}"); time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise err


def list_symbols(include_delisted: bool = True) -> pd.DataFrame:
    info = _get("/fapi/v1/exchangeInfo")
    rows = []
    for s in info["symbols"]:
        if s["quoteAsset"] != "USDT" or s["contractType"] != "PERPETUAL" or s.get("underlyingType") != "COIN":
            continue
        if not include_delisted and s["status"] != "TRADING":
            continue
        rows.append({
            "symbol": s["symbol"], "base": s["baseAsset"], "status": s["status"],
            "onboard": pd.to_datetime(s["onboardDate"], unit="ms", utc=True),
            "delivery": pd.to_datetime(s["deliveryDate"], unit="ms", utc=True) if s.get("deliveryDate") else pd.NaT,
        })
    return pd.DataFrame(rows)


def all_perp_bases(onboarded_before: str | None = None) -> list[str]:
    """Every USDT perp Binance has listed, delisted ones included (status != TRADING), so the
    universe is not selected on survival. Optionally keep only contracts listed before a date."""
    df = list_symbols(include_delisted=True)
    if onboarded_before:
        df = df[df["onboard"] < pd.Timestamp(onboarded_before, tz="UTC")]
    return sorted(df["base"].unique().tolist())


def top_by_volume(n: int = 50) -> list[str]:
    t = _get("/fapi/v1/ticker/24hr")
    df = pd.DataFrame(t)
    # Only crypto perps that are trading now; the ticker also carries stock/commodity contracts (TSLA, XAU).
    coins = set(list_symbols(include_delisted=False)["symbol"])
    df = df[df["symbol"].isin(coins)]
    df["quoteVolume"] = df["quoteVolume"].astype(float)
    return df.sort_values("quoteVolume", ascending=False)["symbol"].head(n).str.replace("USDT", "").tolist()


_INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
SPOT_BASE = "https://api.binance.com"


def klines(base: str, start: str, end: str, interval: str = "1m", market: str = "perp") -> pd.DataFrame:
    """Klines at any interval, perp (fapi) or spot (api). Returns ts-indexed OHLCV + trades + taker buy vol."""
    sym = base.upper() + "USDT"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    step = _INTERVAL_MS[interval]
    out = []
    while t0 < t1:
        if market == "perp":
            rows = _get("/fapi/v1/klines", symbol=sym, interval=interval, startTime=t0, endTime=t1, limit=1500)
        else:
            rows = _get("/api/v3/klines", base=SPOT_BASE, ok_missing=True, symbol=sym, interval=interval,
                        startTime=t0, endTime=t1, limit=1000)
            if rows is None:                     # no spot market for this base
                return pd.DataFrame()
        if not rows:
            break
        out += rows
        t0 = rows[-1][0] + step
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


def klines_1m(base: str, start: str, end: str) -> pd.DataFrame:
    return klines(base, start, end, "1m", "perp")


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
    df["mark"] = pd.to_numeric(df.get("markPrice", pd.Series(dtype=float)), errors="coerce")  # "" on old rows
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


def download(dataset: str, bases: list[str], start: str, end: str, interval: str = "1m") -> None:
    for b in bases:
        try:
            if dataset == "klines":
                df = klines(b, start, end, interval, "perp"); ds, suf = f"klines_{interval}", "-PERP"
            elif dataset == "spot":
                df = klines(b, start, end, interval, "spot"); ds, suf = f"klines_{interval}", "-SPOT"
            elif dataset == "funding":
                df = funding(b, start, end); ds, suf = "funding", "-PERP"
            else:
                df = open_interest_5m(b, start, end); ds, suf = "oi", "-PERP"
        except (requests.RequestException, ValueError) as e:
            print(f"{b}: skipped ({e})"); continue
        n = append_rows(df, "raw", "binance", ds, f"{b.upper()}{suf}")
        print(f"{b} {dataset}: {len(df)} rows -> {len(n)} day files", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["klines", "spot", "funding", "oi", "universe", "universe-all"])
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=pd.Timestamp.now('UTC').strftime("%Y-%m-%d"))
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--top", type=int, default=50)
    a = ap.parse_args()
    if a.dataset == "universe":
        print("\n".join(top_by_volume(a.top)))
    elif a.dataset == "universe-all":
        print("\n".join(all_perp_bases(onboarded_before=a.end)))
    else:
        download(a.dataset, a.symbols, a.start, a.end, a.interval)
