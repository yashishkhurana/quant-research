"""Bybit USDT linear perps history: funding, hourly klines (with dollar turnover), daily open
interest, delisting times, and a point-in-time snapshot. Bybit is the one venue here that serves
open-interest HISTORY, so it is where an OI cap can be applied over the backtest.

Same store layout as Binance: data/raw/bybit/<dataset>/<BASE>-PERP/<day>.parquet, with BASE the
Bybit base coin (e.g. 1000PEPE, 10000SATS); qr.data.venues maps names across venues.

    python -m qr.data.bybit_history funding --start 2023-01-01      # every perp, delisted included
    python -m qr.data.bybit_history klines  --start 2023-01-01
    python -m qr.data.bybit_history oi      --start 2023-01-01
    python -m qr.data.bybit_history snapshot

Rate limit: 600 requests / 5 s per IP on public endpoints; a 0.05 s pause keeps well under it.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from qr.data.store import ROOT, append_rows

BASE = "https://api.bybit.com"
_S = requests.Session()


def _get(path: str, pause: float = 0.05, **params):
    err = None
    for attempt in range(6):
        try:
            r = _S.get(BASE + path, params=params, timeout=30)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = e; time.sleep(2 ** attempt); continue
        if r.status_code == 429 or r.status_code >= 500:
            err = requests.HTTPError(f"{r.status_code} on {path}"); time.sleep(2 ** attempt); continue
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") == 10006:                       # rate limited
            err = requests.HTTPError("rate limit"); time.sleep(2 ** attempt); continue
        if j.get("retCode") != 0:
            raise ValueError(f"{path} {params}: {j.get('retCode')} {j.get('retMsg')}")
        time.sleep(pause)
        return j["result"]
    raise err


def instruments(include_delisted: bool = True) -> pd.DataFrame:
    """USDT linear perpetuals. status Trading or Closed (delisted). delist_time = deliveryTime for
    closed contracts when Bybit reports one (else NaT; use the last funding settlement)."""
    rows = []
    for st in (["Trading", "Closed"] if include_delisted else ["Trading"]):
        cursor = ""
        while True:
            res = _get("/v5/market/instruments-info", category="linear", status=st, limit=1000, cursor=cursor)
            for x in res["list"]:
                if x.get("quoteCoin") != "USDT" or x.get("contractType") != "LinearPerpetual":
                    continue
                dt = int(x.get("deliveryTime") or 0)
                rows.append({"symbol": x["symbol"], "base": x["baseCoin"], "status": st,
                             "launch_time": pd.to_datetime(int(x.get("launchTime") or 0), unit="ms", utc=True),
                             "delist_time": pd.to_datetime(dt, unit="ms", utc=True) if dt else pd.NaT,
                             "funding_interval_min": int(x.get("fundingInterval") or 0)})
            cursor = res.get("nextPageCursor") or ""
            if not cursor:
                break
    df = pd.DataFrame(rows)
    return df.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def _ms(t: str | None) -> int:
    return int(pd.Timestamp(t, tz="UTC").timestamp() * 1000) if t else int(time.time() * 1000)


def funding(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Settlement rates (fraction per settlement, any interval). Pages backwards 200 at a time."""
    t0, t1 = _ms(start), _ms(end)
    out = []
    while t1 > t0:
        lst = _get("/v5/market/funding/history", category="linear", symbol=symbol, startTime=t0, endTime=t1, limit=200)["list"]
        if not lst:
            break
        out += lst
        oldest = min(int(x["fundingRateTimestamp"]) for x in lst)
        if len(lst) < 200 or oldest <= t0:
            break
        t1 = oldest - 1
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    df["rate"] = df["fundingRate"].astype(float)
    df = df.set_index("ts")[["rate"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


def klines(symbol: str, start: str, end: str | None = None, interval: str = "60") -> pd.DataFrame:
    """Hourly OHLCV; quote_vol is Bybit's turnover (USDT). Pages backwards 1000 at a time."""
    t0, t1 = _ms(start), _ms(end)
    out = []
    while t1 > t0:
        lst = _get("/v5/market/kline", category="linear", symbol=symbol, interval=interval, start=t0, end=t1, limit=1000)["list"]
        if not lst:
            break
        out += lst
        oldest = min(int(x[0]) for x in lst)
        if len(lst) < 1000 or oldest <= t0:
            break
        t1 = oldest - 1
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume", "quote_vol"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)   # bar OPEN time, like Binance
    df = df.set_index("ts").astype(float).sort_index()
    return df[~df.index.duplicated(keep="last")]


def open_interest(symbol: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    """Open interest in base units at `interval` (5min..1d). Cursor-paginated."""
    t0, t1 = _ms(start), _ms(end)
    out, cursor = [], ""
    while True:
        res = _get("/v5/market/open-interest", category="linear", symbol=symbol, intervalTime=interval,
                   startTime=t0, endTime=t1, limit=200, cursor=cursor)
        out += res["list"]
        cursor = res.get("nextPageCursor") or ""
        if not cursor or not res["list"]:
            break
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df["oi"] = df["openInterest"].astype(float)
    df = df.set_index("ts")[["oi"]].sort_index()
    return df[~df.index.duplicated(keep="last")]


def snapshot() -> Path:
    """Live tickers: OI (base and USD), 24h turnover, mark, funding, for every linear USDT perp."""
    lst = _get("/v5/market/tickers", category="linear")["list"]
    rows = [{"symbol": x["symbol"], "mark_px": float(x["markPrice"] or 0), "oi": float(x["openInterest"] or 0),
             "oi_usd": float(x.get("openInterestValue") or 0), "day_turnover": float(x.get("turnover24h") or 0),
             "funding": float(x.get("fundingRate") or 0)} for x in lst if x["symbol"].endswith("USDT")]
    df = pd.DataFrame(rows).set_index("symbol").sort_values("oi_usd", ascending=False)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = ROOT / "processed" / "bybit" / f"snapshot_{day}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"{len(df)} perps, total OI ${df['oi_usd'].sum() / 1e9:,.2f}bn -> {out}")
    return out


def save_instruments() -> pd.DataFrame:
    df = instruments(include_delisted=True)
    out = ROOT / "processed" / "bybit" / "instruments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def shared_bases(inst: pd.DataFrame) -> pd.DataFrame:
    """Only contracts whose underlying also trades on Binance (data/universe.txt) or Hyperliquid:
    a Bybit-only coin has no second venue to spread against."""
    from qr.data.hl_history import all_names
    from qr.data.venues import canonical
    other = {canonical(b, "BN")[0] for b in (l.strip() for l in open("data/universe.txt")) if b}
    other |= {canonical(n, "HL")[0] for n in all_names(True)}
    return inst[[canonical(b, "BY")[0] in other for b in inst.base]]


def download(dataset: str, start: str, end: str | None = None, symbols: list[str] | None = None,
             workers: int = 8, shared_only: bool = True) -> None:
    from concurrent.futures import ThreadPoolExecutor
    inst = save_instruments()
    if symbols:
        inst = inst[inst.symbol.isin(symbols)]
    elif shared_only:
        inst = shared_bases(inst)
    print(f"{len(inst)} perps ({(inst.status == 'Closed').sum()} delisted)", flush=True)
    fn = {"funding": funding, "klines": klines, "oi": open_interest}[dataset]
    ds = {"funding": "funding", "klines": "klines_1h", "oi": "oi_1d"}[dataset]

    def one(row):
        sym, base, launch = row
        s = max(pd.Timestamp(start, tz="UTC"), launch).strftime("%Y-%m-%d") if pd.notna(launch) else start
        try:
            df = fn(sym, s, end)
        except (requests.RequestException, ValueError, KeyError) as e:
            return f"{sym}: skipped ({e})"
        n = append_rows(df, "raw", "bybit", ds, f"{base}-PERP")      # one directory per contract: thread-safe
        return f"{sym} {dataset}: {len(df)} rows -> {len(n)} day files"

    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(one, zip(inst.symbol, inst.base, inst.launch_time)):
            print(msg, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["funding", "klines", "oi", "snapshot", "instruments"])
    ap.add_argument("--start", default="2023-01-01"); ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", nargs="*")
    a = ap.parse_args()
    if a.dataset == "snapshot":
        snapshot()
    elif a.dataset == "instruments":
        d = save_instruments(); print(d.status.value_counts().to_dict())
    else:
        download(a.dataset, a.start, a.end, a.symbols)
