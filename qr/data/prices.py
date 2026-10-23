"""Hourly kline columns as matrices (bar time x base), cached per (market, column, range).

    close = load_kline_matrix(bases, "PERP", "close", "2023-01-01", "2026-10-23")
    dvol  = load_kline_matrix(bases, "PERP", "quote_vol", "2023-01-01", "2026-10-23")
    spot  = load_kline_matrix(bases, "SPOT", "close", "2023-01-01", "2026-10-23")

Reading ~1,400 day files per symbol is the slow part (minutes for a few hundred symbols); the
first call reads every requested column for the market at once and caches them all.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from qr.data.store import ROOT, read_range

CACHE_DIR = ROOT / "processed" / "binance"
COLUMNS = ("close", "quote_vol", "volume")


def _path(market: str, col: str, start: str, end: str, interval: str):
    return CACHE_DIR / f"klines_{interval}_{market}_{col}_{start}_{end}.parquet"


def _one(base: str, market: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = read_range("raw", "binance", f"klines_{interval}", f"{base}-{market}", start, end)
    if df.empty:
        return df
    return df[~df.index.duplicated(keep="last")][[c for c in COLUMNS if c in df.columns]]


def load_kline_matrix(bases: list[str], market: str, col: str, start: str, end: str,
                      interval: str = "1h", cache: bool = True) -> pd.DataFrame:
    """market = 'PERP' or 'SPOT'. Columns of the result = bases that have data. Cache is rebuilt
    if it lacks a requested base that has data on disk; delete the files after a new download."""
    path = _path(market, col, start, end, interval)
    if cache and path.exists():
        df = pd.read_parquet(path)
        missing = [b for b in bases if b not in df.columns]
        if not missing or all(_one(b, market, start, end, interval).empty for b in missing[:5]):
            return df[[b for b in bases if b in df.columns]]
    with ThreadPoolExecutor(8) as ex:
        frames = dict(zip(bases, ex.map(lambda b: _one(b, market, start, end, interval), bases)))
    frames = {b: f for b, f in frames.items() if not f.empty}
    out = {}
    for c in COLUMNS:
        m = pd.concat({b: f[c] for b, f in frames.items() if c in f}, axis=1, sort=True)
        m.index.name = "ts"
        out[c] = m
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            m.to_parquet(_path(market, c, start, end, interval))
    return out[col]
