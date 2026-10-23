"""Funding as a matrix on a common clock.

Binance settles most alts every 4h, majors every 8h, and switches a contract to 1h while its
funding sits at the cap. Raw per-settlement rates are therefore not comparable across coins or
across time: a 4h coin's rate is half an 8h coin's for the same carry. Everything here is put on
fixed 8h UTC windows (00:00, 08:00, 16:00) by summing the settlements inside each window.

The value labelled T is the funding settled in (T-8h, T], so it is known at T.

    raw = load_funding(bases, "2023-01-01", "2026-10-23")    # settlement times x bases, cached
    f8 = per_window(raw)                                       # funding paid per 8h window
    apr = f8 * WINDOWS_PER_YEAR                                # annualised
    iv = interval_hours(raw)                                   # settlement interval in force
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from qr.data.store import ROOT, read_range

WINDOW = "8h"
WINDOWS_PER_YEAR = 3 * 365
CACHE_DIR = ROOT / "processed" / "binance"


def _one(base: str, start: str, end: str) -> pd.Series:
    f = read_range("raw", "binance", "funding", f"{base}-PERP", start, end)
    if f.empty:
        return pd.Series(dtype=float, name=base)
    s = f["rate"].copy()
    s.index = s.index.round("min")                      # fundingTime carries ms jitter
    return s[~s.index.duplicated(keep="last")].rename(base)


def load_funding(bases: list[str], start: str, end: str, cache: bool = True) -> pd.DataFrame:
    """Raw settlement rates, index = settlement time (UTC), columns = bases. NaN = no settlement
    for that coin at that time (other interval, or not listed). Cached per (start, end); the
    cache is rebuilt if it is missing any requested base. Delete the file after a new download."""
    path = CACHE_DIR / f"funding_matrix_{start}_{end}.parquet"
    if cache and path.exists():
        df = pd.read_parquet(path)
        if set(bases) <= set(df.columns):
            return df[bases]
    with ThreadPoolExecutor(8) as ex:
        cols = list(ex.map(lambda b: _one(b, start, end), bases))
    df = pd.concat([c for c in cols if not c.empty], axis=1, sort=True)
    df.index.name = "ts"
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    return df


def per_window(raw: pd.DataFrame) -> pd.DataFrame:
    """Funding paid per fixed 8h window, summed over the settlements inside it. NaN where the
    coin had no settlement in the window (before listing, after delisting).
    Timestamps are rounded to the minute first: Binance's fundingTime carries ms jitter, and with
    closed='right' an 08:00:00.008 settlement would fall into the window ending 16:00."""
    raw = raw.copy(); raw.index = raw.index.round("min")
    g = raw.resample(WINDOW, label="right", closed="right")
    out = g.sum(min_count=1)
    return out


def interval_hours(raw: pd.DataFrame) -> pd.DataFrame:
    """Hours since the coin's previous settlement, at each settlement. 8, 4 or 1 in practice."""
    return raw.apply(lambda c: c.dropna().index.to_series().diff().dt.total_seconds().div(3600)
                     .reindex(raw.index))


def has_spot(bases: list[str]) -> list[str]:
    """Bases with Binance spot hourly klines on disk, i.e. the ones a hedged book can trade."""
    d = ROOT / "raw" / "binance" / "klines_1h"
    return [b for b in bases if (d / f"{b}-SPOT").is_dir() and any((d / f"{b}-SPOT").iterdir())]
