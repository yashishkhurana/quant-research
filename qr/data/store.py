"""Parquet store with a CSV fallback.

Layout:  data/<layer>/<exchange>/<dataset>/<symbol>/<YYYY-MM-DD>.parquet
layer    = raw | processed
dataset  = trades | book | klines_1m | funding | oi
Each file holds one UTC day. Appends within a day dedupe on the index.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    _EXT = ".parquet"
except ImportError:  # workspace without pyarrow: fall back to csv
    _EXT = ".csv"

ROOT = Path(os.environ.get("QR_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def _path(layer: str, exchange: str, dataset: str, symbol: str, day: str) -> Path:
    sym = symbol.replace("/", "-").replace(":", "_")
    p = ROOT / layer / exchange / dataset / sym / f"{day}{_EXT}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write(df: pd.DataFrame, p: Path) -> None:
    """Atomic: write a temp file in the same directory, then rename over the target. A reader (or a
    crash) mid-write sees the old complete file, never a truncated one; the collector rewrites the
    whole day file on every flush, so a non-atomic write could lose the day."""
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    if _EXT == ".parquet":
        df.to_parquet(tmp)
    else:
        df.to_csv(tmp)
    os.replace(tmp, p)


def _read(p: Path) -> pd.DataFrame:
    if _EXT == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, index_col=0, parse_dates=True)


def write_day(df: pd.DataFrame, layer: str, exchange: str, dataset: str, symbol: str, day: str) -> Path:
    """Write (or merge into) one day's file. df must have a UTC DatetimeIndex named 'ts'."""
    if df.empty:
        return _path(layer, exchange, dataset, symbol, day)
    p = _path(layer, exchange, dataset, symbol, day)
    if p.exists():
        old = _read(p)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    _write(df, p)
    return p


def append_rows(df: pd.DataFrame, layer: str, exchange: str, dataset: str, symbol: str) -> list[Path]:
    """Split a frame by UTC day and merge each day into its file."""
    if df.empty:
        return []
    out = []
    for day, chunk in df.groupby(df.index.strftime("%Y-%m-%d")):
        out.append(write_day(chunk, layer, exchange, dataset, symbol, str(day)))
    return out


def read_range(layer: str, exchange: str, dataset: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Read [start, end] inclusive by day. Missing days are skipped silently."""
    days = pd.date_range(start, end, freq="D", tz="UTC").strftime("%Y-%m-%d")
    frames = []
    for d in days:
        p = _path(layer, exchange, dataset, symbol, d)
        if p.exists():
            frames.append(_read(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df.index.name = "ts"
    return df


def available_days(layer: str, exchange: str, dataset: str, symbol: str) -> list[str]:
    p = _path(layer, exchange, dataset, symbol, "x").parent
    return sorted(f.stem for f in p.glob(f"*{_EXT}"))
