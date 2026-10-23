"""Binance, Hyperliquid and Bybit perps on one set of names and one clock.

Each venue names contracts its own way: Binance '1000PEPE', '1MBABYDOGE'; Hyperliquid 'kPEPE';
Bybit '1000PEPE', '10000SATS', '1000000BABYDOGE'. `canonical` strips the size prefix and returns
(underlying, multiplier), so 'kPEPE' and '1000PEPE' are both ('PEPE', 1000). Funding rates are per
notional, so the multiplier does not matter for funding; it matters only when prices are compared.

A shared ticker is not proof of a shared token. `price_matched` checks every Binance/Bybit match on
multiplier-adjusted prices and drops pairs whose median ratio is off by more than `tol`.

Loaders return matrices with canonical columns:
    funding   raw settlement rates at settlement times, rounded to the minute (any interval;
              qr.data.funding.per_window puts them on 8h windows)
    close     hourly close in the venue's own contract units (bar open time index)
    dvol      hourly dollar volume
    oi_usd    open interest in USD, hourly (forward-filled from daily; Bybit only, where history exists)
Delisting times per venue: Binance deliveryDate, Bybit deliveryTime, Hyperliquid the last
funding settlement of a coin flagged isDelisted.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from qr.data.store import ROOT, read_range

VENUES = ("BN", "HL", "BY")
TAKER_FEE_BPS = {"BN": 4.0, "HL": 4.5, "BY": 5.5}       # base tier, taker; slippage is added separately
_PREFIX = re.compile(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])")


def canonical(name: str, venue: str) -> tuple[str, float]:
    """(underlying, contract multiplier) for a venue's perp name."""
    if venue == "HL" and len(name) > 1 and name[0] == "k" and name[1:].isupper():
        return name[1:], 1000.0
    m = _PREFIX.match(name)
    if not m:
        return name, 1.0
    p = m.group(1)
    return name[len(p):], 1e6 if p == "1M" else float(p)


def canon_map(names: list[str], venue: str) -> dict[str, str]:
    """raw name -> canonical. If two of a venue's contracts map to the same underlying (Binance
    lists both 'BOB' and '1000000BOB'), the prefixed one keeps its raw name: they are not assumed
    to be the same token."""
    out, seen = {}, {}
    for n in names:
        seen.setdefault(canonical(n, venue)[0], []).append(n)
    for c, ns in seen.items():
        if len(ns) == 1:
            out[ns[0]] = c
        else:
            for n in ns:
                out[n] = c if canonical(n, venue)[1] == 1.0 else n
    return out


@dataclass
class Venue:
    name: str
    funding: pd.DataFrame
    close: pd.DataFrame | None = None
    dvol: pd.DataFrame | None = None
    oi_usd: pd.DataFrame | None = None
    multiplier: dict[str, float] = field(default_factory=dict)
    delisted: dict[str, pd.Timestamp] = field(default_factory=dict)
    raw_names: dict[str, str] = field(default_factory=dict)     # canonical -> venue's own name


def _cache(venue: str, kind: str, start: str, end: str, n: int):
    """n = contracts on disk: a cache built from a partial download is not reused after it completes."""
    return ROOT / "processed" / "venues" / f"{venue}_{kind}_{start}_{end}_n{n}.parquet"


def _round(f: pd.DataFrame | pd.Series):
    f = f.copy()
    f.index = f.index.round("min")
    return f[~f.index.duplicated(keep="last")]


def _matrix(exchange: str, dataset: str, names: dict[str, str], col: str, start: str, end: str,
            rnd: bool = False) -> pd.DataFrame:
    """names: canonical -> store symbol (without -PERP)."""
    def one(item):
        c, raw = item
        d = read_range("raw", exchange, dataset, f"{raw}-PERP", start, end)
        if d.empty or col not in d:
            return c, None
        s = d[col]
        s = _round(s) if rnd else s[~s.index.duplicated(keep="last")]
        return c, s
    with ThreadPoolExecutor(8) as ex:
        got = [x for x in ex.map(one, names.items()) if x[1] is not None]
    if not got:
        return pd.DataFrame()
    m = pd.concat({c: s for c, s in got}, axis=1, sort=True)
    m.index.name = "ts"
    return m


def _utc(m: pd.DataFrame) -> pd.DataFrame:
    """Every matrix on the UTC clock. Hyperliquid candles are stored tz-naive; left that way they
    reindex onto the UTC price index as all-NaN and every HL leg looks unsizable."""
    if isinstance(m.index, pd.DatetimeIndex) and m.index.tz is None:
        m = m.copy()
        m.index = m.index.tz_localize("UTC")
    return m


def _cached(venue, kind, start, end, build, n):
    p = _cache(venue, kind, start, end, n)
    if p.exists():
        return _utc(pd.read_parquet(p))
    m = _utc(build())
    p.parent.mkdir(parents=True, exist_ok=True)
    if not m.empty:
        m.to_parquet(p)
    return m


def _names_on_disk(exchange: str, dataset: str) -> list[str]:
    d = ROOT / "raw" / exchange / dataset
    return sorted(x.name[:-5] for x in d.glob("*-PERP")) if d.exists() else []


def _mask_after(m: pd.DataFrame, times: dict[str, pd.Timestamp]) -> pd.DataFrame:
    m = m.copy()
    for c, t in times.items():
        if c in m.columns and pd.notna(t):
            m.loc[m.index >= t, c] = np.nan
    return m


def load_binance(start: str, end: str) -> Venue:
    from qr.data.funding import load_funding
    from qr.data.history import delisting_times
    from qr.data.prices import load_kline_matrix
    bases = [l.strip() for l in open("data/universe.txt") if l.strip()]
    f = load_funding(bases, start, end)
    close = load_kline_matrix(list(f.columns), "PERP", "close", start, end)          # cut at delisting
    dvol = load_kline_matrix(list(f.columns), "PERP", "quote_vol", start, end)
    ren = canon_map(list(f.columns), "BN")
    mult = {ren[b]: canonical(b, "BN")[1] if ren[b] != b or canonical(b, "BN")[0] == b else 1.0 for b in f.columns}
    dl = {ren[b]: t for b, t in delisting_times().items() if b in ren}
    return Venue("BN", f.rename(columns=ren), close.rename(columns=ren), dvol.rename(columns=ren), None,
                 mult, dl, {v: k for k, v in ren.items()})


def load_bybit(start: str, end: str) -> Venue:
    names = _names_on_disk("bybit", "funding")
    canon = {c: n for n, c in canon_map(names, "BY").items()}
    mult = {c: canonical(n, "BY")[1] if c != n else 1.0 for c, n in canon.items()}
    f = _cached("BY", "funding", start, end, lambda: _matrix("bybit", "funding", canon, "rate", start, end, rnd=True), len(canon))
    close = _cached("BY", "close", start, end, lambda: _matrix("bybit", "klines_1h", canon, "close", start, end), len(canon))
    dvol = _cached("BY", "dvol", start, end, lambda: _matrix("bybit", "klines_1h", canon, "quote_vol", start, end), len(canon))
    oi = _cached("BY", "oi", start, end, lambda: _matrix("bybit", "oi_1d", canon, "oi", start, end), len(canon))
    inst = pd.read_csv(ROOT / "processed" / "bybit" / "instruments.csv", parse_dates=["delist_time"])
    dl = {}
    for _, r in inst[inst.status == "Closed"].iterrows():
        c = canonical(r["base"], "BY")[0]
        if c in canon and pd.notna(r["delist_time"]):
            dl[c] = pd.Timestamp(r["delist_time"]).tz_convert("UTC") if pd.Timestamp(r["delist_time"]).tzinfo else pd.Timestamp(r["delist_time"], tz="UTC")
    close = _mask_after(close, dl)
    oi_usd = None
    if not oi.empty and not close.empty:
        cols = oi.columns.intersection(close.columns)
        oi_h = oi[cols].reindex(close.index, method="ffill", limit=24 * 2)          # daily print, valid ~1 day
        oi_usd = oi_h * close[cols]
    return Venue("BY", f, close, dvol, oi_usd, mult, dl, canon)


def load_hyperliquid(start: str, end: str) -> Venue:
    names = _names_on_disk("hyperliquid", "funding")
    canon = {c: n for n, c in canon_map(names, "HL").items()}
    mult = {c: canonical(n, "HL")[1] if c != n else 1.0 for c, n in canon.items()}
    f = _cached("HL", "funding", start, end, lambda: _matrix("hyperliquid", "funding", canon, "rate", start, end, rnd=True), len(canon))
    close = dvol = None
    if _names_on_disk("hyperliquid", "klines_1h"):
        close = _cached("HL", "close", start, end, lambda: _matrix("hyperliquid", "klines_1h", canon, "close", start, end), len(canon))
        dvol = _cached("HL", "dvol", start, end, lambda: _matrix("hyperliquid", "klines_1h", canon, "quote_vol", start, end), len(canon))
    dl = {}
    meta_path = ROOT / "processed" / "hyperliquid" / "delisted.json"
    if meta_path.exists():
        delisted = set(json.loads(meta_path.read_text()))
        for c, n in canon.items():
            if n in delisted and c in f.columns and f[c].notna().any():
                dl[c] = f[c].last_valid_index() + pd.Timedelta(hours=1)
    return Venue("HL", f, close, dvol, None, mult, dl, canon)


def price_matched(a: Venue, b: Venue, tol: float = 0.10, share_within: float = 0.99,
                  min_overlap: int = 24 * 7) -> list[str]:
    """Canonical names on both venues whose multiplier-adjusted closes are within 1 +/- tol on at
    least `share_within` of the overlapping hours. A median test is not enough: Binance 'H' and
    Bybit 'H' agree at the median but sat 2.4x apart for hours in June 2026."""
    if a.close is None or b.close is None:
        return []
    out = []
    for c in a.close.columns.intersection(b.close.columns):
        pa = a.close[c] / a.multiplier.get(c, 1.0)
        pb = b.close[c] / b.multiplier.get(c, 1.0)
        both = pd.concat([pa, pb], axis=1).dropna()
        if len(both) < min_overlap:
            continue
        r = both.iloc[:, 0] / both.iloc[:, 1]
        if ((r - 1).abs() <= tol).mean() >= share_within:
            out.append(c)
    return out


def save_hl_delisted() -> list[str]:
    from qr.data.hl_history import all_names
    d = sorted(set(all_names(True)) - set(all_names(False)))
    p = ROOT / "processed" / "hyperliquid" / "delisted.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return d
