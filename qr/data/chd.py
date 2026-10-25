"""CryptoHFTData (cryptohftdata.com) as a second, independent source for perp data on Binance,
Bybit and Hyperliquid: open interest, mark/index price with funding, and 24h ticker volume.
History starts 2025-06-28 (Binance, Bybit) and 2025-09-29 (Hyperliquid); one Parquet file per
symbol, data type and UTC hour. Used for (a) point-in-time open-interest caps in the capacity
work, (b) the Hyperliquid mark-vs-index basis, (c) cross-checking our own exchange pulls.

Auth: API key from the CHD_API_KEY environment variable, exchanged for a 4h JWT. The key is
never written to disk by this module.

    export CHD_API_KEY=...
    python -m qr.data.chd open_interest --exchange binance_futures --bases-file data/universe.txt \
        --start 2025-06-28 --hours 0 8 16
    python -m qr.data.chd mark_price --exchange hyperliquid_futures --bases BTC ETH --start 2025-09-29

Each downloaded hour is reduced to ONE row (the last observation in the hour, plus a few
within-hour statistics) and appended to data/raw/chd_<exchange>/<dtype>/<BASE>-PERP/<day>.
The raw hourly files are not kept; re-run to refetch.
"""
from __future__ import annotations

import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from qr.data.store import append_rows

API = "https://api.cryptohftdata.com/v1"
EXCHANGES = ("binance_futures", "bybit", "hyperliquid_futures")       # the vendor's names; "bybit" is linear perps
DTYPES = ("open_interest", "mark_price", "ticker")
START = {"binance_futures": "2025-06-28", "bybit": "2025-06-28", "hyperliquid_futures": "2025-09-29"}
_S = requests.Session()
_JWT: dict = {"token": None, "exp": 0.0}


def chd_symbol(exchange: str, base: str) -> str:
    """A venue's own base name (or our Binance name) -> this provider's symbol. HL uses bare coin
    names with a capital K for 1000x: our HL raw 'kPEPE' and Binance '1000PEPE' are both 'KPEPE'."""
    if exchange == "hyperliquid_futures":
        if len(base) > 1 and base[0] == "k" and base[1:].isupper():
            return "K" + base[1:]
        return ("K" + base[4:]) if base.startswith("1000") and not base.startswith("1000000") else base
    return f"{base}USDT"


def _jwt() -> str:
    if _JWT["token"] and time.time() < _JWT["exp"] - 300:
        return _JWT["token"]
    key = os.environ.get("CHD_API_KEY")
    if not key:
        raise RuntimeError("set CHD_API_KEY in the environment (never in the repo)")
    r = _S.post(f"{API}/jwt-token", headers={"X-API-Key": key}, timeout=30)
    r.raise_for_status()
    _JWT["token"] = r.json()["jwt_token"]; _JWT["exp"] = time.time() + 4 * 3600
    return _JWT["token"]


def fetch_file(exchange: str, day: str, hour: int, symbol: str, dtype: str) -> pd.DataFrame | None:
    """One hourly Parquet file as a DataFrame; None if the file does not exist (404)."""
    path = f"{exchange}/{day}/{hour:02d}/{symbol}_{dtype}.parquet"
    for attempt in range(6):
        try:
            r = _S.get(f"{API}/download", params={"file": path},
                       headers={"Authorization": f"Bearer {_jwt()}"}, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 ** attempt); continue
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt))); continue
        if r.status_code == 401:
            _JWT["token"] = None; continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt); continue
        r.raise_for_status()
        return _read(r.content)
    raise RuntimeError(f"gave up on {path}")


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _read(content: bytes) -> pd.DataFrame:
    """Older files are served as zstd-compressed Parquet (application/zstd), newer ones as plain
    Parquet. pyarrow decompresses the stream itself; no extra dependency."""
    if content[:4] == _ZSTD_MAGIC:
        import pyarrow as pa
        with pa.input_stream(pa.py_buffer(content), compression="zstd") as f:
            content = f.read()
    return pd.read_parquet(io.BytesIO(content))


def _f(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _event_ts(df: pd.DataFrame, col: str) -> pd.Series:
    """Exchange event times come in ms or ns depending on venue; received_time is always ns."""
    t = _f(df[col])
    unit = "ns" if t.median() > 1e15 else "ms"
    return pd.to_datetime(t, unit=unit, utc=True)


def hourly_summary(df: pd.DataFrame, dtype: str, hour_start: pd.Timestamp) -> dict | None:
    """Reduce one hour's file to one row. 'last' values are the final observation in the hour;
    within-hour stats (premium mean/std, n_rows) are kept where they mean something."""
    if df is None or len(df) == 0:
        return None
    row = {"ts": hour_start + pd.Timedelta(hours=1), "n_rows": int(len(df))}   # labelled at hour END
    if dtype == "open_interest":
        d = df.sort_values("timestamp").drop_duplicates(["symbol", "timestamp"], keep="last")
        oi, oiv = _f(d["sum_open_interest"]).dropna(), _f(d["sum_open_interest_value"]).dropna()
        if oi.empty:
            return None
        row.update({"oi": float(oi.iloc[-1]), "oi_usd": float(oiv.iloc[-1]) if not oiv.empty else float("nan"),
                    "oi_mean": float(oi.mean()), "oi_ts": _event_ts(d.loc[oi.index[-1]:oi.index[-1]], "timestamp").iloc[0]})
    elif dtype == "mark_price":
        d = df.sort_values("received_time")
        mark, idx, fr = _f(d["mark_price"]).ffill(), _f(d["index_price"]).ffill(), _f(d["funding_rate"]).ffill()
        if mark.dropna().empty:
            return None
        prem = (mark / idx - 1).dropna()
        nft = _f(d["next_funding_time"]).dropna()
        row.update({"mark": float(mark.dropna().iloc[-1]), "index": float(idx.dropna().iloc[-1]) if not idx.dropna().empty else float("nan"),
                    "funding_rate": float(fr.dropna().iloc[-1]) if not fr.dropna().empty else float("nan"),
                    "funding_rate_first": float(fr.dropna().iloc[0]) if not fr.dropna().empty else float("nan"),
                    "premium_mean": float(prem.mean()) if not prem.empty else float("nan"),
                    "premium_std": float(prem.std()) if len(prem) > 1 else float("nan"),
                    "premium_last": float(prem.iloc[-1]) if not prem.empty else float("nan"),
                    "next_funding_time": pd.to_datetime(float(nft.iloc[-1]), unit="ms", utc=True) if not nft.empty else pd.NaT})
    elif dtype == "ticker":
        d = df.sort_values("received_time")
        last, qv, bv = _f(d["last_price"]).ffill(), _f(d["quote_asset_volume"]).ffill(), _f(d["base_asset_volume"]).ffill()
        if last.dropna().empty:
            return None
        row.update({"last_price": float(last.dropna().iloc[-1]),
                    "quote_vol_24h": float(qv.dropna().iloc[-1]) if not qv.dropna().empty else float("nan"),
                    "base_vol_24h": float(bv.dropna().iloc[-1]) if not bv.dropna().empty else float("nan"),
                    "high_24h": float(_f(d["high_price"]).dropna().iloc[-1]) if not _f(d["high_price"]).dropna().empty else float("nan"),
                    "low_24h": float(_f(d["low_price"]).dropna().iloc[-1]) if not _f(d["low_price"]).dropna().empty else float("nan")})
    return row


def pull(exchange: str, bases: list[str], dtype: str, start: str, end: str | None = None,
         hours: tuple[int, ...] = tuple(range(24)), workers: int = 8) -> None:
    """One coin at a time: its files in parallel, then written, so a crash loses at most one coin
    and progress is visible. Coins already stored for this dtype are skipped (resume)."""
    from qr.data.store import ROOT
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    print(f"{exchange} {dtype}: {len(bases)} symbols x {len(days)} days x {len(hours)} hours = "
          f"{len(bases) * len(days) * len(hours):,} files", flush=True)
    t0, stored, missing = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        for i, b in enumerate(bases, 1):
            if (ROOT / "raw" / f"chd_{exchange}" / dtype / f"{b}-PERP").exists():
                print(f"  [{i}/{len(bases)}] {b}: already stored, skipped", flush=True)
                continue
            sym = chd_symbol(exchange, b)
            jobs = [(d, h) for d in days for h in hours]
            failed: list[str] = []

            def one(j):
                try:
                    df = fetch_file(exchange, j[0].strftime("%Y-%m-%d"), j[1], sym, dtype)
                except RuntimeError as e:                 # one file that keeps failing must not end the run
                    failed.append(str(e).removeprefix("gave up on "))
                    return None
                return hourly_summary(df, dtype, j[0] + pd.Timedelta(hours=j[1]))
            rows = list(ex.map(one, jobs))
            if failed:
                print(f"  {b}: {len(failed)} files failed after retries, e.g. {failed[0]}", flush=True)
            got = [r for r in rows if r is not None]
            stored += len(got); missing += len(rows) - len(got)
            if got:
                df = pd.DataFrame(got).set_index("ts").sort_index()
                append_rows(df[~df.index.duplicated(keep="last")], "raw", f"chd_{exchange}", dtype, f"{b}-PERP")
            print(f"  [{i}/{len(bases)}] {b}: {len(got):,}/{len(rows):,} hours, {time.time() - t0:,.0f}s elapsed", flush=True)
    print(f"{exchange} {dtype}: done, {stored:,} hours stored, {missing:,} files missing (404)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dtype", choices=DTYPES)
    ap.add_argument("--exchange", required=True, choices=EXCHANGES)
    ap.add_argument("--bases", nargs="*"); ap.add_argument("--bases-file")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--hours", type=int, nargs="*", default=list(range(24)))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    bases = a.bases or [l.strip() for l in open(a.bases_file) if l.strip()]
    pull(a.exchange, bases, a.dtype, a.start or START[a.exchange], a.end, tuple(a.hours), a.workers)
