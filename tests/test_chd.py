import numpy as np
import pandas as pd

from qr.data.chd import chd_symbol, hourly_summary

H = pd.Timestamp("2026-10-01 08:00", tz="UTC")


def test_symbol_mapping():
    assert chd_symbol("binance_futures", "BTC") == "BTCUSDT"
    assert chd_symbol("bybit", "1000PEPE") == "1000PEPEUSDT"
    assert chd_symbol("hyperliquid_futures", "kPEPE") == "KPEPE"              # our HL raw name
    assert chd_symbol("hyperliquid_futures", "BTC") == "BTC"
    assert chd_symbol("hyperliquid_futures", "1000PEPE") == "KPEPE"
    assert chd_symbol("hyperliquid_futures", "1000000MOG") == "1000000MOG"   # no K rule for 1e6 contracts


def test_open_interest_summary_takes_last_and_dedupes():
    ts = (H.value // 10**6) + np.array([0, 60_000, 60_000, 120_000])          # ms, with a duplicate
    df = pd.DataFrame({"symbol": "BTCUSDT", "timestamp": ts, "received_time": ts * 10**6,
                       "sum_open_interest": ["10", "11", "11", "12"], "sum_open_interest_value": ["1000", "1100", "1100", "1200"]})
    r = hourly_summary(df, "open_interest", H)
    assert r["ts"] == H + pd.Timedelta(hours=1) and r["oi"] == 12.0 and r["oi_usd"] == 1200.0
    assert np.isclose(r["oi_mean"], (10 + 11 + 12) / 3)


def test_mark_price_summary_premium_and_funding():
    rt = np.arange(5) * 10**9 + H.value
    df = pd.DataFrame({"symbol": "BTCUSDT", "received_time": rt, "event_time": rt // 10**6,
                       "mark_price": ["100", "101", None, "102", "103"], "index_price": ["100", "100", "100", "100", "100"],
                       "funding_rate": ["0.0001", None, "0.0002", "0.0002", "0.0003"], "next_funding_time": [H.value // 10**6] * 5,
                       "estimated_settle_price": [None] * 5})
    r = hourly_summary(df, "mark_price", H)
    assert r["mark"] == 103.0 and r["funding_rate"] == 0.0003 and r["funding_rate_first"] == 0.0001
    assert np.isclose(r["premium_last"], 0.03) and np.isclose(r["premium_mean"], np.mean([0, .01, .01, .02, .03]))


def test_empty_file_gives_none():
    assert hourly_summary(None, "ticker", H) is None
    assert hourly_summary(pd.DataFrame(), "ticker", H) is None


def test_reads_plain_and_zstd_parquet():
    import io
    import pyarrow as pa
    from qr.data.chd import _read
    df = pd.DataFrame({"x": [1, 2, 3]})
    buf = io.BytesIO(); df.to_parquet(buf); raw = buf.getvalue()
    out = pa.BufferOutputStream()
    with pa.CompressedOutputStream(out, "zstd") as f:
        f.write(raw)
    for content in (raw, out.getvalue().to_pybytes()):
        assert _read(content)["x"].tolist() == [1, 2, 3]
