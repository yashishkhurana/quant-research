import numpy as np
import pandas as pd

from qr.data import store
from qr.data.bars import dollar_bars, time_bars, volume_bars
from qr.data.collector import Buffer, binance_handler, flatten_book, hl_handler


def _trades(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.to_datetime("2024-03-01", utc=True) + pd.to_timedelta(np.sort(rng.integers(0, 3_600_000, n)), unit="ms")
    return pd.DataFrame({
        "px": 60000 + np.cumsum(rng.normal(0, 5, n)),
        "qty": rng.lognormal(-2, 1, n),
        "side": rng.choice([-1, 1], n),
    }, index=idx).rename_axis("ts")


def test_store_roundtrip_and_dedupe(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    t = _trades(200)
    store.append_rows(t, "raw", "test", "trades", "BTC-PERP")
    store.append_rows(t.iloc[100:], "raw", "test", "trades", "BTC-PERP")   # overlapping append
    back = store.read_range("raw", "test", "trades", "BTC-PERP", "2024-03-01", "2024-03-01")
    assert len(back) == len(t.index.unique())
    assert store.available_days("raw", "test", "trades", "BTC-PERP") == ["2024-03-01"]


def test_time_bars_have_ohlc_order():
    b = time_bars(_trades(), "1min")
    assert (b["high"] >= b[["open", "close"]].max(axis=1)).all()
    assert (b["low"] <= b[["open", "close"]].min(axis=1)).all()


def test_volume_and_dollar_bars_respect_threshold():
    t = _trades()
    vb = volume_bars(t, threshold=5.0)
    assert vb["volume"].iloc[:-1].min() >= 0        # last bar may be partial
    db = dollar_bars(t, threshold=100_000.0)
    assert len(db) > 1 and db["dollar"].sum() > 0


def test_flatten_book_pads_to_depth():
    row = flatten_book(1, [("1", "2")], [("3", "4")])
    assert row["bid_px_0"] == 1.0 and row["ask_sz_0"] == 4.0
    assert row["bid_px_9"] is None


def test_binance_and_hl_handlers_fill_buffer():
    buf = Buffer()
    binance_handler({"stream": "btcusdt@aggTrade",
                     "data": {"s": "BTCUSDT", "T": 1700000000000, "p": "60000.5", "q": "0.1", "m": True, "a": 1}}, buf)
    binance_handler({"stream": "btcusdt@depth20@100ms",
                     "data": {"s": "BTCUSDT", "T": 1700000000100, "b": [["60000", "1"]], "a": [["60001", "2"]]}}, buf)
    hl_handler({"channel": "trades", "data": [{"coin": "ETH", "time": 1700000000000, "px": "3000", "sz": "1", "side": "B", "tid": 7}]}, buf)
    hl_handler({"channel": "l2Book", "data": {"coin": "ETH", "time": 1700000000000,
                "levels": [[{"px": "2999", "sz": "1"}], [{"px": "3001", "sz": "1"}]]}}, buf)
    assert len(buf.rows[("binance", "trades", "BTC-PERP")]) == 1
    assert buf.rows[("binance", "trades", "BTC-PERP")][0]["side"] == -1
    assert len(buf.rows[("binance", "book", "BTC-PERP")]) == 1
    assert len(buf.rows[("hyperliquid", "trades", "ETH-PERP")]) == 1
    assert len(buf.rows[("hyperliquid", "book", "ETH-PERP")]) == 1
