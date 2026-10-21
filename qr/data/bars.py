"""Bar construction from raw trades: time, volume and dollar bars. Plus resampling of 1m klines."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ohlcv_from_groups(trades: pd.DataFrame, group_key) -> pd.DataFrame:
    g = trades.groupby(group_key)
    out = pd.DataFrame({
        "open": g["px"].first(), "high": g["px"].max(), "low": g["px"].min(), "close": g["px"].last(),
        "volume": g["qty"].sum(), "dollar": (trades["px"] * trades["qty"]).groupby(group_key).sum(),
        "n_trades": g.size(),
        "buy_vol": trades["qty"].where(trades["side"] > 0, 0).groupby(group_key).sum(),
        "ts_end": g.apply(lambda x: x.index[-1]) if not isinstance(group_key, pd.Grouper) else g["px"].last().index,
    })
    return out


def time_bars(trades: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """OHLCV at a fixed clock interval. Empty intervals are dropped (no forward fill of fake prices)."""
    out = _ohlcv_from_groups(trades, pd.Grouper(freq=rule)).drop(columns=["ts_end"])
    out.index.name = "ts"
    return out.dropna(subset=["open"])


def volume_bars(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One bar per `threshold` units of base volume. Index = bar close time."""
    cum = trades["qty"].cumsum()
    bar_id = (cum // threshold).astype(int)
    out = _ohlcv_from_groups(trades, bar_id)
    return out.set_index("ts_end").rename_axis("ts")


def dollar_bars(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One bar per `threshold` quote-currency traded. Closer to i.i.d. returns than time bars."""
    cum = (trades["px"] * trades["qty"]).cumsum()
    bar_id = (cum // threshold).astype(int)
    out = _ohlcv_from_groups(trades, bar_id)
    return out.set_index("ts_end").rename_axis("ts")


def resample_klines(k1m: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "quote_vol": "sum", "n_trades": "sum", "taker_buy_vol": "sum"}
    agg = {k: v for k, v in agg.items() if k in k1m.columns}
    return k1m.resample(rule).agg(agg).dropna(subset=["open"])


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()
