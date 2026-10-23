import numpy as np
import pandas as pd

from qr.backtest.engine import run
from qr.data.prices import mask_delisted


def test_frozen_perp_after_delisting_is_cut_and_force_closed():
    idx = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    perp = pd.DataFrame({"X": [10, 10.5, 11, 11, 11, 11, 11, 11, 11, 11.0]}, index=idx)   # frozen from bar 3
    spot = pd.DataFrame({"X": [10, 10.5, 11, 12, 13, 14, 15, 16, 17, 18.0]}, index=idx)   # spot keeps trading
    cut = mask_delisted(perp, {"X": idx[3]})
    assert cut["X"].iloc[:3].notna().all() and cut["X"].iloc[3:].isna().all()

    prices = pd.DataFrame({"X-SPOT": spot["X"], "X-PERP": cut["X"]})
    both = prices.notna().all(axis=1)
    w = pd.DataFrame({"X-SPOT": 0.5, "X-PERP": -0.5}, index=idx).where(both, 0.0)   # hedged book exits both legs
    res = run(prices, w, fee_bps=0, slippage_bps=0)
    assert (res.weights.iloc[3:] == 0).all().all()
    # without the cut, the frozen perp leaves the spot leg unhedged and the book books spot's rally
    w_frozen = pd.DataFrame({"X-SPOT": 0.5, "X-PERP": -0.5}, index=idx)
    res_frozen = run(pd.DataFrame({"X-SPOT": spot["X"], "X-PERP": perp["X"]}), w_frozen, fee_bps=0, slippage_bps=0)
    assert res_frozen.equity.iloc[-1] > res.equity.iloc[-1] + 0.1
