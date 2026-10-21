"""End-to-end smoke test on synthetic data: build prices, run buy-and-hold, write a tearsheet.
Run: python scripts/smoke.py   -> reports/smoke_tearsheet.png
"""
import numpy as np
import pandas as pd

from qr.backtest.engine import run, summary
from qr.backtest.report import tearsheet
from qr.strategies.base import BuyAndHold

rng = np.random.default_rng(42)
n = 288 * 60                                            # 60 days of 5-minute bars
idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
prices = pd.DataFrame({s: 100 * np.exp(np.cumsum(rng.normal(0.00001, 0.002, n)))
                       for s in ["BTC-PERP", "ETH-PERP", "SOL-PERP"]}, index=idx)
funding_ts = idx[idx.hour.isin([0, 8, 16]) & (idx.minute == 0)]
funding = pd.DataFrame(rng.normal(0.0001, 0.0002, (len(funding_ts), 3)), index=funding_ts, columns=prices.columns)

res = run(prices, BuyAndHold(rebalance=288).weights(prices), fee_bps=4, slippage_bps=2, funding=funding)
print(summary(res))
print(tearsheet(res, "Buy & hold (synthetic)", out="reports/smoke_tearsheet.png"))
