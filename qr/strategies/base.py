"""Strategy interface: a strategy turns data into target weights. Nothing else.

Every strategy in this repo is a function or class with `.weights(data) -> DataFrame` where
the row at t uses only information available at t. The engine applies the weight at t to the
return from t to t+1. If you ever find yourself using `shift(-1)` inside a strategy, stop.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def weights(self, prices: pd.DataFrame, **data) -> pd.DataFrame: ...


class BuyAndHold:
    """Equal-weight long, rebalanced every `rebalance` bars. The sanity benchmark."""
    name = "buy_and_hold"

    def __init__(self, rebalance: int = 1440, leverage: float = 1.0):
        self.rebalance = rebalance
        self.leverage = leverage

    def weights(self, prices: pd.DataFrame, **data) -> pd.DataFrame:
        n = prices.notna().sum(axis=1).replace(0, pd.NA)
        w = prices.notna().astype(float).div(n, axis=0).fillna(0.0) * self.leverage
        # hold between rebalances: forward-fill the weights chosen on rebalance bars
        mask = pd.Series(range(len(w)), index=w.index) % self.rebalance == 0
        return w.where(mask).ffill().fillna(0.0)


class Flat:
    """Zero weights. The backtester must return exactly zero PnL and zero cost on this."""
    name = "flat"

    def weights(self, prices: pd.DataFrame, **data) -> pd.DataFrame:
        return prices * 0.0
