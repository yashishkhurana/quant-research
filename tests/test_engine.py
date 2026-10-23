import numpy as np
import pandas as pd
import pytest

from qr.backtest.engine import run
from qr.strategies.base import BuyAndHold, Flat


def _prices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    p = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))),
        "B": 50 * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
    }, index=idx)
    return p


def test_flat_strategy_has_zero_pnl_and_cost():
    p = _prices()
    res = run(p, Flat().weights(p), fee_bps=10, slippage_bps=10)
    assert res.returns.abs().sum() == 0
    assert res.costs.sum() == 0
    assert res.equity.iloc[-1] == 1.0


def test_buy_and_hold_single_asset_matches_price_ratio_before_costs():
    p = _prices()[["A"]]
    w = pd.DataFrame(1.0, index=p.index, columns=p.columns)
    res = run(p, w, fee_bps=0, slippage_bps=0)
    expected = p["A"].iloc[-1] / p["A"].iloc[0]
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    # once invested, no rebalancing turnover apart from the first bar
    assert res.turnover.iloc[2:].abs().max() < 1e-12


def test_costs_reduce_equity_and_scale_with_turnover():
    p = _prices()
    w = BuyAndHold(rebalance=1).weights(p)        # rebalance every bar -> some turnover
    cheap = run(p, w, fee_bps=0, slippage_bps=0)
    dear = run(p, w, fee_bps=20, slippage_bps=10)
    assert dear.equity.iloc[-1] < cheap.equity.iloc[-1]
    assert dear.costs.sum() == pytest.approx(dear.turnover.sum() * 30 / 1e4)


def test_no_lookahead_weight_at_t_earns_return_t_to_t1():
    p = _prices(n=10)
    w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    w.iloc[3, 0] = 1.0                            # long A decided at bar 3 only
    res = run(p, w, fee_bps=0, slippage_bps=0)
    r = p["A"].pct_change()
    assert res.gross_returns.iloc[4] == pytest.approx(r.iloc[4])
    assert res.gross_returns.iloc[3] == 0
    assert res.gross_returns.iloc[5] == 0


def test_delisted_asset_is_force_closed_and_charged():
    p = _prices(n=50)
    p.loc[p.index[30]:, "B"] = np.nan               # B delists at bar 30
    w = pd.DataFrame(0.5, index=p.index, columns=p.columns)
    res = run(p, w, fee_bps=10, slippage_bps=0)
    assert (res.weights.loc[p.index[30]:, "B"] == 0).all()
    assert res.returns.iloc[31:].notna().all()
    assert res.costs.iloc[30] > 0                   # the exit is paid for


def test_funding_long_pays_short_receives():
    p = _prices(n=20)
    p[:] = 100.0                                    # flat prices: only funding moves PnL
    idx = p.index
    funding = pd.DataFrame({"A": [0.001], "B": [0.0]}, index=[idx[10]])
    long_w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=idx)
    short_w = -long_w
    long_res = run(p, long_w, 0, 0, funding=funding)
    short_res = run(p, short_w, 0, 0, funding=funding)
    assert long_res.funding_pnl.sum() == pytest.approx(-0.001)
    assert short_res.funding_pnl.sum() == pytest.approx(0.001)


def test_metrics_present():
    p = _prices()
    res = run(p, BuyAndHold(rebalance=12).weights(p))
    for k in ["sharpe", "sortino", "max_drawdown", "ann_turnover", "cost_drag_ann"]:
        assert k in res.metrics


def test_rebalance_bars_hold_drift_between_rebalances():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41, 161.051], "B": 100.0}, index=idx)
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    rb = pd.Series([True, False, False, True, False, False], index=idx)
    res = run(prices, w, fee_bps=0, slippage_bps=0, rebalance_bars=rb)
    # no trading on held bars (bar 0 is the entry trade), A's weight drifts up until bar 3 resets it
    assert res.turnover.iloc[1] == 0 and res.turnover.iloc[2] == 0
    assert res.turnover.iloc[3] > 0
    assert res.weights["A"].iloc[2] > 0.5 and abs(res.weights["A"].iloc[3] - 0.5) < 1e-12
    # a forced exit still trades on a held bar
    prices2 = prices.copy(); prices2.loc[idx[4]:, "B"] = np.nan
    res2 = run(prices2, w, fee_bps=0, slippage_bps=0, rebalance_bars=rb)
    assert res2.weights["B"].iloc[4] == 0 and res2.turnover.iloc[4] > 0


def test_per_column_fees_charge_each_column_its_own_rate():
    p = _prices(50)
    w = pd.DataFrame({"A": 0.5, "B": -0.5}, index=p.index)
    flat = run(p, w, fee_bps=4.0, slippage_bps=0.0)
    per = run(p, w, fee_bps=pd.Series({"A": 4.0, "B": 4.0}), slippage_bps=0.0)
    assert np.allclose(flat.costs, per.costs)                        # same rate either way
    mixed = run(p, w, fee_bps=pd.Series({"A": 2.0, "B": 6.0}), slippage_bps=1.0)
    exp = (mixed.trades["A"] * 3.0 + mixed.trades["B"] * 7.0) / 1e4
    assert np.allclose(mixed.costs, exp)
