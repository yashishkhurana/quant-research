import numpy as np
import pandas as pd

from qr.strategies.carry import CarryParams, build_weights, kill_mask, run_with_risk


def _world(n_days=60, bases=("AAA", "BBB", "CCC"), seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    prices = {}
    for b in bases:
        spot = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        basis = 1 + rng.normal(0, 0.0005, len(idx))
        prices[f"{b}-SPOT"] = spot
        prices[f"{b}-PERP"] = spot * basis
    prices = pd.DataFrame(prices, index=idx)
    sett = idx[idx.hour.isin([0, 8, 16])]
    funding = pd.DataFrame({
        "AAA": 0.0005,                       # 55% APR: always on
        "BBB": -0.0002,                      # negative: never on
        "CCC": rng.normal(0.0001, 0.0003, len(sett)),
    }, index=sett)
    return prices, funding


def test_weights_are_delta_neutral_and_respect_caps():
    prices, funding = _world()
    p = CarryParams(max_weight=0.3, max_gross=2.0)
    w = build_weights(prices, funding, p)
    for b in ["AAA", "BBB", "CCC"]:
        assert np.allclose(w[f"{b}-SPOT"], -w[f"{b}-PERP"])
    assert (w.abs().max() <= 0.3 + 1e-12).all()
    assert (w.abs().sum(axis=1) <= 2.0 + 1e-9).all()
    assert (w["BBB-PERP"] == 0).all()                # negative funding never enters
    assert (w["AAA-PERP"].iloc[200:] < 0).all()      # high funding always short perp


def test_no_lookahead_in_funding_signal():
    prices, funding = _world()
    p = CarryParams(lookback=3, entry_apr=0.1)
    w = build_weights(prices, funding, p)
    # before enough settlements have happened, nothing is on
    first_on = w["AAA-PERP"].ne(0).idxmax()
    assert first_on >= funding.index[1]


def test_carry_collects_funding_on_flat_prices():
    prices, funding = _world()
    prices[:] = 100.0
    p = CarryParams(lookback=3)
    res = run_with_risk(prices, funding, p)
    assert res.funding_pnl.sum() > 0
    assert res.equity.iloc[-1] > 1.0


def test_kill_mask_flattens_after_breach_and_grows_monotonically():
    eq = pd.Series(np.r_[np.linspace(1, 1.2, 50), np.linspace(1.2, 1.0, 50), np.linspace(1.0, 1.3, 50)])
    p = CarryParams(kill_dd=0.10, cooldown_bars=20)
    m1 = kill_mask(eq, p)
    assert m1.any() and not m1[:50].any()
    m2 = kill_mask(eq, p, m1)
    assert (m2 >= m1).all()


def test_point_in_time_universe_uses_only_past_volume():
    from qr.strategies.carry import point_in_time_universe
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    v = pd.DataFrame({"A": 10.0, "B": 1.0, "C": np.nan}, index=idx)
    v.loc[idx[60]:, "B"] = 100.0                       # B becomes liquid at bar 60
    m = point_in_time_universe(v, n=1, window=10)
    assert m["A"].iloc[30] and not m["B"].iloc[30]
    assert m["B"].iloc[90] and not m["A"].iloc[90]
    assert not m["B"].iloc[59]                          # the bar before it turns liquid: not yet
    assert not m["C"].any()
