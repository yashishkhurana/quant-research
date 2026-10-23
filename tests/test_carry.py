import numpy as np
import pandas as pd

from qr.strategies.carry import CarryParams, build_weights, kill_mask, run_with_risk, point_in_time_universe


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


def test_kill_mask_flattens_after_breach_then_re_enters():
    # equity: up 20%, down 17% (breach), then recovers. The old mask never let the book back in
    # because drawdown stayed below threshold against the all-time high while flat.
    eq = np.r_[np.linspace(1, 1.2, 50), np.linspace(1.2, 1.0, 50), np.linspace(1.0, 1.3, 100)]
    r = pd.Series(np.diff(eq, prepend=1.0) / np.r_[1.0, eq[:-1]])
    p = CarryParams(kill_dd=0.10, cooldown_bars=20)
    m = kill_mask(r, p)
    assert m.any() and not m[:50].any()
    first_kill = int(np.argmax(m))
    assert m[first_kill:first_kill + 20].all()
    assert not m[first_kill + 20:first_kill + 40].any()      # back in after cooldown
    assert m.sum() < 60                                      # not flat for the rest of the sample


def test_kill_switch_re_enters_in_full_pipeline():
    prices, funding = _world(n_days=120)
    prices[:] = 100.0
    # a 10% one-day loss on AAA spot at day 30 while the perp is flat -> drawdown on the hedged book
    prices.loc[prices.index[30 * 24]:prices.index[30 * 24 + 23], "AAA-SPOT"] = 90.0
    prices.loc[prices.index[30 * 24 + 24]:, "AAA-SPOT"] = 90.0
    p = CarryParams(lookback=3, kill_dd=0.02, cooldown_bars=24, max_weight=0.5, max_gross=2.0)
    res = run_with_risk(prices, funding, p)
    assert res.metrics["kill_events"] >= 1
    active = (res.weights.abs().sum(axis=1) > 0).values
    assert active[-24:].any()                                # trading again by the end


def test_rebalance_band_cuts_turnover():
    prices, funding = _world(n_days=60)
    p0 = CarryParams(lookback=3, rebalance_band=0.0)
    p1 = CarryParams(lookback=3, rebalance_band=0.25)
    t0 = run_with_risk(prices, funding, p0).turnover.iloc[48:].mean()
    t1 = run_with_risk(prices, funding, p1).turnover.iloc[48:].mean()
    target_changes = build_weights(prices, funding, p1).diff().abs().sum(axis=1).iloc[48:].mean()
    assert t1 < t0                                           # drift is no longer traded...
    assert abs(t1 - target_changes) < 0.1 * target_changes   # ...and what remains is signal changes


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


def test_unhedged_has_no_spot_leg_and_trades_perp_only_coins():
    prices, funding = _world()
    prices = prices.drop(columns=["CCC-SPOT"])          # CCC has no spot market
    funding["CCC"] = 0.0006                               # but pays well
    w_h = build_weights(prices, funding, CarryParams(lookback=3, hedged=True))
    w_u = build_weights(prices, funding, CarryParams(lookback=3, hedged=False))
    assert (w_h["CCC-PERP"] == 0).all()                   # hedged: cannot trade it
    assert (w_u["CCC-PERP"].iloc[50:] < 0).all()          # unhedged: shorts it
    assert (w_u.filter(like="-SPOT") == 0).all().all()


def test_universe_hysteresis_stops_boundary_flipping():
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    v = pd.DataFrame({"A": 100.0, "B": 100.0 + rng.normal(0, 30, 200), "C": 100.0 + rng.normal(0, 30, 200)}, index=idx)
    plain = point_in_time_universe(v, n=2, window=1)
    hyst = point_in_time_universe(v, n=2, window=1, exit_n=3)
    flips_plain = plain.astype(int).diff().abs().sum().sum()
    flips_hyst = hyst.astype(int).diff().abs().sum().sum()
    assert flips_hyst < flips_plain
