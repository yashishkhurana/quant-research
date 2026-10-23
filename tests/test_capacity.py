import numpy as np
import pandas as pd

from qr.backtest.engine import run
from qr.capacity import (adv, adv_by_column, cap_weights, daily_vol, deployed_share, impact_drag,
                         pair_cap, static_capacity, uncapped_held)


def _idx(n_days=40):
    return pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")


def test_adv_uses_completed_bars_only_and_is_daily():
    idx = _idx(40)
    v = pd.DataFrame({"A": 1000.0}, index=idx)            # $1000 per hour -> $24k per day
    a = adv(v, window_days=30)
    assert np.isclose(a["A"].iloc[-1], 24_000.0)
    v2 = v.copy(); v2.iloc[-1] = 1e9                       # a huge last bar must not leak into its own ADV
    assert np.isclose(adv(v2, 30)["A"].iloc[-1], 24_000.0)
    assert np.isnan(a["A"].iloc[0])
    v3 = v.copy(); v3.iloc[: 30 * 24] = np.nan              # listed 10 days before the last bar
    assert np.isclose(adv(v3, 30)["A"].iloc[-1], 24_000.0)  # a partial window is not diluted


def test_pair_cap_is_min_of_legs_and_unknown_without_data():
    idx = _idx(2)
    ap = pd.DataFrame({"A": 100.0, "B": 50.0}, index=idx)
    as_ = pd.DataFrame({"A": 10.0}, index=idx)             # B has no spot
    cols = pd.Index(["A-PERP", "A-SPOT", "B-PERP", "C-PERP"])
    c = pair_cap(ap, as_, 0.02, cols)
    assert np.allclose(c["A-PERP"], 0.2) and np.allclose(c["A-SPOT"], 0.2)   # 2% of min(100, 10)
    assert np.allclose(c["B-PERP"], 1.0)                                      # perp-only: 2% of 50
    assert c["C-PERP"].isna().all()                                          # no ADV -> unknown, not zero
    assert (cap_weights(pd.DataFrame({"C-PERP": -0.1}, index=idx), c, 1e6)["C-PERP"] == 0).all()  # ...and not traded


def test_cap_weights_keeps_pair_neutral_and_static_capacity_is_binding_name():
    idx = _idx(1)
    w = pd.DataFrame({"A-PERP": -0.15, "A-SPOT": 0.15, "B-PERP": -0.15, "B-SPOT": 0.15}, index=idx)
    cap = pd.DataFrame({"A-PERP": 3e6, "A-SPOT": 3e6, "B-PERP": 3e5, "B-SPOT": 3e5}, index=idx)
    C = static_capacity(w, cap)
    assert np.allclose(C, 3e5 / 0.15)                      # B binds at $2m
    wb = cap_weights(w, cap, book_size=10e6)               # at $10m: A wants 1.5m (ok), B wants 1.5m (cap 0.3m)
    assert np.allclose(wb["A-PERP"], -0.15) and np.allclose(wb["B-PERP"], -0.03)
    assert np.allclose(wb["B-SPOT"], -wb["B-PERP"])
    assert np.allclose(deployed_share(wb, w), (0.15 * 2 + 0.03 * 2) / 0.6)
    small = cap_weights(w, cap, book_size=1e6)
    assert small.equals(w)                                 # under capacity nothing changes


def test_impact_drag_scales_with_three_halves_power_of_size():
    idx = _idx(1)
    tr = pd.DataFrame({"A-PERP": 0.1}, index=idx)
    sig = pd.DataFrame({"A-PERP": 0.05}, index=idx)
    a = pd.DataFrame({"A-PERP": 1e7}, index=idx)
    d1 = impact_drag(tr, sig, a, book_size=1e6).iloc[0]
    d4 = impact_drag(tr, sig, a, book_size=4e6).iloc[0]
    assert np.isclose(d4 / d1, 2.0)                        # sqrt(4): cost fraction doubles, so $ cost x8
    assert np.isclose(d1, 0.1 * 0.05 * np.sqrt(1e5 / 1e7))
    assert impact_drag(tr * 0, sig, a, 1e6).sum() == 0


def test_engine_trades_sum_to_turnover_and_feed_impact():
    rng = np.random.default_rng(0)
    idx = _idx(10)
    p = pd.DataFrame({"A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))),
                      "B": 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx)
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    res = run(p, w, rebalance_band=0.1)
    assert res.trades is not None
    assert np.allclose(res.trades.sum(axis=1), res.turnover)
    assert (res.trades >= 0).all().all()
    sig = daily_vol(p); a = pd.DataFrame(1e8, index=idx, columns=p.columns)
    drag = impact_drag(res.trades, sig, a, book_size=1e7)
    assert drag.iloc[24 * 8:].sum() > 0 and (drag >= 0).all()


def test_adv_by_column_routes_legs():
    idx = _idx(1)
    ap = pd.DataFrame({"A": 1.0}, index=idx); as_ = pd.DataFrame({"A": 2.0}, index=idx)
    out = adv_by_column(ap, as_, pd.Index(["A-PERP", "A-SPOT", "B-PERP"]))
    assert (out["A-PERP"] == 1.0).all() and (out["A-SPOT"] == 2.0).all() and out["B-PERP"].isna().all()


def test_static_capacity_skips_unknown_caps_and_flags_them():
    idx = _idx(1)[:2]
    w = pd.DataFrame({"A-PERP": [-0.1, -0.1], "B-PERP": [-0.1, 0.0]}, index=idx)
    cap = pd.DataFrame({"A-PERP": [1e6, 1e6], "B-PERP": [np.nan, np.nan]}, index=idx)
    assert np.allclose(static_capacity(w, cap), 1e7)      # B (no cap) is left out of the min, not a $0 cap
    assert uncapped_held(w, cap).tolist() == [True, False]
    assert uncapped_held(w * 0, cap).isna().all()          # flat bars: undefined
