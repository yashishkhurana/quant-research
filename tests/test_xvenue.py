import numpy as np
import pandas as pd

from qr.strategies.xvenue import XVenueParams, basis_risk, build_weights, run_with_risk, spread_apr


def _world(n_days=60, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_days * 24, freq="1h", tz="UTC")
    px = pd.DataFrame({b: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for b in ("AAA", "BBB", "CCC")}, index=idx)
    bn_t = idx[idx.hour.isin([0, 8, 16])]
    f_bn = pd.DataFrame({"AAA": 0.0001, "BBB": 0.0001, "CCC": 0.0001}, index=bn_t)          # default everywhere
    f_hl = pd.DataFrame({"AAA": 0.0001 / 8 + 0.0004 / 8,                                    # HL pays 4bp/8h more
                         "BBB": 0.0001 / 8 - 0.0004 / 8,                                    # HL pays 4bp/8h less
                         "CCC": 0.0001 / 8}, index=idx)                                     # same -> no trade
    return px, f_bn, f_hl


def test_spread_is_per_8h_window_and_hourly_hl_is_summed():
    px, f_bn, f_hl = _world()
    s = spread_apr(f_bn, f_hl, lookback=3).dropna().iloc[3:]   # skip the partial first window's influence
    assert np.allclose(s["AAA"], 0.0004 * 1095, rtol=1e-6)
    assert np.allclose(s["BBB"], -0.0004 * 1095, rtol=1e-6)
    assert np.allclose(s["CCC"], 0.0, atol=1e-12)


def test_direction_short_the_expensive_venue_and_pair_is_neutral():
    px, f_bn, f_hl = _world()
    w = build_weights(px, f_bn, f_hl, XVenueParams(lookback=3, entry_apr=0.10))
    late = w.iloc[200:]
    assert (late["AAA-HL"] < 0).all() and (late["AAA-BN"] > 0).all()   # HL pays more: short HL
    assert (late["BBB-HL"] > 0).all() and (late["BBB-BN"] < 0).all()   # HL pays less: long HL
    assert (w["CCC-HL"] == 0).all() and (w["CCC-BN"] == 0).all()
    assert np.allclose(w.filter(like="-HL").values, -w.filter(like="-BN").values)


def test_collects_spread_on_flat_prices_and_selection_moves_on_window_ends_only():
    px, f_bn, f_hl = _world()
    px[:] = 100.0
    res = run_with_risk(px, f_bn, f_hl, XVenueParams(lookback=3))
    assert res.funding_pnl.sum() > 0 and res.equity.iloc[-1] > 1.0
    changes = (res.weights.diff().abs().sum(axis=1) > 1e-12)
    assert set(res.weights.index[changes].hour) <= {0, 8, 16}


def test_delisting_exits_between_decisions():
    px, f_bn, f_hl = _world()
    px.loc[px.index[500]:, "AAA"] = np.nan                      # delists at bar 500 (not a window end)
    w = build_weights(px, f_bn, f_hl, XVenueParams(lookback=3))
    assert (w["AAA-HL"].iloc[500:] == 0).all() and (w["AAA-HL"].iloc[300:500] != 0).all()


def test_basis_risk_reports_bps():
    idx = pd.date_range("2024-01-01", periods=24 * 10, freq="1h", tz="UTC")
    prem = pd.DataFrame({"AAA": np.sin(np.arange(len(idx)) / 24) * 0.001}, index=idx)
    b = basis_risk(prem)
    assert b.loc["AAA", "worst_day_bps"] <= 20.0 + 1e-9 and b.loc["AAA", "daily_std_bps"] > 0
