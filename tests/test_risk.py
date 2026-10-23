import numpy as np
import pandas as pd

from qr.risk import adl_exposure, liquidation_mask, venue_basis, venue_gross_cap, vol_kill_mask


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


def test_vol_kill_scales_with_book_vol():
    rng = np.random.default_rng(0)
    n = 24 * 60
    calm = pd.Series(rng.normal(0, 0.0005, n), index=_idx(n))
    wild = calm * 20
    hit = -0.03                                           # the same 3% loss on day 40
    for s in (calm, wild):
        s.iloc[24 * 40] = hit
    assert vol_kill_mask(calm, k=3).any()                 # 3% is many sigma for the calm book: stop
    m = vol_kill_mask(wild, k=3)
    assert not m[24 * 40: 24 * 40 + 5].any()              # ...and noise for the wild one: no stop there


def test_vol_kill_uses_only_past_vol():
    n = 24 * 30
    r = pd.Series(0.0, index=_idx(n)); r.iloc[24 * 20] = -0.05
    early = vol_kill_mask(r.iloc[: 24 * 20 + 1])
    full = vol_kill_mask(r)
    assert (early == full[: 24 * 20 + 1]).all()           # the decision at t does not change with later data


def test_venue_gross_cap_keeps_pairs_neutral():
    idx = _idx(2)
    w = pd.DataFrame({"A-BN": [0.5, 0.2], "A-HL": [-0.5, -0.2], "B-BN": [0.5, 0.2], "B-BY": [-0.5, -0.2]}, index=idx)
    c = venue_gross_cap(w, cap=0.5)
    assert np.allclose(c["A-BN"], -c["A-HL"]) and np.allclose(c["B-BN"], -c["B-BY"])
    assert c[["A-BN", "B-BN"]].abs().sum(axis=1).max() <= 0.5 + 1e-12   # BN carried 1.0, now 0.5
    assert np.allclose(c.iloc[1], w.iloc[1])               # under the cap nothing changes


def test_liquidation_closes_both_legs_until_episode_ends():
    idx = _idx(6)
    p = pd.DataFrame({"A-BN": [100, 110, 130, 150, 150, 150.0], "A-HL": [100, 110, 130, 150, 150, 150.0]}, index=idx)
    w = pd.DataFrame({"A-BN": [0.5] * 4 + [0, 0.5], "A-HL": [-0.5] * 4 + [0, -0.5]}, index=idx)
    w2, ev = liquidation_mask(p, w, leverage=3.0, maintenance=0.01)   # threshold 32.3%: short leg hit at 150
    assert len(ev) == 1 and ev.iloc[0]["leg"] == "A-HL"
    assert (w2.iloc[3] == 0).all()                         # both legs closed on the breach bar
    assert (w2.iloc[5] == w.iloc[5]).all()                 # a new episode re-enters


def test_adl_exposure_counts_profitable_leg_in_shock():
    idx = _idx(4)
    p = pd.DataFrame({"A-HL": [100, 100, 80, 80.0]}, index=idx)          # -20% hour; short is +20% in profit
    w = pd.DataFrame({"A-HL": -0.5}, index=idx)
    e = adl_exposure(p, w, shock_pct=0.10, profit_pct=0.10).set_index("leg")
    assert e.loc["A-HL", "adl_risk_hours"] == 1


def test_venue_basis_uses_multipliers():
    idx = _idx(24 * 10)
    a = pd.DataFrame({"PEPE": 0.01}, index=idx)            # per 1000 PEPE
    b = pd.DataFrame({"PEPE": 0.00001}, index=idx)         # per 1 PEPE
    vb = venue_basis(a, b, {"PEPE": 1000.0}, {"PEPE": 1.0})
    assert abs(vb.loc["PEPE", "mean_bps"]) < 1e-6
