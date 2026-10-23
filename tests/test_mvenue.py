import numpy as np
import pandas as pd

from qr.strategies.mvenue import MVParams, build_weights, expected_hold, pair_spreads, round_trip_bps, spell_ends


def _bars(days):
    return pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")


def _setup(days=120, hl_rate=0.0003, coins=("A", "B")):
    """BN pays the 0.01%/8h default, HL pays hl_rate per 8h (as 8 hourly settlements)."""
    idx8 = pd.date_range("2024-01-01", periods=3 * days, freq="8h", tz="UTC")
    idx1 = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    bn = pd.DataFrame({c: 0.0001 for c in coins}, index=idx8)
    hl = pd.DataFrame({c: hl_rate / 8 for c in coins}, index=idx1)
    px = pd.DataFrame({f"{c}-{v}": 100.0 for c in coins for v in ("BN", "HL")}, index=_bars(days))
    return {"BN": bn, "HL": hl}, px


def test_round_trip_counts_both_legs_both_ways():
    assert round_trip_bps("BN", "HL", 3.0) == 2 * (4.0 + 3) + 2 * (4.5 + 3)        # 29 bps


def test_direction_short_the_venue_that_pays_more():
    f, px = _setup(hl_rate=0.0005)
    w, _ = build_weights(px, f, MVParams(rule="raw", entry_apr=0.10))
    last = w.iloc[-1]
    assert last["A-HL"] < 0 and last["A-BN"] > 0 and np.isclose(last["A-HL"], -last["A-BN"])


def test_spell_ends_and_expected_hold_are_point_in_time():
    idx = pd.date_range("2024-01-01", periods=12, freq="8h", tz="UTC")
    t = pd.DataFrame({"A": [0, .001, .001, .001, 0, 0, .001, .001, 0, 0, 0, 0]}, index=idx)
    e = spell_ends(t, spell_apr=0.10)
    assert e["length"].tolist() == [3, 2] and e["end"].tolist() == [idx[4], idx[8]]
    eh = expected_hold({("BN", "HL"): t}, idx, MVParams(min_spells=1, hold_window_days=30, spell_apr=0.10))
    assert np.isnan(eh.iloc[3])                     # no spell has ended yet
    assert eh.iloc[4] == 3 and eh.iloc[7] == 3      # only the first spell is known before idx[8]
    assert eh.iloc[8] == 2.5


def test_net_rule_waits_for_enough_spells_and_needs_spread_to_pay():
    f, px = _setup(hl_rate=0.0005)                  # constant spread: one spell that never ends
    w, d = build_weights(px, f, MVParams(rule="net", min_spells=1))
    assert d["expected_hold"].isna().all() and (w == 0).all().all()     # no ended spell -> no estimate -> no trade


def test_net_rule_rejects_spread_that_cannot_cover_costs():
    # spread alternates on/off in short spells: E[hold] is short, and a 0.2 bp/window gap never pays 29 bps
    idx8 = pd.date_range("2024-01-01", periods=3 * 120, freq="8h", tz="UTC")
    on = (np.arange(len(idx8)) // 3) % 2 == 0
    bn = pd.DataFrame({"A": 0.0001, "B": 0.0001}, index=idx8)
    hl8 = pd.DataFrame({"A": np.where(on, 0.0001 + 0.00002, 0.0001), "B": np.where(on, 0.0001 + 0.00002, 0.0001)}, index=idx8)
    px = pd.DataFrame({f"{c}-{v}": 100.0 for c in "AB" for v in ("BN", "HL")}, index=_bars(120))
    p = MVParams(rule="net", lookback=1, spell_apr=0.01, min_spells=2, exit_apr=0.0)
    w, d = build_weights(px, {"BN": bn, "HL": hl8}, p)
    assert d["expected_hold"].notna().any()
    assert (w == 0).all().all()
    raw, _ = build_weights(px, {"BN": bn, "HL": hl8}, MVParams(rule="raw", lookback=1, entry_apr=0.01, exit_apr=0.0))
    assert (raw != 0).any().any()                   # the raw rule would have traded it


def test_missing_price_exits_until_next_decision():
    f, px = _setup(hl_rate=0.0005)
    px.loc[px.index[100]:px.index[102], "A-HL"] = np.nan
    w, _ = build_weights(px, f, MVParams(rule="raw"))
    assert (w.loc[px.index[100]:px.index[103], ["A-BN", "A-HL"]] == 0).all().all()   # 103 is not a decision bar
    assert w["A-HL"].iloc[-1] < 0                   # re-entered later


def test_expected_remaining_conditions_on_spell_age():
    from qr.strategies.mvenue import expected_remaining, spell_ages
    idx = pd.date_range("2024-01-01", periods=20, freq="8h", tz="UTC")
    # three spells, lengths 2, 4, 6, all ended by idx[18]
    x = [0, .001, .001, 0, .001, .001, .001, .001, 0, .001, .001, .001, .001, .001, .001, 0, 0, 0, 0, 0]
    t = pd.DataFrame({"A": x}, index=idx)
    er = expected_remaining({("BN", "HL"): t}, idx, MVParams(min_spells=1, hold_window_days=30, spell_apr=0.10), kmax=8)
    last = er[-1]
    assert np.isclose(last[0], 4.0)                 # unconditional mean length (2+4+6)/3
    assert np.isclose(last[1], 3.0)                 # age 1: all three continue, mean L-1
    assert np.isclose(last[2], 3.0)                 # age 2: only 4 and 6 are longer -> (2+4)/2
    assert np.isclose(last[5], 1.0)                 # age 5: only the 6 -> 1
    assert np.isnan(last[6])                        # no ended spell lasted longer than 6
    ag = spell_ages(t, 0.10)["A"].tolist()
    assert ag[:6] == [0, 1, 2, 0, 1, 2]
