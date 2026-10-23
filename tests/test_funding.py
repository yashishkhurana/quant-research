import numpy as np
import pandas as pd

from qr.data.funding import interval_hours, per_window


def _raw():
    # A settles every 8h, B every 4h, C switches from 4h to 1h on day 2. All pay 0.01% per settlement.
    idx_a = pd.date_range("2024-01-01 08:00", periods=6, freq="8h", tz="UTC")
    idx_b = pd.date_range("2024-01-01 04:00", periods=12, freq="4h", tz="UTC")
    idx_c = pd.date_range("2024-01-01 04:00", periods=6, freq="4h", tz="UTC").append(
        pd.date_range("2024-01-02 01:00", periods=24, freq="1h", tz="UTC"))
    a = pd.Series(0.0001, idx_a, name="A")
    b = pd.Series(0.0001, idx_b, name="B")
    c = pd.Series(0.0001, idx_c, name="C")
    return pd.concat([a, b, c], axis=1, sort=True)


def test_per_window_puts_every_interval_on_8h():
    f8 = per_window(_raw())
    assert np.isclose(f8["A"].dropna(), 0.0001).all()       # one 8h settlement per window
    assert np.isclose(f8["B"].dropna(), 0.0002).all()       # two 4h settlements per window
    assert np.isclose(f8.loc["2024-01-02 08:00", "C"], 0.0008)  # eight 1h settlements


def test_window_label_is_the_window_end_so_value_is_known_at_label():
    f8 = per_window(_raw())
    # B's settlements at 04:00 and 08:00 belong to the window labelled 08:00, not 16:00
    assert np.isclose(f8.loc["2024-01-01 08:00", "B"], 0.0002)
    assert f8.index.min() == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_interval_hours():
    iv = interval_hours(_raw())
    assert set(iv["A"].dropna().unique()) == {8.0}
    assert set(iv["B"].dropna().unique()) == {4.0}
    assert {4.0, 1.0} <= set(iv["C"].dropna().unique())
