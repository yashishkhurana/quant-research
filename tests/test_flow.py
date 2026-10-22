import numpy as np
import pandas as pd

from qr.signals.flow import (bar_features, flow_strategy_weights, newey_west_t, ofi_events,
                             signal_study)


def _book(n=20000, seed=5, impact=0.0):
    """Synthetic 100ms snapshots. With impact>0, the mid drifts in the direction of the previous
    snapshot's depth imbalance, so the study should find a positive slope on depth_imb."""
    rng = np.random.default_rng(seed)
    idx = pd.to_datetime("2026-10-22", utc=True) + pd.to_timedelta(np.arange(n) * 100, unit="ms")
    mid = 100.0
    rows, prev_e, imb = [], 0.0, 0.0
    for i in range(n):
        mid += rng.normal(0, 0.002) + impact * prev_e
        spread = 0.01
        imb = 0.995 * imb + rng.normal(0, 0.1)          # persistent imbalance so it predicts the NEXT bar
        qb, qa = np.exp(imb), np.exp(-imb)
        row = {"ts": idx[i]}
        for l in range(10):
            row[f"bid_px_{l}"] = mid - spread / 2 - l * 0.01; row[f"bid_sz_{l}"] = qb * (1 + l)
            row[f"ask_px_{l}"] = mid + spread / 2 + l * 0.01; row[f"ask_sz_{l}"] = qa * (1 + l)
        rows.append(row)
        prev_e = (qb - qa) / (qb + qa)
    book = pd.DataFrame(rows).set_index("ts")
    tr = pd.DataFrame({"px": book["bid_px_0"].values, "qty": rng.lognormal(-1, 1, n),
                       "side": rng.choice([-1, 1], n)}, index=idx)
    return book, tr


def test_ofi_sign_convention():
    book, _ = _book(50)
    b = book.copy()
    b.iloc[1] = b.iloc[0]                                      # identical book...
    b.iloc[1, b.columns.get_loc("bid_px_0")] += 0.001          # ...then the bid lifts -> buying pressure
    e = ofi_events(b)
    assert e.iloc[1] > 0


def test_bar_features_shape():
    book, tr = _book(6000)
    f = bar_features(book, tr, "1min")
    assert {"ofi", "depth_imb", "micro_dev", "spread_bps", "mid", "tfi"} <= set(f.columns)
    assert len(f) == 10


def test_newey_west_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000); y = 0.5 * x + rng.normal(size=5000)
    b, t, r2 = newey_west_t(x, y, lags=3)
    assert abs(b - 0.5) < 0.05 and t > 10 and 0.15 < r2 < 0.35


def test_study_finds_impact_when_present():
    book, tr = _book(60000, impact=0.003)
    f = bar_features(book, tr, "10s")
    s = signal_study(f, features=("depth_imb",), horizons=(1, 5), z_window=60)
    row = s[s.horizon == 1].iloc[0]
    assert row["slope_bps_per_sd"] > 0 and row["t_nw"] > 3


def test_strategy_weights_hold_period():
    book, tr = _book(6000)
    f = bar_features(book, tr, "10s")
    w = flow_strategy_weights(f, "ofi", horizon=3, threshold=0.5, z_window=10)
    assert set(np.unique(w.values)) <= {-1.0, 0.0, 1.0}
    runs = (w != 0).astype(int).diff().fillna(0)
    assert (runs.abs() <= 1).all()
