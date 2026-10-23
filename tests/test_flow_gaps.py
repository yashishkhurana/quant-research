import numpy as np
import pandas as pd

from qr.signals.flow import bar_features, flow_strategy_weights, forward_returns, ofi_events, zscore


def _book(index):
    n = len(index)
    rng = np.random.default_rng(0)
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    d = {}
    for i in range(5):
        d[f"bid_px_{i}"] = mid - 0.01 * (i + 1); d[f"ask_px_{i}"] = mid + 0.01 * (i + 1)
        d[f"bid_sz_{i}"] = rng.uniform(1, 2, n); d[f"ask_sz_{i}"] = rng.uniform(1, 2, n)
    return pd.DataFrame(d, index=index)


def _gapped_index():
    a = pd.date_range("2026-01-01 00:00", "2026-01-01 03:00", freq="1s", tz="UTC", inclusive="left")
    b = pd.date_range("2026-01-01 05:00", "2026-01-01 08:00", freq="1s", tz="UTC", inclusive="left")
    return a.append(b)


def test_segments_split_at_missing_bars():
    f = bar_features(_book(_gapped_index()), None, "1min")
    assert f["segment"].nunique() == 2
    assert (f.groupby("segment").apply(lambda g: g.index.to_series().diff().dropna().eq(pd.Timedelta("1min")).all())).all()


def test_ofi_event_after_gap_is_zeroed():
    idx = _gapped_index(); book = _book(idx)
    e = ofi_events(book)
    first_after = idx.get_loc(pd.Timestamp("2026-01-01 05:00", tz="UTC"))
    assert e.iloc[first_after] == 0.0


def test_zscore_and_forward_returns_do_not_cross_gap():
    f = bar_features(_book(_gapped_index()), None, "1min")
    seg = f["segment"]
    z = zscore(f["depth_imb"], 120, seg)
    second = f.index[seg == 1]
    assert z.loc[second[:59]].isna().all()                  # warm-up restarts after the gap
    fr = forward_returns(f["mid"], (5,), seg)
    first = f.index[seg == 0]
    assert fr.loc[first[-5:], 5].isna().all()               # last 5 bars before the gap have no 5-bar return
    assert fr.loc[first[:-5], 5].notna().all()


def test_strategy_never_holds_into_a_gap():
    f = bar_features(_book(_gapped_index()), None, "1min")
    f["depth_imb"] = np.where(np.arange(len(f)) % 7 == 0, 5.0, 0.0) + np.random.default_rng(1).normal(0, 0.01, len(f))
    w = flow_strategy_weights(f, "depth_imb", horizon=30, threshold=1.0, z_window=60)
    last_before_gap = f.index[f["segment"] == 0][-1]
    assert w.loc[last_before_gap] == 0.0
    assert (w != 0).any()
