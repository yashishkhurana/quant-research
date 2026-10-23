import pandas as pd

from qr.data.venues import canon_map, canonical


def test_canonical_strips_size_prefixes_per_venue():
    assert canonical("kPEPE", "HL") == ("PEPE", 1000.0)
    assert canonical("1000PEPE", "BN") == ("PEPE", 1000.0)
    assert canonical("10000SATS", "BY") == ("SATS", 10000.0)
    assert canonical("1MBABYDOGE", "BN") == ("BABYDOGE", 1e6)
    assert canonical("1INCH", "BN") == ("1INCH", 1.0)          # a digit in the name is not a prefix
    assert canonical("BTC", "HL") == ("BTC", 1.0)


def test_canon_map_keeps_colliding_contracts_apart():
    m = canon_map(["BOB", "1000000BOB", "1000PEPE"], "BN")
    assert m == {"BOB": "BOB", "1000000BOB": "1000000BOB", "1000PEPE": "PEPE"}


def test_price_matched_needs_agreement_most_of_the_time_not_just_at_the_median():
    import numpy as np, pandas as pd
    from qr.data.venues import Venue, price_matched
    idx = pd.date_range("2026-01-01", periods=24 * 20, freq="h", tz="UTC")
    a = pd.DataFrame({"H": 1.0, "OK": 1.0}, index=idx)
    b = a.copy(); b.iloc[: 24 * 3, 0] = 2.4                 # 15% of hours 2.4x apart; median still 1.0
    va, vb = Venue("BN", pd.DataFrame(), a), Venue("BY", pd.DataFrame(), b)
    assert price_matched(va, vb) == ["OK"]


def test_cached_matrices_are_utc(tmp_path, monkeypatch):
    import qr.data.venues as v
    monkeypatch.setattr(v, "_cache", lambda *a: tmp_path / "m.parquet")
    naive = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2026-03-29", periods=2, freq="h"))
    built = v._cached("HL", "close", "s", "e", lambda: naive, 1)
    reread = v._cached("HL", "close", "s", "e", lambda: None, 1)
    for m in (built, reread):
        assert str(m.index.tz) == "UTC"
        assert m.reindex(pd.date_range("2026-03-29", periods=2, freq="h", tz="UTC"))["A"].notna().all()
