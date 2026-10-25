"""Regime-filtered hedged carry at size on Binance. Parameters in notes/04_regime_params.md,
fixed before this was run. Question: does a $10m hedged book pay when it holds only sizable names,
only while the cross-section is paying above the default, at maker fees, rebalanced monthly?

    python scripts/regime_carry.py

Writes reports/regime_carry/: summary.csv (the 2x2: regime {filtered, always-on} x fees {maker, taker},
by period), regime.csv (the monthly switch series with the universe funding behind it),
tearsheet_*.png, and note_autogen.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qr.backtest.engine import run
from qr.backtest.report import tearsheet
from qr.capacity import adv, pair_cap, static_capacity
from qr.data.funding import WINDOWS_PER_YEAR, has_spot, load_funding, per_window
from qr.data.prices import load_kline_matrix
from qr.risk import vol_kill_mask

START, END = "2023-01-01", "2026-10-23"
DEFAULT_APR = 0.0001 * WINDOWS_PER_YEAR            # 10.95%
PARTICIPATION, LEG_USD = 0.02, 1_000_000           # 2% of min(spot, perp ADV) must cover $1m a leg
MAX_NAMES, MAX_LEG = 20, 0.10
FEES = {"maker": {"perp": 2.0, "spot": 3.0, "slip": 1.0}, "taker": {"perp": 4.0, "spot": 4.0, "slip": 3.0}}
PERIODS = {"2023-24": ("2023-01-01", "2024-12-31 23:00"), "2025-26": ("2025-01-01", END), "full": (START, END)}
OUT = Path("reports/regime_carry"); OUT.mkdir(parents=True, exist_ok=True)


def build():
    u = [l.strip() for l in open("data/universe.txt") if l.strip()]
    H = has_spot(u)
    perp = load_kline_matrix(H, "PERP", "close", START, END)
    spot = load_kline_matrix(H, "SPOT", "close", START, END)
    dv_perp = load_kline_matrix(H, "PERP", "quote_vol", START, END)
    dv_spot = load_kline_matrix(H, "SPOT", "quote_vol", START, END)
    bases = sorted(set(perp.columns) & set(spot.columns))
    idx = perp.index.union(spot.index)
    perp, spot, dv_perp, dv_spot = (x.reindex(index=idx, columns=bases) for x in (perp, spot, dv_perp, dv_spot))
    raw = load_funding(bases, START, END)
    f8 = per_window(raw)                                                  # 8h windows, label = window end
    f30 = f8.rolling(90, min_periods=45).mean() * WINDOWS_PER_YEAR        # trailing 30 days, annualised, known at window end
    f30 = f30.reindex(idx, method="ffill")                                # last completed window at each bar
    a_perp, a_spot = adv(dv_perp), adv(dv_spot)                           # completed bars only
    sizable = (PARTICIPATION * np.minimum(a_perp, a_spot) >= LEG_USD) & perp.notna() & spot.notna()
    return perp, spot, raw, f30, sizable, a_perp, a_spot, idx, bases


def weights(perp, spot, f30, sizable, idx, bases, regime_filter: bool):
    per = idx.tz_localize(None).to_period("M")
    month_start = pd.Series(np.r_[True, per[1:] != per[:-1]], index=idx)
    both = perp.notna() & spot.notna()
    sel = pd.DataFrame(False, index=idx, columns=bases)
    regime_rows = []
    for t in idx[month_start.values]:
        S = sizable.loc[t]
        S = S[S].index
        univ_f = f30.loc[t, S].dropna()
        mean_excess = float(univ_f.mean() - DEFAULT_APR) if len(univ_f) else np.nan
        on = (mean_excess > 0) if regime_filter else True
        paying = univ_f[univ_f > DEFAULT_APR].index if len(univ_f) else []
        chosen = list(paying[:MAX_NAMES]) if on else []
        if on and len(paying) > MAX_NAMES:                              # cap at 20: the highest trailing funding
            chosen = list(univ_f.loc[paying].nlargest(MAX_NAMES).index)
        sel.loc[t, chosen] = True
        regime_rows.append({"month": t, "sizable_n": int(len(S)), "universe_mean_apr": float(univ_f.mean()) if len(univ_f) else np.nan,
                            "mean_excess_apr": mean_excess, "on": bool(on), "paying_n": int(len(paying)), "held_n": len(chosen)})
    keep = np.broadcast_to(month_start.values[:, None], sel.shape)
    sel = sel.astype(float).where(keep).ffill().fillna(0.0).astype(bool) & both
    n = sel.sum(axis=1).replace(0, np.nan)
    leg = (1.0 / n).clip(upper=MAX_LEG)
    w = pd.DataFrame(0.0, index=idx, columns=[f"{b}-{m}" for b in bases for m in ("SPOT", "PERP")])
    for b in bases:
        w[f"{b}-SPOT"] = sel[b].astype(float).mul(leg, axis=0).fillna(0.0)
        w[f"{b}-PERP"] = -w[f"{b}-SPOT"]
    gross = w.abs().sum(axis=1); w = w.mul((2.0 / gross).clip(upper=1.0).fillna(1.0), axis=0)
    return w, month_start, pd.DataFrame(regime_rows).set_index("month")


OI_SHARE, OI_FROM = 0.01, "2025-07-01"


def oi_capped(cap: pd.DataFrame, w: pd.DataFrame, idx, bases):
    """Tighten both legs of a base to 1% of its Binance perp OI where the vendor OI history exists
    (qr.data.chd; ~110 most-held coins, from 2025-06-28). Elsewhere the volume cap stands."""
    from qr.data.venues import _matrix, _names_on_disk
    have = set(_names_on_disk("chd_binance_futures", "open_interest"))
    oi = _matrix("chd_binance_futures", "open_interest", {b: b for b in bases if b in have}, "oi_usd", START, END)
    out = cap.copy()
    if oi.empty:
        return out, 0.0
    oi = oi.reindex(idx, method="ffill", limit=8)
    for b in oi.columns:
        for leg in (f"{b}-SPOT", f"{b}-PERP"):
            if leg in out.columns:
                out[leg] = np.fmin(out[leg], OI_SHARE * oi[b]).where(out[leg].notna())
    held = (w.loc[OI_FROM:, [c for c in w.columns if c.endswith("-PERP")]] != 0)
    known = oi.reindex(columns=[c[:-5] for c in held.columns]).loc[OI_FROM:].notna().values
    share = float((held.values & known).sum() / held.values.sum()) if held.values.sum() else float("nan")
    return out, share


def fee_series(columns: pd.Index, f: dict):
    fee = pd.Series([f["perp"] if c.endswith("-PERP") else f["spot"] for c in columns], index=columns)
    return fee, pd.Series(f["slip"], index=columns)


def run_book(prices, w, f_perp, month_start, f: dict, max_iter: int = 8):
    fee, slip = fee_series(prices.columns, f)
    kw = dict(fee_bps=fee, slippage_bps=slip, funding=f_perp, rebalance_bars=month_start)
    res = run(prices, w, **kw)
    mask = np.zeros(len(w), dtype=bool)
    for _ in range(max_iter):
        active = (res.weights.abs().sum(axis=1) > 0).values
        new = vol_kill_mask(res.returns, active=active)
        if (new == mask).all():
            break
        mask = new
        res = run(prices, w * (~mask)[:, None], **kw)
    res.metrics["kill_events"] = int((mask[1:] & ~mask[:-1]).sum()); res.metrics["kill_bars"] = int(mask.sum())
    return res


def stats(res, a, b, on_bars: pd.Series) -> dict:
    r = res.returns.loc[a:b]; bpy = res.bars_per_year
    eq = (1 + r).cumprod(); on = on_bars.loc[a:b]
    ann = lambda x: float(x.loc[a:b].mean() * bpy)
    r_on = r[on]
    return {"ann_return": float(eq.iloc[-1] ** (bpy / len(r)) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(bpy)) if r.std() > 0 else np.nan,
            "max_dd": float((eq / eq.cummax() - 1).min()), "funding_ann": ann(res.funding_pnl), "price_basis_ann": ann(res.gross_returns),
            "cost_ann": -ann(res.costs), "turnover_ann": ann(res.turnover), "share_time_on": float(on.mean()),
            "ann_return_while_on": float(r_on.mean() * bpy) if len(r_on) else np.nan,
            "sharpe_while_on": float(r_on.mean() / r_on.std() * np.sqrt(bpy)) if len(r_on) > 1 and r_on.std() > 0 else np.nan,
            "kill_events": res.metrics["kill_events"]}


def main():
    perp, spot, raw, f30, sizable, a_perp, a_spot, idx, bases = build()
    prices = pd.concat({f"{b}-SPOT": spot[b] for b in bases} | {f"{b}-PERP": perp[b] for b in bases}, axis=1)
    f_perp = raw.rename(columns=lambda b: f"{b}-PERP")
    rows, regimes = [], None
    for regime_filter in (True, False):
        w, month_start, reg = weights(perp, spot, f30, sizable, idx, bases, regime_filter)
        prices_w = prices[w.columns]
        on_bars = (w.abs().sum(axis=1) > 0)
        if regime_filter:
            regimes = reg; regimes.to_csv(OUT / "regime.csv")
        for fname, f in FEES.items():
            res = run_book(prices_w, w, f_perp, month_start, f)
            label = f"{'regime' if regime_filter else 'always_on'}_{fname}"
            tearsheet(res, f"Sizable hedged carry, {label}", out=OUT / f"tearsheet_{label}.png")
            for pname, (a, b) in PERIODS.items():
                rows.append({"book": label, "period": pname, **stats(res, a, b, on_bars)})
            if regime_filter and fname == "maker":
                cap = pair_cap(a_perp, a_spot, PARTICIPATION, res.weights.columns)
                C = static_capacity(res.weights, cap).dropna()
                cap_oi, oi_share_held = oi_capped(cap, res.weights, idx, bases)
                C_oi = static_capacity(res.weights, cap_oi).dropna()
                since = C.loc[OI_FROM:]
                json.dump({"static_capacity_2pct_median": float(C.median()), "p10": float(C.quantile(0.1)), "p90": float(C.quantile(0.9)),
                           "active_bars": int(len(C)),
                           f"since_{OI_FROM}_volume_only_median": float(since.median()) if len(since) else None,
                           f"since_{OI_FROM}_with_1pct_oi_median": float(C_oi.loc[OI_FROM:].median()) if len(C_oi.loc[OI_FROM:]) else None,
                           f"since_{OI_FROM}_share_held_perp_legs_with_oi": oi_share_held},
                          open(OUT / "capacity.json", "w"), indent=2)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(regimes.round(4).to_string())
    print(f"\nmonths ON: {int(regimes.on.sum())} of {len(regimes)}; ON months by year:")
    print(regimes.groupby(regimes.index.year)["on"].agg(["sum", "count"]).to_string())
    print("\n" + table.round(4).to_string(index=False))
    md = ["# Regime-filtered hedged carry at size (autogenerated by scripts/regime_carry.py)", "",
          "Parameters: notes/04_regime_params.md (fixed before the run).", "",
          "## Monthly regime switch", regimes.round(4).to_string(), "", "## 2x2 by period", table.round(4).to_string(index=False), ""]
    (OUT / "note_autogen.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
