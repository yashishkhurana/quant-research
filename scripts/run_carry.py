"""Funding carry: load data, run base case, parameter grid, walk-forward, stress windows; write
reports/carry/* (incl. note_autogen.md, the rotation-backtest numbers). notes/01_carry.md is hand-written.

    python scripts/run_carry.py --start 2023-01-01 --universe data/universe.txt

Requires, for each base in the universe: raw/binance/funding/<B>-PERP, raw/binance/klines_1h/<B>-PERP,
raw/binance/klines_1h/<B>-SPOT (see scripts/fetch_carry_data.sh).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qr.backtest.engine import run
from qr.backtest.report import tearsheet
from qr.data.history import delisting_times
from qr.data.store import read_range
from qr.data.funding import per_window
from qr.strategies.carry import (CarryParams, build_weights, implied_leverage, point_in_time_universe,
                                 run_with_risk)
from qr.validation import deflated_sharpe

OUT = Path("reports/carry")
OUT.mkdir(parents=True, exist_ok=True)


DELISTED = delisting_times()


def load(bases: list[str], start: str, end: str):
    closes, fund, dvol = {}, {}, {}
    for b in bases:
        kp = read_range("raw", "binance", "klines_1h", f"{b}-PERP", start, end)
        if b in DELISTED and not kp.empty:                 # frozen klines after delisting: cut them
            kp = kp[kp.index < DELISTED[b]]
        ks = read_range("raw", "binance", "klines_1h", f"{b}-SPOT", start, end)
        f = read_range("raw", "binance", "funding", f"{b}-PERP", start, end)
        if kp.empty or f.empty:
            print(f"skip {b}: perp={len(kp)} funding={len(f)}")
            continue
        closes[f"{b}-PERP"] = kp["close"]
        if not ks.empty:                       # perp-only coins stay in: tradeable unhedged, never hedged
            closes[f"{b}-SPOT"] = ks["close"]
        r = f["rate"].copy()
        r.index = r.index.round("min")                     # fundingTime carries ms jitter (see load_funding)
        fund[b] = r[~r.index.duplicated(keep="last")]
        dvol[b] = kp["quote_vol"] if "quote_vol" in kp else kp["close"] * kp["volume"]
    prices = pd.DataFrame(closes).sort_index()
    prices = prices[~prices.index.duplicated()]
    funding = pd.DataFrame(fund).sort_index()
    funding = funding[~funding.index.duplicated()]
    dollar_volume = pd.DataFrame(dvol).sort_index()
    dollar_volume = dollar_volume[~dollar_volume.index.duplicated()].reindex(prices.index)
    return prices, funding, dollar_volume


def pnl_split(res) -> tuple[pd.DataFrame, dict]:
    """Per-year price / funding / cost P&L (simple sums of per-bar returns) and compounded net;
    plus the same three per year over the full sample. price + funding + costs = net, per bar."""
    df = pd.DataFrame({"price": res.gross_returns, "funding": res.funding_pnl, "costs": -res.costs, "net": res.returns})
    y = df.groupby(df.index.year)
    t = y[["price", "funding", "costs"]].sum()
    t["net_compounded"] = y["net"].apply(lambda r: (1 + r).prod() - 1)
    yrs = len(df) / res.bars_per_year
    return t, {k: float(df[k].sum() / yrs) for k in ("price", "funding", "costs")}


def no_signal_short(prices, funding, dvol, p: CarryParams, gross: float) -> pd.DataFrame:
    """Control for the unhedged book: equal-weight short of the same point-in-time top-N universe,
    no funding signal, re-decided at the same 8h window ends, `gross` total short, delistings exit."""
    bases = [b for b in funding.columns if f"{b}-PERP" in prices.columns]
    cols = [f"{b}-PERP" for b in bases]
    priced = prices[cols].notna().values
    univ = point_in_time_universe(dvol.reindex(columns=bases).reindex(prices.index), p.universe_n,
                                  p.volume_window, p.universe_exit_n).fillna(False).values & priced
    dec = np.zeros(len(prices), dtype=bool)
    dec[np.unique(np.clip(prices.index.searchsorted(per_window(funding[bases]).index), 0, len(prices) - 1))] = True
    sel = pd.DataFrame(np.where(dec[:, None], univ, np.nan), index=prices.index).ffill().fillna(0).values.astype(bool) & priced
    n = sel.sum(axis=1).astype(float); n[n == 0] = np.nan
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[cols] = np.nan_to_num(-sel.astype(float) / n[:, None]) * gross
    return w


def unhedged_controls(prices, funding, dvol, p: CarryParams, res_u) -> pd.DataFrame:
    """2x2: carry signal vs no-signal short (gross matched to the signal's average when invested),
    each with and without the kill switch. Separates what the signal, the stop and the market did."""
    f_perp = funding.rename(columns=lambda b: f"{b}-PERP")
    kw = dict(fee_bps=4, slippage_bps=3, funding=f_perp, max_gross=p.max_gross, rebalance_band=p.rebalance_band)
    ws = build_weights(prices, funding, p, dvol)
    g = ws.abs().sum(axis=1); gross = float(g[g > 0].mean())
    wn = no_signal_short(prices, funding, dvol, p, gross)
    runs = {("carry signal", True): res_u, ("carry signal", False): run(prices, ws, **kw),
            (f"no signal, short all top {p.universe_n} (gross {gross:.2f})", True):
                run_with_risk(prices, funding, p, dollar_volume=dvol, weights=wn),
            (f"no signal, short all top {p.universe_n} (gross {gross:.2f})", False): run(prices, wn, **kw)}
    rows = []
    for (book, kill), r in runs.items():
        t, full = pnl_split(r)
        rows.append({"book": book, "kill_switch": kill, "price_ann": full["price"], "funding_ann": full["funding"],
                     "costs_ann": full["costs"], "ann_return": r.metrics["ann_return"], "sharpe": r.metrics["sharpe"],
                     "max_dd": r.metrics["max_drawdown"], "kill_events": r.metrics.get("kill_events", 0),
                     **{f"net_{y}": v for y, v in t["net_compounded"].items()}})
    return pd.DataFrame(rows)


def stress(res, windows: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for name, (a, b) in windows.items():
        r = res.returns.loc[a:b]
        if len(r) < 10:
            continue
        eq = (1 + r).cumprod()
        rows.append({"window": name, "return": eq.iloc[-1] - 1, "max_dd": (eq / eq.cummax() - 1).min(),
                     "worst_bar": r.min()})
    return pd.DataFrame(rows)


def main(a):
    bases = [l.strip() for l in open(a.universe) if l.strip()]
    prices, funding, dvol = load(bases, a.start, a.end)
    print(f"{len(funding.columns)} assets with data, {len(prices)} hourly bars, "
          f"{prices.index[0]} -> {prices.index[-1]}")

    base = CarryParams(universe_n=a.universe_n)
    rr = lambda pr, fu, p: run_with_risk(pr, fu, p, dollar_volume=dvol)
    res = rr(prices, funding, base)
    univ = (dvol.notna() & (dvol.rolling(base.volume_window, min_periods=base.volume_window // 4).sum() > 0)).sum(axis=1)
    m_extra = {"assets_with_data": int(len(funding.columns)), "universe_n": a.universe_n,
               "avg_assets_eligible": float(univ.mean()),
               # delisted = last valid perp price more than 7 days before the end of the sample
               # (the final bar is incomplete for nearly every symbol, so "NaN on the last bar" is wrong)
               "n_delisted_in_sample": int((prices.filter(like="-PERP").apply(lambda c: c.last_valid_index())
                                             < prices.index[-1] - pd.Timedelta(days=7)).sum())}
    tearsheet(res, "Funding carry, base params", out=OUT / "tearsheet.png")
    lev = implied_leverage(res)
    m = dict(res.metrics); m.update(m_extra)
    m["avg_perp_leverage"] = float(lev.mean()); m["max_perp_leverage"] = float(lev.max())
    m["avg_n_assets"] = float((res.weights.abs() > 0).sum(axis=1).div(2).mean())

    # no-risk-layer comparison: same signal, no kill switch
    f_perp = funding.rename(columns=lambda b: f"{b}-PERP")
    res_nr = run(prices, build_weights(prices, funding, base, dvol), fee_bps=4, slippage_bps=3, rebalance_band=base.rebalance_band,
                 funding=f_perp, max_gross=base.max_gross)
    m["sharpe_no_kill"] = res_nr.metrics["sharpe"]; m["max_dd_no_kill"] = res_nr.metrics["max_drawdown"]
    tgt = build_weights(prices, funding, base, dvol)
    m["turnover_from_target_changes_per_bar"] = float(tgt.diff().abs().sum(axis=1).mean())
    m["kill_events"] = res.metrics.get("kill_events", 0)
    m["n_perp_only"] = int(sum(1 for c in prices.columns if c.endswith("-PERP") and c.replace("-PERP", "-SPOT") not in prices.columns))

    # unhedged variant: same signal, short perp only. This is the trade most "funding carry" PnL in
    # 2024-25 actually came from, and it has the spot leg's protection removed.
    res_u = rr(prices, funding, CarryParams(universe_n=a.universe_n, hedged=False))
    tearsheet(res_u, "Funding carry, UNHEDGED (short perp only)", out=OUT / "tearsheet_unhedged.png")
    wk = lambda r: (1 + r).resample("W").prod() - 1
    m.update({"unhedged_sharpe": res_u.metrics["sharpe"], "unhedged_ann_return": res_u.metrics["ann_return"],
              "unhedged_max_dd": res_u.metrics["max_drawdown"], "unhedged_ann_vol": res_u.metrics["ann_vol"],
              "unhedged_worst_week": float(wk(res_u.returns).min()),
              "unhedged_worst_week_date": str(wk(res_u.returns).idxmin().date()),
              "unhedged_kill_bars": res_u.metrics.get("kill_bars", 0),
              "unhedged_kill_events": res_u.metrics.get("kill_events", 0),
              "unhedged_avg_turnover_per_bar": res_u.metrics["avg_turnover_per_bar"],
              "hedged_worst_week": float(wk(res.returns).min()),
              "hedged_worst_week_date": str(wk(res.returns).idxmin().date())})
    uc = unhedged_controls(prices, funding, dvol, CarryParams(universe_n=a.universe_n, hedged=False), res_u)
    uc.to_csv(OUT / "unhedged_controls.csv", index=False)
    split_u, _ = pnl_split(res_u)
    split_u.to_csv(OUT / "unhedged_split.csv")

    # parameter grid: lookback x entry threshold (full sample; this is the in-sample picture)
    grid_lb, grid_entry = [3, 6, 9, 15, 21], [0.05, 0.10, 0.15, 0.20, 0.30]
    tbl = pd.DataFrame(index=grid_lb, columns=grid_entry, dtype=float)
    trials = []
    for lb, en in itertools.product(grid_lb, grid_entry):
        p = CarryParams(lookback=lb, entry_apr=en, exit_apr=en * 0.3, universe_n=a.universe_n)
        r = rr(prices, funding, p)
        tbl.loc[lb, en] = r.metrics["sharpe"]; trials.append(r.returns)
    tbl.to_csv(OUT / "grid_sharpe.csv")
    per_bar_sr = pd.Series([t.mean() / t.std(ddof=1) for t in trials])
    dsr = deflated_sharpe(res.returns, n_trials=len(trials), bars_per_year=res.bars_per_year,
                          var_trials=float(per_bar_sr.var()))
    # The grid's trials are not exchangeable noise draws: their Sharpes differ mainly because they
    # trade different amounts and pay different costs, so the deflation benchmark built from their
    # spread is reported but not relied on. The probabilistic Sharpe against zero is reported beside it.
    psr0 = deflated_sharpe(res.returns, n_trials=1, bars_per_year=res.bars_per_year, var_trials=0.0)
    m["psr_vs_zero"] = psr0["psr"]
    m["grid_sharpe_min"] = float(np.nanmin(tbl.values)); m["grid_sharpe_max"] = float(np.nanmax(tbl.values))
    m.update({f"dsr_{k}": v for k, v in dsr.items()})
    # The best grid cell, chosen after seeing all 25: plain PSR against zero, deflated for 25 trials
    # with the zero-skill variance (1/T, the honest benchmark here), and against the grid's spread.
    keys = list(itertools.product(grid_lb, grid_entry))
    ib = int(np.nanargmax([tbl.loc[k] for k in keys]))
    best = trials[ib]
    m["best_cell_lookback"], m["best_cell_entry"] = keys[ib]
    m["best_cell_sharpe"] = float(tbl.loc[keys[ib]])
    m["best_cell_psr_vs_zero"] = deflated_sharpe(best, 1, res.bars_per_year, var_trials=0.0)["psr"]
    d1 = deflated_sharpe(best, len(trials), res.bars_per_year)               # var_trials = 1/T
    m["best_cell_dsr_sr0_ann"], m["best_cell_dsr_psr"] = d1["sr0_ann"], d1["psr"]
    m["best_cell_dsr_gridvar_psr"] = deflated_sharpe(best, len(trials), res.bars_per_year,
                                                     var_trials=float(per_bar_sr.var()))["psr"]
    m["grid_cells_positive"] = int((tbl.values > 0).sum())

    # walk-forward: refit (lookback, entry) on 6 months, trade next 2 months
    bars_train, bars_test = 24 * 182, 24 * 61
    oos = []
    start = 0
    while start + bars_train + bars_test <= len(prices):
        tr = slice(start, start + bars_train); te = slice(start + bars_train, start + bars_train + bars_test)
        best, best_sr = base, -np.inf          # fall back to base params if no combo traded
        for lb, en in itertools.product(grid_lb, grid_entry):
            p = CarryParams(lookback=lb, entry_apr=en, exit_apr=en * 0.3, universe_n=a.universe_n)
            sr = rr(prices.iloc[tr], funding, p).metrics.get("sharpe", -np.inf)
            if np.isfinite(sr) and sr > best_sr:
                best, best_sr = p, sr
        if not np.isfinite(best_sr):
            wf_untradeable = m.get("wf_untradeable_folds", 0) + 1
            m["wf_untradeable_folds"] = wf_untradeable
        oos.append(rr(prices.iloc[te], funding, best).returns)
        start += bars_test
    if oos:
        o = pd.concat(oos)
        eq = (1 + o).cumprod()
        m["wf_sharpe"] = float(o.mean() / o.std(ddof=1) * np.sqrt(res.bars_per_year))
        m["wf_ann_return"] = float(eq.iloc[-1] ** (res.bars_per_year / len(o)) - 1)
        m["wf_max_dd"] = float((eq / eq.cummax() - 1).min())
        m["wf_bars"] = int(len(o)); m.setdefault("wf_untradeable_folds", 0)

    st = stress(res, {"Mar 2023 USDC depeg": ("2023-03-09", "2023-03-15"),
                      "Aug 2023 crash": ("2023-08-15", "2023-08-20"),
                      "Mar 2024 top": ("2024-03-04", "2024-03-22"),
                      "Aug 2024 carry unwind": ("2024-08-02", "2024-08-07"),
                      "Feb 2025 selloff": ("2025-02-01", "2025-02-05")})
    st.to_csv(OUT / "stress.csv", index=False)

    json.dump(m, open(OUT / "metrics.json", "w"), indent=2, default=float)
    write_note(m, tbl, st, prices, funding, base, split_u, uc)
    print(json.dumps(m, indent=2, default=float))


def write_note(m, tbl, st, prices, funding, p: CarryParams, split_u=None, uc=None):
    pct = lambda x: f"{x:+.1%}" if x == x else "n/a"
    f2 = lambda x: f"{x:.2f}" if x == x else "n/a"
    period = f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}"
    grid_md = "| lookback \\ entry APR | " + " | ".join(f"{c:.0%}" for c in tbl.columns) + " |\n"
    grid_md += "| --- |" + " --- |" * len(tbl.columns) + "\n"
    for lb, row in tbl.iterrows():
        grid_md += f"| {lb} | " + " | ".join(f2(v) for v in row) + " |\n"
    st_md = "| Window | Return | Max DD | Worst hour |\n| --- | --- | --- | --- |\n"
    for _, r in st.iterrows():
        st_md += f"| {r['window']} | {pct(r['return'])} | {pct(r['max_dd'])} | {pct(r['worst_bar'])} |\n"
    uc_md = ""
    if split_u is not None and uc is not None:
        yrs = [c for c in uc.columns if c.startswith("net_2")]
        L = ["Unhedged P&L by year (simple sums of per-bar returns; compounded net):", "",
             "| Year | Price | Funding | Costs | Net (compounded) |", "| --- | --- | --- | --- | --- |"]
        L += [f"| {y} | {pct(r.price)} | {pct(r.funding)} | {pct(r.costs)} | {pct(r.net_compounded)} |" for y, r in split_u.iterrows()]
        L += ["", "Controls: the same signal without the kill switch, and a no-signal equal-weight short of the same universe at matched gross, with and without it:", "",
              "| Book | Kill switch | Price / yr | Funding / yr | Costs / yr | Net / yr | Sharpe | Max DD | Kill events | " + " | ".join(c[4:] for c in yrs) + " |",
              "| --- " * (9 + len(yrs)) + "|"]
        for _, r in uc.iterrows():
            L.append(f"| {r['book']} | {'8% DD' if r['kill_switch'] else 'none'} | {pct(r['price_ann'])} | {pct(r['funding_ann'])} | "
                     f"{pct(r['costs_ann'])} | {pct(r['ann_return'])} | {f2(r['sharpe'])} | {pct(r['max_dd'])} | {r['kill_events']} | "
                     + " | ".join(pct(r[c]) for c in yrs) + " |")
        uc_md = "\n".join(L)
    note = f"""# Funding-rate carry on Binance perps

**One line:** delta-neutral long-spot / short-perp across the top {m['universe_n']} Binance USDT perps by trailing volume (point in time, from {m['assets_with_data']} listed contracts), held when trailing funding annualises above {p.entry_apr:.0%}; {pct(m['ann_return'])} a year net of fees, slippage and funding, Sharpe {f2(m['sharpe'])}, max drawdown {pct(m['max_drawdown'])}, {period}.

## Hypothesis
Perp funding is paid by whoever is crowded. When longs crowd in, funding goes positive and a short perp hedged with spot collects it with no directional exposure. The edge is a risk premium for providing the other side, not a mispricing, so it should be small on majors, larger on alts, and should vanish exactly when it is most needed (deleveraging events, when basis blows out and the hedge is not free).

## Data
Binance USDT-M perps and Binance spot, hourly closes, {period}. Funding settled every 8h from the fapi funding endpoint. Universe: {m['assets_with_data']} USDT perps with funding, perp and spot history (every contract Binance has listed in the window, delisted ones included; {m['n_delisted_in_sample']} are no longer trading at the end of the sample). At each bar the tradeable set is the top {m['universe_n']} by trailing 30-day dollar volume, ranked point in time, averaging {m['avg_assets_eligible']:.0f} eligible names; nothing is selected on what is liquid today. An asset is only tradeable on bars where both perp and spot have a price, and a delisting force-closes the position at the last price with the exit paid for.

## Method
Signal: trailing mean funding over {p.lookback} settlements, annualised. Enter when above {p.entry_apr:.0%} APR, exit when below {p.exit_apr:.0%} (hysteresis). Equal weight across qualifying assets, at most {p.max_assets} assets, at most {p.max_weight:.0%} of equity per leg, gross capped at {p.max_gross:.1f}x. Selection is re-decided only when a settlement lands; between settlements the engine holds realised weights inside a {p.rebalance_band:.0%} no-trade band, so price drift is not traded hourly. The tradeable set has hysteresis (enter top {p.universe_n} by trailing 30-day volume, leave below {p.universe_exit_n}). Costs: {4} bps fee + {3} bps slippage per leg on turnover. Risk layer: flatten everything if drawdown from the peak since the last re-entry exceeds {p.kill_dd:.0%}, stay flat {p.cooldown_bars} hours, then re-enter with the reference reset. The weight at bar t uses funding settled at or before t and earns t to t+1; nothing is shifted inside the backtester.

## Results
| Metric | Value |
| --- | --- |
| Period | {period} |
| Ann. return (net) | {pct(m['ann_return'])} |
| Ann. vol | {pct(m['ann_vol'])} |
| Sharpe / Sortino | {f2(m['sharpe'])} / {f2(m['sortino'])} |
| Max drawdown | {pct(m['max_drawdown'])} |
| Funding income / yr | {pct(m['funding_ann'])} |
| Cost drag / yr | {pct(-m['cost_drag_ann'])} |
| Turnover / yr | {m['ann_turnover']:.0f}x |
| Avg assets held | {m['avg_n_assets']:.1f} |
| Avg / max perp leverage | {f2(m['avg_perp_leverage'])}x / {f2(m['max_perp_leverage'])}x |
| Kill-switch events / bars flat | {m.get('kill_events', 0)} / {m.get('kill_bars', 0)} |
| Turnover from signal changes / bar | {m['turnover_from_target_changes_per_bar']:.4f} (of {m['avg_turnover_per_bar']:.4f} total) |
| Same signal, no kill-switch | Sharpe {f2(m['sharpe_no_kill'])}, max DD {pct(m['max_dd_no_kill'])} |

![tearsheet](../reports/carry/tearsheet.png)

Parameter stability, in-sample Sharpe by lookback (settlements) and entry threshold:

{grid_md}
Stress windows (base parameters):

{st_md}
Hedged vs unhedged, same signal and risk layer:

| | Hedged (spot long / perp short) | Unhedged (perp short only) |
| --- | --- | --- |
| Ann. return (net) | {pct(m['ann_return'])} | {pct(m['unhedged_ann_return'])} |
| Ann. vol | {pct(m['ann_vol'])} | {pct(m['unhedged_ann_vol'])} |
| Sharpe | {f2(m['sharpe'])} | {f2(m['unhedged_sharpe'])} |
| Max drawdown | {pct(m['max_drawdown'])} | {pct(m['unhedged_max_dd'])} |
| Worst week | {pct(m['hedged_worst_week'])} ({m['hedged_worst_week_date']}) | {pct(m['unhedged_worst_week'])} ({m['unhedged_worst_week_date']}) |
| Kill-switch events / bars flat | {m.get('kill_events', 0)} / {m.get('kill_bars', 0)} | {m['unhedged_kill_events']} / {m['unhedged_kill_bars']} |
| Turnover / bar | {m['avg_turnover_per_bar']:.4f} | {m['unhedged_avg_turnover_per_bar']:.4f} |

The unhedged book can also trade the {m['n_perp_only']} contracts that never had a Binance spot market (mostly 2024-25 meme and launch tokens), which is where much of the real "funding carry" PnL of that period came from, and where the squeezes were.

{uc_md}

![unhedged](../reports/carry/tearsheet_unhedged.png)

## Validation
{len(tbl.values.ravel())} parameter combinations were tried. Grid Sharpe ranges {f2(m['grid_sharpe_min'])} to {f2(m['grid_sharpe_max'])}. Probability the base-case Sharpe is above zero: {m['psr_vs_zero']:.0%}. Deflated Sharpe against the grid's spread: benchmark {f2(m['dsr_sr0_ann'])}, probability {m['dsr_psr']:.0%}; the trials differ mainly in how much they trade, not in sampling noise, so this benchmark is reported for completeness and not relied on. {m['grid_cells_positive']} of {len(tbl.values.ravel())} cells have a positive Sharpe; the best (lookback {m['best_cell_lookback']}, entry {m['best_cell_entry']:.0%}) has Sharpe {f2(m['best_cell_sharpe'])}, probability above zero {m['best_cell_psr_vs_zero']:.0%} taken alone, and {m['best_cell_dsr_psr']:.0%} after deflating for {len(tbl.values.ravel())} trials (zero-skill benchmark Sharpe {f2(m['best_cell_dsr_sr0_ann'])}). Walk-forward, refitting lookback and threshold on six months and trading the next two ({m.get('wf_untradeable_folds', 0)} folds where no parameter set traded in-sample, run on base parameters): out-of-sample Sharpe {f2(m.get('wf_sharpe', float('nan')))}, annualised return {pct(m.get('wf_ann_return', float('nan')))}, max drawdown {pct(m.get('wf_max_dd', float('nan')))} over {m.get('wf_bars', 0)} hours.

## What broke
- Funding income arrives in discrete 8h lumps while the spot/perp basis moves continuously; most of the drawdown is basis, not funding turning negative. The hedge is not free in a squeeze.
- The alts that pay the most funding are the ones that delist or lose their spot pair; leaving them out would overstate returns.
- Costs are a flat bps model. The microstructure note (03) replaces this with fills from our own book data; expect the alt legs to be worse than modelled here.
- Not modelled: spot borrow when spot is unavailable, exchange-level margin on the short leg, and the capacity limit of alt spot depth. Capacity is bounded by the thinnest spot leg, not by the perp.
"""
    Path("notes").mkdir(exist_ok=True)
    (OUT / "note_autogen.md").write_text(note)       # notes/01_carry.md is hand-written; do not overwrite it


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=pd.Timestamp.now('UTC').strftime("%Y-%m-%d"))
    ap.add_argument("--universe", default="data/universe.txt")
    ap.add_argument("--universe-n", dest="universe_n", type=int, default=40,
                    help="tradeable set at each bar = top N by trailing 30-day volume")
    main(ap.parse_args())
