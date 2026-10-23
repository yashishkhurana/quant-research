"""Cross-venue funding carry: long the perp on the venue with lower funding, short it on the
venue with higher funding, same coin, same notional. No spot leg.

Mechanism. Binance and Hyperliquid each settle funding on their own perp. When HL funding is
above Binance's for a coin, short HL / long Binance collects the difference; when below, the
reverse. Price exposure nets out except for the cross-venue basis (HL mark vs Binance mark),
which is small and mean-reverting but not zero.

Signal. Trailing mean of (f_HL - f_BN) per fixed 8h UTC window (HL settles hourly, Binance
every 8h/4h/1h; both are summed onto the same windows), annualised. Enter when |spread| >
`entry_apr`, exit when |spread| < `exit_apr`. Direction = sign of the spread at the decision.
Selection only changes at window ends; between them only a delisting (price NaN) exits.

Pricing in the backtest. HL serves only ~7 months of hourly candles, so BOTH legs are priced off
the Binance perp close: the simulated price P&L of a pair is zero and the return is funding
spread minus costs. The cross-venue basis risk is measured separately from HL's hourly premium
series (which does go back) and reported, not simulated. State this in any write-up.

Columns: '<BASE>-BN' and '<BASE>-HL'. Funding for the engine: f_BN on -BN, f_HL on -HL, each at
its own settlement times; the engine charges long-pays-positive on each column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qr.data.funding import WINDOWS_PER_YEAR, per_window
from qr.strategies.carry import kill_mask, point_in_time_universe


@dataclass
class XVenueParams:
    lookback: int = 9             # 8h windows
    entry_apr: float = 0.10       # |f_HL - f_BN| annualised
    exit_apr: float = 0.03
    max_assets: int = 10
    max_weight: float = 0.15      # per leg
    max_gross: float = 2.0
    kill_dd: float = 0.08
    cooldown_bars: int = 72
    universe_n: int = 60          # by trailing Binance perp dollar volume, point in time
    universe_exit_n: int = 80
    volume_window: int = 24 * 30
    rebalance_band: float = 0.25


def spread_apr(f_bn: pd.DataFrame, f_hl: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Annualised trailing mean of (HL - Binance) funding per 8h window, on the shared coins.
    NaN where either venue has no settlement in the window."""
    coins = sorted(set(f_bn.columns) & set(f_hl.columns))
    s = per_window(f_hl[coins]) - per_window(f_bn[coins])
    return s.rolling(lookback, min_periods=max(2, lookback // 2)).mean() * WINDOWS_PER_YEAR


def build_weights(prices_bn: pd.DataFrame, f_bn: pd.DataFrame, f_hl: pd.DataFrame, p: XVenueParams,
                  dollar_volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """prices_bn: hourly Binance perp close, columns = bases (NaN = not listed / delisted).
    Returns weights on columns '<b>-BN' and '<b>-HL' for the shared coins."""
    apr = spread_apr(f_bn, f_hl, p.lookback)
    coins = [c for c in apr.columns if c in prices_bn.columns]
    apr = apr[coins]
    window_ends = apr.index
    apr = apr.reindex(prices_bn.index, method="ffill")

    priced = prices_bn[coins].notna().values.copy()
    tradeable = priced.copy()
    if dollar_volume is not None:
        in_univ = point_in_time_universe(dollar_volume.reindex(columns=coins).reindex(prices_bn.index),
                                         p.universe_n, p.volume_window, p.universe_exit_n)
        tradeable &= in_univ.fillna(False).values
    apr = apr.where(tradeable, np.nan)

    av = apr.abs().values
    state = np.zeros(len(coins), dtype=bool); on = np.zeros_like(av, dtype=bool)
    for i in range(len(av)):
        row = av[i]
        state = np.where(np.isnan(row), False, np.where(state, row > p.exit_apr, row > p.entry_apr))
        on[i] = state
    strength = apr.abs().where(on)
    fresh = (strength.rank(axis=1, ascending=False).le(p.max_assets) & on).values
    sett = np.zeros(len(prices_bn.index), dtype=bool)
    sett[np.unique(np.clip(prices_bn.index.searchsorted(window_ends, side="left"), 0, len(sett) - 1))] = True

    sel = np.zeros_like(fresh); sign = np.zeros(fresh.shape, dtype=float)
    prev = np.zeros(len(coins), dtype=bool); prev_sign = np.zeros(len(coins))
    sgn = np.sign(np.nan_to_num(apr.values))
    for i in range(len(sel)):
        if sett[i]:
            prev = fresh[i]; prev_sign = np.where(prev, sgn[i], 0.0)
        else:
            prev = prev & priced[i]; prev_sign = np.where(prev, prev_sign, 0.0)
        sel[i] = prev; sign[i] = prev_sign

    n = sel.sum(axis=1).astype(float); n[n == 0] = np.nan
    leg = np.minimum(1.0 / n, p.max_weight)[:, None] * sel
    leg = np.nan_to_num(leg)
    w = pd.DataFrame(0.0, index=prices_bn.index, columns=[f"{c}-BN" for c in coins] + [f"{c}-HL" for c in coins])
    for j, c in enumerate(coins):
        w[f"{c}-HL"] = -sign[:, j] * leg[:, j]          # spread > 0: HL pays more -> short HL
        w[f"{c}-BN"] = sign[:, j] * leg[:, j]
    gross = w.abs().sum(axis=1)
    return w.mul((p.max_gross / gross).clip(upper=1.0).fillna(1.0), axis=0)


def engine_inputs(prices_bn: pd.DataFrame, f_bn: pd.DataFrame, f_hl: pd.DataFrame, coins: list[str]):
    """Prices for both columns from the Binance perp close (see module docstring); funding per column."""
    prices = pd.concat([prices_bn[coins].add_suffix("-BN"), prices_bn[coins].add_suffix("-HL")], axis=1)
    funding = pd.concat([f_bn[coins].add_suffix("-BN"), f_hl[coins].add_suffix("-HL")], axis=1).sort_index()
    return prices, funding


def run_with_risk(prices_bn: pd.DataFrame, f_bn: pd.DataFrame, f_hl: pd.DataFrame, p: XVenueParams,
                  fee_bps: float = 4.5, slippage_bps: float = 3.0, max_iter: int = 8,
                  dollar_volume: pd.DataFrame | None = None, weights: pd.DataFrame | None = None):
    from qr.backtest.engine import run
    w = build_weights(prices_bn, f_bn, f_hl, p, dollar_volume) if weights is None else weights
    coins = sorted({c.rsplit("-", 1)[0] for c in w.columns})
    prices, funding = engine_inputs(prices_bn, f_bn, f_hl, coins)
    kw = dict(fee_bps=fee_bps, slippage_bps=slippage_bps, funding=funding, max_gross=p.max_gross,
              rebalance_band=p.rebalance_band)
    res = run(prices, w, **kw)
    mask = np.zeros(len(w), dtype=bool)
    for _ in range(max_iter):
        active = (res.weights.abs().sum(axis=1) > 0).values
        new = kill_mask(res.returns, p, active)
        if (new == mask).all():
            break
        mask = new
        res = run(prices, w * (~mask)[:, None], **kw)
    res.metrics["kill_bars"] = int(mask.sum()); res.metrics["kill_events"] = int((mask[1:] & ~mask[:-1]).sum())
    return res


def basis_risk(premium_hl: pd.DataFrame, bars_per_day: int = 24) -> pd.DataFrame:
    """What the backtest does not simulate: the HL-vs-oracle premium per coin. Returns per-coin
    daily std of the premium change and its worst 1-day move, in bps of notional. A pair held
    through a premium move of x bps gains/loses x bps on the HL leg against the Binance leg
    (Binance's own premium is not in this measure; the two are correlated, so this is an upper bound)."""
    d = premium_hl.diff(bars_per_day)
    return pd.DataFrame({"daily_std_bps": d.std() * 1e4, "worst_day_bps": d.abs().max() * 1e4,
                         "mean_premium_bps": premium_hl.mean() * 1e4})
