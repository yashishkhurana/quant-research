"""Tear-sheet plots for a backtest Result."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from qr.backtest.engine import Result


def tearsheet(res: Result, title: str = "", out: str | Path | None = None):
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2, 1, 1]})
    eq = res.equity
    axes[0].plot(eq.index, eq.values, lw=1.2, label="net")
    gross_eq = (1 + res.gross_returns.fillna(0)).cumprod()
    axes[0].plot(gross_eq.index, gross_eq.values, lw=0.8, alpha=0.6, label="gross")
    axes[0].set_ylabel("equity")
    axes[0].legend(loc="upper left")
    m = res.metrics
    axes[0].set_title(f"{title}  Sharpe {m.get('sharpe', float('nan')):.2f}  "
                      f"MaxDD {m.get('max_drawdown', float('nan')):.1%}  "
                      f"AnnRet {m.get('ann_return', float('nan')):.1%}  "
                      f"Turnover/yr {m.get('ann_turnover', float('nan')):.0f}x")
    dd = eq / eq.cummax() - 1
    axes[1].fill_between(dd.index, dd.values, 0, alpha=0.4)
    axes[1].set_ylabel("drawdown")
    to = res.turnover.iloc[1:]                      # skip the initial entry
    axes[2].plot(to.index, to.rolling(50, min_periods=1).mean(), lw=0.8)
    axes[2].set_ylabel("turnover (50-bar)")
    gross = res.weights.abs().sum(axis=1)
    axes[3].plot(gross.index, gross.values, lw=0.8)
    axes[3].set_ylabel("gross exposure")
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return Path(out)
    return fig


def monthly_table(res: Result) -> pd.DataFrame:
    r = res.returns
    m = (1 + r).resample("ME").prod() - 1
    return m.to_frame("ret").assign(year=m.index.year, month=m.index.month).pivot(index="year", columns="month", values="ret")
