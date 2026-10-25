"""Interactive tear sheet (Plotly, one self-contained HTML file) for a backtest Result, and a
side-by-side comparison of several Results. Same panels as the matplotlib tearsheet plus a
monthly funding-vs-costs bar and a daily P&L strip with stop-active bars flagged.

    from qr.backtest.report_html import tearsheet_html, compare_html
    tearsheet_html(res, "Funding carry, base", "reports/carry/tearsheet.html")
    compare_html({"signal+stop": r1, "signal": r2, "no signal+stop": r3, "no signal": r4},
                 "reports/carry/unhedged_2x2.html")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qr.backtest.engine import Result


def _daily(res: Result) -> pd.DataFrame:
    r = res.returns.fillna(0.0)
    d = pd.DataFrame({"net": (1 + r).resample("D").prod() - 1,
                      "funding": res.funding_pnl.resample("D").sum(),
                      "costs": -res.costs.resample("D").sum(),
                      "price": res.gross_returns.resample("D").sum()})
    active = (res.weights.abs().sum(axis=1) > 0).resample("D").mean()
    d["active_share"] = active
    d["stopped"] = (active < 0.5) & (active.shift(1) >= 0.5)   # went flat during the day
    return d


def tearsheet_html(res: Result, title: str, out: str | Path) -> Path:
    eq = res.equity; dd = eq / eq.cummax() - 1
    gross_eq = (1 + res.gross_returns.fillna(0.0)).cumprod()
    d = _daily(res)
    m = res.metrics
    monthly = pd.DataFrame({"funding": res.funding_pnl.resample("ME").sum() * 100,
                            "costs": -res.costs.resample("ME").sum() * 100,
                            "price/basis": res.gross_returns.resample("ME").sum() * 100})
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.32, 0.14, 0.18, 0.18, 0.18],
                        subplot_titles=("Equity (net and gross)", "Drawdown", "Daily net P&L, % (stops marked)",
                                        "Monthly funding / costs / price, %", "Gross exposure and turnover"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="net", line=dict(width=1.4)), 1, 1)
    fig.add_trace(go.Scatter(x=gross_eq.index, y=gross_eq, name="gross (before costs, funding)", line=dict(width=0.8), opacity=0.6), 1, 1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, name="drawdown %", fill="tozeroy", line=dict(width=0.6)), 2, 1)
    fig.add_trace(go.Bar(x=d.index, y=d["net"] * 100, name="daily net %", marker_color=np.where(d["net"] >= 0, "#2a9d8f", "#e76f51")), 3, 1)
    st = d[d["stopped"]]
    if len(st):
        fig.add_trace(go.Scatter(x=st.index, y=st["net"] * 100, mode="markers", name="stop fired",
                                 marker=dict(symbol="x", size=7, color="black")), 3, 1)
    for c in monthly.columns:
        fig.add_trace(go.Bar(x=monthly.index, y=monthly[c], name=c), 4, 1)
    fig.update_layout(barmode="relative")
    gross = res.weights.abs().sum(axis=1)
    fig.add_trace(go.Scatter(x=gross.index, y=gross, name="gross exposure", line=dict(width=0.8)), 5, 1)
    to = res.turnover.rolling(24 * 7, min_periods=1).sum()
    fig.add_trace(go.Scatter(x=to.index, y=to, name="turnover, 7d sum", line=dict(width=0.8), yaxis="y6"), 5, 1)
    fig.update_layout(
        title=f"{title}   Sharpe {m.get('sharpe', float('nan')):.2f}   AnnRet {m.get('ann_return', float('nan')):.1%}   "
              f"MaxDD {m.get('max_drawdown', float('nan')):.1%}   Turnover/yr {m.get('ann_turnover', float('nan')):.0f}x",
        height=1250, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-0.02))
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def compare_html(results: dict[str, Result], out: str | Path, title: str = "Books side by side") -> Path:
    """Equity curves on one chart, plus a bar panel of annualised funding / costs / price / net per book."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12, row_heights=[0.6, 0.4],
                        subplot_titles=("Equity", "Per year: funding, costs, price/basis, net"))
    rows = []
    for name, res in results.items():
        fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name=name, line=dict(width=1.2)), 1, 1)
        bpy = res.bars_per_year
        rows.append({"book": name, "funding": res.funding_pnl.mean() * bpy * 100, "costs": -res.costs.mean() * bpy * 100,
                     "price/basis": res.gross_returns.mean() * bpy * 100, "net": res.metrics.get("ann_return", np.nan) * 100})
    t = pd.DataFrame(rows).set_index("book")
    for c in t.columns:
        fig.add_trace(go.Bar(x=t.index, y=t[c], name=c), 2, 1)
    fig.update_layout(title=title, barmode="group", height=850, template="plotly_white", hovermode="x unified")
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def excel_tables(tables: dict[str, pd.DataFrame], out: str | Path) -> Path:
    """One workbook, one sheet per table, for anyone who reads results in a spreadsheet."""
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out) as xw:
        for name, df in tables.items():
            df.to_excel(xw, sheet_name=name[:31])
    return out
