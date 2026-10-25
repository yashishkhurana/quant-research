# quant-research

Mid-frequency crypto quant research on perpetual futures: a shared data pipeline, an honest
backtester, a validation layer, and the strategies built on top of them.

| # | Study | Horizon | Status | Note |
| --- | --- | --- | --- | --- |
| 1 | Research infrastructure: collector, Parquet store, backtester, validation, capacity, risk | — | built; 74 tests | — |
| 2 | Funding-rate carry, Binance (hedged) | 8h | done: does not pay after costs, 2023–26 | [`01_carry.md`](notes/01_carry.md) |
| 3 | Cross-venue funding: Binance, Bybit, Hyperliquid | 8h | done: real spread, under $200k of capacity, negative after impact | [`02_mvenue.md`](notes/02_mvenue.md) |
| 4 | Order-flow signals on self-collected book data | 1m | superseded: the first sample lost ~70% to host sleep; rerun pending | [`03_flow.md`](notes/03_flow.md) |
| 5 | Regime-filtered carry at size | monthly | done: almost nothing to switch on after Jan 2025 | [`04_regime.md`](notes/04_regime.md) |
| 6 | Classifier for above-default funding | 8h | done: matches the persistence rule, does not cover costs | [`05_predict.md`](notes/05_predict.md) |
| 7 | Yield leg: staking token or PT hedged with a perp short | snapshot | first scan | [`06_yield_leg.md`](notes/06_yield_leg.md) |
| 8 | Intraday pairs stat arb | 5m bars | planned | — |
| 9 | On-chain ML (triple-barrier, purged CV) | 1h | planned | — |

Results in this table are after fees, slippage and funding, on a universe that includes delisted
contracts. If a strategy loses money after costs, the note says so.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 74 tests
python scripts/smoke.py     # synthetic end-to-end run -> reports/smoke_tearsheet.png
```

## Start collecting order-book data (do this on day 1)

```bash
nohup python -m qr.data.collector --symbols BTC ETH SOL --exchanges binance hyperliquid \
      > logs/collector.log 2>&1 &
```

Trades and 10-level book snapshots land in `data/raw/<exchange>/{trades,book}/<SYM>-PERP/<day>.parquet`,
flushed every 30 s. Check gaps with `python -c "from qr.data.store import available_days as a; print(a('raw','binance','book','BTC-PERP'))"`.
Under tmux or a systemd unit it survives logout.

## Historical data

```bash
python -m qr.data.history universe --top 50                       # top 50 perps by 24h volume
python -m qr.data.history klines  --symbols BTC ETH SOL --start 2023-01-01
python -m qr.data.history funding --symbols BTC ETH SOL --start 2023-01-01
python -m qr.data.history oi      --symbols BTC --start 2025-09-20   # ~30 days back only
```

`history.list_symbols(include_delisted=True)` returns delisted contracts with onboard and
delivery dates; build universes from it so the tokens that went to zero stay in the backtest.

## Layout

```
qr/
  data/
    store.py        Parquet store (one UTC day per file, dedupe on append; CSV fallback without pyarrow)
    collector.py    Binance + Hyperliquid websocket collector: trades + L2 book
    history.py      Binance REST loaders: 1m klines, funding, open interest, universe
    bars.py         time / volume / dollar bars from trades; kline resampling
  backtest/
    engine.py       vectorised weight-based backtester: fees, slippage, funding, delistings, metrics
    report.py       tearsheet (equity, drawdown, turnover, exposure) + monthly table
  strategies/
    base.py         Strategy protocol, BuyAndHold and Flat benchmarks
  signals/          order-flow features (flow.py)
  validation/       deflated Sharpe, walk-forward, parameter-stability grid
tests/              engine invariants: zero strategy = zero PnL, no look-ahead, costs, funding sign, delisting
notes/              one 2-page research note per strategy (template in notes/TEMPLATE.md)
scripts/            smoke.py and later per-strategy runners
```

## Backtester contract

The weight at bar *t* is decided with information up to *t* and earns the return from *t* to
*t+1*. The engine does not lag signals for you; lag them in the strategy so the look-ahead
boundary is visible in the code. Long perps pay positive funding. Untradeable prices (NaN)
force-close the position at the last valid price and the exit is charged. See
`tests/test_engine.py` for the invariants this is held to.
