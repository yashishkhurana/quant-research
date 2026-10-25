# quant-research

Crypto perp research: data pipeline, backtester, and one note per study. Binance, Bybit and
Hyperliquid, 2023 to now.

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
```

Each note ends with the commands that reproduce it.

## Data

- Binance perps and spot: `python -m qr.data.history {universe-all,klines,spot,funding} ...`. `list_symbols(include_delisted=True)` keeps delisted contracts in the universe.
- Bybit: `python -m qr.data.bybit_history {funding,klines,oi} --start 2024-01-01`.
- Hyperliquid: `python -m qr.data.hl_history {funding,candles,snapshot}`. Its candle API only goes back ~5,000 hours.
- Second source (CryptoHFTData, open interest and mark prices from mid-2025): `qr/data/chd.py`, key in `CHD_API_KEY`.
- Own order-book and trade collector, Binance and Hyperliquid: `python -m qr.data.collector --symbols BTC ETH SOL`. Writes `data/raw/<exchange>/{trades,book}/<SYM>-PERP/<day>.parquet` every 30 s. The host must not sleep (`caffeinate -is` on a Mac).

## Layout

```
qr/
  data/        store, collector, Binance/Bybit/Hyperliquid loaders, vendor client, funding on 8h windows, cross-venue name matching
  backtest/    weight-based engine (fees, slippage, funding, delistings), PNG and HTML tearsheets
  strategies/  carry, cross-venue, multi-venue
  signals/     order-flow features
  validation/  deflated Sharpe, walk-forward, parameter grid
  capacity.py  participation caps, square-root impact
  risk.py      vol-scaled stop, liquidation model
scripts/       one runner per study
notes/         one note per study; *_params.md files are written and committed before the run
```

## Backtester contract

The weight at bar *t* is decided with information up to *t* and earns the return from *t* to
*t+1*. The engine does not lag signals; strategies do, so the look-ahead boundary is visible in
the code. Long perps pay positive funding. A missing price force-closes the position at the last
valid price and charges the exit. `tests/test_engine.py` holds the invariants.
