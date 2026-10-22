#!/usr/bin/env bash
# Pull everything the carry study needs. The universe is EVERY USDT perp Binance has listed
# before END (delisted included), so the backtest is not selected on what survived. The strategy
# then picks the top-N by trailing volume at each bar, point in time.
# ~1–2 h for ~300 symbols on a normal connection; safe to re-run (appends dedupe).
#   bash scripts/fetch_carry_data.sh 2023-01-01
set -euo pipefail
START=${1:-2023-01-01}
mkdir -p data
python -m qr.data.history universe-all --end "$(date -u +%Y-%m-%d)" > data/universe.txt
echo "universe: $(wc -l < data/universe.txt) symbols (all listed USDT perps, delisted included)"
SYMS=$(tr '\n' ' ' < data/universe.txt)
python -m qr.data.history funding --symbols $SYMS --start "$START"
python -m qr.data.history klines  --symbols $SYMS --start "$START" --interval 1h
python -m qr.data.history spot    --symbols $SYMS --start "$START" --interval 1h
echo done
