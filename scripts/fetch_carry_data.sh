#!/usr/bin/env bash
# Pull everything the carry study needs: top-N perp universe, funding since 2023, hourly perp and
# spot closes. ~20–40 min on a normal connection; safe to re-run (appends dedupe).
#   bash scripts/fetch_carry_data.sh 40 2023-01-01
set -euo pipefail
TOP=${1:-40}
START=${2:-2023-01-01}
mkdir -p data
python -m qr.data.history universe --top "$TOP" > data/universe.txt
echo "universe: $(wc -l < data/universe.txt) symbols"
SYMS=$(tr '\n' ' ' < data/universe.txt)
python -m qr.data.history funding --symbols $SYMS --start "$START"
python -m qr.data.history klines  --symbols $SYMS --start "$START" --interval 1h
python -m qr.data.history spot    --symbols $SYMS --start "$START" --interval 1h
echo done
