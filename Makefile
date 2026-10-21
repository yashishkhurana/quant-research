.PHONY: install test smoke collect
install: ; pip install -e ".[dev]"
test:    ; pytest -q
smoke:   ; python scripts/smoke.py
collect: ; mkdir -p logs && nohup python -m qr.data.collector --symbols BTC ETH SOL > logs/collector.log 2>&1 &
