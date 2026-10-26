# Cross-venue funding: Binance, Bybit, Hyperliquid

Same perp, two exchanges, two funding rates. Long where funding is lower, short where it is higher. No spot leg. Parameters fixed before the run: `notes/02_mvenue_params.md`.

**Result:** the spread is real. Liquidations take most of it. 2026 is negative. Capacity is under $200k.

## Why it might work
Each exchange has its own crowd. The same coin can have longs paying shorts on Bybit while shorts pay longs on Binance. Being short where the crowd is long collects the gap. It also puts you on the wrong side of every squeeze.

## Data
- Funding: all Binance USDT perps (655, incl. delisted), all Bybit linear perps, all Hyperliquid perps (233, incl. 56 delisted). 2024-01-01 to 2026-10-23. Everything on 8h UTC windows.
- Prices: hourly closes, Binance and Bybit. Hyperliquid only serves ~7 months of candles, so **HL legs are priced off the Binance close**. Their price P&L is close to zero by construction (−0.5 to +0.7%/yr in the table below). Their basis is measured separately (section 4).
- Names: `kPEPE` = `1000PEPE`. Binance `BOB` and `1000000BOB` are not merged. A pair is kept only if both venues' prices agree within 10% on 99% of hours. Bybit ICX and ON and HL PURR fail and are dropped. Binance `H` and Bybit `H` pass, though they sat up to 2.8× apart for hours on 9 June 2026. 185 coins trade on all three venues.
- Volume: full period on Binance and Bybit; Hyperliquid from 2026-03-29 only.
- Open interest: Bybit daily; 8-hourly for ~110 coins on Binance and Bybit from mid-2025 (CryptoHFTData).
- Cross-check: 10 coins, 3 venues (8 on HL; the vendor has no LSK or MARSCOIN), 1–14 Oct 2026, against CryptoHFTData. No funding settlement off by more than 0.3 bp; hourly price gaps p99 under 22 bps. A 14-day sample, not a validation.

## Method
Trailing 9-window spread per coin and venue pair, annualised. Enter when |spread| × expected remaining hold > round trip + 5 bps. Expected remaining hold = mean remaining length of spells that have already lasted this long, over the trailing 180 days. Exit on sign flip, spread under 3% APR, or missing price. Top 10 pairs, 15% per leg, 2× gross, 1× per venue, stop sized to the book's 30-day vol. Taker fees: Binance 4, HL 4.5, Bybit 5.5 bps, plus 3 bps slippage per leg.

Two liquidity floors: $5m ADV (does the spread exist) and $50m (can it be sized).

## Results

### The spread exists

| | $5m floor | $50m floor |
| --- | --- | --- |
| Coins passing (median) | 291 | 90 |
| Funding | +37.3% | +33.7% |
| Costs | −16.0% | −17.6% |
| Price / basis | −12.7% | −8.4% |
| **Net / Sharpe** | **+8.5% / 1.07** | **+7.7% / 1.00** |
| Grid, 12 cells | all positive; best 1.43; deflated 0.77 | all positive; best 1.50; deflated 0.80 |
| With liquidation at 3× / 2× | +1.3% / +2.3% | +2.3% / +2.6% |

Funding collected is 37% a year. All 12 grid cells are positive. Deflated for 12 trials, the best cell's probability is 0.77 (0.80 at $50m). Costs take 16 points, basis 13. Net 8.5%. Then liquidations take it to 1.3%.

Best cell: lookback 3 windows, margin 20 bps. Fast entry, high hurdle. In the Binance rotation the least-bad cell was the slowest one. Here speed pays because the spread is short-lived.

### 2026 is negative
Each year's sum. 2026 = 1 Jan to 23 Oct, not annualised.

| Net | 2024 | 2025 | 2026 |
| --- | --- | --- | --- |
| $5m | +12.5% | +16.9% | −5.6% |
| $50m | +15.2% | +20.4% | −14.0% |

| Pair | Year | Funding $5m | Price $5m | Funding $50m | Price $50m |
| --- | --- | --- | --- | --- | --- |
| BN–BY | 2024 | +4.1 | −2.0 | +3.1 | −0.8 |
| BN–BY | 2025 | +24.6 | −2.2 | +27.6 | −0.8 |
| BN–BY | 2026 | +26.1 | −26.3 | +22.7 | −21.8 |
| BN–HL | 2024 | +15.0 | −0.3 | +15.2 | +0.1 |
| BN–HL | 2025 | +7.2 | 0.0 | +8.6 | +0.7 |
| BN–HL | 2026 | +6.9 | 0.0 | +3.4 | −0.5 |
| BY–HL | 2024 | +6.9 | −0.3 | +7.0 | +1.1 |
| BY–HL | 2025 | +5.5 | −2.5 | +4.1 | −1.3 |
| BY–HL | 2026 | +8.5 | −2.1 | +3.0 | −0.1 |

The Binance–Bybit legs lost on price what they collected in funding. Funding on that pair didn't fall; the book had moved into it (coin-hours roughly tripled 2024→2025). BN–HL pairs show close to zero price P&L by construction. The BY–HL price column is the Binance–Bybit move on the Bybit leg, not HL basis.

### Liquidation
Leg liquidated when its adverse move since entry exceeds 1/L − 1%. Hourly closes, no fee: optimistic. At 3×, 327 liquidations (152 at 2×), 57% on the short leg (72% at 2×). $5m book: +8.5% → +1.3%. The short is on the venue where the crowd is long. That venue squeezes.

### Basis not in the P&L
Daily moves of held coins, bps:

| | Daily std, median (p90) | Worst day, median | Worst day, max |
| --- | --- | --- | --- |
| HL premium vs index (not simulated) | 6.4 (26) | ~210 | 1,991 (HYPER) |
| Binance–Bybit basis (simulated) | 14.8 (31) | 389 | 10,228 (H) |

The HL number is what the backtest leaves out. It's about half the size of what it includes.

## Capacity
Static capacity, 2% of venue ADV per leg (1% of OI where history exists): 2026 median **$86k** at the $5m floor, **$175k** at $50m. After square-root impact: **negative at every size from $1m to $100m**, both floors. Adding the vendor's 8-hourly OI to Bybit's daily OI moved static capacity by under 1%; without any OI cap it would be about 2× ($5m floor) and 8× ($50m) higher.

A Sharpe-1 spread with $86–175k of capacity, 10 pairs at a time, about 210× turnover a year (380× in the best cell), three venues. That's a trade for individuals and market makers. Not for a mandate.

## Validation
- Parameters and both floors committed before the run (`2095782`).
- Grid deflated for 12 trials.
- Two floors agree on sign, Sharpe and the 2026 turn.
- Data cross-checked against a second vendor; vendor OI is within a median 0.9% of our 23 Oct snapshot in USD (0.3% in coins) across 230 coins, at the nearest 8-hourly print.
- Liquidation at 2× and 3×; capacity with and without OI.
- In-sample. No walk-forward. HL basis not simulated. Liquidation has no fees or wicks.

## What broke
- HL candle timestamps had no timezone. Every HL leg looked unsizable. Fixed, with a test.
- The price-agreement check compared HL's live price to a 40-hour-old Binance close and dropped 11 real coins. Now hourly.
- First run was on partial HL data: +48% funding, −35% basis, −26% costs, −12.6% net. Disclosed in the parameters file, not used.
- 2026 was first read as a full year. "Funding fell at $50m" was the partial-year artefact. The loss is the Binance–Bybit price leg at both floors.

## Reproduce
```bash
python -m qr.data.hl_history funding --start 2024-01-01 && python -m qr.data.hl_history candles && python -m qr.data.hl_history snapshot
python -m qr.data.bybit_history funding --start 2024-01-01 && python -m qr.data.bybit_history klines --start 2024-01-01 && python -m qr.data.bybit_history oi --start 2024-01-01
python scripts/chd_crosscheck.py --start 2026-10-01 --end 2026-10-14     # needs CHD_API_KEY
python scripts/run_mvenue.py --min-adv 5e6 --out reports/mvenue/adv5m
python scripts/run_mvenue.py --min-adv 5e7 --out reports/mvenue/adv50m
```
