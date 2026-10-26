# Yield leg: a staking or fixed-yield token hedged with a perp short

*Sixth note. A first scan, not a backtest. Snapshot of 25 Oct 2026.*

**One line:** hedging ETH, SOL and HYPE liquid-staking tokens with a Hyperliquid short pays 10–12% a year before costs today and 7.5–8.8% after a 25% margin reserve, about the same as an unhedged stablecoin PT (9.4%). It is the only version of the trade with eight-figure capacity. On Pendle, the hedged trades that beat fixed income have $1–96k of capacity each.

## Idea
Plain hedged carry earns only funding: the spot leg yields nothing. Put a yield on the spot leg and funding no longer has to be above the default for the book to pay. Long a staking token or a fixed-yield PT, short the underlying's perp on the venue that pays the short most. Carry = token yield + funding − costs.

## Pendle PTs
92 active markets on all chains (`/v2/markets/all`). Each hedgeable PT is joined to 30-day funding on Binance, Bybit and Hyperliquid (stocks and gold also via Binance and Bybit's equity and gold perps, Hyperliquid's HIP-3 `xyz` dex and its PAXG perp). Capacity is 2% of the PT pool, or 1% of perp OI if smaller.

| Kind | Markets | Pool liquidity |
| --- | --- | --- |
| USD fixed income (no hedge) | 72 | $146.7m |
| Crypto | 12 | $10.6m |
| Stock | 6 | $2.6m |
| Gold | 2 | $0.1m |

The USD PTs set the benchmark: 9.36%, their pool-weighted median implied APY. Hedged stock and gold PTs beat it by 7–16 points (superTSLA 25.7%, superNVDA 20.1%, XGLD 20.2%, PAXGy 16.3%), in pools of $48–85k: $1–2k of capacity each. The large pools pay less than the benchmark once hedged (STRCx 6.7%, NVDAc 7.6%, GOOGLc 7.9%). The largest hedged capacity is wstETH, $96k. The PT pool binds in every hedged row.

## Liquid-staking tokens
Issuer pools from DefiLlama (`yields.llama.fi/pools`), staking APY = `apyBase`. Same funding join. Reserve: 25% of capital idle as margin, 75% in the token hedged one-for-one, so carry on capital = 0.75 × combined.

| Token | Staking APY | TVL | 30-day HL funding | Combined | With 25% reserve |
| --- | --- | --- | --- | --- | --- |
| jitoSOL | 4.80% | $1.06bn | 6.88% | 11.7% | 8.8% |
| mSOL | 4.71% | $229m | 6.88% | 11.6% | 8.7% |
| bnSOL | 4.56% | $1.07bn | 6.88% | 11.4% | 8.6% |
| stETH/wstETH | 2.78% | $22.7bn | 8.15% | 10.9% | 8.2% |
| ezETH | 2.43% | $101m | 8.15% | 10.6% | 7.9% |
| cbETH | 2.27% | $462m | 8.15% | 10.4% | 7.8% |
| weETH | 2.25% | $5.20bn | 8.15% | 10.4% | 7.8% |
| kHYPE | 1.92% | $778m | 8.33% | 10.2% | 7.7% |
| rETH | 1.82% | $1.14bn | 8.15% | 10.0% | 7.5% |
| slisBNB | 0.64% | $765m | 6.88% | 7.5% | 5.6% |

Hyperliquid pays the short most on every underlying; Binance pays 1.8–4.5% and Bybit −0.3 to 3.6%. Capacity at 1% of Hyperliquid OI: ETH $25.8m, HYPE $15.8m, SOL $5.5m, BNB $0.4m. It is shared by every token on the same underlying.

## Funding over time
Hyperliquid funding by quarter, APR (`reports/yield_leg/lst_hl_funding_quarterly.csv`):

| Quarter | ETH | SOL | HYPE |
| --- | --- | --- | --- |
| 2024 Q1 | 35.4% | 56.2% | — |
| 2024 Q2 | 18.7% | 17.3% | — |
| 2024 Q3 | 8.9% | 9.6% | — |
| 2024 Q4 | 25.3% | 29.6% | 118.8% (from 5 Dec) |
| 2025 Q1 | 6.6% | 3.6% | 36.1% |
| 2025 Q2 | 5.6% | 3.9% | 21.6% |
| 2025 Q3 | 13.8% | 14.2% | 19.4% |
| 2025 Q4 | 8.1% | −0.5% | 11.5% |
| 2026 Q1 | 4.2% | −7.8% | 8.6% |
| 2026 Q2 | 3.6% | 0.2% | 8.8% |
| 2026 Q3 | 9.5% | 7.2% | 10.1% |

Today's 7–8% is a quiet quarter. ETH and SOL funding sat above the default (10.95%) in 2024 Q1, Q2 and Q4 and in 2025 Q3; HYPE in every quarter through 2025. SOL was negative for two quarters: the short pays then.

## Caveats
- One dated snapshot for yields, TVL and OI. Staking yields over time are not measured; the quarterly table is funding only.
- Before costs: fees, entry spread, cross-venue transfers.
- Depeg and duration risk not modelled: an LST can trade below its underlying, a PT moves with rates until maturity.
- Exchange margin cannot be the token itself, so the reserve is dead capital. 25% is an assumption, not sized to a liquidation model.
- stHYPE has no issuer pool on DefiLlama.

## Reproduce
```bash
python scripts/yield_leg_scan.py          # Pendle -> reports/yield_leg/scan.csv
python scripts/yield_leg_scan.py --lst    # staking tokens -> reports/yield_leg/lst_scan.csv, lst_hl_funding_quarterly.csv
```
