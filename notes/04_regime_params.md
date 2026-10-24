# Regime-filtered hedged carry at size: parameters fixed before the run

*Written 2026-10-25 03:20 IST, before any result existed. The question: is there a version of
single-venue hedged carry on Binance that a $10m book could run, given studies 1–3 and the
capacity section of the carry note? The design is the one a desk would actually try — hold the
sizable names, only when the market is paying, at maker fees, with almost no trading — and the
test is whether the data supports it. These values are not changed after the run.*

| Parameter | Value | Why |
| --- | --- | --- |
| Universe (sizable) | at each month start, coins with both a perp and a spot price whose 2% × min(trailing-30d spot ADV, perp ADV) ≥ $1m per leg, i.e. min ADV ≥ $50m, point in time | a $10m book at 10 names is $1m a leg; the carry note's capacity section uses the same 2% participation |
| Regime switch | ON when the equal-weight mean of trailing 30-day per-window funding across the sizable universe, annualised, exceeds the default (10.95% APR); decided at each month start on windows ending before it | per-coin spells die in 16h (study 1) but the cross-section moves in regimes lasting months (study 3, 2023–24 vs 2025–26); the switch is the only thing that could have kept the default-rate book out of 2025–26 |
| Selection when ON | every sizable coin whose own trailing 30-day funding is above the default, equal weight, per-leg weight = min(1/n, 10%), gross ≤ 2.0×; n capped at 20 | no ranking, no rotation inside the month; the loss in the rotation book was turnover |
| When OFF | flat; cash earns 0 in the backtest (a T-bill rate is not added) | the result must stand without a risk-free yield painted on the flat periods |
| Rebalance | month starts only; drift not traded; a coin losing either price exits both legs | as study 2 |
| Fees (maker case) | perp 2 bps, spot 3 bps, slippage 1 bp per leg | VIP-tier maker fees a $10m book with ~$2–4bn/yr of flow would pay; slippage for maker fills that are adversely selected |
| Fees (taker case, control) | 4 + 3 bps per leg, as the carry note | isolates the fee effect from the regime effect |
| Controls | the 2×2: regime {ON-filtered, always-on} × fees {maker, taker}; always-on + taker is study 2 on the sizable universe | the regime filter and the fee tier are the two claims; each needs its own column |
| Stop | 3 × the book's trailing 30-day daily vol × √5, floor 1%, 72h cooldown | as the multi-venue book; sized to the book's vol, not a fixed 8% |
| Periods reported | 2023–24, 2025–26, full; share of months ON; months ON that were positive | the regime filter's value is in 2025–26; the ON/OFF split is the whole result |
| Capacity | static capacity at 2% ADV of the realised book, and 1% of OI where the vendor OI history (from 2025-06-28) exists | the book is sized to pass; the table shows what it could actually hold |

What a positive result would and would not mean: a positive ON-regime return at maker fees says a
$10m hedged book pays *when the market is paying*, with the switch deciding when; it does not say
the switch can be timed better than monthly, and the share of time ON is part of the result, not
a nuisance. A negative result closes single-venue hedged carry at size on Binance for 2023–26.

In-sample; the regime threshold (the default rate) and the 30-day windows are not tuned; no grid.
