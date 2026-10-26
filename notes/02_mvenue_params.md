# Multi-venue funding book: parameters fixed before the full run

*Written 2026-10-24 00:15 IST, before any full-sample result existed. The only result seen at this
point was a smoke run on Mar–Oct 2026 with incomplete Hyperliquid data (funding +48%/yr, basis −41%/yr,
net −12.6%/yr). These values are not changed after the full run.*

| Parameter | Value | Why |
| --- | --- | --- |
| Entry rule | expected net > 0: \|trailing spread\| × E[remaining hold] − round trip − margin | trade only where the spread is expected to pay for the trade; the fix for what lost money in the Binance rotation |
| E[remaining hold] | E[L − k \| L > k] over spread spells that ended in the trailing 180 days, k = the spell's age at the decision, pooled over coins and venue pairs | the unconditional mean spell length overstates the hold for any entry after a spell's first window; conditioning on age raises the hurdle |
| Spell threshold | 10% APR | same as the carry note's entry level |
| E[hold] window | 180 days | long enough for hundreds of ended spells, short enough to follow a regime change within months |
| Margin | 5 bps of leg notional | small cushion on top of the round trip; not tuned |
| Round trip | 2 × (fee + 3 bps) per venue: BN–HL 29 bps, BN–BY 31, HL–BY 32 | base-tier taker fees (Binance 4, Hyperliquid 4.5, Bybit 5.5 bps) |
| Lookback | 9 windows (3 days) | same as the carry note |
| Exit | sign flip, or \|trailing spread\| below 3% APR, or a missing price | same hysteresis as the carry note |
| Liquidity floor | two runs: $5m ADV ("does the spread exist") and $50m ADV ("can it be sized"); BN ADV, else BY | at $5m a 2% cap is $100k a leg and cannot hold a large book; at $50m it is ~$1m a leg. With 10 names that is a ~$10m book; $25m would need ~25 names passing the floor, reported per window |
| Book | top 10 coins by expected net, ≤ 15% per leg, 2.0× gross | as the carry rotation |
| Venue cap | 1.0× equity gross per venue | exchange / withdrawal risk; not binding often at 2.0× gross over three venues |
| Drawdown stop | 3 × the book's trailing 30-day daily vol × √5, floor 1%, 72 h cooldown | sized to the book's own volatility (the fixed 8% stop became the strategy in the carry note) |
| Leverage | 3×, with 2× reported alongside | a leg is liquidated when its adverse move since entry exceeds 1/L − 1% |
| Capacity | 2% of each venue's ADV per leg, tightened to 1% of OI where Bybit has OI history | as the carry note |
| Grid | lookback {3, 9, 21} × margin {0, 5, 10, 20} bps, best cell deflated for 12 trials | shows plateau vs single cell |

Known limits fixed in advance: OKX unreachable from this network; Hyperliquid volume exists only for its last ~7 months of candles, so HL legs cannot be sized (and are not traded in capped runs) before that; Hyperliquid legs are priced off the Binance close, so their basis is reported from HL's premium series rather than simulated; Binance–Bybit pairs carry real basis P&L. In-sample; no walk-forward.

**Erratum, 26 Oct 2026:** the smoke run's basis was −35%/yr, not −41% (funding +48.4%, costs −26.1%, net −12.6%; `reports/mvenue/summary.json`). The parameters above are unchanged.
