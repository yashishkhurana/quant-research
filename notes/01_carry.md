# Funding-rate carry on Binance perps

*The first of a planned set of notes on Binance perps. It tests the simplest delta-neutral carry book before anything is built on top of it, and it rules it out.*

**One line:** hedged funding carry (long spot / short perp) on Binance does not pay after costs in 2023–26: rotating into high-funding coins spends the income on turnover, holding liquid coins at the default rate loses −4.3% a year because below-default funding outweighs it, and the above-default spells that funded the trade in 2023–24 became about 20× rarer in 2025–26, with the most liquid coins hit hardest. The coins that could hold $2.5m per leg, about seven at a time, paid above-default funding in under 1% of 2025–26 windows: the premium and the capacity are in different coins.

## Hypothesis
When leveraged longs crowd a perp, it trades above spot and longs pay shorts funding. A short perp hedged with long spot collects that funding without directional exposure. The edge is a premium for taking the other side of leverage demand, so it should exist only while that demand outweighs the capital willing to supply it.

Binance's funding has a floor-like default: an interest component of 0.01% per 8h (≈10.95% APR) that the rate sits at whenever the perp premium is small. So there are two distinct trades: collecting spikes **above** the default, and simply holding coins **at** the default.

## Data
- Binance USDT-M perps and Binance spot, hourly closes; funding from the fapi funding endpoint. Funding 2023-01-01 to 2026-10-21, hourly prices to 2026-10-23.
- Universe: every USDT perp Binance has listed (655, including 133 delisted). 447 have a spot market and so can be hedged; 89 of those are delisted contracts (FTT, FTM, WAVES, MKR, OCEAN, AGIX, …), so the sample includes the casualties. Perp-only coins are excluded from every hedged study.
- Settlement intervals differ: at the end of the sample 509 coins settle every 4h, 134 every 8h, 12 every 1–2h, and 169 dropped to 1h at some point (Binance does this when funding is at its cap). All funding is put on fixed 8h UTC windows by summing the settlements inside each window (`qr/data/funding.py`); the default is 0.01% per 8h window at every interval (0.005% per 4h settlement, 0.00125% per 1h).
- Liquidity is point in time: trailing 30-day perp dollar volume on completed bars only.
- Binance keeps serving perp klines after a contract is delisted, with a frozen close. Perp prices are cut at each contract's delisting time (Binance's `deliveryDate`, 134 contracts), so a delisting force-closes the position in the backtester instead of leaving a frozen perp hedged against live spot.

## Method
Three studies, each answering one question, plus a rotation backtest (`scripts/run_carry.py`, results 1 and 1b) and a capacity section with its own scripts, all listed under Reproduce:

1. **Persistence** (`scripts/funding_persistence.py`): a *spell* is a run of consecutive 8h windows with funding strictly above the default. Trade per spell: enter at the end of the first window above default (when it is known), earn from the next window, exit at the end of the first window back at or below default. Compared with a 28 bps round trip (4 legs × (4 bps fee + 3 bps slippage)). Funding only; basis PnL excluded.
2. **Default-rate carry** (`scripts/default_carry.py`): at each month start, the top 20 hedgeable coins by trailing volume, equal weight, +1/20 spot and −1/20 perp (2.0× gross), held through the month (drift not traded; delistings exit both legs). Full engine accounting: price and basis PnL, funding, 7 bps per leg. Funding is attributed to settlements at, above and below their interval's default.
3. **Funding by liquidity** (`scripts/funding_by_volume.py`): coins bucketed each window by trailing-volume rank (1–20, 21–50, 51–100, 101+); share of windows below the default, averaged per coin (coins with ≥ 30 windows in a cell), bootstrap CIs over coins.

## Results

### 1. Above-default spells are short and mostly do not pay for the trade
17,068 spells over 447 coins. Median length 2 windows (16h), mean 5.1, p90 11.

| After k windows (8h) | 2 | 3 (1d) | 9 (3d) | 21 (7d) | 42 (14d) | 153 (51d) |
| --- | --- | --- | --- | --- | --- | --- |
| Share of spells still above default | 57% | 40% | 14% | 5.6% | 1.1% | 0.0% |

A swap between coins costs ~28 bps and earns the carry difference for as long as it lasts; for a 2% APR gap the break-even is ~51 days. No spell population survives that long. Only 12.8% of spells earn their round trip; the mean spell nets −9.6 bps.

By the APR of the first window above default, mean funding net of the round trip (bps), bootstrap 95% CI:

| First-window APR | 2023–24 | 2025–26 |
| --- | --- | --- |
| 11–20% | −13.6 (−14.8, −12.2) | −23.8 (−24.6, −23.0) |
| 20–50% | −3.3 (−4.8, −1.6) | −21.2 (−23.1, −18.8) |
| 50–100% | **+10.5 (+4.7, +16.6)** | **−8.1 (−14.6, −0.4)** |
| > 100% | +25.1 (+2.4, +52.6) | +10.7 (−3.0, +27.0) |

The rule *enter above 50% APR, hold until back at default* was chosen from this table. It worked in 2023–24 and failed out of sample: the 50–100% bucket turned significantly negative in 2025–26, and > 100% kept its sign but is indistinguishable from zero (181 spells).

Spells also became rarer. The share of listed coin-windows above default fell about 20× for both settlement intervals (8h coins 13.9% → 0.7%, 4h coins 25.2% → 1.4%), while the number of listed hedgeable coins rose from 218 to 365. My first explanation, that 4h settlement shortened spells, is rejected: within each period 4h coins have slightly *longer* spells than 8h coins (2025–26 mean 2.5 vs 2.0 windows), and 8h coins, which never changed interval, collapsed just as much.

> *Corrected 2026-10-23.* An earlier version of this paragraph read raw Binance funding timestamps, which carry millisecond jitter (08:00:00.008). About a third of settlements fell into the following 8h window, which misaligned the rotation signal by one window for those settlements and made it trade more than it should. With timestamps rounded to the minute, the rotation result moved from −20.7% to −13.4% a year and the unhedged result from −0.03% to +23.3% (section 1b). Studies 1–3 already used rounded timestamps and are unchanged.

The rotation backtest shows the same thing in PnL. Book: top 10 by trailing 9-window funding among the top 40 by trailing volume, entry above 10% APR and exit below 3%, decisions at 8h window ends, 25% no-trade band, 8% drawdown kill switch, 2.0× gross, 7 bps per leg. It collected **+11.4% a year in funding and paid −26.3% a year in costs**: net −13.4% a year, Sharpe −4.2, max drawdown −42.3%, turnover 376× a year, almost all of it from selection changes (0.044 per bar from target changes against 0.043 traded; the no-trade band absorbs the difference). Six of the 25 parameter combinations have a positive Sharpe, all in the slowest-trading corner of the grid (entry 15–30% APR, lookback 9–21 windows); the best is 0.49, with a 79% probability of being above zero taken alone and 19% after deflating for 25 trials. Walk-forward out of sample returns −0.1% a year (Sharpe −0.03). The cells that stop losing are the ones that trade least, and once the rule trades that little nothing is left over: the loss is in the trading rule, and the signal is worth about zero after costs.

![rotation carry](../reports/carry/tearsheet.png)

### 1b. The unhedged book's return is its stop, not its carry
The same signal as a short-perp-only book (perp-only coins allowed) returns +23.3% a year, Sharpe 0.65, with a −77% max drawdown and a −19% worst week (week ending 2025-10-12). Four runs separate the signal from the 8% drawdown stop. The control is an equal-weight short of the same point-in-time top-40 universe with no funding signal, at the same average gross (0.95), decision clock and costs:

| Book | Stop | Price / yr | Funding / yr | Costs / yr | Net / yr | Max DD | Stop events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Carry signal | 8% DD | +51.3% | +10.5% | −14.6% | **+23.3%** | −77% | 177 |
| Carry signal | none | +4.2% | +20.6% | −14.3% | **−34.3%** | −94% | 0 |
| No signal, short top 40 | 8% DD | +49.5% | −12.6% | −5.4% | **+15.0%** | −57% | 127 |
| No signal, short top 40 | none | +13.3% | −16.2% | −1.4% | **−26.6%** | −84% | 0 |

(Price, funding and costs are simple annual sums; net is compounded.)

Removing the stop turns +23% into −34%, and the plain short of the same universe does nearly as well with the stop (+15%) and also loses without it (−27%). The signal is worth about +8 points a year with the stop and −8 without it, which is indistinguishable from zero. On a book with about 72% annual volatility an 8% stop fired 177 times and behaved like a trend filter, flat after each squeeze and back in after the 72-hour cooldown; the full-sample number records when it happened to fire, and it is not stable (the no-signal short with the stop returns +16% at gross 1.0 and −7% at 0.55). The stop also reverses the carry: flat about 40% of the time, the book collects half the funding while every stop and re-entry is a full round trip, so funding (+10.5%) no longer covers costs (−14.6%).

| Year | Price | Funding | Costs | Net (compounded) |
| --- | --- | --- | --- | --- |
| 2023 | −9.8% | +8.6% | −15.5% | −29.3% |
| 2024 | −13.7% | +10.4% | −16.6% | −37.1% |
| 2025 | +140.8% | +7.8% | −13.5% | +178.8% |
| 2026 (to Oct) | +78.1% | +13.0% | −10.2% | +79.2% |

On 1 January 2025 the book was down about 55% from its start. The two years that pay come after the two that would have shut it down.

### 2. Holding liquid coins at the default does not pay either

| | 2023–24 | 2025–26 | Full |
| --- | --- | --- | --- |
| Ann. return (net) | +2.3% | −11.2% | −4.3% |
| Ann. vol | 7.3% | 2.0% | 5.5% |
| Sharpe | 0.35 | −5.99 | −0.78 |
| Max drawdown | −9.3% | −19.7% | −19.7% |
| Price / basis PnL / yr | +0.5% | −1.0% | −0.2% |
| Funding / yr | +3.5% | −9.7% | −2.8% |
| from settlements at default | +4.7% | +3.4% | +4.1% |
| from settlements above default | +7.7% | +0.3% | +4.2% |
| from settlements below default | −8.9% | −13.4% | −11.0% |
| Costs / yr | −1.4% | −1.2% | −1.3% |
| Turnover / yr | 20× | 17× | 19× |

Costs are about a twentieth of the rotation book's. The default component pays about 4% a year, not 10.95%, because only part of the time is spent at the default; the below-default settlements cost more than that in both periods.

![default-rate carry](../reports/default_carry/tearsheet_n20.png)

### 3. Below-default funding is concentrated in the most liquid coins, increasingly so
Share of 8h windows below the default, per coin (≥ 30 windows in the cell), 95% CI:

| Volume rank | 2023–24 | 2025–26 |
| --- | --- | --- |
| 1–20 | 33.7% (29.9, 37.7) | 72.5% (68.0, 76.7) |
| 21–50 | 27.0% (24.4, 29.9) | 67.0% (63.5, 70.3) |
| 51–100 | 22.1% (20.3, 24.1) | 56.4% (53.7, 58.9) |
| 101+ | 18.1% (16.7, 19.6) | 37.5% (35.7, 39.4) |

Below-default funding rose in every bucket and is monotone in liquidity in both periods. On the odds scale, the top 20 coins' odds of below-default funding rose 5.2× against 2.7× for the tail; the ratio, 1.91 (bootstrap 95% CI 1.41–2.62 over coins), says the most liquid coins moved further. This is consistent with delta-neutral capital crowding the liquid perps: the same trade this note studies, run at scale, pushing the perp below spot so that shorts pay longs.

The same table is also consistent with **hedging that is not carry**: holders of spot BTC, ETH and SOL (treasury companies, funds, miners, ETF-adjacent desks) shorting the liquid perps to hedge their inventory, and directional shorts concentrating in the most liquid contracts in a weak market. Both put persistent short pressure on exactly the coins in the top bucket. Funding data alone cannot separate them from delta-neutral carry; that would need positioning data (open interest and its growth against funding) or the disclosed books of known delta-neutral issuers. So this note claims consistency with crowding, not proof of it.

## Capacity
How large can a hedged book get before it is the market in the coins it holds? A name's cap is a participation rate times its trailing 30-day dollar volume, taking the smaller of the perp and spot legs, on completed bars only (`qr/capacity.py`). Binance serves no open-interest history, so open interest enters only as a dated snapshot.

**The coins that can hold size do not pay.** Coins that could each take $2.5m per leg at 2% of volume, re-evaluated at every 8h window, 2025-01 to 2026-10, funding only (`scripts/sizable_funding.py`):

| Coins | Per window (median) | Below / at / above default | Mean funding (APR) |
| --- | --- | --- | --- |
| Can hold $2.5m per leg (point in time) | 7 (range 2–28) | 75.4% / 23.8% / **0.7%** | −9.1% |
| Same, filtered on the 23 Oct OI snapshot instead | 6 | 75.8% / 24.1% / 0.1% | +0.3% |
| All other hedgeable coins | 359 | 44.1% / 54.7% / 1.2% | −13.3% |

The sizable coins are mostly ETH, BTC, SOL, XRP, BNB, DOGE and SUI. A $25m hedged book at 2× gross needs about ten names at $2.5m a leg; there are about seven, and they sit below the default three quarters of the time. At size the trade becomes the default-rate book of section 2, which lost 4.3% a year. The point-in-time filter and the snapshot filter agree, so the result does not depend on choosing coins by today's open interest.

**What the signal's own book can hold.** Static capacity is the largest book at which every held name stays inside its cap (the minimum over held names of cap / weight), over the bars the rotation book is invested:

| Hedged, 2% of volume | 2023 | 2024 | 2025 | 2026 | All |
| --- | --- | --- | --- | --- | --- |
| Median | $1.4m | $4.8m | $3.4m | $1.1m | **$2.4m** |
| 10th–90th percentile | $0.7–3.8m | $2.0–11.7m | $1.7–6.8m | $0.7–2.3m | $0.9–7.5m |

The median is $1.2m at 1% of volume and $6.1m at 5%, linear in the participation rate by construction. Capacity peaked in 2024 and in 2026 is about a quarter of that: as the premium moved into smaller coins, the book followed it into coins that cannot hold size. The book holds a coin in its first week as a tradeable pair (too new to have a volume record, so no cap) on about 9% of bars, but those positions are about 1% of its exposure and 2–3% of its funding, so they barely move the figure. The unhedged book's median is $18m at 2% of volume, but its perp-only holdings are where the volume cap is least reliable (below).

**Impact.** The capacity curve (`scripts/run_capacity.py`, `reports/carry/capacity_curve.png`) runs the capped book from $1m to $100m and charges square-root impact on every trade (coefficient 1, on top of the flat 7 bps). No size is positive, hedged or unhedged. At $1m and 2% of volume, impact alone costs the hedged book about 100% a year at 376× turnover, against 11% of funding. Losses shrink at larger sizes only because the caps leave most of the capital idle (35% deployed at $25m). The coefficient is a rough convention, but the conclusion does not depend on it: the book already loses at the flat 7 bps.

**Open-interest check on the current book.** Target weights held in the seven days to 23 Oct 2026, against open interest on 23 Oct (`scripts/oi_check.py`):

| Held name | Share of the week held | Volume / OI | Book at 1% of OI | Book at 2% of volume |
| --- | --- | --- | --- | --- |
| LSK | 28% | 9.3 | $0.34m | $0.64m |
| MARSCOIN | 98% | 4.0 | $0.83m | $1.8m |
| MUBARAK | 98% | 4.0 | $1.06m | $1.6m |
| 牛来 | 19% | 2.1 | $1.11m | $1.16m |
| STRK | 34% | 7.9 | $1.7m | $3.1m |
| PUMP | 86% | 3.7 | $5.1m | $4.9m |
| PAXG | 28% | 0.6 | $7.8m | $1.9m |
| ZEC | 86% | 3.4 | $37m | $31m |

The book's size is set by its least liquid holding, and the signal's holdings are small: LSK, MARSCOIN, MUBARAK and STRK each cap the hedged book at $0.3–1.7m before the position is 1% of the contract's open interest, and only ZEC could take eight figures. Open interest is the tighter limit for five of the eight hedged names and ten of the thirteen unhedged ones (`scripts/oi_check.py`, `reports/carry/oi_check_*.csv`); for perp-only coins whose daily volume is many times their open interest (US, 龙虾, BR) the volume cap overstates capacity 26–37×, so the unhedged $18m is an upper bound. This is one week of target holdings against one day of open interest and cannot be run over the backtest; at size the book's own short would also be part of the open interest it is measured against.

*Added 2026-10-25.* With open-interest history from a second data source (CryptoHFTData, from mid-2025; agrees with our 23 Oct snapshot to a median 0.6% across 230 coins), the same check runs over every 8h window from 1 Jul 2025 to 23 Oct 2026 (`scripts/oi_check.py --history`, `reports/carry/oi_history_*.csv`):

| Book | Windows | Median book at 2% of volume | Median book also capped at 1% of OI | OI is the tighter cap |
| --- | --- | --- | --- | --- |
| Hedged | 1,429 | $1.55m | $0.89m | 73% of windows |
| Unhedged | 1,437 | $13.4m | $0.50m | 100% of windows |

Restricting to windows where every held name has OI (88% hedged, 78% unhedged) gives $0.85m and $0.47m. The check said above to be impossible over the backtest now runs over its last 16 months: the hedged book's median is $0.89m, inside the snapshot's $0.3–1.7m range, and in the snapshot week itself it is $0.86m, set by the same names (MARSCOIN, LSK). The unhedged $18m above is wrong, not just an upper bound: over the same windows the volume cap gives $13.4m, and adding open interest cuts it to $0.50m, 27× less, because open interest is the tighter limit in every window. By quarter, the hedged book's median with open interest falls from $1.6m in Q3 2025 to $0.4m in Q2 2026 and returns to $1.6m only in October, close to the decline the volume caps show (`reports/carry/oi_history_hedged.csv`), now measured against a second data source.

## Validation
- **Out of sample:** the one rule chosen from the data (> 50% APR entry) was tested on the period split it was not tuned on and failed in 2025–26.
- **Survivorship:** the universe is every listed contract, including 89 hedgeable delisted coins; liquidity is ranked point in time. An earlier version that used today's top 40 by volume was discarded for selecting on survival.
- **Unit check:** default per settlement verified per interval (0.01% at 8h, 0.005% at 4h, 0.00125% at 1h) before attributing funding; funding attribution sums exactly to the engine's funding PnL.
- **Robustness:** the volume-bucket ordering is unchanged when coins that passed briefly through a bucket are dropped; bootstrap CIs resample coins, the independent unit.
- **Multiple testing:** 6 of the 25 rotation cells are positive. The best (lookback 21, entry 20% APR, Sharpe 0.49) has a 79% probability of a Sharpe above zero taken alone and 19% after deflating for 25 trials against a zero-skill benchmark (the best of 25 skill-less strategies would be expected to show a Sharpe of about 1.0).
- **Controls:** the unhedged result was checked against the same book without its stop and against a no-signal short of the same universe at matched gross (section 1b); the capacity result was checked with a point-in-time filter and a snapshot filter that agree.
- **Not done:** the persistence study is funding only (no basis PnL); 7 bps per leg is optimistic for small-cap spot; capacity is a volume cap with square-root impact, not a fill simulation.

## What broke
- **Rotation spends the carry on turnover.** Re-ranking every 8h swaps about one position every one to two settlements; each swap costs ~28 bps against a fraction of a basis point of extra carry per window, and the break-even (~51 days for a 2% APR gap) is longer than almost every spell.
- **The spell rule was in-sample.** The 50% threshold came from the same table that justified it, and the 2025–26 split rejected it.
- **The default is not free.** Holding liquid coins at the default earns ~4% a year from default settlements and gives back ~11% to below-default ones.
- **The opportunity shrank.** Above-default funding became ~20× rarer in 2025–26 across all settlement intervals, and below-default funding rose most in the coins with the most capacity.
- **My first explanation was wrong.** 4h settlement does not explain shorter spells; the collapse is market-wide.
- **A timestamp bug inflated the rotation's losses.** The rotation script read raw funding times, which carry millisecond jitter; with 8h windows closed on the right, a 08:00:00.008 settlement landed in the window ending 16:00. About a third of settlements were misaligned, the signal churned, and turnover was 511× a year instead of 376×. It was found when the capacity study, which loaded funding through the rounded loader, could not reproduce the note's baseline; the two loaders now agree to four decimals and `per_window` rounds timestamps itself.
- **The unhedged book's risk layer was its strategy.** An 8% drawdown stop built for a hedged book fired 177 times on a 72%-vol short book and produced the whole of its +23%; without the stop the same book loses 34% a year.

- **Why nobody takes the other side.** Collecting below-default funding means long perp and **short spot**, and shorting spot means borrowing the coin on margin; the borrow rate, its limited and recallable supply, and the small size of the shortfall below default (a few bps per window on majors) mean the reverse trade does not pay either, so the imbalance persists. Borrow costs were not measured here; that is the next thing to check before calling it closed.

- **What predicts below-default funding at decision time.** Liquidity: a coin in the top 20 by trailing volume was below default in ~72% of 2025–26 windows against ~37% for the tail, and that rank is known in advance. It is only half an answer because it says *where*, not *when*: it is a cross-sectional base rate, not a timing signal, and avoiding the liquid coins leaves the illiquid tail, where 7 bps per leg is optimistic, spot depth caps size, and delistings live.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest                  # 50 tests
bash scripts/fetch_carry_data.sh 2023-01-01        # universe + funding + hourly perp/spot (~1–2 h)
# the first study run builds the funding and price matrices in data/processed/ (~15 min, once)
python scripts/funding_persistence.py              # study 1
python scripts/default_carry.py --n 20             # study 2 -> reports/default_carry/
python scripts/funding_by_volume.py --min-windows 30   # study 3
python scripts/run_carry.py --end 2026-10-23       # rotation, grid, walk-forward, unhedged 2x2 (~40 min)
python scripts/fetch_oi_snapshot.py                # today's open interest (dated; needs network)
python scripts/sizable_funding.py                  # capacity: funding in coins that can hold $2.5m a leg
python scripts/run_capacity.py                     # static capacity and capacity curve
python scripts/oi_check.py                         # current holdings vs the OI snapshot
```
