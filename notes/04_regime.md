# Regime-filtered hedged carry on Binance

*Fourth note. Parameters were fixed before the run (`notes/04_regime_params.md`).*

**One line:** there was almost nothing to switch on: most sizable coins paid above the default in just a few months around 2024 and never since, and a switch couldn't create a regime that didn't exist.

## Question
Can a $10m book run hedged carry if it holds only coins that take $1m a leg, only in months when they pay above the default, at maker fees, rebalanced monthly?

## Method
Data as the carry note, Jan 2023 – Oct 2026. Each month start:
- Sizable universe: 2% × min(trailing 30-day spot, perp volume) ≥ $1m.
- Switch ON if the equal-weight mean of their trailing 30-day funding is above the default (10.95% APR).
- ON: hold the sizable coins above the default; if more than 20 qualify, the 20 with the highest trailing funding (26–53 qualified in every ON month). ≤ 10% a leg, 2.0× gross.
- OFF: flat, earning nothing.

Controls: the same book always on; taker fees. Code: `scripts/regime_carry.py`. Output: `reports/regime_carry/`.

## Results

| Book | Period | Return / yr | Funding / yr | Costs / yr | Share of time ON | Return / yr while ON |
| --- | --- | --- | --- | --- | --- | --- |
| Regime, maker | 2023–24 | +4.6% | +4.5% | −0.21% | 17% | +26.8% |
| Regime, maker | 2025–26 | −0.03% | 0.0% | −0.03% | 0% | — |
| Regime, maker | full | +2.4% | +2.4% | −0.12% | 8.8% | +26.8% |
| Always on, maker | 2023–24 | +9.0% | +9.0% | −0.50% | 54% | +16.0% |
| Always on, maker | 2025–26 | +0.5% | +0.6% | −0.11% | 18% | +2.6% |
| Regime, taker | full | +2.2% | +2.4% | −0.25% | 8.8% | +25.9% |

For the always-on book, "time ON" is time holding anything.

**The switch.** The mean was above the default in 4 of the 45 months with a sizable universe (Jan 2023 had none): Jan, Mar, Apr and Dec 2024. Most sizable coins paid above the default in 7: those four, plus Dec 2023, Feb 2024 and Jan 2025. In Dec 2023 and Jan 2025 most coins paid but the mean did not. A median switch would have added the three extra months. Their return was not run. Neither measure was above the default in the 21 months from Feb 2025. The expectation going in, ON in most of 2023–24, failed.

**Size.** Static capacity at 2% of volume, while holding: median $14.3m (p10 $9.2m, p90 $21.4m). No open-interest cap: the vendor's OI starts mid-2025, after the last ON month. No impact charge.

## Caveats
- +26.8% is annualised over about 2,950 ON hours in four months. It is not a yearly return. Do not pair it with $14.3m.
- Sharpe ratios of 4.6–7.2 (full and 2023–24) and up to 18 while ON reflect smooth returns over few hours, not significance.
- In-sample. Threshold and windows not tuned. No grid, no impact. Cash earns nothing when OFF.

## Reproduce
```bash
python scripts/regime_carry.py      # -> reports/regime_carry/
```
