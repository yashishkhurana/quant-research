# Predicting above-default funding with a classifier

*Fifth note. Does a model find anything in Binance funding that study 1's persistence rule does not?*

**One line:** the classifier is a random forest that predicts whether funding stays above the default, and it doesn't cover the trade: the model matches the rule, and the windows it picks pay only 1/7th of the round-trip cost, adding up to nothing.

## Question
At the end of an 8h window, will the next window pay above the default? A model is useful only if it beats the rule "it was above the default last window", and only if the windows it picks cover a 28 bp hedged round trip.

## Method
- Data: every hedgeable Binance coin, 8h windows, Jan 2023 – Oct 2026. 1.13m coin-windows, 444 coins. 7.7% of next windows are above the default.
- Features, all known at the window end: last, 3-, 9- and 90-window mean funding; above the default now; spell length; trailing-volume rank; log volume; days since first funding in the sample (from 2023-01-01, so truncated for older coins); settlement interval (forward-filled); hour.
- Models: the rule (no fitting; the score is last window's funding), logistic regression, random forest (300 trees, depth 8, ≥ 500 per leaf).
- Validation: five expanding time-ordered folds, one-window gap. Only out-of-fold predictions are scored: 1.03m rows from Sep 2023.
- Comparison: each model's k most confident windows, where k is the number of windows the rule fires on.

Code: `scripts/predict_above_default.py`. Output: `reports/predict/`.

## Results

| Model | AUC | Precision at the rule's coverage (k = 79,529) | Next-window funding there |
| --- | --- | --- | --- |
| Rule: above the default last window | 0.935 | 81.2% | 3.65 bp |
| Random forest | 0.969 | 81.2% | 3.67 bp |
| Logistic | 0.799 | 57.5% | 2.93 bp |

By year, k set within each year:

| Year | Windows | Above default | Rule fires | Rule precision | Forest precision | Next funding, rule / forest |
| --- | --- | --- | --- | --- | --- | --- |
| 2023 (Sep–Dec) | 67,806 | 26.9% | 18,078 | 81.0% | 81.0% | 3.05 / 3.05 bp |
| 2024 | 266,397 | 20.2% | 53,953 | 84.8% | 84.7% | 3.87 / 3.88 bp |
| 2025 | 370,328 | 1.1% | 3,977 | 57.9% | 58.8% | 4.34 / 4.41 bp |
| 2026 (to 21 Oct) | 324,906 | 1.1% | 3,521 | 53.9% | 54.5% | 2.63 / 2.67 bp |

Forest importances: last window's funding 0.30, above the default now 0.21, spell length 0.21, 3- and 9-window means 0.12 each, volume rank 0.001. The forest's higher AUC comes from windows the rule does not fire on. A trade would not take those.

## Caveats
- The rule's top-decile precision (12.4%) means nothing. Most windows sit exactly at the default, so ties decide its top decile. Comparisons use the rule's coverage instead.
- Volume rank's 0.001 is untested. It may predict where funding is low rather than whether it stays above the default. It may simply add nothing to past funding.
- In-sample feature choice. One forest configuration. Funding only, no basis.

## Reproduce
```bash
python scripts/predict_above_default.py      # -> reports/predict/
```
