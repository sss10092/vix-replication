# VIX Replication

The CBOE VIX method for any underlying and any target tenor. It works on SPX-style bid/ask chains and on CME options-on-futures settlement files, such as Henry Hub natural gas, and is checked against the exact model-free variance it approximates.

## Method

For each expiry:

1. Forward from put-call parity at the strike where call minus put is smallest.
2. K0 is the first strike at or below the forward.
3. OTM puts below K0 and calls above it, walking outward until two zero bids in a row. At K0 the put/call average is used.
4. Variance from the CBOE formula.

The two expiries that bracket the target tenor are interpolated in total variance, in minutes, then annualised.

## Results

| Check | Result |
|---|---|
| Flat 12% / 25% / 60% vol | recovered within 0.05 vol pts |
| Skewed SPX-like smile, 30-day index | 18.486 vs exact 18.488 |
| Flat term structure through interpolation | 20.00 recovered |
| Target tenor not bracketed | raises instead of extrapolating |
| Strikes and prices in different units | detected from the put-call parity slope |

Error vs strike spacing (SPX-like, 30 days, 18% ATM):

| Strike spacing | Strikes used | Error (vol pts) |
|---|---|---|
| 5 | 436 | -0.004 |
| 25 | 87 | +0.009 |
| 50 | 44 | +0.051 |
| 100 | 22 | +0.213 |

Coarse strike grids bias the index upward, which matters for commodity options with few liquid strikes.

## The original 2.998 natural gas reading

Natural gas implied vol in late 2020 was around 50-60%, so 3.0 was not plausible. Leaving out K0, which the original notebook did, only costs about 0.13 vol points. The more likely cause is extrapolation: the notebook targeted 90 days using options that expire around 45 and 75 days out, which gives negative interpolation weights. A strike unit mismatch would also produce a 10x error. Both cases now raise an error, and `--strike-scale 0.01` handles strikes stored in cents. The original CSVs were not available, so run `vixcalc cme` on them to confirm.

`vixcalc sample` writes a synthetic options file and Treasury file in the same CME and treasury.gov formats, along with the exact answer for that chain. On it the CME path gives 59.02 against an exact 59.21.

## Run

```bash
pip install -e '.[dev]'
vixcalc demo
vixcalc budget
vixcalc sample
vixcalc cme --options data/sample/ng_options_synthetic.csv --cmt data/sample/treasury_cmt_synthetic.csv --date 2020-11-12 --days 30
pytest -q
```

For real data, put the CME settlement CSV (columns tradeDate, futures-expirationDate, options-strikePrice, options-optiontype, options-priorSettle) and the treasury.gov daily yield curve CSV in the folder and pass them with `--options` and `--cmt`. Add `--strike-scale 0.01` if strikes are in cents.
