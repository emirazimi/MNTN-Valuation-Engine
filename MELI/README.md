# MELI Valuation Setup

This folder uses the same valuation engine as MNTN, but with MELI-specific company inputs and a separate MELI config.

## Valuation Approach

MELI should not be treated exactly like MNTN from a modeling perspective.

The shared engine is still useful for MELI in these areas:

- driver-based DCF forecasting
- scenario analysis
- latent-state filtering
- regime-aware Monte Carlo
- dilution and reinvestment modeling
- horizon return analysis

But some of the MNTN-oriented assumptions should be used much more lightly for MELI:

- peer multiples
- peer shrinkage
- small-cap rerating logic
- aggressive event-jump framing

Why: MELI is a more differentiated platform business. It combines commerce, payments, logistics, advertising, and credit in one ecosystem, so direct comparable-company valuation is inherently weaker than it is for a narrower business model. For that reason, the MELI config leans much harder on the operating thesis and DCF path than on the peer-multiple rerating layer.

The current MELI config specifically does four things:

- lowers shrinkage to peer priors so MELI-specific thesis assumptions matter more
- raises DCF weight in blended horizon valuation
- uses stronger base and bull operating cases
- keeps the market-rerating layer as a secondary cross-check rather than a primary valuation driver

## Files

- `../data/companies/MELI/snapshot.json`: MELI snapshot anchors
- `../data/companies/MELI/history.csv`: MELI operating history inputs
- `../configs/meli.json`: MELI-specific run, thesis, math, and market assumptions
- `./run_meli.sh`: helper script to run the valuation

## Run

From the repo root:

```bash
python3 -m mntn_valuation run \
  --company MELI \
  --data-dir data \
  --config configs/meli.json \
  --output-dir MELI/out
```

From inside this folder:

```bash
./run_meli.sh
```

Disable plots:

```bash
./run_meli.sh --no-plots
```

Outputs will be written to `MELI/out`.

## Notes

This setup reuses the existing engine and current generic peer files. The next quality upgrade for MELI would be to replace the shared peer panel and peer multiples with a MELI-specific comp framework or to build a dedicated MELI relative-value module around ecosystem/platform analogs rather than direct one-to-one comps.
