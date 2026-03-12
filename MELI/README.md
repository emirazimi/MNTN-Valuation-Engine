# MELI Valuation Setup

This folder uses the same valuation engine as MNTN, but with MELI-specific company inputs and a separate MELI config.

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

This setup reuses the existing engine and current generic peer files. The next quality upgrade for MELI would be to replace the shared peer panel and peer multiples with a MELI-specific comp set.
