# MNTN Valuation Engine

Data-backed Python valuation engine for MNTN with a reusable package core. The model now runs off local input files instead of hard-coded sample builders and exposes a stable library + CLI workflow.

## Features

- Discounted cash flow (DCF)
- Monte Carlo simulation
- Empirical-Bayes priors from comparable companies
- Macro-aware regime switching
- EV/Revenue and EV/EBITDA cross-checks
- Tornado and Sobol sensitivity analysis
- CSV export and optional plot export

## Repo Layout

- `mntn_valuation/`: package modules for data loading, model logic, reporting, and CLI
- `data/`: local source inputs
- `configs/`: run and regime settings
- `tests/`: unit and integration coverage
- `MTMN Model Updated.py`: legacy research script kept for reference

## Input Files

- `data/companies/MNTN/snapshot.json`
- `data/companies/MNTN/history.csv`
- `data/peers/peer_panel.csv`
- `data/peers/peer_multiples.csv`
- `configs/mntn.json`

`configs/mntn.json` now has three sections:

- `run`: simulation size, seed, and shares basis
- `regime`: regime-dependent growth/margin/WACC shocks
- `forecast`: operating forecast assumptions such as fade speed, reinvestment intensity, and dilution

## Usage

Run the model:

```bash
python3 -m mntn_valuation run \
  --company MNTN \
  --data-dir data \
  --config configs/mntn.json \
  --output-dir out/mntn
```

Disable plots:

```bash
python3 -m mntn_valuation run \
  --company MNTN \
  --data-dir data \
  --config configs/mntn.json \
  --output-dir out/mntn \
  --no-plots
```

## Library API

```python
from mntn_valuation import export_results, load_inputs, run_valuation

inputs = load_inputs("MNTN", "data", "configs/mntn.json")
results = run_valuation(inputs)
export_results(results, "out/mntn")
```

## Outputs

The CLI writes:

- scenario table
- transition matrices
- Monte Carlo summary and simulation detail
- multiples cross-check
- tornado sensitivity
- Sobol sensitivity tables
- run manifest
- optional PNG charts

## Testing

```bash
python3 -m unittest discover -s tests -v
```
