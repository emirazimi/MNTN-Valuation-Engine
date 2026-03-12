# MNTN Valuation Engine

Data-backed Python valuation engine for MNTN with a reusable package core. The model now runs off local input files instead of hard-coded sample builders and exposes a stable library + CLI workflow.

## Features

- Discounted cash flow (DCF)
- Monte Carlo simulation
- Empirical-Bayes priors from comparable companies
- Macro-aware regime switching
- Hidden Markov Model regime smoothing
- Kalman-filtered and particle-filtered latent business quality state
- Wasserstein-distance peer relevance weighting
- t-copula joint prior sampling for correlated valuation drivers
- Regime-switching jump diffusion with stochastic volatility
- EV/Revenue and EV/EBITDA cross-checks
- Blended horizon valuation with rerating-aware market paths
- Return and alpha summary versus a benchmark CAGR
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

`configs/mntn.json` now has these sections:

- `run`: simulation size, seed, and shares basis
- `regime`: regime-dependent growth/margin/WACC shocks
- `forecast`: operating forecast assumptions such as fade speed, reinvestment intensity, and dilution
  The default config keeps a 5-year explicit forecast and also exports horizon-specific valuation summaries for 1Y, 2Y, 5Y, and a long-dated 10Y view derived from the terminal-state assumptions.
  The share-count bridge now separates residual dilution, SBC-driven issuance, and unwind of the current basic-to-diluted overhang.
- `thesis`: explicit bear/base/bull operating cases that shape the medium-term MNTN story instead of relying only on generic parameter spreads
- `math`: advanced stochastic settings for weighted peer priors, Wasserstein peer similarity, HMM smoothing, Kalman / particle filters, copula sampling, and stochastic-volatility path generation
- `market`: benchmark return assumptions and rerating-aware horizon valuation settings

## Advanced Math Layer

The current model uses several statistical layers beyond a basic DCF:

- `Wasserstein distance`: peer weighting now accounts for full-distribution similarity in growth, margin, and FCFF margin rather than only comparing the latest quarter.
- `Kalman filtering`: historical MNTN growth, margin, and FCFF margin are filtered into smoother latent state estimates before they feed the stochastic engine.
- `Particle filtering`: a latent business-quality scalar is estimated sequentially and then sampled into the Monte Carlo to tilt long-run growth and margin outcomes.
- `Hidden Markov Model`: peer-derived regime emissions and transition probabilities are used to smooth the starting regime probabilities from MNTN’s recent operating history.
- `t-copula sampling`: growth, margin, D&A, capex, NWC, and terminal-growth priors are drawn jointly with tail dependence rather than as independent random variables.
- `Stochastic volatility`: annual growth and margin shocks now have volatility paths that mean-revert and change by regime.
- `Jump diffusion`: jump probabilities and magnitudes scale by regime, allowing asymmetric event shocks in small-cap scenarios.

## Methodology

This engine is designed as a latent-state, regime-aware valuation framework for MNTN. The objective is not only to estimate a static intrinsic value, but to translate a medium-term investment thesis into a probabilistic range of operating and valuation outcomes over a 1-5 year horizon.

The process begins with local company and peer inputs. MNTN's recent operating history is filtered through Kalman methods to estimate smoother latent levels of growth, margin, and cash generation. A particle filter is then used to estimate a latent business-quality factor, which acts as a hidden state variable inside the simulation engine and influences long-run growth and profitability paths.

Peer calibration is performed using both point-in-time similarity and Wasserstein-distance similarity. This means peer relevance is determined not only by who looks similar in the latest quarter, but also by which companies have historically exhibited operating behavior that resembles MNTN across growth, margin, and cash-flow distributions. These peer weights feed the empirical-Bayes prior construction for the core valuation drivers.

The regime engine models the business environment through latent Bear, Base, and Bull states. A transition matrix is estimated from peer histories, adjusted for macro conditions, and refined using a Hidden Markov Model. The HMM uses MNTN's observed operating history to infer a smoothed posterior over latent regimes, producing a more disciplined starting regime distribution than a purely heuristic assignment.

The Monte Carlo layer draws fundamental assumptions jointly through a t-copula. This preserves realistic dependence across long-run growth, margins, reinvestment intensity, and terminal assumptions, including tail dependence in stronger and weaker outcomes. Simulated operating paths then evolve through mean-reverting processes driven by latent factors, stochastic volatility, and regime-scaled jump shocks, which is intended to better reflect the behavior of a smaller-cap growth company.

Each simulated path is converted into a full operating forecast covering revenue, margins, cash flow, reinvestment, dilution, and share count. Those paths are then valued in two ways: a DCF-based intrinsic-value path and a market-based rerating path derived from peer multiple logic. The model blends these views by horizon so that shorter and medium-term outputs reflect both operating execution and the potential for market revaluation.

The final outputs are designed to support investment decisions rather than only valuation mechanics. In addition to scenario analysis and Monte Carlo valuation summaries, the engine reports 1Y, 2Y, and 5Y horizon values, benchmark-relative return and alpha estimates, regime diagnostics, and simulation-level path detail. The result is a framework built to answer not only what MNTN may be worth, but what distribution of outcomes an investor is underwriting over the actual holding period.

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
- start probabilities including peer, macro, and HMM posterior views
- Monte Carlo summary and simulation detail
- horizon valuation summary (1Y / 2Y / 5Y / 10Y)
- return summary with CAGR, alpha vs benchmark, and probability of beating the benchmark
- multiples cross-check
- tornado sensitivity
- Sobol sensitivity tables
- run manifest
- optional PNG charts

## Testing

```bash
python3 -m unittest discover -s tests -v
```
