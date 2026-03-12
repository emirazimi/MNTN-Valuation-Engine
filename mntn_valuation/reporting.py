from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import ValuationResults


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_scenario_valuation_bar(results: ValuationResults, output_dir: Path) -> None:
    df = results.scenario_df
    plt.figure(figsize=(9, 5))
    plt.bar(df["Scenario"], df["Value / Share"])
    plt.axhline(results.snapshot.current_price, linestyle="--", label=f"Current Price = ${results.snapshot.current_price:.2f}")
    plt.title(f"{results.company_id} Scenario DCF")
    plt.ylabel("Value / Share ($)")
    plt.legend()
    _savefig(output_dir / "scenario_valuation_bar.png")


def plot_histogram(results: ValuationResults, output_dir: Path) -> None:
    values = results.values
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=60, edgecolor="black", alpha=0.85)
    plt.axvline(results.snapshot.current_price, linestyle="--", label=f"Current Price = ${results.snapshot.current_price:.2f}")
    plt.axvline(np.median(values), linestyle="--", label=f"Median = ${np.median(values):.2f}")
    plt.title(f"{results.company_id} Valuation Distribution")
    plt.xlabel("Value / Share ($)")
    plt.ylabel("Frequency")
    plt.legend()
    _savefig(output_dir / "valuation_histogram.png")


def plot_regime_heatmap(results: ValuationResults, output_dir: Path) -> None:
    states = ["Bear", "Base", "Bull"]
    counts = np.zeros((results.regime_paths.shape[1], len(states)))
    for t in range(results.regime_paths.shape[1]):
        for j, state in enumerate(states):
            counts[t, j] = np.mean(results.regime_paths[:, t] == state)
    plt.figure(figsize=(8, 4))
    plt.imshow(counts, aspect="auto")
    plt.xticks([0, 1, 2], states)
    plt.yticks(range(results.regime_paths.shape[1]), [f"Year {i}" for i in range(1, results.regime_paths.shape[1] + 1)])
    plt.colorbar(label="Probability")
    plt.title("Projected Regime Probabilities by Year")
    _savefig(output_dir / "regime_heatmap.png")


def plot_mc_fan_chart(results: ValuationResults, output_dir: Path) -> None:
    valuation_paths = results.valuation_paths
    years = np.arange(1, valuation_paths.shape[1] + 1)
    p10 = np.percentile(valuation_paths, 10, axis=0)
    p25 = np.percentile(valuation_paths, 25, axis=0)
    p50 = np.percentile(valuation_paths, 50, axis=0)
    p75 = np.percentile(valuation_paths, 75, axis=0)
    p90 = np.percentile(valuation_paths, 90, axis=0)
    plt.figure(figsize=(12, 7))
    plt.fill_between(years, p10, p90, alpha=0.2, label="P10-P90")
    plt.fill_between(years, p25, p75, alpha=0.35, label="P25-P75")
    plt.plot(years, p50, linewidth=2, label="Median")
    plt.axhline(results.snapshot.current_price, linestyle="--", label=f"Current Price = ${results.snapshot.current_price:.2f}")
    plt.title(f"{results.company_id} Intrinsic Value Fan Chart")
    plt.xlabel("Projection Year")
    plt.ylabel("Value / Share ($)")
    plt.legend()
    _savefig(output_dir / "valuation_fan_chart.png")


def export_results(results: ValuationResults, output_dir: str | Path, include_plots: bool | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_plots = results.run_config.include_plots if include_plots is None else include_plots

    results.scenario_df.to_csv(output_dir / "scenario_table.csv", index=False)
    results.start_probs_df.to_csv(output_dir / "start_probabilities.csv", index=False)
    results.base_transition_df.to_csv(output_dir / "base_transition_matrix.csv")
    results.macro_transition_df.to_csv(output_dir / "macro_transition_matrix.csv")
    results.summary_df.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    results.simulation_df.to_csv(output_dir / "monte_carlo_simulations.csv", index=False)
    results.horizon_summary_df.to_csv(output_dir / "horizon_valuation_summary.csv", index=False)
    results.return_summary_df.to_csv(output_dir / "return_summary.csv", index=False)
    results.ending_regime_df.to_csv(output_dir / "ending_regime_breakdown.csv", index=False)
    results.driver_corr_df.to_csv(output_dir / "driver_correlations.csv", index=False)
    results.multiples_df.to_csv(output_dir / "multiple_cross_check.csv", index=False)
    results.tornado_df.to_csv(output_dir / "tornado_sensitivity.csv", index=False)
    results.sobol_df.to_csv(output_dir / "sobol_sensitivity.csv", index=False)
    results.structural_sobol_df.to_csv(output_dir / "structural_sobol_sensitivity.csv", index=False)

    manifest = {
        "company_id": results.company_id,
        "run_config": results.run_config.__dict__,
        "current_price": results.snapshot.current_price,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    if include_plots:
        plot_scenario_valuation_bar(results, output_dir)
        plot_histogram(results, output_dir)
        plot_regime_heatmap(results, output_dir)
        plot_mc_fan_chart(results, output_dir)

    return output_dir
