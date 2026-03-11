from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CompanySnapshot:
    company_id: str
    current_price: float
    revenue_2025: float
    ebit_2025: float
    adj_ebitda_2025: float
    d_and_a_2025: float
    capex_2025: float
    cash_2025: float
    debt_2025: float
    shares_outstanding: float
    shares_outstanding_diluted: float
    revenue_2026_mid: float
    adj_ebitda_2026_mid: float


@dataclass(frozen=True)
class HistoricalSeries:
    company_id: str
    data: pd.DataFrame


@dataclass(frozen=True)
class PeerPanel:
    data: pd.DataFrame


@dataclass(frozen=True)
class PeerMultiples:
    data: pd.DataFrame


@dataclass(frozen=True)
class RegimeConfig:
    regime_wacc_base: dict[str, float]
    regime_growth_shift: dict[str, float]
    regime_margin_shift: dict[str, float]
    jump_params: dict[str, float]


@dataclass(frozen=True)
class ForecastConfig:
    projection_years: int
    tax_rate: float
    revenue_fade_exponent: float
    margin_fade_exponent: float
    capex_intensity_floor: float
    capex_growth_sensitivity: float
    nwc_intensity_floor: float
    nwc_growth_sensitivity: float
    dilution_rate: float


@dataclass(frozen=True)
class ValuationRunConfig:
    n_sims: int = 30000
    seed: int = 42
    macro_stress_level: float = 0.50
    shrinkage: float = 0.65
    sobol_n_base: int = 512
    structural_sobol_n_base: int = 256
    shares_basis: str = "basic"
    include_plots: bool = True

    def resolve_shares(self, snapshot: CompanySnapshot) -> float:
        if self.shares_basis == "diluted":
            return snapshot.shares_outstanding_diluted
        return snapshot.shares_outstanding


@dataclass(frozen=True)
class ValuationInputs:
    company_id: str
    snapshot: CompanySnapshot
    history: HistoricalSeries
    peer_panel: PeerPanel
    peer_multiples: PeerMultiples
    regime_config: RegimeConfig
    forecast_config: ForecastConfig
    run_config: ValuationRunConfig
    config_path: Path
    data_dir: Path


@dataclass
class ValuationResults:
    company_id: str
    snapshot: CompanySnapshot
    run_config: ValuationRunConfig
    priors: dict[str, dict[str, float]]
    scenario_df: pd.DataFrame
    scenario_detailed: dict[str, dict[str, Any]]
    start_probs_df: pd.DataFrame
    base_transition_df: pd.DataFrame
    macro_transition_df: pd.DataFrame
    summary_df: pd.DataFrame
    simulation_df: pd.DataFrame
    ending_regime_df: pd.DataFrame
    driver_corr_df: pd.DataFrame
    multiples_df: pd.DataFrame
    tornado_df: pd.DataFrame
    sobol_df: pd.DataFrame
    structural_sobol_df: pd.DataFrame
    values: Any
    regime_paths: Any
    growth_paths: Any
    margin_paths: Any
    wacc_paths: Any
    valuation_paths: Any
