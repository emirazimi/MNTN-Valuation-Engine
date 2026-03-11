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
    capex_fade_exponent: float
    nwc_fade_exponent: float
    dilution_fade_exponent: float
    capex_intensity_floor: float
    capex_growth_sensitivity: float
    nwc_intensity_floor: float
    nwc_growth_sensitivity: float
    terminal_capex_pct: float
    terminal_nwc_pct: float
    dilution_rate: float
    terminal_dilution_rate: float
    sbc_pct_revenue_start: float
    sbc_pct_revenue_end: float
    sbc_fade_exponent: float
    sbc_share_issuance_price_factor: float
    overhang_years: int


@dataclass(frozen=True)
class ThesisCase:
    name: str
    start_growth_adjustment: float
    long_run_growth_override: float | None
    ebit_margin_start_adjustment: float
    terminal_ebit_margin_override: float | None
    d_and_a_pct_adjustment: float
    capex_pct_adjustment: float
    nwc_pct_adjustment: float
    terminal_growth_override: float | None
    residual_dilution_adjustment: float
    regime_label: str


@dataclass(frozen=True)
class ThesisConfig:
    base: ThesisCase
    bull: ThesisCase
    bear: ThesisCase
    bear_weight: float
    base_weight: float
    bull_weight: float


@dataclass(frozen=True)
class MathConfig:
    peer_recency_half_life_quarters: float
    peer_relevance_strength: float
    hierarchical_shrinkage: float
    growth_mean_reversion: float
    margin_mean_reversion: float
    factor_persistence: float
    macro_factor_vol: float
    execution_factor_vol: float
    capital_factor_vol: float
    idiosyncratic_growth_vol: float
    idiosyncratic_margin_vol: float
    growth_factor_loadings: dict[str, float]
    margin_factor_loadings: dict[str, float]


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
    thesis_config: ThesisConfig
    math_config: MathConfig
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
    horizon_summary_df: pd.DataFrame
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
    share_paths: Any
    valuation_paths: Any
