from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from mntn_valuation import export_results, load_inputs, run_valuation
from mntn_valuation.model import (
    apply_growth_margin_premium,
    assign_empirical_states,
    build_filtered_history_state,
    build_operating_forecast,
    build_prior_copula_corr,
    dcf_fcff_regime_discounted,
    dcf_from_operating_forecast,
    estimate_hmm_emissions,
    estimate_transition_matrix,
    enforce_transition_floor,
    fit_empirical_bayes_priors,
    hmm_smoothed_state_probs,
    sample_bayesian_prior,
    sample_t_copula_uniforms,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "configs" / "mntn.json"


class ModelTests(unittest.TestCase):
    def test_dcf_invariant_positive_value(self) -> None:
        result = dcf_fcff_regime_discounted(
            revenue0=100.0,
            growth_rates=[0.10] * 5,
            ebit_margin_start=0.10,
            ebit_margin_end=0.15,
            tax_rate=0.25,
            d_and_a_pct=0.03,
            capex_pct=0.04,
            nwc_pct_of_incremental_rev=0.02,
            regime_path=["Base"] * 5,
            regime_wacc_path=[0.10] * 5,
            terminal_growth=0.03,
            cash=10.0,
            debt=0.0,
            shares=10.0,
        )
        self.assertGreater(result["value_per_share"], 0)

    def test_transition_floor_normalizes_rows(self) -> None:
        matrix = pd.DataFrame([[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]], columns=list("ABC"), index=list("ABC"))
        floored = enforce_transition_floor(matrix, floor=1e-4)
        for value in floored.sum(axis=1):
            self.assertAlmostEqual(value, 1.0, places=6)
        self.assertTrue((floored.values > 0).all())

    def test_sample_bayesian_prior_stays_bounded(self) -> None:
        prior = {"mu": 0.2, "sigma": 0.05, "low": 0.1, "high": 0.3}
        draw = sample_bayesian_prior(0.99, prior)
        self.assertGreaterEqual(draw, prior["low"])
        self.assertLessEqual(draw, prior["high"])

    def test_growth_margin_premium_is_floored(self) -> None:
        premium = apply_growth_margin_premium(0.2, -1.0, 1.0, -1.0, 1.0)
        self.assertGreaterEqual(premium, 0.1)

    def test_operating_forecast_builds_driver_paths(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        forecast = build_operating_forecast(
            inputs=inputs,
            revenue0=inputs.snapshot.revenue_2026_mid,
            start_growth=0.20,
            long_run_growth=0.10,
            ebit_margin_start=0.08,
            terminal_ebit_margin=0.18,
            d_and_a_pct=0.03,
            capex_base_pct=0.04,
            nwc_base_pct=0.02,
            terminal_growth=0.03,
            regime_path=["Base"] * inputs.forecast_config.projection_years,
            regime_wacc_path=[0.10] * inputs.forecast_config.projection_years,
            shares_start=inputs.snapshot.shares_outstanding,
        )
        self.assertEqual(len(forecast), inputs.forecast_config.projection_years)
        self.assertIn("growth_rate", forecast.columns)
        self.assertIn("sbc_share_issuance", forecast.columns)
        self.assertIn("overhang_release", forecast.columns)
        self.assertGreater(forecast["sbc_share_issuance"].sum(), 0)
        self.assertGreater(forecast["shares"].iloc[-1], inputs.snapshot.shares_outstanding)

    def test_dcf_from_operating_forecast_positive(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        forecast = build_operating_forecast(
            inputs=inputs,
            revenue0=inputs.snapshot.revenue_2026_mid,
            start_growth=0.18,
            long_run_growth=0.08,
            ebit_margin_start=0.08,
            terminal_ebit_margin=0.17,
            d_and_a_pct=0.03,
            capex_base_pct=0.04,
            nwc_base_pct=0.02,
            terminal_growth=0.03,
            regime_path=["Base"] * inputs.forecast_config.projection_years,
            regime_wacc_path=[0.10] * inputs.forecast_config.projection_years,
            shares_start=inputs.snapshot.shares_outstanding,
        )
        result = dcf_from_operating_forecast(forecast, 0.03, inputs.snapshot.cash_2025, inputs.snapshot.debt_2025)
        self.assertGreater(result["value_per_share"], 0)

    def test_filtered_history_state_returns_latent_metrics(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        state = build_filtered_history_state(inputs.history.data, inputs.math_config)
        self.assertIn("filtered_growth", state)
        self.assertIn("latent_quality", state)
        self.assertIn("particle_quality", state)

    def test_t_copula_uniforms_are_bounded(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        corr = build_prior_copula_corr(inputs.peer_panel.data)
        uniforms = sample_t_copula_uniforms(32, corr, inputs.math_config.copula_df, seed=42)
        self.assertEqual(uniforms.shape, (32, 6))
        self.assertTrue(((uniforms > 0) & (uniforms < 1)).all())

    def test_hmm_smoothed_probs_normalize(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        state_df = assign_empirical_states(inputs.peer_panel.data)
        transition_df = estimate_transition_matrix(state_df)
        emissions = estimate_hmm_emissions(state_df, inputs.math_config.hmm_emission_scale)
        probs = hmm_smoothed_state_probs(
            inputs.history.data,
            start_probs=np.array([0.2, 0.5, 0.3]),
            transition_df=transition_df,
            emissions=emissions,
        )
        self.assertEqual(probs.shape[1], 3)
        for row_sum in probs.sum(axis=1):
            self.assertAlmostEqual(row_sum, 1.0, places=6)


class IntegrationTests(unittest.TestCase):
    def test_load_inputs_and_priors(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        priors = fit_empirical_bayes_priors(inputs, inputs.peer_panel.data)
        self.assertEqual(inputs.snapshot.company_id, "MNTN")
        self.assertIn("long_run_growth", priors)
        self.assertEqual(inputs.thesis_config.base.name, "Base")
        self.assertGreater(inputs.math_config.growth_mean_reversion, 0)

    def test_end_to_end_run_and_export(self) -> None:
        inputs = load_inputs("MNTN", DATA_DIR, CONFIG_PATH)
        fast_config = inputs.run_config.__class__(
            n_sims=250,
            seed=inputs.run_config.seed,
            macro_stress_level=inputs.run_config.macro_stress_level,
            shrinkage=inputs.run_config.shrinkage,
            sobol_n_base=32,
            structural_sobol_n_base=16,
            shares_basis=inputs.run_config.shares_basis,
            include_plots=False,
        )
        results = run_valuation(inputs, run_config=fast_config)
        self.assertFalse(results.summary_df.empty)
        self.assertFalse(results.horizon_summary_df.empty)
        self.assertFalse(results.return_summary_df.empty)
        self.assertIn("median", results.summary_df["Metric"].tolist())
        self.assertIn("5Y", results.horizon_summary_df["Horizon"].tolist())
        self.assertIn("5Y", results.return_summary_df["Horizon"].tolist())
        self.assertIn("ending_shares", results.simulation_df.columns)
        self.assertIn("year1_sbc_shares", results.simulation_df.columns)
        self.assertIn("year5_blended_value", results.simulation_df.columns)
        self.assertIn("year5_alpha_vs_benchmark", results.simulation_df.columns)
        self.assertEqual(set(results.scenario_df["Scenario"].tolist()), {"Bear", "Base", "Bull"})
        self.assertGreater(results.summary_df.loc[results.summary_df["Metric"] == "median", "Value"].iloc[0], 0)

        output_dir = Path(tempfile.mkdtemp(prefix="mntn-valuation-test-"))
        try:
            export_results(results, output_dir, include_plots=False)
            self.assertTrue((output_dir / "scenario_table.csv").exists())
            self.assertTrue((output_dir / "monte_carlo_summary.csv").exists())
            self.assertTrue((output_dir / "horizon_valuation_summary.csv").exists())
            self.assertTrue((output_dir / "return_summary.csv").exists())
            exported = pd.read_csv(output_dir / "scenario_table.csv")
            self.assertFalse(exported.empty)
        finally:
            shutil.rmtree(output_dir)

    def test_missing_inputs_fail_clearly(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Missing required input file"):
            load_inputs("NOPE", DATA_DIR, CONFIG_PATH)

    def test_meli_inputs_load(self) -> None:
        inputs = load_inputs("MELI", DATA_DIR, ROOT / "configs" / "meli.json")
        self.assertEqual(inputs.snapshot.company_id, "MELI")
        self.assertGreater(inputs.snapshot.revenue_2025, 0)
        self.assertFalse(inputs.history.data.empty)


if __name__ == "__main__":
    unittest.main()
