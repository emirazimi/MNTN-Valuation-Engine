from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sample
from scipy import stats
from scipy.stats import qmc

from .types import ThesisCase, ValuationInputs, ValuationResults


def _company_metrics(inputs: ValuationInputs) -> dict[str, float]:
    snapshot = inputs.snapshot
    return {
        "current_price": snapshot.current_price,
        "ebit_margin_2025": snapshot.ebit_2025 / snapshot.revenue_2025,
        "adj_ebitda_margin_2026_mid": snapshot.adj_ebitda_2026_mid / snapshot.revenue_2026_mid,
        "company_anchor_growth": snapshot.revenue_2026_mid / snapshot.revenue_2025 - 1,
        "company_anchor_margin": 0.23,
        "company_anchor_dna": snapshot.d_and_a_2025 / snapshot.revenue_2025,
        "company_anchor_capex": snapshot.capex_2025 / snapshot.revenue_2025,
        "company_anchor_nwc": 0.040,
    }


def _fade_series(start: float, end: float, periods: int, exponent: float) -> np.ndarray:
    if periods <= 1:
        return np.array([end], dtype=float)
    progress = np.linspace(0.0, 1.0, periods)
    shaped = progress**exponent
    return start + (end - start) * shaped


def _scenario_regime_path(label: str, years: int) -> list[str]:
    if label == "Bear":
        return ["Bear"] * min(3, years) + ["Base"] * max(0, years - min(3, years))
    if label == "Bull":
        bull_years = min(3, years)
        base_years = years - bull_years
        return ["Bull"] * bull_years + ["Base"] * base_years
    return ["Base"] * years


def _scenario_wacc_path(label: str, years: int) -> list[float]:
    if label == "Bear":
        return list(np.linspace(0.116, 0.102, years))
    if label == "Bull":
        midpoint = min(4, years)
        first = np.linspace(0.094, 0.091, midpoint)
        second = np.linspace(0.092, 0.095, years - midpoint) if years > midpoint else np.array([])
        return list(np.concatenate([first, second]))
    return list(np.linspace(0.101, 0.098, years))


def _thesis_cases(inputs: ValuationInputs) -> list[ThesisCase]:
    return [
        inputs.thesis_config.bear,
        inputs.thesis_config.base,
        inputs.thesis_config.bull,
    ]


def _build_share_count_bridge(
    inputs: ValuationInputs,
    revenues: np.ndarray,
    shares_start: float,
    residual_dilution_start: float | None = None,
) -> pd.DataFrame:
    cfg = inputs.forecast_config
    snapshot = inputs.snapshot
    periods = len(revenues)
    residual_start = cfg.dilution_rate if residual_dilution_start is None else residual_dilution_start
    residual_dilution_path = np.maximum(
        0.0,
        _fade_series(residual_start, cfg.terminal_dilution_rate, periods, cfg.dilution_fade_exponent),
    )
    sbc_pct_path = np.maximum(
        0.0,
        _fade_series(cfg.sbc_pct_revenue_start, cfg.sbc_pct_revenue_end, periods, cfg.sbc_fade_exponent),
    )
    issuance_price = max(0.01, snapshot.current_price * cfg.sbc_share_issuance_price_factor)
    opening_overhang = max(snapshot.shares_outstanding_diluted - shares_start, 0.0)

    rows = []
    shares_prev = shares_start
    for idx in range(periods):
        residual_shares = shares_prev * residual_dilution_path[idx]
        sbc_expense = revenues[idx] * sbc_pct_path[idx]
        sbc_share_issuance = sbc_expense / issuance_price
        if idx < cfg.overhang_years and opening_overhang > 0:
            overhang_release = opening_overhang / cfg.overhang_years
        else:
            overhang_release = 0.0
        ending_shares = shares_prev + residual_shares + sbc_share_issuance + overhang_release
        rows.append(
            {
                "residual_dilution_rate": residual_dilution_path[idx],
                "residual_shares_issued": residual_shares,
                "sbc_pct_revenue": sbc_pct_path[idx],
                "sbc_expense": sbc_expense,
                "sbc_share_issuance": sbc_share_issuance,
                "overhang_release": overhang_release,
                "shares": ending_shares,
            }
        )
        shares_prev = ending_shares
    return pd.DataFrame(rows)


def build_operating_forecast(
    inputs: ValuationInputs,
    revenue0: float,
    start_growth: float,
    long_run_growth: float,
    ebit_margin_start: float,
    terminal_ebit_margin: float,
    d_and_a_pct: float,
    capex_base_pct: float,
    nwc_base_pct: float,
    terminal_growth: float,
    regime_path: list[str],
    regime_wacc_path: list[float],
    shares_start: float,
    growth_adjustments: list[float] | None = None,
    margin_end_adjustment: float = 0.0,
    dilution_rate: float | None = None,
    growth_path_override: np.ndarray | None = None,
    margin_path_override: np.ndarray | None = None,
) -> pd.DataFrame:
    cfg = inputs.forecast_config
    periods = cfg.projection_years
    if len(regime_path) != periods or len(regime_wacc_path) != periods:
        raise ValueError("Regime path and WACC path must match projection years.")

    growth_adjustments = np.zeros(periods) if growth_adjustments is None else np.asarray(growth_adjustments, dtype=float)
    if growth_path_override is None:
        base_growth = _fade_series(start_growth, long_run_growth, periods, cfg.revenue_fade_exponent)
        growth_rates = np.clip(base_growth + growth_adjustments, 0.01, 0.40)
    else:
        growth_rates = np.clip(np.asarray(growth_path_override, dtype=float), 0.01, 0.40)
    terminal_margin = np.clip(terminal_ebit_margin + margin_end_adjustment, 0.08, 0.35)
    terminal_capex_pct = max(cfg.capex_intensity_floor, cfg.terminal_capex_pct)
    terminal_nwc_pct = max(cfg.nwc_intensity_floor, cfg.terminal_nwc_pct)
    residual_dilution_start = cfg.dilution_rate if dilution_rate is None else dilution_rate

    revenues = []
    rev_prev = revenue0
    for growth_rate in growth_rates:
        revenue = rev_prev * (1 + growth_rate)
        revenues.append(revenue)
        rev_prev = revenue
    revenues = np.array(revenues)

    if margin_path_override is None:
        time_progress = np.linspace(0.0, 1.0, periods) ** cfg.margin_fade_exponent
        scale_progress = (revenues / revenues[-1]) if revenues[-1] > 0 else np.ones(periods)
        scale_progress = np.clip(scale_progress, 0.0, 1.0)
        margin_progress = np.clip(0.55 * time_progress + 0.45 * scale_progress, 0.0, 1.0)
        margin_path = np.clip(ebit_margin_start + (terminal_margin - ebit_margin_start) * margin_progress, 0.03, 0.40)
    else:
        margin_path = np.clip(np.asarray(margin_path_override, dtype=float), 0.03, 0.40)

    capex_fade = _fade_series(capex_base_pct, terminal_capex_pct, periods, cfg.capex_fade_exponent)
    capex_pct_path = np.maximum(
        cfg.capex_intensity_floor,
        capex_fade + cfg.capex_growth_sensitivity * np.maximum(growth_rates - long_run_growth, 0.0),
    )
    nwc_fade = _fade_series(nwc_base_pct, terminal_nwc_pct, periods, cfg.nwc_fade_exponent)
    nwc_pct_path = np.maximum(
        cfg.nwc_intensity_floor,
        nwc_fade + cfg.nwc_growth_sensitivity * np.maximum(growth_rates - long_run_growth, 0.0),
    )
    share_bridge_df = _build_share_count_bridge(
        inputs=inputs,
        revenues=revenues,
        shares_start=shares_start,
        residual_dilution_start=residual_dilution_start,
    )

    rows = []
    rev_prev = revenue0
    for idx in range(periods):
        revenue = revenues[idx]
        ebit = revenue * margin_path[idx]
        nopat = ebit * (1 - cfg.tax_rate)
        d_and_a = revenue * d_and_a_pct
        capex = revenue * capex_pct_path[idx]
        delta_nwc = nwc_pct_path[idx] * max(revenue - rev_prev, 0.0)
        fcff = nopat + d_and_a - capex - delta_nwc
        share_row = share_bridge_df.iloc[idx]
        shares = share_row["shares"]
        rows.append(
            {
                "year": idx + 1,
                "regime": regime_path[idx],
                "revenue": revenue,
                "growth_rate": growth_rates[idx],
                "ebit_margin": margin_path[idx],
                "ebit": ebit,
                "d_and_a_pct": d_and_a_pct,
                "d_and_a": d_and_a,
                "capex_pct": capex_pct_path[idx],
                "capex": capex,
                "nwc_pct": nwc_pct_path[idx],
                "delta_nwc": delta_nwc,
                "fcff": fcff,
                "wacc": regime_wacc_path[idx],
                "residual_dilution_rate": share_row["residual_dilution_rate"],
                "residual_shares_issued": share_row["residual_shares_issued"],
                "sbc_pct_revenue": share_row["sbc_pct_revenue"],
                "sbc_expense": share_row["sbc_expense"],
                "sbc_share_issuance": share_row["sbc_share_issuance"],
                "overhang_release": share_row["overhang_release"],
                "shares": shares,
                "terminal_growth": terminal_growth,
            }
        )
        rev_prev = revenue
    return pd.DataFrame(rows)


def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    mean = _weighted_mean(values, weights)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    var = np.sum(weights * (values - mean) ** 2) / np.sum(weights)
    return float(np.sqrt(max(var, 1e-8)))


def _build_peer_weights(inputs: ValuationInputs, comp_panel: pd.DataFrame) -> pd.Series:
    cfg = inputs.math_config
    history = inputs.history.data
    latest = history.iloc[-1]
    latest_quarter = comp_panel["quarter"].max()
    quarter_delta = latest_quarter.ordinal - comp_panel["quarter"].map(lambda period: period.ordinal)
    recency_weight = np.exp(-np.log(2) * quarter_delta / cfg.peer_recency_half_life_quarters)

    peer_latest = comp_panel.sort_values(["company", "quarter"]).groupby("company").tail(1).copy()
    peer_latest["growth_distance"] = (peer_latest["rev_growth_yoy"] - latest["rev_growth_yoy"]) ** 2
    peer_latest["margin_distance"] = (peer_latest["ebit_margin"] - latest["ebit_margin"]) ** 2
    peer_latest["fcff_distance"] = (peer_latest["fcff_margin"] - latest["fcff_margin"]) ** 2
    peer_latest["relevance_weight"] = np.exp(
        -cfg.peer_relevance_strength
        * (peer_latest["growth_distance"] + 0.75 * peer_latest["margin_distance"] + 0.50 * peer_latest["fcff_distance"])
    )
    relevance = peer_latest.set_index("company")["relevance_weight"]
    weights = recency_weight * comp_panel["company"].map(relevance).fillna(1.0)
    return pd.Series(weights, index=comp_panel.index, dtype=float)


def fit_empirical_bayes_priors(inputs: ValuationInputs, comp_panel: pd.DataFrame) -> dict[str, dict[str, float]]:
    df = comp_panel.copy()
    cols = ["rev_growth_yoy", "ebit_margin", "d_and_a_pct", "capex_pct", "nwc_pct"]
    for col in cols:
        df[col] = winsorize_series(df[col])
    weights = _build_peer_weights(inputs, df)
    shrink = inputs.math_config.hierarchical_shrinkage

    company_agg = (
        df.assign(weight=weights)
        .groupby("company")
        .apply(
            lambda grp: pd.Series(
                {
                    "rev_growth_yoy": _weighted_mean(grp["rev_growth_yoy"], grp["weight"]),
                    "ebit_margin": _weighted_mean(grp["ebit_margin"], grp["weight"]),
                    "d_and_a_pct": _weighted_mean(grp["d_and_a_pct"], grp["weight"]),
                    "capex_pct": _weighted_mean(grp["capex_pct"], grp["weight"]),
                    "nwc_pct": _weighted_mean(grp["nwc_pct"], grp["weight"]),
                    "weight": grp["weight"].sum(),
                }
            )
        )
        .reset_index()
    )

    priors = {
        "long_run_growth": {
            "mu": shrink * _weighted_mean(df["rev_growth_yoy"], weights) + (1 - shrink) * _weighted_mean(company_agg["rev_growth_yoy"], company_agg["weight"]),
            "sigma": max(shrink * _weighted_std(df["rev_growth_yoy"], weights) + (1 - shrink) * _weighted_std(company_agg["rev_growth_yoy"], company_agg["weight"]), 1e-4),
            "low": float(df["rev_growth_yoy"].quantile(0.05)),
            "high": float(df["rev_growth_yoy"].quantile(0.95)),
        },
        "terminal_ebit_margin": {
            "mu": shrink * _weighted_mean(df["ebit_margin"], weights) + (1 - shrink) * _weighted_mean(company_agg["ebit_margin"], company_agg["weight"]),
            "sigma": max(shrink * _weighted_std(df["ebit_margin"], weights) + (1 - shrink) * _weighted_std(company_agg["ebit_margin"], company_agg["weight"]), 1e-4),
            "low": float(df["ebit_margin"].quantile(0.05)),
            "high": float(df["ebit_margin"].quantile(0.95)),
        },
        "d_and_a_pct": {
            "mu": shrink * _weighted_mean(df["d_and_a_pct"], weights) + (1 - shrink) * _weighted_mean(company_agg["d_and_a_pct"], company_agg["weight"]),
            "sigma": max(shrink * _weighted_std(df["d_and_a_pct"], weights) + (1 - shrink) * _weighted_std(company_agg["d_and_a_pct"], company_agg["weight"]), 1e-4),
            "low": float(df["d_and_a_pct"].quantile(0.05)),
            "high": float(df["d_and_a_pct"].quantile(0.95)),
        },
        "capex_pct": {
            "mu": shrink * _weighted_mean(df["capex_pct"], weights) + (1 - shrink) * _weighted_mean(company_agg["capex_pct"], company_agg["weight"]),
            "sigma": max(shrink * _weighted_std(df["capex_pct"], weights) + (1 - shrink) * _weighted_std(company_agg["capex_pct"], company_agg["weight"]), 1e-4),
            "low": float(df["capex_pct"].quantile(0.05)),
            "high": float(df["capex_pct"].quantile(0.95)),
        },
        "nwc_pct": {
            "mu": shrink * _weighted_mean(df["nwc_pct"], weights) + (1 - shrink) * _weighted_mean(company_agg["nwc_pct"], company_agg["weight"]),
            "sigma": max(shrink * _weighted_std(df["nwc_pct"], weights) + (1 - shrink) * _weighted_std(company_agg["nwc_pct"], company_agg["weight"]), 1e-4),
            "low": float(df["nwc_pct"].quantile(0.05)),
            "high": float(df["nwc_pct"].quantile(0.95)),
        },
        "terminal_growth": {
            "mu": min(0.03, max(0.02, 0.35 * _weighted_mean(df["rev_growth_yoy"], weights))),
            "sigma": 0.004,
            "low": 0.018,
            "high": 0.038,
        },
    }
    return priors


def clipped_ppf(u: float, dist_name: str, params: tuple[float, ...]) -> float:
    u = np.clip(u, 1e-6, 1 - 1e-6)
    if dist_name == "truncnorm":
        mean, std, low, high = params
        a = (low - mean) / std
        b = (high - mean) / std
        return float(stats.truncnorm.ppf(u, a, b, loc=mean, scale=std))
    if dist_name == "triangular":
        low, mode, high = params
        c = (mode - low) / (high - low)
        return float(stats.triang.ppf(u, c, loc=low, scale=(high - low)))
    raise ValueError("Unsupported distribution")


def sample_bayesian_prior(u: float, prior_dict: dict[str, float]) -> float:
    return clipped_ppf(
        u,
        "truncnorm",
        (prior_dict["mu"], prior_dict["sigma"], prior_dict["low"], prior_dict["high"]),
    )


def correlated_lhs_uniforms(n_sims: int, corr_matrix: np.ndarray, seed: int = 42) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=corr_matrix.shape[0], seed=seed)
    uniforms = sampler.random(n=n_sims)
    z = stats.norm.ppf(np.clip(uniforms, 1e-6, 1 - 1e-6))
    cholesky = np.linalg.cholesky(corr_matrix)
    correlated = z @ cholesky.T
    return np.clip(stats.norm.cdf(correlated), 1e-6, 1 - 1e-6)


def sample_latent_factor_paths(inputs: ValuationInputs, n_years: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    cfg = inputs.math_config
    macro = np.zeros(n_years)
    execution = np.zeros(n_years)
    capital = np.zeros(n_years)
    for t in range(n_years):
        macro_prev = macro[t - 1] if t > 0 else 0.0
        execution_prev = execution[t - 1] if t > 0 else 0.0
        capital_prev = capital[t - 1] if t > 0 else 0.0
        macro[t] = cfg.factor_persistence * macro_prev + rng.normal(0.0, cfg.macro_factor_vol)
        execution[t] = cfg.factor_persistence * execution_prev + rng.normal(0.0, cfg.execution_factor_vol)
        capital[t] = cfg.factor_persistence * capital_prev + rng.normal(0.0, cfg.capital_factor_vol)
    return {"macro": macro, "execution": execution, "capital": capital}


def generate_mean_reverting_operating_paths(
    inputs: ValuationInputs,
    start_growth: float,
    long_run_growth: float,
    ebit_margin_start: float,
    terminal_ebit_margin: float,
    regime_path: list[str],
    growth_adjustments: list[float],
    margin_end_adjustment: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = inputs.math_config
    n_years = len(regime_path)
    factors = sample_latent_factor_paths(inputs, n_years, rng)
    growth = np.zeros(n_years)
    margin = np.zeros(n_years)
    growth_prev = start_growth
    margin_prev = ebit_margin_start
    terminal_margin = np.clip(terminal_ebit_margin + margin_end_adjustment, 0.08, 0.35)
    for t in range(n_years):
        growth_factor = (
            cfg.growth_factor_loadings["macro"] * factors["macro"][t]
            + cfg.growth_factor_loadings["execution"] * factors["execution"][t]
            + cfg.growth_factor_loadings["capital"] * factors["capital"][t]
        )
        margin_factor = (
            cfg.margin_factor_loadings["macro"] * factors["macro"][t]
            + cfg.margin_factor_loadings["execution"] * factors["execution"][t]
            + cfg.margin_factor_loadings["capital"] * factors["capital"][t]
        )
        growth_t = (
            long_run_growth
            + (1 - cfg.growth_mean_reversion) * (growth_prev - long_run_growth)
            + growth_factor
            + growth_adjustments[t]
            + rng.normal(0.0, cfg.idiosyncratic_growth_vol)
        )
        margin_t = (
            terminal_margin
            + (1 - cfg.margin_mean_reversion) * (margin_prev - terminal_margin)
            + margin_factor
            + rng.normal(0.0, cfg.idiosyncratic_margin_vol)
        )
        growth[t] = np.clip(growth_t, 0.01, 0.40)
        margin[t] = np.clip(margin_t, 0.03, 0.40)
        growth_prev = growth[t]
        margin_prev = margin[t]
    return growth, margin


def soften_probabilities(probs: np.ndarray, temperature: float = 2.5) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    logits = np.log(np.clip(probs, 1e-12, None))
    softened = np.exp(logits / temperature)
    return softened / softened.sum()


def apply_probability_floor(probs: np.ndarray, floor: np.ndarray) -> np.ndarray:
    floored = np.maximum(np.asarray(probs, dtype=float), floor)
    return floored / floored.sum()


def assign_empirical_states(comp_panel: pd.DataFrame) -> pd.DataFrame:
    df = comp_panel.copy()
    df["growth_pctile"] = df["rev_growth_yoy"].rank(pct=True)
    df["margin_pctile"] = df["ebit_margin"].rank(pct=True)
    df["fcff_pctile"] = df["fcff_margin"].rank(pct=True)
    df["state_score"] = 0.45 * df["growth_pctile"] + 0.35 * df["margin_pctile"] + 0.20 * df["fcff_pctile"]
    df["state"] = pd.qcut(df["state_score"], q=3, labels=["Bear", "Base", "Bull"])
    return df


def enforce_transition_floor(matrix: pd.DataFrame, floor: float = 1e-4) -> pd.DataFrame:
    matrix = matrix.copy().astype(float).clip(lower=floor)
    return matrix.div(matrix.sum(axis=1), axis=0)


def estimate_transition_matrix(state_df: pd.DataFrame) -> pd.DataFrame:
    df = state_df.copy().sort_values(["company", "quarter"])
    df["next_state"] = df.groupby("company")["state"].shift(-1)
    trans = df.dropna(subset=["next_state"]).copy()
    matrix = pd.crosstab(trans["state"], trans["next_state"], normalize="index")
    matrix = matrix.reindex(index=["Bear", "Base", "Bull"], columns=["Bear", "Base", "Bull"]).fillna(0.0)
    return enforce_transition_floor(matrix)


def adjust_transition_for_macro(base_transition_df: pd.DataFrame, stress_level: float) -> pd.DataFrame:
    matrix = base_transition_df.copy()
    for state in matrix.index:
        matrix.loc[state, "Bear"] += 0.12 * stress_level
        matrix.loc[state, "Bull"] = max(1e-6, matrix.loc[state, "Bull"] - 0.09 * stress_level)
        matrix.loc[state, "Base"] = max(1e-6, matrix.loc[state, "Base"] - 0.03 * stress_level)
        matrix.loc[state] = matrix.loc[state] / matrix.loc[state].sum()
    return enforce_transition_floor(matrix)


def cap_transition_persistence(matrix: pd.DataFrame, max_diag: float = 0.55) -> pd.DataFrame:
    matrix = matrix.copy().astype(float)
    for state in matrix.index:
        diag_val = matrix.loc[state, state]
        if diag_val > max_diag:
            excess = diag_val - max_diag
            matrix.loc[state, state] = max_diag
            others = [col for col in matrix.columns if col != state]
            matrix.loc[state, others] += excess / len(others)
        matrix.loc[state] = matrix.loc[state] / matrix.loc[state].sum()
    return enforce_transition_floor(matrix)


def infer_peer_likelihood_start_probs(
    state_df: pd.DataFrame,
    mntn_growth: float,
    mntn_margin: float,
    mntn_fcff_margin: float,
) -> np.ndarray:
    probs = {}
    x = np.array([mntn_growth, mntn_margin, mntn_fcff_margin])
    for state in ["Bear", "Base", "Bull"]:
        sub = state_df[state_df["state"] == state][["rev_growth_yoy", "ebit_margin", "fcff_margin"]].copy()
        mu = sub.mean().values
        cov = np.cov(sub.values.T) + np.eye(3) * 5e-4
        cov = cov * 1.75
        probs[state] = stats.multivariate_normal.pdf(x, mean=mu, cov=cov)
    arr = np.array([probs["Bear"], probs["Base"], probs["Bull"]], dtype=float)
    arr = np.clip(arr, 1e-12, None)
    return arr / arr.sum()


def build_macro_start_prior(stress_level: float) -> np.ndarray:
    bear = 0.20 + 0.25 * stress_level
    base = 0.50 - 0.10 * stress_level
    bull = 1.0 - bear - base
    prior = np.array([bear, base, bull], dtype=float)
    return prior / prior.sum()


def build_final_start_probs(state_df: pd.DataFrame, history: pd.DataFrame, stress_level: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latest = history.iloc[-1]
    peer_probs = infer_peer_likelihood_start_probs(
        state_df,
        mntn_growth=latest["rev_growth_yoy"],
        mntn_margin=latest["ebit_margin"],
        mntn_fcff_margin=latest["fcff_margin"],
    )
    macro_prior = build_macro_start_prior(stress_level)
    raw = 0.45 * peer_probs + 0.55 * macro_prior
    raw = raw / raw.sum()
    raw = soften_probabilities(raw, temperature=2.75)
    final = apply_probability_floor(raw, floor=np.array([0.20, 0.30, 0.20]))
    return final, peer_probs, macro_prior


def simulate_regime_path(n_years: int, start_probs: np.ndarray, transition_matrix_df: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    states = ["Bear", "Base", "Bull"]
    current = rng.choice(states, p=start_probs)
    path = [current]
    for _ in range(n_years - 1):
        row = transition_matrix_df.loc[current, states].values
        row = np.clip(row, 1e-12, None)
        row = row / row.sum()
        current = rng.choice(states, p=row)
        path.append(current)
    return path


def deterministic_expected_regime_path(start_probs: np.ndarray, transition_matrix_df: pd.DataFrame, n_years: int = 5) -> list[str]:
    states = ["Bear", "Base", "Bull"]
    probs_t = np.array(start_probs, dtype=float)
    path = []
    transition = transition_matrix_df.loc[states, states].values
    for _ in range(n_years):
        path.append(states[int(np.argmax(probs_t))])
        probs_t = probs_t @ transition
        probs_t = np.clip(probs_t, 1e-12, None)
        probs_t = probs_t / probs_t.sum()
    return path


def sample_jump_shock(rng: np.random.Generator, params: dict[str, float]) -> tuple[float, float, int]:
    jump_occurs = rng.uniform() < params["annual_prob"]
    if not jump_occurs:
        return 0.0, 0.0, 0
    growth_hit = min(-0.005, rng.normal(params["growth_jump_mean"], params["growth_jump_std"]))
    margin_hit = min(-0.002, rng.normal(params["margin_jump_mean"], params["margin_jump_std"]))
    return growth_hit, margin_hit, 1


def dcf_fcff_regime_discounted(
    revenue0: float,
    growth_rates: list[float] | np.ndarray,
    ebit_margin_start: float,
    ebit_margin_end: float,
    tax_rate: float,
    d_and_a_pct: float,
    capex_pct: float,
    nwc_pct_of_incremental_rev: float,
    regime_path: list[str],
    regime_wacc_path: list[float],
    terminal_growth: float,
    cash: float,
    debt: float,
    shares: float,
) -> dict[str, np.ndarray | float]:
    n_years = len(growth_rates)
    rev_prev = revenue0
    revenues, ebits, fcffs, margins = [], [], [], []
    margin_path = np.linspace(ebit_margin_start, ebit_margin_end, n_years)

    for idx in range(n_years):
        rev_t = rev_prev * (1 + growth_rates[idx])
        ebit_t = rev_t * margin_path[idx]
        nopat_t = ebit_t * (1 - tax_rate)
        d_and_a_t = rev_t * d_and_a_pct
        capex_t = rev_t * capex_pct
        delta_nwc_t = nwc_pct_of_incremental_rev * max(rev_t - rev_prev, 0.0)
        fcff_t = nopat_t + d_and_a_t - capex_t - delta_nwc_t
        revenues.append(rev_t)
        ebits.append(ebit_t)
        fcffs.append(fcff_t)
        margins.append(margin_path[idx])
        rev_prev = rev_t

    running = 1.0
    discount = []
    for wacc in regime_wacc_path:
        running *= 1 + wacc
        discount.append(running)
    discount = np.array(discount)
    revenues = np.array(revenues)
    ebits = np.array(ebits)
    fcffs = np.array(fcffs)
    margins = np.array(margins)

    pv_fcffs = np.sum(fcffs / discount)
    terminal_wacc = regime_wacc_path[-1]
    if terminal_wacc <= terminal_growth:
        terminal_wacc = terminal_growth + 0.005
    terminal_fcff = fcffs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcff / (terminal_wacc - terminal_growth)
    pv_terminal = terminal_value / discount[-1]
    enterprise_value = pv_fcffs + pv_terminal
    equity_value = enterprise_value + cash - debt
    value_per_share = equity_value / shares
    return {
        "revenues": revenues,
        "ebits": ebits,
        "fcffs": fcffs,
        "margins": margins,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
    }


def dcf_from_operating_forecast(
    forecast_df: pd.DataFrame,
    terminal_growth: float,
    cash: float,
    debt: float,
) -> dict[str, np.ndarray | float]:
    fcffs = forecast_df["fcff"].to_numpy()
    margins = forecast_df["ebit_margin"].to_numpy()
    revenues = forecast_df["revenue"].to_numpy()
    ebits = forecast_df["ebit"].to_numpy()
    wacc_path = forecast_df["wacc"].to_numpy()
    discount = np.cumprod(1 + wacc_path)
    pv_fcffs = np.sum(fcffs / discount)
    terminal_wacc = max(wacc_path[-1], terminal_growth + 0.005)
    terminal_fcff = fcffs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcff / (terminal_wacc - terminal_growth)
    pv_terminal = terminal_value / discount[-1]
    enterprise_value = pv_fcffs + pv_terminal
    equity_value = enterprise_value + cash - debt
    value_per_share = equity_value / forecast_df["shares"].iloc[-1]
    return {
        "revenues": revenues,
        "ebits": ebits,
        "fcffs": fcffs,
        "margins": margins,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
        "forecast_df": forecast_df,
    }


def run_scenario_table_calibrated(inputs: ValuationInputs, priors: dict[str, dict[str, float]], shares: float) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | float]]]:
    metrics = _company_metrics(inputs)
    snapshot = inputs.snapshot
    years = inputs.forecast_config.projection_years
    g = priors["long_run_growth"]
    m = priors["terminal_ebit_margin"]
    d = priors["d_and_a_pct"]
    c = priors["capex_pct"]
    n = priors["nwc_pct"]
    tg = priors["terminal_growth"]

    rows = []
    detailed = {}
    for case in _thesis_cases(inputs):
        scenario = {
            "start_growth": np.clip(metrics["company_anchor_growth"] + case.start_growth_adjustment, 0.01, 0.40),
            "long_run_growth": np.clip(case.long_run_growth_override if case.long_run_growth_override is not None else g["mu"], 0.01, 0.30),
            "ebit_margin_start": max(0.05, metrics["ebit_margin_2025"] + case.ebit_margin_start_adjustment),
            "terminal_ebit_margin": np.clip(case.terminal_ebit_margin_override if case.terminal_ebit_margin_override is not None else m["mu"], 0.08, 0.35),
            "d_and_a_pct": max(0.0, d["mu"] + case.d_and_a_pct_adjustment),
            "capex_pct": max(0.0, c["mu"] + case.capex_pct_adjustment),
            "nwc_pct": max(0.0, n["mu"] + case.nwc_pct_adjustment),
            "terminal_growth": np.clip(case.terminal_growth_override if case.terminal_growth_override is not None else tg["mu"], 0.018, 0.040),
            "regime_path": _scenario_regime_path(case.regime_label, years),
            "regime_wacc_path": _scenario_wacc_path(case.regime_label, years),
            "residual_dilution_adjustment": case.residual_dilution_adjustment,
        }
        forecast_df = build_operating_forecast(
            inputs=inputs,
            revenue0=snapshot.revenue_2026_mid,
            start_growth=scenario["start_growth"],
            long_run_growth=scenario["long_run_growth"],
            ebit_margin_start=scenario["ebit_margin_start"],
            terminal_ebit_margin=scenario["terminal_ebit_margin"],
            d_and_a_pct=scenario["d_and_a_pct"],
            capex_base_pct=scenario["capex_pct"],
            nwc_base_pct=scenario["nwc_pct"],
            terminal_growth=scenario["terminal_growth"],
            regime_path=scenario["regime_path"],
            regime_wacc_path=scenario["regime_wacc_path"],
            shares_start=shares,
            dilution_rate=max(0.0, inputs.forecast_config.dilution_rate + scenario["residual_dilution_adjustment"]),
        )
        result = dcf_from_operating_forecast(forecast_df, scenario["terminal_growth"], snapshot.cash_2025, snapshot.debt_2025)
        rows.append(
            {
                "Scenario": case.name,
                "Value / Share": result["value_per_share"],
                "Upside / Downside": result["value_per_share"] / snapshot.current_price - 1,
                "Avg WACC": np.mean(scenario["regime_wacc_path"]),
                "Terminal Growth": scenario["terminal_growth"],
            }
        )
        detailed[case.name] = result
    scenario_df = pd.DataFrame(rows).sort_values("Value / Share").reset_index(drop=True)
    return scenario_df, detailed


def apply_growth_margin_premium(
    base_multiple: float,
    mntn_growth: float,
    peer_growth_median: float,
    mntn_margin: float,
    peer_margin_median: float,
    growth_weight: float = 0.50,
    margin_weight: float = 0.35,
    cap_up: float = 0.35,
    cap_down: float = -0.35,
) -> float:
    premium = growth_weight * (mntn_growth - peer_growth_median) + margin_weight * (mntn_margin - peer_margin_median)
    premium = np.clip(premium, cap_down, cap_up)
    return max(0.1, base_multiple * (1 + premium))


def multiple_cross_check(peer_multiples_df: pd.DataFrame, snapshot, metrics: dict[str, float], shares: float) -> pd.DataFrame:
    med_ev_rev = peer_multiples_df["ev_revenue_ntm"].median()
    med_ev_ebitda = peer_multiples_df["ev_ebitda_ntm"].median()
    peer_growth_median = peer_multiples_df["ntm_revenue_growth"].median()
    peer_margin_median = peer_multiples_df["ntm_ebitda_margin"].median()

    adj_ev_rev = apply_growth_margin_premium(
        med_ev_rev,
        metrics["company_anchor_growth"],
        peer_growth_median,
        metrics["adj_ebitda_margin_2026_mid"],
        peer_margin_median,
    )
    adj_ev_ebitda = apply_growth_margin_premium(
        med_ev_ebitda,
        metrics["company_anchor_growth"],
        peer_growth_median,
        metrics["adj_ebitda_margin_2026_mid"],
        peer_margin_median,
    )
    methods = {
        "Peer Median EV/Revenue": med_ev_rev * snapshot.revenue_2026_mid,
        "Peer Median EV/EBITDA": med_ev_ebitda * snapshot.adj_ebitda_2026_mid,
        "Adj EV/Revenue": adj_ev_rev * snapshot.revenue_2026_mid,
        "Adj EV/EBITDA": adj_ev_ebitda * snapshot.adj_ebitda_2026_mid,
    }
    rows = []
    for name, enterprise_value in methods.items():
        equity_value = enterprise_value + snapshot.cash_2025 - snapshot.debt_2025
        rows.append(
            {
                "Method": name,
                "Implied EV": enterprise_value,
                "Implied Equity Value": equity_value,
                "Implied Value / Share": equity_value / shares,
            }
        )
    return pd.DataFrame(rows)


def build_mc_valuation_paths_rolling_terminal(
    snapshot,
    growth_paths: np.ndarray,
    margin_paths: np.ndarray,
    wacc_paths: np.ndarray,
    d_and_a_draws: np.ndarray,
    capex_draws: np.ndarray,
    nwc_draws: np.ndarray,
    share_paths: np.ndarray,
    terminal_growth_draws: np.ndarray,
    tax_rate: float = 0.25,
) -> np.ndarray:
    n_sims, n_years = growth_paths.shape
    valuation_paths = np.zeros((n_sims, n_years))
    for i in range(n_sims):
        rev_prev = snapshot.revenue_2026_mid
        running_discount = 1.0
        cumulative_pv = 0.0
        for t in range(n_years):
            rev_t = rev_prev * (1 + growth_paths[i, t])
            ebit_t = rev_t * margin_paths[i, t]
            nopat_t = ebit_t * (1 - tax_rate)
            d_and_a_t = rev_t * d_and_a_draws[i]
            capex_t = rev_t * capex_draws[i]
            delta_nwc_t = nwc_draws[i] * max(rev_t - rev_prev, 0.0)
            fcff_t = nopat_t + d_and_a_t - capex_t - delta_nwc_t
            running_discount *= 1 + wacc_paths[i, t]
            cumulative_pv += fcff_t / running_discount
            terminal_wacc = max(wacc_paths[i, t], terminal_growth_draws[i] + 0.005)
            terminal_fcff = fcff_t * (1 + terminal_growth_draws[i])
            terminal_value_t = terminal_fcff / (terminal_wacc - terminal_growth_draws[i])
            terminal_pv_t = terminal_value_t / running_discount
            equity_value_t = cumulative_pv + terminal_pv_t + snapshot.cash_2025 - snapshot.debt_2025
            valuation_paths[i, t] = equity_value_t / share_paths[i, t]
            rev_prev = rev_t
    return valuation_paths


def build_horizon_summary(
    valuation_paths: np.ndarray,
    current_price: float,
    terminal_growth_draws: np.ndarray,
    terminal_dilution_rate: float,
) -> pd.DataFrame:
    horizon_map = {"1Y": 1, "2Y": 2, "5Y": 5}
    rows = []
    max_horizon = valuation_paths.shape[1]
    for label, horizon in horizon_map.items():
        if horizon > max_horizon:
            continue
        values = valuation_paths[:, horizon - 1]
        rows.append(
            {
                "Horizon": label,
                "mean": values.mean(),
                "median": np.median(values),
                "p10": np.percentile(values, 10),
                "p25": np.percentile(values, 25),
                "p75": np.percentile(values, 75),
                "p90": np.percentile(values, 90),
                "prob_above_current": np.mean(values > current_price),
            }
        )
    if max_horizon >= 5:
        extension_factor = ((1 + terminal_growth_draws) / (1 + terminal_dilution_rate)) ** 5
        values_10y = valuation_paths[:, 4] * extension_factor
        rows.append(
            {
                "Horizon": "10Y",
                "mean": values_10y.mean(),
                "median": np.median(values_10y),
                "p10": np.percentile(values_10y, 10),
                "p25": np.percentile(values_10y, 25),
                "p75": np.percentile(values_10y, 75),
                "p90": np.percentile(values_10y, 90),
                "prob_above_current": np.mean(values_10y > current_price),
            }
        )
    return pd.DataFrame(rows)


def advanced_monte_carlo_empirical(inputs: ValuationInputs, priors: dict[str, dict[str, float]], shares: float) -> dict[str, object]:
    metrics = _company_metrics(inputs)
    snapshot = inputs.snapshot
    history = inputs.history.data
    comp_panel = inputs.peer_panel.data
    peer_multiples_df = inputs.peer_multiples.data
    run_config = inputs.run_config
    regime_cfg = inputs.regime_config
    rng = np.random.default_rng(run_config.seed)

    state_df = assign_empirical_states(comp_panel)
    base_transition_df = estimate_transition_matrix(state_df)
    macro_transition_df = adjust_transition_for_macro(base_transition_df, run_config.macro_stress_level)
    macro_transition_df = cap_transition_persistence(macro_transition_df, max_diag=0.55)
    macro_transition_df = enforce_transition_floor(macro_transition_df)
    final_start_probs, peer_start_probs, macro_prior = build_final_start_probs(state_df, history, run_config.macro_stress_level)
    start_probs_df = pd.DataFrame(
        {
            "Regime": ["Bear", "Base", "Bull"],
            "Peer Likelihood": peer_start_probs,
            "Macro Prior": macro_prior,
            "Final Start Probability": final_start_probs,
        }
    )

    uniforms = qmc.LatinHypercube(d=6, seed=run_config.seed).random(n=run_config.n_sims)
    values = np.empty(run_config.n_sims)
    all_regime_paths = []
    all_wacc_paths = []
    all_growth_paths = []
    all_margin_paths = []
    all_share_paths = []
    all_sbc_paths = []
    jump_counts = np.zeros(run_config.n_sims)
    long_run_growth_draws = np.zeros(run_config.n_sims)
    terminal_margin_draws = np.zeros(run_config.n_sims)
    d_and_a_draws = np.zeros(run_config.n_sims)
    capex_draws = np.zeros(run_config.n_sims)
    nwc_draws = np.zeros(run_config.n_sims)
    terminal_growth_draws = np.zeros(run_config.n_sims)
    latest = history.iloc[-1]
    years = inputs.forecast_config.projection_years
    base_case = inputs.thesis_config.base

    for i in range(run_config.n_sims):
        prior_growth = sample_bayesian_prior(uniforms[i, 0], priors["long_run_growth"])
        prior_margin = sample_bayesian_prior(uniforms[i, 1], priors["terminal_ebit_margin"])
        prior_dna = sample_bayesian_prior(uniforms[i, 2], priors["d_and_a_pct"])
        prior_capex = sample_bayesian_prior(uniforms[i, 3], priors["capex_pct"])
        prior_nwc = sample_bayesian_prior(uniforms[i, 4], priors["nwc_pct"])
        terminal_growth = sample_bayesian_prior(uniforms[i, 5], priors["terminal_growth"])

        thesis_start_growth = np.clip(metrics["company_anchor_growth"] + base_case.start_growth_adjustment, 0.01, 0.40)
        thesis_long_run_growth = base_case.long_run_growth_override if base_case.long_run_growth_override is not None else metrics["company_anchor_growth"]
        thesis_terminal_margin = base_case.terminal_ebit_margin_override if base_case.terminal_ebit_margin_override is not None else metrics["company_anchor_margin"]
        long_run_growth = run_config.shrinkage * prior_growth + (1 - run_config.shrinkage) * thesis_long_run_growth
        terminal_ebit_margin = run_config.shrinkage * prior_margin + (1 - run_config.shrinkage) * thesis_terminal_margin
        d_and_a_pct = run_config.shrinkage * prior_dna + (1 - run_config.shrinkage) * metrics["company_anchor_dna"]
        capex_pct = run_config.shrinkage * prior_capex + (1 - run_config.shrinkage) * metrics["company_anchor_capex"]
        nwc_pct = run_config.shrinkage * prior_nwc + (1 - run_config.shrinkage) * metrics["company_anchor_nwc"]

        long_run_growth_draws[i] = long_run_growth
        terminal_margin_draws[i] = terminal_ebit_margin
        d_and_a_draws[i] = d_and_a_pct
        capex_draws[i] = capex_pct
        nwc_draws[i] = nwc_pct
        terminal_growth_draws[i] = terminal_growth

        regime_path = simulate_regime_path(years, final_start_probs, macro_transition_df, rng)
        ebit_margin_start = latest["ebit_margin"] + regime_cfg.regime_margin_shift[regime_path[0]]
        margin_shock_accum = 0.0
        regime_wacc_path = []
        jump_total = 0
        growth_adjustments = []

        for t in range(years):
            reg = regime_path[t]
            jump_g, jump_m, jump_flag = sample_jump_shock(rng, regime_cfg.jump_params)
            growth_adjustments.append(regime_cfg.regime_growth_shift[reg] + jump_g)
            margin_shock_accum += jump_m
            jump_total += jump_flag
            wacc_t = regime_cfg.regime_wacc_base[reg] + rng.normal(0.0, 0.004)
            regime_wacc_path.append(np.clip(wacc_t, 0.08, 0.14))

        growth_path, margin_path = generate_mean_reverting_operating_paths(
            inputs=inputs,
            start_growth=thesis_start_growth,
            long_run_growth=long_run_growth,
            ebit_margin_start=ebit_margin_start,
            terminal_ebit_margin=terminal_ebit_margin,
            regime_path=regime_path,
            growth_adjustments=growth_adjustments,
            margin_end_adjustment=regime_cfg.regime_margin_shift[regime_path[-1]] + margin_shock_accum,
            rng=rng,
        )
        forecast_df = build_operating_forecast(
            inputs=inputs,
            revenue0=snapshot.revenue_2026_mid,
            start_growth=thesis_start_growth,
            long_run_growth=long_run_growth,
            ebit_margin_start=ebit_margin_start,
            terminal_ebit_margin=terminal_ebit_margin,
            d_and_a_pct=d_and_a_pct,
            capex_base_pct=capex_pct,
            nwc_base_pct=max(0.0, nwc_pct),
            terminal_growth=terminal_growth,
            regime_path=regime_path,
            regime_wacc_path=regime_wacc_path,
            shares_start=shares,
            dilution_rate=max(0.0, inputs.forecast_config.dilution_rate + base_case.residual_dilution_adjustment),
            growth_path_override=growth_path,
            margin_path_override=margin_path,
        )
        result = dcf_from_operating_forecast(forecast_df, terminal_growth, snapshot.cash_2025, snapshot.debt_2025)

        values[i] = result["value_per_share"]
        all_regime_paths.append(regime_path)
        all_wacc_paths.append(regime_wacc_path)
        all_growth_paths.append(forecast_df["growth_rate"].to_numpy())
        all_margin_paths.append(forecast_df["ebit_margin"].to_numpy())
        all_share_paths.append(forecast_df["shares"].to_numpy())
        all_sbc_paths.append(forecast_df["sbc_share_issuance"].to_numpy())
        jump_counts[i] = jump_total

    all_regime_paths = np.array(all_regime_paths)
    all_wacc_paths = np.array(all_wacc_paths)
    all_growth_paths = np.array(all_growth_paths)
    all_margin_paths = np.array(all_margin_paths)
    all_share_paths = np.array(all_share_paths)
    all_sbc_paths = np.array(all_sbc_paths)
    horizon_5_idx = min(4, years - 1)

    simulation_df = pd.DataFrame(
        {
            "value_per_share": values,
            "start_regime": all_regime_paths[:, 0],
            "end_regime": all_regime_paths[:, -1],
            "avg_wacc": np.mean(all_wacc_paths, axis=1),
            "year1_wacc": all_wacc_paths[:, 0],
            "year5_wacc": all_wacc_paths[:, horizon_5_idx],
            "avg_growth": np.mean(all_growth_paths, axis=1),
            "year1_growth": all_growth_paths[:, 0],
            "year5_growth": all_growth_paths[:, horizon_5_idx],
            "avg_margin": np.mean(all_margin_paths, axis=1),
            "year1_margin": all_margin_paths[:, 0],
            "year5_margin": all_margin_paths[:, horizon_5_idx],
            "jump_count": jump_counts,
            "long_run_growth_draw": long_run_growth_draws,
            "terminal_margin_draw": terminal_margin_draws,
            "d_and_a_pct_draw": d_and_a_draws,
            "capex_pct_draw": capex_draws,
            "nwc_pct_draw": nwc_draws,
            "terminal_growth_draw": terminal_growth_draws,
            "year1_sbc_shares": all_sbc_paths[:, 0],
            "year5_sbc_shares": all_sbc_paths[:, horizon_5_idx],
            "ending_shares": all_share_paths[:, -1],
            "upside_downside": values / snapshot.current_price - 1,
        }
    )
    summary_df = pd.DataFrame(
        [
            {
                "mean": simulation_df["value_per_share"].mean(),
                "median": simulation_df["value_per_share"].median(),
                "std": simulation_df["value_per_share"].std(),
                "p5": simulation_df["value_per_share"].quantile(0.05),
                "p10": simulation_df["value_per_share"].quantile(0.10),
                "p25": simulation_df["value_per_share"].quantile(0.25),
                "p75": simulation_df["value_per_share"].quantile(0.75),
                "p90": simulation_df["value_per_share"].quantile(0.90),
                "p95": simulation_df["value_per_share"].quantile(0.95),
                "prob_above_current": (simulation_df["value_per_share"] > snapshot.current_price).mean(),
            }
        ]
    ).T.reset_index()
    summary_df.columns = ["Metric", "Value"]

    ending_regime_df = simulation_df.groupby("end_regime")["value_per_share"].agg(["count", "mean", "median", "min", "max"]).sort_values("mean").reset_index()
    driver_corr_df = (
        simulation_df[["value_per_share", "avg_wacc", "avg_growth", "avg_margin", "jump_count", "long_run_growth_draw", "terminal_margin_draw", "capex_pct_draw", "nwc_pct_draw", "terminal_growth_draw"]]
        .corr()[["value_per_share"]]
        .sort_values("value_per_share", ascending=False)
        .reset_index()
        .rename(columns={"index": "Variable", "value_per_share": "Correlation_to_Value"})
    )
    multiples_df = multiple_cross_check(peer_multiples_df, snapshot, metrics, shares)
    valuation_paths = build_mc_valuation_paths_rolling_terminal(
        snapshot=snapshot,
        growth_paths=all_growth_paths,
        margin_paths=all_margin_paths,
        wacc_paths=all_wacc_paths,
        d_and_a_draws=d_and_a_draws,
        capex_draws=capex_draws,
        nwc_draws=nwc_draws,
        share_paths=all_share_paths,
        terminal_growth_draws=terminal_growth_draws,
    )
    horizon_summary_df = build_horizon_summary(
        valuation_paths,
        snapshot.current_price,
        terminal_growth_draws,
        inputs.forecast_config.terminal_dilution_rate,
    )
    return {
        "state_df": state_df,
        "base_transition_df": base_transition_df,
        "macro_transition_df": macro_transition_df,
        "start_probs_df": start_probs_df,
        "summary_df": summary_df,
        "simulation_df": simulation_df,
        "horizon_summary_df": horizon_summary_df,
        "ending_regime_df": ending_regime_df,
        "driver_corr_df": driver_corr_df,
        "multiples_df": multiples_df,
        "values": values,
        "regime_paths": all_regime_paths,
        "growth_paths": all_growth_paths,
        "margin_paths": all_margin_paths,
        "wacc_paths": all_wacc_paths,
            "share_paths": all_share_paths,
            "valuation_paths": valuation_paths,
    }


def run_tornado_sensitivity_calibrated(inputs: ValuationInputs, priors: dict[str, dict[str, float]], shares: float) -> pd.DataFrame:
    snapshot = inputs.snapshot
    metrics = _company_metrics(inputs)
    years = inputs.forecast_config.projection_years
    base_case = {
        "start_growth": metrics["company_anchor_growth"],
        "long_run_growth": priors["long_run_growth"]["mu"],
        "ebit_margin_start": metrics["ebit_margin_2025"] + 0.015,
        "terminal_ebit_margin": priors["terminal_ebit_margin"]["mu"],
        "d_and_a_pct": priors["d_and_a_pct"]["mu"],
        "capex_pct": priors["capex_pct"]["mu"],
        "nwc_pct": max(0.0, priors["nwc_pct"]["mu"]),
        "regime_path": _scenario_regime_path("Base", years),
        "regime_wacc_path": _scenario_wacc_path("Base", years),
        "terminal_growth": priors["terminal_growth"]["mu"],
    }
    base_forecast = build_operating_forecast(
        inputs=inputs,
        revenue0=snapshot.revenue_2026_mid,
        start_growth=base_case["start_growth"],
        long_run_growth=base_case["long_run_growth"],
        ebit_margin_start=base_case["ebit_margin_start"],
        terminal_ebit_margin=base_case["terminal_ebit_margin"],
        d_and_a_pct=base_case["d_and_a_pct"],
        capex_base_pct=base_case["capex_pct"],
        nwc_base_pct=base_case["nwc_pct"],
        terminal_growth=base_case["terminal_growth"],
        regime_path=base_case["regime_path"],
        regime_wacc_path=base_case["regime_wacc_path"],
        shares_start=shares,
    )
    base_val = dcf_from_operating_forecast(base_forecast, base_case["terminal_growth"], snapshot.cash_2025, snapshot.debt_2025)["value_per_share"]
    tests = {
        "Start Growth": {"low": max(0.01, metrics["company_anchor_growth"] - 0.06), "high": min(0.35, metrics["company_anchor_growth"] + 0.06)},
        "Long-Run Growth": {"low": priors["long_run_growth"]["low"], "high": priors["long_run_growth"]["high"]},
        "Terminal EBIT Margin": {"low": priors["terminal_ebit_margin"]["low"], "high": priors["terminal_ebit_margin"]["high"]},
        "Capex % Revenue": {"low": priors["capex_pct"]["low"], "high": priors["capex_pct"]["high"]},
        "NWC % Incremental Rev": {"low": max(0.0, priors["nwc_pct"]["low"]), "high": max(0.0, priors["nwc_pct"]["high"])},
        "WACC Path": {"low": list(np.linspace(0.093, 0.091, years)), "high": list(np.linspace(0.109, 0.105, years))},
        "Terminal Growth": {"low": priors["terminal_growth"]["low"], "high": priors["terminal_growth"]["high"]},
    }
    rows = []
    for variable, spec in tests.items():
        low_case = base_case.copy()
        high_case = base_case.copy()
        if variable == "Start Growth":
            low_case["start_growth"] = spec["low"]
            high_case["start_growth"] = spec["high"]
        elif variable == "Long-Run Growth":
            low_case["long_run_growth"] = spec["low"]
            high_case["long_run_growth"] = spec["high"]
        elif variable == "Terminal EBIT Margin":
            low_case["terminal_ebit_margin"] = spec["low"]
            high_case["terminal_ebit_margin"] = spec["high"]
        elif variable == "Capex % Revenue":
            low_case["capex_pct"] = spec["low"]
            high_case["capex_pct"] = spec["high"]
        elif variable == "NWC % Incremental Rev":
            low_case["nwc_pct"] = spec["low"]
            high_case["nwc_pct"] = spec["high"]
        elif variable == "WACC Path":
            low_case["regime_wacc_path"] = spec["low"]
            high_case["regime_wacc_path"] = spec["high"]
        elif variable == "Terminal Growth":
            low_case["terminal_growth"] = spec["low"]
            high_case["terminal_growth"] = spec["high"]
        low_forecast = build_operating_forecast(
            inputs=inputs,
            revenue0=snapshot.revenue_2026_mid,
            start_growth=low_case["start_growth"],
            long_run_growth=low_case["long_run_growth"],
            ebit_margin_start=low_case["ebit_margin_start"],
            terminal_ebit_margin=low_case["terminal_ebit_margin"],
            d_and_a_pct=low_case["d_and_a_pct"],
            capex_base_pct=low_case["capex_pct"],
            nwc_base_pct=low_case["nwc_pct"],
            terminal_growth=low_case["terminal_growth"],
            regime_path=low_case["regime_path"],
            regime_wacc_path=low_case["regime_wacc_path"],
            shares_start=shares,
        )
        high_forecast = build_operating_forecast(
            inputs=inputs,
            revenue0=snapshot.revenue_2026_mid,
            start_growth=high_case["start_growth"],
            long_run_growth=high_case["long_run_growth"],
            ebit_margin_start=high_case["ebit_margin_start"],
            terminal_ebit_margin=high_case["terminal_ebit_margin"],
            d_and_a_pct=high_case["d_and_a_pct"],
            capex_base_pct=high_case["capex_pct"],
            nwc_base_pct=high_case["nwc_pct"],
            terminal_growth=high_case["terminal_growth"],
            regime_path=high_case["regime_path"],
            regime_wacc_path=high_case["regime_wacc_path"],
            shares_start=shares,
        )
        low_val = dcf_from_operating_forecast(low_forecast, low_case["terminal_growth"], snapshot.cash_2025, snapshot.debt_2025)["value_per_share"]
        high_val = dcf_from_operating_forecast(high_forecast, high_case["terminal_growth"], snapshot.cash_2025, snapshot.debt_2025)["value_per_share"]
        rows.append(
            {
                "Variable": variable,
                "Low Case Value": low_val,
                "Base Case Value": base_val,
                "High Case Value": high_val,
                "Low Delta": low_val - base_val,
                "High Delta": high_val - base_val,
                "Range": abs(high_val - low_val),
            }
        )
    return pd.DataFrame(rows).sort_values("Range", ascending=False).reset_index(drop=True)


def sobol_valuation_wrapper(x: np.ndarray, inputs: ValuationInputs, shares: float) -> float:
    snapshot = inputs.snapshot
    metrics = _company_metrics(inputs)
    long_run_growth, terminal_ebit_margin, d_and_a_pct, capex_pct, nwc_pct, terminal_growth, wacc = x
    periods = inputs.forecast_config.projection_years
    forecast_df = build_operating_forecast(
        inputs=inputs,
        revenue0=snapshot.revenue_2026_mid,
        start_growth=metrics["company_anchor_growth"],
        long_run_growth=long_run_growth,
        ebit_margin_start=max(0.05, metrics["ebit_margin_2025"] + 0.01),
        terminal_ebit_margin=np.clip(terminal_ebit_margin, 0.08, 0.35),
        d_and_a_pct=d_and_a_pct,
        capex_base_pct=capex_pct,
        nwc_base_pct=max(0.0, nwc_pct),
        terminal_growth=terminal_growth,
        regime_path=["Base"] * periods,
        regime_wacc_path=[wacc] * periods,
        shares_start=shares,
    )
    result = dcf_from_operating_forecast(forecast_df, terminal_growth, snapshot.cash_2025, snapshot.debt_2025)
    return float(result["value_per_share"])


def run_sobol_sensitivity(inputs: ValuationInputs, priors: dict[str, dict[str, float]], shares: float) -> pd.DataFrame:
    problem = {
        "num_vars": 7,
        "names": ["long_run_growth", "terminal_ebit_margin", "d_and_a_pct", "capex_pct", "nwc_pct", "terminal_growth", "wacc"],
        "bounds": [
            [priors["long_run_growth"]["low"], priors["long_run_growth"]["high"]],
            [priors["terminal_ebit_margin"]["low"], priors["terminal_ebit_margin"]["high"]],
            [priors["d_and_a_pct"]["low"], priors["d_and_a_pct"]["high"]],
            [priors["capex_pct"]["low"], priors["capex_pct"]["high"]],
            [max(0.0, priors["nwc_pct"]["low"]), max(0.0, priors["nwc_pct"]["high"])],
            [priors["terminal_growth"]["low"], priors["terminal_growth"]["high"]],
            [0.085, 0.125],
        ],
    }
    param_values = sobol_sample.sample(problem, inputs.run_config.sobol_n_base, calc_second_order=False)
    y = np.array([sobol_valuation_wrapper(x, inputs, shares) for x in param_values])
    si = sobol.analyze(problem, y, calc_second_order=False, print_to_console=False)
    return pd.DataFrame(
        {
            "Variable": problem["names"],
            "First Order (S1)": si["S1"],
            "First Order Conf": si["S1_conf"],
            "Total Order (ST)": si["ST"],
            "Total Order Conf": si["ST_conf"],
        }
    ).sort_values("Total Order (ST)", ascending=False).reset_index(drop=True)


def structural_sobol_valuation_wrapper(x: np.ndarray, inputs: ValuationInputs, priors: dict[str, dict[str, float]], shares: float) -> float:
    snapshot = inputs.snapshot
    metrics = _company_metrics(inputs)
    history = inputs.history.data
    comp_panel = inputs.peer_panel.data
    regime_cfg = inputs.regime_config
    long_run_growth, terminal_ebit_margin, terminal_growth, macro_stress_level, persistence_cap, jump_prob, bear_wacc, bull_wacc, shrinkage = x
    macro_stress_level = float(np.clip(macro_stress_level, 0.0, 1.0))
    persistence_cap = float(np.clip(persistence_cap, 0.35, 0.85))
    jump_prob = float(np.clip(jump_prob, 0.00, 0.30))
    shrinkage = float(np.clip(shrinkage, 0.20, 0.90))
    state_df = assign_empirical_states(comp_panel)
    base_transition_df = estimate_transition_matrix(state_df)
    macro_transition_df = adjust_transition_for_macro(base_transition_df, macro_stress_level)
    macro_transition_df = cap_transition_persistence(macro_transition_df, max_diag=persistence_cap)
    macro_transition_df = enforce_transition_floor(macro_transition_df)
    final_start_probs, _, _ = build_final_start_probs(state_df, history, macro_stress_level)
    years = inputs.forecast_config.projection_years
    regime_path = deterministic_expected_regime_path(final_start_probs, macro_transition_df, n_years=years)
    blended_long_run_growth = shrinkage * long_run_growth + (1.0 - shrinkage) * metrics["company_anchor_growth"]
    blended_terminal_margin = shrinkage * terminal_ebit_margin + (1.0 - shrinkage) * metrics["company_anchor_margin"]
    blended_long_run_growth = float(np.clip(blended_long_run_growth, 0.01, 0.35))
    blended_terminal_margin = float(np.clip(blended_terminal_margin, 0.08, 0.35))
    terminal_growth = float(np.clip(terminal_growth, 0.015, 0.040))
    expected_growth_jump = jump_prob * regime_cfg.jump_params["growth_jump_mean"]
    expected_margin_jump_total = years * jump_prob * regime_cfg.jump_params["margin_jump_mean"]
    growth_adjustments = [regime_cfg.regime_growth_shift[regime_path[t]] + expected_growth_jump for t in range(years)]
    base_wacc = 0.5 * (bear_wacc + bull_wacc)
    regime_wacc_map = {"Bear": float(np.clip(bear_wacc, 0.09, 0.16)), "Base": float(np.clip(base_wacc, 0.085, 0.14)), "Bull": float(np.clip(bull_wacc, 0.07, 0.12))}
    regime_wacc_path = [regime_wacc_map[state] for state in regime_path]
    latest = history.iloc[-1]
    ebit_margin_start = float(np.clip(latest["ebit_margin"] + regime_cfg.regime_margin_shift[regime_path[0]], 0.03, 0.30))
    forecast_df = build_operating_forecast(
        inputs=inputs,
        revenue0=snapshot.revenue_2026_mid,
        start_growth=metrics["company_anchor_growth"],
        long_run_growth=blended_long_run_growth,
        ebit_margin_start=ebit_margin_start,
        terminal_ebit_margin=blended_terminal_margin,
        d_and_a_pct=priors["d_and_a_pct"]["mu"],
        capex_base_pct=priors["capex_pct"]["mu"],
        nwc_base_pct=max(0.0, priors["nwc_pct"]["mu"]),
        terminal_growth=terminal_growth,
        regime_path=regime_path,
        regime_wacc_path=regime_wacc_path,
        shares_start=shares,
        growth_adjustments=growth_adjustments,
        margin_end_adjustment=regime_cfg.regime_margin_shift[regime_path[-1]] + expected_margin_jump_total,
    )
    result = dcf_from_operating_forecast(forecast_df, terminal_growth, snapshot.cash_2025, snapshot.debt_2025)
    return float(result["value_per_share"])


def run_structural_sobol_sensitivity(inputs: ValuationInputs, priors: dict[str, dict[str, float]], shares: float) -> pd.DataFrame:
    problem = {
        "num_vars": 9,
        "names": ["long_run_growth", "terminal_ebit_margin", "terminal_growth", "macro_stress_level", "persistence_cap", "jump_prob", "bear_wacc", "bull_wacc", "shrinkage"],
        "bounds": [
            [priors["long_run_growth"]["low"], priors["long_run_growth"]["high"]],
            [priors["terminal_ebit_margin"]["low"], priors["terminal_ebit_margin"]["high"]],
            [priors["terminal_growth"]["low"], priors["terminal_growth"]["high"]],
            [0.00, 1.00],
            [0.40, 0.75],
            [0.00, 0.20],
            [0.105, 0.135],
            [0.082, 0.100],
            [0.35, 0.85],
        ],
    }
    param_values = sobol_sample.sample(problem, inputs.run_config.structural_sobol_n_base, calc_second_order=False)
    y = np.array([structural_sobol_valuation_wrapper(x, inputs, priors, shares) for x in param_values])
    si = sobol.analyze(problem, y, calc_second_order=False, print_to_console=False)
    return pd.DataFrame(
        {
            "Variable": problem["names"],
            "First Order (S1)": si["S1"],
            "First Order Conf": si["S1_conf"],
            "Total Order (ST)": si["ST"],
            "Total Order Conf": si["ST_conf"],
        }
    ).sort_values("Total Order (ST)", ascending=False).reset_index(drop=True)


def run_valuation(inputs: ValuationInputs, run_config=None) -> ValuationResults:
    if run_config is not None:
        inputs = ValuationInputs(
            company_id=inputs.company_id,
            snapshot=inputs.snapshot,
            history=inputs.history,
            peer_panel=inputs.peer_panel,
            peer_multiples=inputs.peer_multiples,
            regime_config=inputs.regime_config,
            forecast_config=inputs.forecast_config,
            thesis_config=inputs.thesis_config,
            math_config=inputs.math_config,
            run_config=run_config,
            config_path=inputs.config_path,
            data_dir=inputs.data_dir,
        )
    shares = inputs.run_config.resolve_shares(inputs.snapshot)
    priors = fit_empirical_bayes_priors(inputs, inputs.peer_panel.data)
    scenario_df, scenario_detailed = run_scenario_table_calibrated(inputs, priors, shares)
    mc = advanced_monte_carlo_empirical(inputs, priors, shares)
    tornado_df = run_tornado_sensitivity_calibrated(inputs, priors, shares)
    sobol_df = run_sobol_sensitivity(inputs, priors, shares)
    structural_sobol_df = run_structural_sobol_sensitivity(inputs, priors, shares)
    return ValuationResults(
        company_id=inputs.company_id,
        snapshot=inputs.snapshot,
        run_config=inputs.run_config,
        priors=priors,
        scenario_df=scenario_df,
        scenario_detailed=scenario_detailed,
        start_probs_df=mc["start_probs_df"],
        base_transition_df=mc["base_transition_df"],
        macro_transition_df=mc["macro_transition_df"],
        summary_df=mc["summary_df"],
        simulation_df=mc["simulation_df"],
        horizon_summary_df=mc["horizon_summary_df"],
        ending_regime_df=mc["ending_regime_df"],
        driver_corr_df=mc["driver_corr_df"],
        multiples_df=mc["multiples_df"],
        tornado_df=tornado_df,
        sobol_df=sobol_df,
        structural_sobol_df=structural_sobol_df,
        values=mc["values"],
        regime_paths=mc["regime_paths"],
        growth_paths=mc["growth_paths"],
        margin_paths=mc["margin_paths"],
        wacc_paths=mc["wacc_paths"],
        share_paths=mc["share_paths"],
        valuation_paths=mc["valuation_paths"],
    )
