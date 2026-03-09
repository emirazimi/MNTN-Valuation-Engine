import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import qmc
from SALib.sample import saltelli
from SALib.analyze import sobol

# ============================================================
# 1) Current company data
# ============================================================

CURRENT_PRICE = 10.57

# 2025 anchors
revenue_2025 = 290.093e6
ebit_2025 = 24.044e6
adj_ebitda_2025 = 67.986e6
d_and_a_2025 = 9.870e6
capex_2025 = 12.503e6
cash_2025 = 210.160e6
debt_2025 = 0.0

shares_outstanding = 73.844625e6
shares_outstanding_diluted = 79.724041e6

# 2026 midpoint guidance anchors
revenue_2026_mid = 350.0e6
adj_ebitda_2026_mid = 97.1e6

ebit_margin_2025 = ebit_2025 / revenue_2025
adj_ebitda_margin_2026_mid = adj_ebitda_2026_mid / revenue_2026_mid
company_anchor_growth = revenue_2026_mid / revenue_2025 - 1

# Approximate recent MNTN operating history

mntn_hist = pd.DataFrame({
    "quarter": pd.period_range("2024Q1", periods=6, freq="Q"),
    "rev_growth_yoy": [0.26, 0.24, 0.23, 0.21, 0.20, 0.205],
    "ebit_margin":    [0.03, 0.045, 0.055, 0.070, 0.078, 0.083],
    "fcff_margin":    [0.01, 0.020, 0.030, 0.045, 0.055, 0.060]
})

# ============================================================
# 2) Comps data builder
# ============================================================

def build_sample_comp_panel(seed=42):
    rng = np.random.default_rng(seed)

    companies = ["TTD", "MGNI", "PUBM", "PERI", "CRTO"]
    quarters = pd.period_range("2019Q1", "2025Q4", freq="Q")

    rows = []
    for c in companies:
        base_growth = {
            "TTD": 0.23,
            "MGNI": 0.14,
            "PUBM": 0.13,
            "PERI": 0.10,
            "CRTO": 0.05
        }[c]

        base_ebitda_margin = {
            "TTD": 0.34,
            "MGNI": 0.23,
            "PUBM": 0.27,
            "PERI": 0.18,
            "CRTO": 0.16
        }[c]

        base_ebit_margin = {
            "TTD": 0.22,
            "MGNI": 0.10,
            "PUBM": 0.14,
            "PERI": 0.09,
            "CRTO": 0.07
        }[c]

        rev = {
            "TTD": 180e6,
            "MGNI": 120e6,
            "PUBM": 55e6,
            "PERI": 95e6,
            "CRTO": 500e6
        }[c]

        for q in quarters:
            macro_noise = rng.normal(0, 0.04)
            growth = np.clip(base_growth + macro_noise, -0.10, 0.50)
            rev = rev * (1 + growth / 4.0)

            ebitda_margin = np.clip(base_ebitda_margin + rng.normal(0, 0.03), -0.05, 0.50)
            ebit_margin = np.clip(base_ebit_margin + rng.normal(0, 0.025), -0.10, 0.35)

            d_and_a_pct = np.clip(0.035 + rng.normal(0, 0.005), 0.015, 0.060)
            capex_pct = np.clip(0.040 + rng.normal(0, 0.008), 0.015, 0.080)
            nwc_pct = np.clip(0.030 + rng.normal(0, 0.015), -0.020, 0.080)

            ebitda = rev * ebitda_margin
            ebit = rev * ebit_margin
            d_and_a = rev * d_and_a_pct
            capex = rev * capex_pct
            delta_nwc = max(rev * nwc_pct * 0.1, -0.01 * rev)

            fcff = ebit * (1 - 0.25) + d_and_a - capex - delta_nwc

            rows.append({
                "company": c,
                "quarter": q,
                "revenue": rev,
                "ebitda": ebitda,
                "ebit": ebit,
                "d_and_a": d_and_a,
                "capex": capex,
                "delta_nwc": delta_nwc,
                "fcff": fcff
            })

    df = pd.DataFrame(rows).sort_values(["company", "quarter"]).reset_index(drop=True)

    df["revenue_lag_4"] = df.groupby("company")["revenue"].shift(4)
    df["rev_growth_yoy"] = df["revenue"] / df["revenue_lag_4"] - 1
    df["ebitda_margin"] = df["ebitda"] / df["revenue"]
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df["fcff_margin"] = df["fcff"] / df["revenue"]
    df["d_and_a_pct"] = df["d_and_a"] / df["revenue"]
    df["capex_pct"] = df["capex"] / df["revenue"]
    df["nwc_pct"] = df["delta_nwc"] / df["revenue"]

    return df.dropna().reset_index(drop=True)


def build_sample_peer_multiples(comp_panel):
    latest = comp_panel.sort_values(["company", "quarter"]).groupby("company").tail(1).copy()

    mapping_rev = {"TTD": 17.0, "MGNI": 2.2, "PUBM": 2.0, "PERI": 1.4, "CRTO": 1.1}
    mapping_ebitda = {"TTD": 42.0, "MGNI": 10.5, "PUBM": 7.8, "PERI": 7.0, "CRTO": 5.8}

    latest["ev_revenue_ntm"] = latest["company"].map(mapping_rev)
    latest["ev_ebitda_ntm"] = latest["company"].map(mapping_ebitda)
    latest["ntm_revenue_growth"] = latest["rev_growth_yoy"]
    latest["ntm_ebitda_margin"] = latest["ebitda_margin"]

    return latest[[
        "company",
        "ev_revenue_ntm",
        "ev_ebitda_ntm",
        "ntm_revenue_growth",
        "ntm_ebitda_margin"
    ]].reset_index(drop=True)

# ============================================================
# 3) Helper functions for sampling and transformations
# ============================================================

def sobol_valuation_wrapper(X):
    """
    Deterministic wrapper for Sobol analysis.
    Input vector X order:
    [long_run_growth, terminal_ebit_margin, d_and_a_pct, capex_pct, nwc_pct, terminal_growth, wacc]
    """
    long_run_growth, terminal_ebit_margin, d_and_a_pct, capex_pct, nwc_pct, terminal_growth, wacc = X

    # Build an 8-year explicit forecast to reduce terminal-value dominance
    growth_rates = np.array([
        long_run_growth + 0.06,
        long_run_growth + 0.04,
        long_run_growth + 0.02,
        long_run_growth + 0.01,
        long_run_growth,
        long_run_growth - 0.01,
        long_run_growth - 0.02,
        long_run_growth - 0.03
    ])
    growth_rates = np.clip(growth_rates, 0.01, 0.40)

    n_years = len(growth_rates)

    # Smooth margin convergence
    ebit_margin_start = max(0.05, ebit_margin_2025 + 0.01)
    ebit_margin_end = np.clip(terminal_ebit_margin, 0.08, 0.35)

    # Use a base-state regime path for deterministic Sobol testing
    regime_path = ["Base"] * n_years
    regime_wacc_path = [wacc] * n_years

    result = dcf_fcff_regime_discounted(
        revenue0=revenue_2026_mid,
        growth_rates=growth_rates,
        ebit_margin_start=ebit_margin_start,
        ebit_margin_end=ebit_margin_end,
        tax_rate=0.25,
        d_and_a_pct=d_and_a_pct,
        capex_pct=capex_pct,
        nwc_pct_of_incremental_rev=max(0.0, nwc_pct),
        regime_path=regime_path,
        regime_wacc_path=regime_wacc_path,
        terminal_growth=terminal_growth,
        cash=cash_2025,
        debt=debt_2025,
        shares=shares_outstanding
    )

    return result["value_per_share"]

def winsorize_series(s, lower=0.05, upper=0.95):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def clipped_ppf(u, dist_name, params):
    u = np.clip(u, 1e-6, 1 - 1e-6)

    if dist_name == "truncnorm":
        mean, std, low, high = params
        a = (low - mean) / std
        b = (high - mean) / std
        return stats.truncnorm.ppf(u, a, b, loc=mean, scale=std)

    if dist_name == "triangular":
        low, mode, high = params
        c = (mode - low) / (high - low)
        return stats.triang.ppf(u, c, loc=low, scale=(high - low))

    raise ValueError("Unsupported distribution")


def correlated_lhs_uniforms(n_sims, corr_matrix, seed=42):
    dim = corr_matrix.shape[0]
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    U = sampler.random(n=n_sims)

    Z = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    L = np.linalg.cholesky(corr_matrix)
    Z_corr = Z @ L.T
    U_corr = stats.norm.cdf(Z_corr)

    return np.clip(U_corr, 1e-6, 1 - 1e-6)


def soften_probabilities(probs, temperature=2.5):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, None)
    logits = np.log(probs)
    softened = np.exp(logits / temperature)
    softened = softened / softened.sum()
    return softened


def apply_probability_floor(probs, floor=np.array([0.20, 0.30, 0.20])):
    probs = np.asarray(probs, dtype=float)
    probs = np.maximum(probs, floor)
    probs = probs / probs.sum()
    return probs

# ============================================================
# 4) Constructing empirical Bayes priors
# ============================================================

def fit_empirical_bayes_priors(comp_panel):
    df = comp_panel.copy()

    cols = ["rev_growth_yoy", "ebit_margin", "d_and_a_pct", "capex_pct", "nwc_pct"]
    for col in cols:
        df[col] = winsorize_series(df[col])

    priors = {
        "long_run_growth": {
            "mu": df["rev_growth_yoy"].mean(),
            "sigma": max(df["rev_growth_yoy"].std(), 1e-4),
            "low": df["rev_growth_yoy"].quantile(0.05),
            "high": df["rev_growth_yoy"].quantile(0.95)
        },
        "terminal_ebit_margin": {
            "mu": df["ebit_margin"].mean(),
            "sigma": max(df["ebit_margin"].std(), 1e-4),
            "low": df["ebit_margin"].quantile(0.05),
            "high": df["ebit_margin"].quantile(0.95)
        },
        "d_and_a_pct": {
            "mu": df["d_and_a_pct"].mean(),
            "sigma": max(df["d_and_a_pct"].std(), 1e-4),
            "low": df["d_and_a_pct"].quantile(0.05),
            "high": df["d_and_a_pct"].quantile(0.95)
        },
        "capex_pct": {
            "mu": df["capex_pct"].mean(),
            "sigma": max(df["capex_pct"].std(), 1e-4),
            "low": df["capex_pct"].quantile(0.05),
            "high": df["capex_pct"].quantile(0.95)
        },
        "nwc_pct": {
            "mu": df["nwc_pct"].mean(),
            "sigma": max(df["nwc_pct"].std(), 1e-4),
            "low": df["nwc_pct"].quantile(0.05),
            "high": df["nwc_pct"].quantile(0.95)
        },
        "terminal_growth": {
            "mu": min(0.03, max(0.02, 0.35 * df["rev_growth_yoy"].mean())),
            "sigma": 0.004,
            "low": 0.018,
            "high": 0.038
        }
    }

    return priors


def sample_bayesian_prior(u, prior_dict):
    return clipped_ppf(
        u,
        "truncnorm",
        (prior_dict["mu"], prior_dict["sigma"], prior_dict["low"], prior_dict["high"])
    )

# ============================================================
# 5) Peer states
# ============================================================

def assign_empirical_states(comp_panel):
    df = comp_panel.copy()

    df["growth_pctile"] = df["rev_growth_yoy"].rank(pct=True)
    df["margin_pctile"] = df["ebit_margin"].rank(pct=True)
    df["fcff_pctile"] = df["fcff_margin"].rank(pct=True)

    df["state_score"] = (
        0.45 * df["growth_pctile"] +
        0.35 * df["margin_pctile"] +
        0.20 * df["fcff_pctile"]
    )

    df["state"] = pd.qcut(df["state_score"], q=3, labels=["Bear", "Base", "Bull"])
    return df

# ============================================================
# 6) Transition matrix
# ============================================================

def enforce_transition_floor(T, floor=1e-4):
    T = T.copy().astype(float)
    T = T.clip(lower=floor)
    T = T.div(T.sum(axis=1), axis=0)
    return T


def estimate_transition_matrix(state_df):
    df = state_df.copy().sort_values(["company", "quarter"])
    df["next_state"] = df.groupby("company")["state"].shift(-1)

    trans = df.dropna(subset=["next_state"]).copy()
    T = pd.crosstab(trans["state"], trans["next_state"], normalize="index")
    T = T.reindex(index=["Bear", "Base", "Bull"], columns=["Bear", "Base", "Bull"]).fillna(0.0)

    return enforce_transition_floor(T, floor=1e-4)


def adjust_transition_for_macro(base_transition_df, stress_level):
    ##  stress_level in [0, 1]
    ##  Higher stress pushes mass toward Bear and away from Bull/Base.
    T = base_transition_df.copy()

    for state in T.index:
        bear_boost = 0.12 * stress_level
        bull_cut = 0.09 * stress_level
        base_cut = 0.03 * stress_level

        T.loc[state, "Bear"] += bear_boost
        T.loc[state, "Bull"] = max(1e-6, T.loc[state, "Bull"] - bull_cut)
        T.loc[state, "Base"] = max(1e-6, T.loc[state, "Base"] - base_cut)

        T.loc[state] = T.loc[state] / T.loc[state].sum()

    return enforce_transition_floor(T, floor=1e-4)


def cap_transition_persistence(T, max_diag=0.55):
    T = T.copy().astype(float)

    for state in T.index:
        diag_val = T.loc[state, state]

        if diag_val > max_diag:
            excess = diag_val - max_diag
            T.loc[state, state] = max_diag

            others = [s for s in T.columns if s != state]
            T.loc[state, others] += excess / len(others)

        T.loc[state] = T.loc[state] / T.loc[state].sum()

    return enforce_transition_floor(T, floor=1e-4)

# ============================================================
# 7) Starting regime probabilities
# ------------------------------------------------------------
# Key choice:
# We DO NOT let short-history MNTN inference dominate.
# We blend (for accuracy):
#   - peer-likelihood
#   - macro prior
#   - softening
#   - uncertainty floor
# ============================================================

def infer_peer_likelihood_start_probs(state_df, mntn_growth, mntn_margin, mntn_fcff_margin):
    probs = {}

    for state in ["Bear", "Base", "Bull"]:
        sub = state_df[state_df["state"] == state][["rev_growth_yoy", "ebit_margin", "fcff_margin"]].copy()
        mu = sub.mean().values
        cov = np.cov(sub.values.T) + np.eye(3) * 5e-4
        cov = cov * 1.75

        x = np.array([mntn_growth, mntn_margin, mntn_fcff_margin])
        probs[state] = stats.multivariate_normal.pdf(x, mean=mu, cov=cov)

    arr = np.array([probs["Bear"], probs["Base"], probs["Bull"]], dtype=float)
    arr = np.clip(arr, 1e-12, None)
    arr = arr / arr.sum()
    return arr


def build_macro_start_prior(stress_level):
    ## Explicit macro-aware prior.
    ## This is intentional: for a short-history stock, this is more honest
    ## than pretending we can infer 99% Base.

    
    # Low stress = more balanced / slightly base-leaning
    # High stress = more bear-leaning
    bear = 0.20 + 0.25 * stress_level
    base = 0.50 - 0.10 * stress_level
    bull = 1.0 - bear - base

    prior = np.array([bear, base, bull], dtype=float)
    prior = prior / prior.sum()
    return prior


def build_final_start_probs(state_df, mntn_hist, stress_level):
    latest = mntn_hist.iloc[-1]

    peer_probs = infer_peer_likelihood_start_probs(
        state_df,
        mntn_growth=latest["rev_growth_yoy"],
        mntn_margin=latest["ebit_margin"],
        mntn_fcff_margin=latest["fcff_margin"]
    )

    macro_prior = build_macro_start_prior(stress_level)

    # Blend peer information with macro prior.
    # Macro gets real weight because short MNTN history cannot support high-confidence inference.
    raw = 0.45 * peer_probs + 0.55 * macro_prior
    raw = raw / raw.sum()
    ## normalize

    # Flatten any overconfidence
    raw = soften_probabilities(raw, temperature=2.75)

    # Explicit uncertainty floor
    final_probs = apply_probability_floor(raw, floor=np.array([0.20, 0.30, 0.20]))

    return final_probs, peer_probs, macro_prior

# ============================================================
# 8) Regime settings and jump shocks
# ============================================================

REGIME_WACC_BASE = {
    "Bear": 0.118,
    "Base": 0.100,
    "Bull": 0.091
}

REGIME_GROWTH_SHIFT = {
    "Bear": -0.05,
    "Base":  0.00,
    "Bull":  0.04
}

REGIME_MARGIN_SHIFT = {
    "Bear": -0.025,
    "Base":  0.00,
    "Bull":  0.020
}

JUMP_PARAMS = {
    "annual_prob": 0.10,
    "growth_jump_mean": -0.04,
    "growth_jump_std": 0.015,
    "margin_jump_mean": -0.015,
    "margin_jump_std": 0.007
}


def sample_jump_shock(rng, params):
    jump_occurs = rng.uniform() < params["annual_prob"]
    if not jump_occurs:
        return 0.0, 0.0, 0

    growth_hit = min(-0.005, rng.normal(params["growth_jump_mean"], params["growth_jump_std"]))
    margin_hit = min(-0.002, rng.normal(params["margin_jump_mean"], params["margin_jump_std"]))
    return growth_hit, margin_hit, 1

# ============================================================
# 9) Simulate regime
# ============================================================

def simulate_regime_path(n_years, start_probs, transition_matrix_df, rng):
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

# ============================================================
# 10) DCF engine
# ============================================================

def dcf_fcff_regime_discounted(
    revenue0,
    growth_rates,
    ebit_margin_start,
    ebit_margin_end,
    tax_rate,
    d_and_a_pct,
    capex_pct,
    nwc_pct_of_incremental_rev,
    regime_path,
    regime_wacc_path,
    terminal_growth,
    cash,
    debt,
    shares
):
    n = len(growth_rates)

    rev_prev = revenue0
    revenues, ebits, fcffs, margins = [], [], [], []
    margin_path = np.linspace(ebit_margin_start, ebit_margin_end, n)

    for t in range(n):
        rev_t = rev_prev * (1 + growth_rates[t])
        ebit_t = rev_t * margin_path[t]
        nopat_t = ebit_t * (1 - tax_rate)
        d_and_a_t = rev_t * d_and_a_pct
        capex_t = rev_t * capex_pct
        delta_nwc_t = nwc_pct_of_incremental_rev * max(rev_t - rev_prev, 0.0)
        fcff_t = nopat_t + d_and_a_t - capex_t - delta_nwc_t

        revenues.append(rev_t)
        ebits.append(ebit_t)
        fcffs.append(fcff_t)
        margins.append(margin_path[t])

        rev_prev = rev_t

    revenues = np.array(revenues)
    ebits = np.array(ebits)
    fcffs = np.array(fcffs)
    margins = np.array(margins)

    cumulative_discount = []
    running = 1.0
    for w in regime_wacc_path:
        running *= (1 + w)
        cumulative_discount.append(running)
    cumulative_discount = np.array(cumulative_discount)

    pv_fcffs = np.sum(fcffs / cumulative_discount)

    terminal_wacc = regime_wacc_path[-1]
    if terminal_wacc <= terminal_growth:
        terminal_wacc = terminal_growth + 0.005

    terminal_fcff = fcffs[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcff / (terminal_wacc - terminal_growth)
    pv_terminal = terminal_value / cumulative_discount[-1]

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
        "value_per_share": value_per_share
    }

# ============================================================
# 11) Scenarios
# ============================================================

def run_scenario_table_calibrated(priors, shares=shares_outstanding):
    g = priors["long_run_growth"]
    m = priors["terminal_ebit_margin"]
    d = priors["d_and_a_pct"]
    c = priors["capex_pct"]
    n = priors["nwc_pct"]
    tg = priors["terminal_growth"]

    scenarios = {
        "Bear": {
            "growth_rates": [
                max(0.01, g["mu"] + g["low"] - 0.02),
                max(0.01, g["mu"] + g["low"] - 0.04),
                max(0.01, g["mu"] + g["low"] - 0.06),
                max(0.01, g["mu"] + g["low"] - 0.08),
                max(0.01, g["low"])
            ],
            "ebit_margin_start": max(0.05, ebit_margin_2025 - 0.02),
            "ebit_margin_end": max(0.08, m["low"]),
            "d_and_a_pct": d["mu"],
            "capex_pct": min(0.10, c["mu"] + 0.005),
            "nwc_pct": max(0.0, n["mu"] + 0.01),
            "terminal_growth": tg["low"],
            "regime_path": ["Bear", "Bear", "Base", "Base", "Base"],
            "regime_wacc_path": [0.116, 0.115, 0.109, 0.106, 0.104]
        },
        "Base": {
            "growth_rates": [
                max(0.01, g["mu"] + 0.05),
                max(0.01, g["mu"] + 0.03),
                max(0.01, g["mu"] + 0.01),
                max(0.01, g["mu"] - 0.01),
                max(0.01, g["mu"] - 0.03)
            ],
            "ebit_margin_start": ebit_margin_2025 + 0.015,
            "ebit_margin_end": m["mu"],
            "d_and_a_pct": d["mu"],
            "capex_pct": c["mu"],
            "nwc_pct": max(0.0, n["mu"]),
            "terminal_growth": tg["mu"],
            "regime_path": ["Base", "Base", "Base", "Base", "Base"],
            "regime_wacc_path": [0.101, 0.100, 0.100, 0.099, 0.099]
        },
        "Bull": {
            "growth_rates": [
                min(0.40, g["high"] + 0.05),
                min(0.35, g["high"] + 0.02),
                min(0.30, g["high"]),
                min(0.25, g["mu"] + 0.02),
                min(0.20, g["mu"])
            ],
            "ebit_margin_start": ebit_margin_2025 + 0.04,
            "ebit_margin_end": m["high"],
            "d_and_a_pct": d["mu"],
            "capex_pct": max(0.0, c["mu"] - 0.002),
            "nwc_pct": max(0.0, n["mu"] - 0.003),
            "terminal_growth": tg["high"],
            "regime_path": ["Bull", "Bull", "Bull", "Base", "Base"],
            "regime_wacc_path": [0.094, 0.093, 0.092, 0.093, 0.094]
        }
    }

    rows = []
    detailed = {}

    for name, s in scenarios.items():
        res = dcf_fcff_regime_discounted(
            revenue0=revenue_2026_mid,
            growth_rates=s["growth_rates"],
            ebit_margin_start=s["ebit_margin_start"],
            ebit_margin_end=s["ebit_margin_end"],
            tax_rate=0.25,
            d_and_a_pct=s["d_and_a_pct"],
            capex_pct=s["capex_pct"],
            nwc_pct_of_incremental_rev=s["nwc_pct"],
            regime_path=s["regime_path"],
            regime_wacc_path=s["regime_wacc_path"],
            terminal_growth=s["terminal_growth"],
            cash=cash_2025,
            debt=debt_2025,
            shares=shares
        )

        rows.append({
            "Scenario": name,
            "Value / Share": res["value_per_share"],
            "Upside / Downside": res["value_per_share"] / CURRENT_PRICE - 1,
            "Avg WACC": np.mean(s["regime_wacc_path"]),
            "Terminal Growth": s["terminal_growth"]
        })
        detailed[name] = res

    scenario_df = pd.DataFrame(rows).sort_values("Value / Share").reset_index(drop=True)
    return scenario_df, detailed

# ============================================================
# 12) Multiple Cross-Check
# ============================================================

def apply_growth_margin_premium(
    base_multiple,
    mntn_growth,
    peer_growth_median,
    mntn_margin,
    peer_margin_median,
    growth_weight=0.50,
    margin_weight=0.35,
    cap_up=0.35,
    cap_down=-0.35
):
    premium = (
        growth_weight * (mntn_growth - peer_growth_median) +
        margin_weight * (mntn_margin - peer_margin_median)
    )
    premium = np.clip(premium, cap_down, cap_up)
    adjusted_multiple = base_multiple * (1 + premium)
    return max(0.1, adjusted_multiple)


def multiple_cross_check(
    peer_multiples_df,
    mntn_forward_revenue,
    mntn_forward_ebitda,
    mntn_forward_growth,
    mntn_forward_ebitda_margin,
    net_cash,
    shares
):
    med_ev_rev = peer_multiples_df["ev_revenue_ntm"].median()
    med_ev_ebitda = peer_multiples_df["ev_ebitda_ntm"].median()
    peer_growth_median = peer_multiples_df["ntm_revenue_growth"].median()
    peer_margin_median = peer_multiples_df["ntm_ebitda_margin"].median()

    adj_ev_rev = apply_growth_margin_premium(
        med_ev_rev,
        mntn_forward_growth,
        peer_growth_median,
        mntn_forward_ebitda_margin,
        peer_margin_median
    )

    adj_ev_ebitda = apply_growth_margin_premium(
        med_ev_ebitda,
        mntn_forward_growth,
        peer_growth_median,
        mntn_forward_ebitda_margin,
        peer_margin_median
    )

    methods = {
        "Peer Median EV/Revenue": med_ev_rev * mntn_forward_revenue,
        "Peer Median EV/EBITDA": med_ev_ebitda * mntn_forward_ebitda,
        "Adj EV/Revenue": adj_ev_rev * mntn_forward_revenue,
        "Adj EV/EBITDA": adj_ev_ebitda * mntn_forward_ebitda
    }

    rows = []
    for name, ev in methods.items():
        eq = ev + net_cash
        vps = eq / shares
        rows.append({
            "Method": name,
            "Implied EV": ev,
            "Implied Equity Value": eq,
            "Implied Value / Share": vps
        })

    return pd.DataFrame(rows)

# ============================================================
# 13) Monte Carlo Simulation
# ============================================================

def advanced_monte_carlo_empirical(
    comp_panel,
    peer_multiples_df,
    mntn_hist,
    macro_stress_level=0.50,
    n_sims=30000,
    shares=shares_outstanding,
    seed=42,
    shrinkage=0.65
):
    rng = np.random.default_rng(seed)

    priors = fit_empirical_bayes_priors(comp_panel)
    state_df = assign_empirical_states(comp_panel)

    base_transition_df = estimate_transition_matrix(state_df)
    macro_transition_df = adjust_transition_for_macro(base_transition_df, macro_stress_level)
    macro_transition_df = cap_transition_persistence(macro_transition_df, max_diag=0.55)
    macro_transition_df = enforce_transition_floor(macro_transition_df, floor=1e-4)

    final_start_probs, peer_start_probs, macro_prior = build_final_start_probs(
        state_df=state_df,
        mntn_hist=mntn_hist,
        stress_level=macro_stress_level
    )

    start_probs_df = pd.DataFrame({
        "Regime": ["Bear", "Base", "Bull"],
        "Peer Likelihood": peer_start_probs,
        "Macro Prior": macro_prior,
        "Final Start Probability": final_start_probs
    })

    corr = np.array([
        [1.00,  0.45,  0.10,  0.15,  0.10, -0.05],
        [0.45,  1.00,  0.10,  0.10,  0.10,  0.10],
        [0.10,  0.10,  1.00,  0.35,  0.00,  0.00],
        [0.15,  0.10,  0.35,  1.00,  0.20,  0.00],
        [0.10,  0.10,  0.00,  0.20,  1.00,  0.00],
        [-0.05, 0.10,  0.00,  0.00,  0.00,  1.00]
    ])

    U = correlated_lhs_uniforms(n_sims, corr, seed)

    values = np.empty(n_sims)
    all_regime_paths = []
    all_wacc_paths = []
    all_growth_paths = []
    all_margin_paths = []
    jump_counts = np.zeros(n_sims)

    long_run_growth_draws = np.zeros(n_sims)
    terminal_margin_draws = np.zeros(n_sims)
    d_and_a_draws = np.zeros(n_sims)
    capex_draws = np.zeros(n_sims)
    nwc_draws = np.zeros(n_sims)
    terminal_growth_draws = np.zeros(n_sims)

    company_anchor_margin = 0.23
    company_anchor_dna = d_and_a_2025 / revenue_2025
    company_anchor_capex = capex_2025 / revenue_2025
    company_anchor_nwc = 0.040

    latest_mntn = mntn_hist.iloc[-1]

    for i in range(n_sims):
        prior_growth = sample_bayesian_prior(U[i, 0], priors["long_run_growth"])
        prior_margin = sample_bayesian_prior(U[i, 1], priors["terminal_ebit_margin"])
        prior_dna = sample_bayesian_prior(U[i, 2], priors["d_and_a_pct"])
        prior_capex = sample_bayesian_prior(U[i, 3], priors["capex_pct"])
        prior_nwc = sample_bayesian_prior(U[i, 4], priors["nwc_pct"])
        terminal_growth = sample_bayesian_prior(U[i, 5], priors["terminal_growth"])

        long_run_growth = shrinkage * prior_growth + (1 - shrinkage) * company_anchor_growth
        terminal_ebit_margin = shrinkage * prior_margin + (1 - shrinkage) * company_anchor_margin
        d_and_a_pct = shrinkage * prior_dna + (1 - shrinkage) * company_anchor_dna
        capex_pct = shrinkage * prior_capex + (1 - shrinkage) * company_anchor_capex
        nwc_pct = shrinkage * prior_nwc + (1 - shrinkage) * company_anchor_nwc

        long_run_growth_draws[i] = long_run_growth
        terminal_margin_draws[i] = terminal_ebit_margin
        d_and_a_draws[i] = d_and_a_pct
        capex_draws[i] = capex_pct
        nwc_draws[i] = nwc_pct
        terminal_growth_draws[i] = terminal_growth

        regime_path = simulate_regime_path(
            n_years=5,
            start_probs=final_start_probs,
            transition_matrix_df=macro_transition_df,
            rng=rng
        )

        base_growth_path = np.array([
        long_run_growth + 0.06,
        long_run_growth + 0.04,
        long_run_growth + 0.02,
        long_run_growth + 0.01,
        long_run_growth,
        long_run_growth - 0.01,
        long_run_growth - 0.02,
        long_run_growth - 0.03
])

        growth_rates = []
        ebit_margin_start = latest_mntn["ebit_margin"] + REGIME_MARGIN_SHIFT[regime_path[0]]
        ebit_margin_end = terminal_ebit_margin + REGIME_MARGIN_SHIFT[regime_path[-1]]
        margin_shock_accum = 0.0
        regime_wacc_path = []
        jump_total = 0

        for t in range(5):
            reg = regime_path[t]
            g_t = base_growth_path[t] + REGIME_GROWTH_SHIFT[reg]

            jump_g, jump_m, jump_flag = sample_jump_shock(rng, JUMP_PARAMS)
            g_t += jump_g
            margin_shock_accum += jump_m
            jump_total += jump_flag

            g_t = np.clip(g_t, 0.01, 0.40)
            growth_rates.append(g_t)

            wacc_t = REGIME_WACC_BASE[reg] + rng.normal(0.0, 0.004)
            wacc_t = np.clip(wacc_t, 0.08, 0.14)
            regime_wacc_path.append(wacc_t)

        ebit_margin_end = np.clip(ebit_margin_end + margin_shock_accum, 0.08, 0.35)
        margin_path = np.linspace(ebit_margin_start, ebit_margin_end, 5)

        result = dcf_fcff_regime_discounted(
            revenue0=revenue_2026_mid,
            growth_rates=growth_rates,
            ebit_margin_start=ebit_margin_start,
            ebit_margin_end=ebit_margin_end,
            tax_rate=0.25,
            d_and_a_pct=d_and_a_pct,
            capex_pct=capex_pct,
            nwc_pct_of_incremental_rev=nwc_pct,
            regime_path=regime_path,
            regime_wacc_path=regime_wacc_path,
            terminal_growth=terminal_growth,
            cash=cash_2025,
            debt=debt_2025,
            shares=shares
        )

        values[i] = result["value_per_share"]
        all_regime_paths.append(regime_path)
        all_wacc_paths.append(regime_wacc_path)
        all_growth_paths.append(growth_rates)
        all_margin_paths.append(margin_path)
        jump_counts[i] = jump_total

    all_regime_paths = np.array(all_regime_paths)
    all_wacc_paths = np.array(all_wacc_paths)
    all_growth_paths = np.array(all_growth_paths)
    all_margin_paths = np.array(all_margin_paths)

    simulation_df = pd.DataFrame({
        "value_per_share": values,
        "start_regime": all_regime_paths[:, 0],
        "end_regime": all_regime_paths[:, -1],
        "avg_wacc": np.mean(all_wacc_paths, axis=1),
        "year1_wacc": all_wacc_paths[:, 0],
        "year5_wacc": all_wacc_paths[:, -1],
        "avg_growth": np.mean(all_growth_paths, axis=1),
        "year1_growth": all_growth_paths[:, 0],
        "year5_growth": all_growth_paths[:, -1],
        "avg_margin": np.mean(all_margin_paths, axis=1),
        "year1_margin": all_margin_paths[:, 0],
        "year5_margin": all_margin_paths[:, -1],
        "jump_count": jump_counts,
        "long_run_growth_draw": long_run_growth_draws,
        "terminal_margin_draw": terminal_margin_draws,
        "d_and_a_pct_draw": d_and_a_draws,
        "capex_pct_draw": capex_draws,
        "nwc_pct_draw": nwc_draws,
        "terminal_growth_draw": terminal_growth_draws,
        "upside_downside": values / CURRENT_PRICE - 1
    })

    summary = {
        "mean": simulation_df["value_per_share"].mean(),
        "median": simulation_df["value_per_share"].median(),
        "std": simulation_df["value_per_share"].std(),
        "p5": simulation_df["value_per_share"].quantile(0.05),
        "p10": simulation_df["value_per_share"].quantile(0.10),
        "p25": simulation_df["value_per_share"].quantile(0.25),
        "p75": simulation_df["value_per_share"].quantile(0.75),
        "p90": simulation_df["value_per_share"].quantile(0.90),
        "p95": simulation_df["value_per_share"].quantile(0.95),
        "prob_above_current": (simulation_df["value_per_share"] > CURRENT_PRICE).mean()
    }

    summary_df = pd.DataFrame([summary]).T.reset_index()
    summary_df.columns = ["Metric", "Value"]

    ending_regime_df = (
        simulation_df.groupby("end_regime")["value_per_share"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("mean")
        .reset_index()
    )

    driver_corr_df = (
        simulation_df[
            [
                "value_per_share",
                "avg_wacc",
                "avg_growth",
                "avg_margin",
                "jump_count",
                "long_run_growth_draw",
                "terminal_margin_draw",
                "capex_pct_draw",
                "nwc_pct_draw",
                "terminal_growth_draw"
            ]
        ]
        .corr()[["value_per_share"]]
        .sort_values("value_per_share", ascending=False)
        .reset_index()
        .rename(columns={"index": "Variable", "value_per_share": "Correlation_to_Value"})
    )

    multiples_df = multiple_cross_check(
        peer_multiples_df=peer_multiples_df,
        mntn_forward_revenue=revenue_2026_mid,
        mntn_forward_ebitda=adj_ebitda_2026_mid,
        mntn_forward_growth=company_anchor_growth,
        mntn_forward_ebitda_margin=adj_ebitda_margin_2026_mid,
        net_cash=cash_2025 - debt_2025,
        shares=shares
    )
    valuation_paths = build_mc_valuation_paths_rolling_terminal(
    revenue0=revenue_2026_mid,
    growth_paths=all_growth_paths,
    margin_paths=all_margin_paths,
    wacc_paths=all_wacc_paths,
    d_and_a_draws=d_and_a_draws,
    capex_draws=capex_draws,
    nwc_draws=nwc_draws,
    terminal_growth_draws=terminal_growth_draws,
    tax_rate=0.25,
    cash=cash_2025,
    debt=debt_2025,
    shares=shares
)

    return {
        "priors": priors,
        "state_df": state_df,
        "base_transition_df": base_transition_df,
        "macro_transition_df": macro_transition_df,
        "start_probs_df": start_probs_df,
        "summary_df": summary_df,
        "simulation_df": simulation_df,
        "ending_regime_df": ending_regime_df,
        "driver_corr_df": driver_corr_df,
        "multiples_df": multiples_df,
        "values": values,
        "regime_paths": all_regime_paths,
        "growth_paths": all_growth_paths,
        "margin_paths": all_margin_paths,
        "wacc_paths": all_wacc_paths,
        "valuation_paths": valuation_paths,
    }

# ============================================================
# 14) Tornado Sensitivity Analysis
# ============================================================

def run_tornado_sensitivity_calibrated(priors, shares=shares_outstanding):
    base_case = {
        "revenue0": revenue_2026_mid,
        "growth_rates": [0.21, 0.19, 0.16, 0.13, 0.10],
        "ebit_margin_start": ebit_margin_2025 + 0.015,
        "ebit_margin_end": priors["terminal_ebit_margin"]["mu"],
        "tax_rate": 0.25,
        "d_and_a_pct": priors["d_and_a_pct"]["mu"],
        "capex_pct": priors["capex_pct"]["mu"],
        "nwc_pct_of_incremental_rev": max(0.0, priors["nwc_pct"]["mu"]),
        "regime_path": ["Base", "Base", "Base", "Base", "Base"],
        "regime_wacc_path": [0.101, 0.100, 0.100, 0.099, 0.099],
        "terminal_growth": priors["terminal_growth"]["mu"],
        "cash": cash_2025,
        "debt": debt_2025,
        "shares": shares
    }

    base_val = dcf_fcff_regime_discounted(**base_case)["value_per_share"]

    tests = {
        "Growth Path": {
            "low": [0.16, 0.14, 0.12, 0.09, 0.07],
            "high": [0.25, 0.22, 0.18, 0.15, 0.12]
        },
        "Terminal EBIT Margin": {
            "low": priors["terminal_ebit_margin"]["low"],
            "high": priors["terminal_ebit_margin"]["high"]
        },
        "Capex % Revenue": {
            "low": priors["capex_pct"]["low"],
            "high": priors["capex_pct"]["high"]
        },
        "NWC % Incremental Rev": {
            "low": max(0.0, priors["nwc_pct"]["low"]),
            "high": max(0.0, priors["nwc_pct"]["high"])
        },
        "WACC Path": {
            "low": [0.093, 0.093, 0.092, 0.092, 0.091],
            "high": [0.109, 0.108, 0.107, 0.106, 0.105]
        },
        "Terminal Growth": {
            "low": priors["terminal_growth"]["low"],
            "high": priors["terminal_growth"]["high"]
        }
    }

    rows = []

    for variable, spec in tests.items():
        low_case = base_case.copy()
        high_case = base_case.copy()

        if variable == "Growth Path":
            low_case["growth_rates"] = spec["low"]
            high_case["growth_rates"] = spec["high"]
        elif variable == "Terminal EBIT Margin":
            low_case["ebit_margin_end"] = spec["low"]
            high_case["ebit_margin_end"] = spec["high"]
        elif variable == "Capex % Revenue":
            low_case["capex_pct"] = spec["low"]
            high_case["capex_pct"] = spec["high"]
        elif variable == "NWC % Incremental Rev":
            low_case["nwc_pct_of_incremental_rev"] = spec["low"]
            high_case["nwc_pct_of_incremental_rev"] = spec["high"]
        elif variable == "WACC Path":
            low_case["regime_wacc_path"] = spec["low"]
            high_case["regime_wacc_path"] = spec["high"]
        elif variable == "Terminal Growth":
            low_case["terminal_growth"] = spec["low"]
            high_case["terminal_growth"] = spec["high"]

        low_val = dcf_fcff_regime_discounted(**low_case)["value_per_share"]
        high_val = dcf_fcff_regime_discounted(**high_case)["value_per_share"]

        rows.append({
            "Variable": variable,
            "Low Case Value": low_val,
            "Base Case Value": base_val,
            "High Case Value": high_val,
            "Low Delta": low_val - base_val,
            "High Delta": high_val - base_val,
            "Range": abs(high_val - low_val)
        })

    return pd.DataFrame(rows).sort_values("Range", ascending=False).reset_index(drop=True)


# ============================================================
# 15) Sobol Method Sensitivity Analysis
# ============================================================

def run_sobol_sensitivity(priors, n_base=2048):
    """
    Run Sobol sensitivity analysis on the deterministic valuation wrapper.
    n_base should be a power of 2 for best performance.
    """
    problem = {
        "num_vars": 7,
        "names": [
            "long_run_growth",
            "terminal_ebit_margin",
            "d_and_a_pct",
            "capex_pct",
            "nwc_pct",
            "terminal_growth",
            "wacc"
        ],
        "bounds": [
            [priors["long_run_growth"]["low"], priors["long_run_growth"]["high"]],
            [priors["terminal_ebit_margin"]["low"], priors["terminal_ebit_margin"]["high"]],
            [priors["d_and_a_pct"]["low"], priors["d_and_a_pct"]["high"]],
            [priors["capex_pct"]["low"], priors["capex_pct"]["high"]],
            [max(0.0, priors["nwc_pct"]["low"]), max(0.0, priors["nwc_pct"]["high"])],
            [priors["terminal_growth"]["low"], priors["terminal_growth"]["high"]],
            [0.085, 0.125]  # chosen valuation range for WACC
        ]
    }

    param_values = saltelli.sample(problem, n_base, calc_second_order=False)

    Y = np.array([sobol_valuation_wrapper(x) for x in param_values])

    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

    sobol_df = pd.DataFrame({
        "Variable": problem["names"],
        "First Order (S1)": Si["S1"],
        "First Order Conf": Si["S1_conf"],
        "Total Order (ST)": Si["ST"],
        "Total Order Conf": Si["ST_conf"]
    }).sort_values("Total Order (ST)", ascending=False).reset_index(drop=True)

    return problem, param_values, Y, Si, sobol_df

# ============================================================
# 16) Plots
# ============================================================

def plot_scenario_valuation_bar(scenario_df):
    plt.figure(figsize=(9, 5))
    plt.bar(scenario_df["Scenario"], scenario_df["Value / Share"])
    plt.axhline(CURRENT_PRICE, linestyle="--", label=f"Current Price = ${CURRENT_PRICE:.2f}")
    plt.title("MNTN Calibrated Scenario DCF")
    plt.ylabel("Value / Share ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scenario_paths(detailed, key, title, ylabel, scale=1.0):
    plt.figure(figsize=(9, 5))
    years = np.arange(1, 6)
    for scenario_name, result in detailed.items():
        plt.plot(years, result[key] / scale, marker="o", label=scenario_name)
    plt.title(title)
    plt.xlabel("Projection Year")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_histogram(values, title):
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=60, edgecolor="black", alpha=0.85)
    plt.axvline(CURRENT_PRICE, linestyle="--", label=f"Current Price = ${CURRENT_PRICE:.2f}")
    plt.axvline(np.median(values), linestyle="--", label=f"Median = ${np.median(values):.2f}")
    plt.axvline(np.percentile(values, 25), linestyle=":", label=f"P25 = ${np.percentile(values, 25):.2f}")
    plt.axvline(np.percentile(values, 75), linestyle=":", label=f"P75 = ${np.percentile(values, 75):.2f}")
    plt.title(title)
    plt.xlabel("Value / Share ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cdf(values, title):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y)
    plt.axvline(CURRENT_PRICE, linestyle="--", label=f"Current Price = ${CURRENT_PRICE:.2f}")
    plt.title(title)
    plt.xlabel("Value / Share ($)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_regime_heatmap(regime_paths):
    states = ["Bear", "Base", "Bull"]
    counts = np.zeros((5, 3))

    for t in range(5):
        for j, state in enumerate(states):
            counts[t, j] = np.mean(regime_paths[:, t] == state)

    plt.figure(figsize=(8, 4))
    plt.imshow(counts, aspect="auto")
    plt.xticks([0, 1, 2], states)
    plt.yticks(range(5), [f"Year {i}" for i in range(1, 6)])
    plt.colorbar(label="Probability")
    plt.title("Projected Regime Probabilities by Year")
    plt.tight_layout()
    plt.show()


def plot_sample_paths(growth_paths, margin_paths, wacc_paths, n_show=50):
    years = np.arange(1, 6)

    plt.figure(figsize=(9, 5))
    for i in range(min(n_show, len(growth_paths))):
        plt.plot(years, growth_paths[i], alpha=0.20)
    plt.title("Sample Growth Paths")
    plt.xlabel("Projection Year")
    plt.ylabel("Revenue Growth")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    for i in range(min(n_show, len(margin_paths))):
        plt.plot(years, margin_paths[i], alpha=0.20)
    plt.title("Sample EBIT Margin Paths")
    plt.xlabel("Projection Year")
    plt.ylabel("EBIT Margin")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    for i in range(min(n_show, len(wacc_paths))):
        plt.plot(years, wacc_paths[i], alpha=0.20)
    plt.title("Sample Regime-Dependent WACC Paths")
    plt.xlabel("Projection Year")
    plt.ylabel("WACC")
    plt.tight_layout()
    plt.show()


def plot_tornado(tornado_df):
    df = tornado_df.sort_values("Range")
    y = np.arange(len(df))

    plt.figure(figsize=(10, 5))
    plt.barh(y, df["Low Delta"], alpha=0.75, label="Low Case")
    plt.barh(y, df["High Delta"], alpha=0.75, label="High Case")
    plt.yticks(y, df["Variable"])
    plt.axvline(0, linestyle="--")
    plt.title("Tornado Sensitivity")
    plt.xlabel("Change in Value / Share ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_empirical_prior_density(comp_panel):
    x = comp_panel["rev_growth_yoy"].values
    y = comp_panel["ebit_margin"].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    kernel = stats.gaussian_kde(values)
    zz = np.reshape(kernel(positions).T, xx.shape)

    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, zz, levels=20)
    plt.scatter(x, y, alpha=0.25, s=10)
    plt.title("Empirical Prior Density: Revenue Growth vs EBIT Margin")
    plt.xlabel("Revenue Growth YoY")
    plt.ylabel("EBIT Margin")
    plt.tight_layout()
    plt.show()

def plot_sobol_indices(sobol_df):
    df = sobol_df.sort_values("Total Order (ST)", ascending=True)

    y = np.arange(len(df))

    plt.figure(figsize=(10, 6))
    plt.barh(y - 0.18, df["First Order (S1)"], height=0.35, label="First Order")
    plt.barh(y + 0.18, df["Total Order (ST)"], height=0.35, label="Total Order")
    plt.yticks(y, df["Variable"])
    plt.xlabel("Sobol Sensitivity Index")
    plt.title("Sobol Sensitivity Analysis: MNTN Valuation")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sobol_response_scatter(param_values, Y, problem, top_k=3):
    df = pd.DataFrame(param_values, columns=problem["names"])
    df["valuation"] = Y

    top_vars = problem["names"][:top_k]

    # Better: choose by simple absolute correlation
    corr = df.corr(numeric_only=True)["valuation"].drop("valuation").abs().sort_values(ascending=False)
    top_vars = corr.index[:top_k]

    for var in top_vars:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[var], df["valuation"], alpha=0.25, s=10)
        plt.xlabel(var)
        plt.ylabel("Value / Share ($)")
        plt.title(f"Sobol Response Scatter: {var} vs Valuation")
        plt.tight_layout()
        plt.show()

def build_mc_valuation_paths_rolling_terminal(
    revenue0,
    growth_paths,
    margin_paths,
    wacc_paths,
    d_and_a_draws,
    capex_draws,
    nwc_draws,
    terminal_growth_draws,
    tax_rate=0.25,
    cash=cash_2025,
    debt=debt_2025,
    shares=shares_outstanding
):
    """
    Build year-by-year intrinsic value per share paths.
    Each year t includes:
      - PV of FCFF through year t
      - terminal value computed at year t and discounted back
    This makes each point a full valuation, not a partial one.
    """
    n_sims, n_years = growth_paths.shape
    valuation_paths = np.zeros((n_sims, n_years))

    for i in range(n_sims):
        rev_prev = revenue0
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

            running_discount *= (1 + wacc_paths[i, t])
            cumulative_pv += fcff_t / running_discount

            # rolling terminal value at year t
            terminal_wacc = max(wacc_paths[i, t], terminal_growth_draws[i] + 0.005)
            terminal_fcff = fcff_t * (1 + terminal_growth_draws[i])
            terminal_value_t = terminal_fcff / (terminal_wacc - terminal_growth_draws[i])
            terminal_pv_t = terminal_value_t / running_discount

            equity_value_t = cumulative_pv + terminal_pv_t + cash - debt
            valuation_paths[i, t] = equity_value_t / shares

            rev_prev = rev_t

    return valuation_paths

def plot_mc_spaghetti(valuation_paths, n_show=200):
    """
    Plot Monte Carlo valuation paths like a stock-path simulation chart.
    """
    n_sims, n_years = valuation_paths.shape
    years = np.arange(0, n_years + 1)

    plt.figure(figsize=(12, 7))

    # Sample a subset so the chart stays readable
    if n_sims > n_show:
        idx = np.linspace(0, n_sims - 1, n_show).astype(int)
        paths_to_plot = valuation_paths[idx]
    else:
        paths_to_plot = valuation_paths

    # Start all paths at current price for visual consistency
    for path in paths_to_plot:
        full_path = np.insert(path, 0, CURRENT_PRICE)
        plt.plot(years, full_path, alpha=0.5, linewidth=1)

    plt.title("Monte Carlo Simulated Intrinsic Value Paths: MNTN")
    plt.xlabel("Projection Year")
    plt.ylabel("Value / Share ($)")
    plt.tight_layout()
    plt.show()

def plot_mc_fan_chart(valuation_paths):
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
    plt.axhline(CURRENT_PRICE, linestyle="--", label=f"Current Price = ${CURRENT_PRICE:.2f}")

    plt.title("Monte Carlo Intrinsic Value Fan Chart: MNTN")
    plt.xlabel("Projection Year")
    plt.ylabel("Value / Share ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 17) Main
# ============================================================

if __name__ == "__main__":
    comp_panel = build_sample_comp_panel(seed=42)
    peer_multiples_df = build_sample_peer_multiples(comp_panel)

    priors = fit_empirical_bayes_priors(comp_panel)
    scenario_df, scenario_detailed = run_scenario_table_calibrated(priors)

    print("\nEMPIRICAL PRIORS")
    print(pd.DataFrame(priors).T.to_string(float_format=lambda x: f"{x:,.4f}"))

    results = advanced_monte_carlo_empirical(
        comp_panel=comp_panel,
        peer_multiples_df=peer_multiples_df,
        mntn_hist=mntn_hist,
        macro_stress_level=0.50,
        n_sims=100000,
        shares=shares_outstanding,
        seed=42,
        shrinkage=0.65
    )

    print("\nSTART PROBABILITIES")
    print(results["start_probs_df"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBASE TRANSITION MATRIX")
    print(results["base_transition_df"].to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nMACRO-ADJUSTED TRANSITION MATRIX")
    print(results["macro_transition_df"].to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\nCALIBRATED SCENARIO TABLE")
    print(scenario_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nADVANCED MONTE CARLO SUMMARY")
    print(results["summary_df"].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nSIMULATION BREAKDOWN BY ENDING REGIME")
    print(results["ending_regime_df"].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nDRIVER CORRELATIONS TO VALUE")
    print(results["driver_corr_df"].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\nMULTIPLE CROSS-CHECKS")
    print(results["multiples_df"].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    tornado_df = run_tornado_sensitivity_calibrated(priors)
    print("\nTORNADO SENSITIVITY")
    print(tornado_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    sobol_problem, sobol_params, sobol_Y, sobol_Si, sobol_df = run_sobol_sensitivity(priors, n_base=2048)

    print("\nSOBOL SENSITIVITY ANALYSIS")
    print(sobol_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Optional exports (AI did this part) - uncomment if needed
    # comp_panel.to_csv("comp_panel.csv", index=False)
    # peer_multiples_df.to_csv("peer_multiples.csv", index=False)
    # scenario_df.to_csv("mntn_scenarios_calibrated.csv", index=False)
    # results["simulation_df"].to_csv("mntn_simulations_empirical.csv", index=False)
    # results["multiples_df"].to_csv("mntn_multiple_cross_check.csv", index=False)
    # tornado_df.to_csv("mntn_tornado_empirical.csv", index=False)

    # Plots
    plot_scenario_valuation_bar(scenario_df)
    plot_scenario_paths(scenario_detailed, "revenues", "Scenario Revenue Paths", "Revenue ($mm)", scale=1e6)
    plot_scenario_paths(scenario_detailed, "fcffs", "Scenario FCFF Paths", "FCFF ($mm)", scale=1e6)
    plot_histogram(results["values"], "MNTN Valuation Distribution")
    plot_cdf(results["values"], "MNTN Valuation CDF")
    plot_regime_heatmap(results["regime_paths"])
    plot_sample_paths(results["growth_paths"], results["margin_paths"], results["wacc_paths"], n_show=50)
    plot_tornado(tornado_df)
    plot_empirical_prior_density(comp_panel)
    plot_mc_spaghetti(results["valuation_paths"], n_show=250)
    plot_mc_fan_chart(results["valuation_paths"])
    plot_sobol_indices(sobol_df)
    plot_sobol_response_scatter(sobol_params, sobol_Y, sobol_problem, top_k=3)

