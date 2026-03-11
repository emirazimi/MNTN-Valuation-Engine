from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .types import (
    CompanySnapshot,
    ForecastConfig,
    HistoricalSeries,
    PeerMultiples,
    PeerPanel,
    RegimeConfig,
    ValuationInputs,
    ValuationRunConfig,
)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required input file: {path}") from exc


def load_company_snapshot(company_id: str, data_dir: Path) -> CompanySnapshot:
    raw = _load_json(data_dir / "companies" / company_id / "snapshot.json")
    return CompanySnapshot(company_id=company_id, **raw)


def load_company_history(company_id: str, data_dir: Path) -> HistoricalSeries:
    path = data_dir / "companies" / company_id / "history.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required input file: {path}") from exc
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    return HistoricalSeries(company_id=company_id, data=df)


def _prepare_peer_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
    df = df.sort_values(["company", "quarter"]).reset_index(drop=True)
    df["revenue_lag_4"] = df.groupby("company")["revenue"].shift(4)
    df["rev_growth_yoy"] = df["revenue"] / df["revenue_lag_4"] - 1
    df["ebitda_margin"] = df["ebitda"] / df["revenue"]
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df["fcff_margin"] = df["fcff"] / df["revenue"]
    df["d_and_a_pct"] = df["d_and_a"] / df["revenue"]
    df["capex_pct"] = df["capex"] / df["revenue"]
    df["nwc_pct"] = df["delta_nwc"] / df["revenue"]
    return df.dropna().reset_index(drop=True)


def load_peer_panel(data_dir: Path) -> PeerPanel:
    path = data_dir / "peers" / "peer_panel.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required input file: {path}") from exc
    return PeerPanel(data=_prepare_peer_panel(df))


def load_peer_multiples(data_dir: Path) -> PeerMultiples:
    path = data_dir / "peers" / "peer_multiples.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required input file: {path}") from exc
    return PeerMultiples(data=df)


def load_run_config(config_path: Path) -> tuple[ValuationRunConfig, RegimeConfig, ForecastConfig]:
    raw = _load_json(config_path)
    run_cfg = ValuationRunConfig(**raw["run"])
    regime_cfg = RegimeConfig(**raw["regime"])
    forecast_cfg = ForecastConfig(**raw["forecast"])
    return run_cfg, regime_cfg, forecast_cfg


def load_inputs(company_id: str, data_dir: str | Path, config_path: str | Path) -> ValuationInputs:
    data_dir = Path(data_dir)
    config_path = Path(config_path)
    run_config, regime_config, forecast_config = load_run_config(config_path)
    return ValuationInputs(
        company_id=company_id,
        snapshot=load_company_snapshot(company_id, data_dir),
        history=load_company_history(company_id, data_dir),
        peer_panel=load_peer_panel(data_dir),
        peer_multiples=load_peer_multiples(data_dir),
        regime_config=regime_config,
        forecast_config=forecast_config,
        run_config=run_config,
        config_path=config_path,
        data_dir=data_dir,
    )
