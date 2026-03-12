from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_inputs
from .model import run_valuation
from .reporting import export_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mntn_valuation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run a valuation")
    run_parser.add_argument("--company", required=True)
    run_parser.add_argument("--data-dir", required=True)
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    inputs = load_inputs(args.company, args.data_dir, args.config)
    results = run_valuation(inputs)
    output_dir = export_results(results, Path(args.output_dir), include_plots=not args.no_plots)

    print(f"\n{args.company} valuation complete")
    print(results.summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nHorizon valuation summary")
    print(results.horizon_summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print("\nReturn summary")
    print(results.return_summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nOutputs written to {output_dir}")
    return 0
