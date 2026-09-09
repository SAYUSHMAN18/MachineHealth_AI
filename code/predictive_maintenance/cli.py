from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_analysis


def analyze_command(args: argparse.Namespace) -> None:
    result = run_analysis(
        args.sos,
        args.telemetry,
        args.output,
        work_orders_path=args.work_orders,
        horizon_days=args.horizon,
        api_key=args.api_key,
        allow_external_ai=args.allow_external_ai,
    )
    print(f"Mode: {result['mode']}")
    print(f"S.O.S samples: {len(result['sos'])}")
    print(f"Unique assets: {result['sos']['asset_id'].nunique(dropna=True)}")
    print(f"Clean telemetry snapshots: {len(result['telemetry'])}")
    print(f"Matched assets: {len(result['matched_assets'])}")
    print(f"Outputs: {Path(args.output).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heavy-machinery predictive-maintenance POC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze S.O.S and telemetry files")
    analyze.add_argument("--sos", required=True)
    analyze.add_argument("--telemetry", help="Path to telemetry Excel/CSV file (Optional)")
    analyze.add_argument("--work-orders", help="Path to work orders Excel/CSV file (Optional)")
    analyze.add_argument("--output", default="outputs/current")
    analyze.add_argument("--horizon", type=int, default=30)
    analyze.add_argument("--api-key", help="Gemini API Key for LLM Insights generation")
    analyze.add_argument(
        "--allow-external-ai",
        action="store_true",
        help="Explicitly allow an aggregated, identifier-free summary prompt to be sent to Gemini",
    )
    analyze.set_defaults(func=analyze_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
