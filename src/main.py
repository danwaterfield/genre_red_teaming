from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loaders import load_experiment_config
from .logging.jsonl_logger import export_csv
from .matrix import csv_path, labels_log_path, raw_log_path, resolve_run_id
from .runner import run_experiment
from .plotting import plot_run
from .rejudge import rejudge_sample


def _cmd_run(args: argparse.Namespace) -> int:
    run_id, out_dir = run_experiment(
        config_path=args.config,
        replicates=args.replicates,
        suite_name=args.suite,
    )
    print(f"run_id={run_id}")
    print(f"outputs_dir={out_dir}")
    return 0


def _cmd_export_csv(args: argparse.Namespace) -> int:
    cfg = load_experiment_config(args.config)
    run_id = args.run_id or cfg.run.run_id
    if not run_id:
        print("Error: run_id is required (set run.run_id in config or pass --run-id)", file=sys.stderr)
        return 2
    out_csv = Path(args.out_csv) if args.out_csv else csv_path(cfg, run_id)
    export_csv(
        raw_jsonl=raw_log_path(cfg, run_id),
        labels_jsonl=labels_log_path(cfg, run_id),
        out_csv=out_csv,
    )
    print(f"wrote_csv={out_csv}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    cfg = load_experiment_config(args.config)
    run_id = args.run_id or cfg.run.run_id
    plots_dir = plot_run(outputs_dir=cfg.run.output_dir, run_id=run_id)
    print(f"plots_dir={plots_dir}")
    return 0


def _cmd_rejudge_sample(args: argparse.Namespace) -> int:
    cfg = load_experiment_config(args.config)
    out = rejudge_sample(
        cfg=cfg,
        run_id=args.run_id,
        n=args.n,
        seed=args.seed,
        judge_temps_csv=args.judge_temperatures,
        out_path=args.out,
    )
    print(f"judge_sweep_jsonl={out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-scenario-harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the scenario×frame matrix and write JSONL logs")
    run_p.add_argument("--config", required=True, help="Path to config YAML")
    run_p.add_argument(
        "--replicates",
        type=int,
        default=1,
        help="Number of replicates per (scenario, frame, model, temperature)",
    )
    run_p.add_argument(
        "--suite",
        help="Suite name from config.suites (defaults to first suite)",
    )
    run_p.set_defaults(func=_cmd_run)

    ex_p = sub.add_parser("export-csv", help="Merge JSONL logs and export a CSV")
    ex_p.add_argument("--config", required=True, help="Path to config YAML")
    ex_p.add_argument("--run-id", help="Run id to export (overrides config.run.run_id)")
    ex_p.add_argument("--out-csv", help="Output CSV path (defaults to outputs/<run_id>/attempts.csv)")
    ex_p.set_defaults(func=_cmd_export_csv)

    plot_p = sub.add_parser("plot", help="Generate PNG plots from JSONL logs")
    plot_p.add_argument("--config", required=True, help="Path to config YAML")
    plot_p.add_argument("--run-id", help="Run id to plot (defaults to latest run under outputs/)")
    plot_p.set_defaults(func=_cmd_plot)

    rj_p = sub.add_parser(
        "rejudge-sample",
        help="Re-run the judge-blind rubric on a random sample of existing successful outputs",
    )
    rj_p.add_argument("--config", required=True, help="Path to config YAML")
    rj_p.add_argument("--run-id", required=True, help="Run id to sample from (under outputs/)")
    rj_p.add_argument("--n", type=int, default=20, help="Number of attempts to sample")
    rj_p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling")
    rj_p.add_argument(
        "--judge-temperatures",
        default="0.0,0.2,0.7",
        help="Comma-separated judge temperatures to sweep (e.g. 0,0.2,0.7)",
    )
    rj_p.add_argument(
        "--out",
        help="Output JSONL path (defaults to outputs/<run_id>/judge_sweep.jsonl)",
    )
    rj_p.set_defaults(func=_cmd_rejudge_sample)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


