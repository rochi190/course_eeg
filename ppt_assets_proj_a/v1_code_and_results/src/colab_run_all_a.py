#!/usr/bin/env python3
"""Self-contained orchestrator for SPEC A on Colab."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run_step(script: Path, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(script)] + extra_args
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=70)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    run_step(src_dir / "01_download_data.py", [])
    run_step(src_dir / "02_synthesis.py", [])
    train_args = ["--smoke"] if args.smoke else ["--epochs", str(args.epochs)]
    run_step(src_dir / "04_train_eval.py", train_args)
    run_step(src_dir / "05_plot_figures.py", [])


if __name__ == "__main__":
    main()
