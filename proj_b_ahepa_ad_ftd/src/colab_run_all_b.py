#!/usr/bin/env python3
"""Self-contained orchestrator for SPEC B on Colab."""

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
    parser.add_argument("--n-seeds", type=int, default=10)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    cache_name = "smoke_cache.npz" if args.smoke else "ahepa_dataset.npz"

    run_step(src_dir / "01_download_bids.py", ["--smoke"] if args.smoke else [])
    run_step(src_dir / "02_preprocess_feat.py", ["--cache-name", cache_name])
    bench_args = ["--cache-name", cache_name]
    if args.smoke:
        bench_args.append("--smoke")
    else:
        bench_args.extend(["--n-seeds", str(args.n_seeds)])
    run_step(src_dir / "04_run_benchmarks.py", bench_args)
    run_step(src_dir / "05_plot_figures.py", ["--cache-name", cache_name])


if __name__ == "__main__":
    main()
