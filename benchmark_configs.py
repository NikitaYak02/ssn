#!/usr/bin/env python3
"""
Benchmark different configurations to find optimal settings.
Runs profile_one_batch.py with different parameters and compares results.

Usage:
    python benchmark_configs.py \
        --img_dir /path/to/images \
        --mask_dir /path/to/masks
"""

import subprocess
import sys
import time
import argparse
from collections import defaultdict


def run_profile(img_dir, mask_dir, **kwargs):
    """Run profile_one_batch.py with given parameters, return timing dict."""
    cmd = [
        sys.executable, 'profile_one_batch.py',
        '--img_dir', img_dir,
        '--mask_dir', mask_dir,
    ]
    for k, v in kwargs.items():
        cmd.extend([f'--{k}', str(v)])

    print(f"  Running: {' '.join(cmd[3:])}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[:200]}")
        return None

    # Parse output to extract TOTAL time
    lines = result.stdout.split('\n')
    total_ms = 0
    for line in lines:
        if 'TOTAL' in line and 'ms' in line:
            parts = line.split()
            try:
                total_ms = float(parts[-2])
            except:
                pass
    return total_ms


def benchmark():
    parser = argparse.ArgumentParser(
        description="Benchmark different training configurations")
    parser.add_argument("--img_dir",  required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--repeats",  default=2, type=int,
                        help="Repeat each config N times")

    args = parser.parse_args()

    print("=" * 70)
    print("CONFIG BENCHMARK")
    print("=" * 70)
    print()

    # Configurations to test
    configs = [
        {
            'name': 'baseline (batch=6, workers=2)',
            'params': {'batchsize': 6, 'nworkers': 2},
        },
        {
            'name': 'large batch (batch=12, workers=2)',
            'params': {'batchsize': 12, 'nworkers': 2},
        },
        {
            'name': 'no workers (batch=6, workers=0)',
            'params': {'batchsize': 6, 'nworkers': 0},
        },
        {
            'name': 'many workers (batch=6, workers=4)',
            'params': {'batchsize': 6, 'nworkers': 4},
        },
        {
            'name': 'small crop (batch=6, crop=128)',
            'params': {'batchsize': 6, 'nworkers': 2, 'crop_size': 128},
        },
        {
            'name': 'large crop (batch=6, crop=320)',
            'params': {'batchsize': 6, 'nworkers': 2, 'crop_size': 320},
        },
        {
            'name': 'large batch + no workers',
            'params': {'batchsize': 12, 'nworkers': 0},
        },
    ]

    results = defaultdict(list)

    for config in configs:
        print(f"Testing: {config['name']}")
        for rep in range(args.repeats):
            ms = run_profile(args.img_dir, args.mask_dir, **config['params'])
            if ms is not None:
                results[config['name']].append(ms)
                it_s = 1000 / ms
                print(f"    Run {rep + 1}: {ms:.1f} ms → {it_s:.1f} it/s")
            else:
                print(f"    Run {rep + 1}: FAILED")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY (lower is better)")
    print("=" * 70)
    summary = []
    for config_name in [c['name'] for c in configs]:
        if config_name in results and results[config_name]:
            times = results[config_name]
            mean_ms = sum(times) / len(times)
            it_s = 1000 / mean_ms
            summary.append((config_name, mean_ms, it_s))

    summary.sort(key=lambda x: x[1])  # sort by time

    for i, (name, ms, it_s) in enumerate(summary, 1):
        speedup = summary[0][2] / it_s
        marker = " ← BEST" if i == 1 else f" ({speedup:.2f}x slower)"
        print(f"{i}. {name:35s}  {ms:6.1f} ms  "
              f"{it_s:6.1f} it/s{marker}")

    print()
    print("Recommended config:")
    print(f"  {summary[0][0]}")
    best_config = next(c for c in configs if c['name'] == summary[0][0])
    print("  Parameters:")
    for k, v in best_config['params'].items():
        print(f"    --{k} {v}")


if __name__ == "__main__":
    benchmark()
