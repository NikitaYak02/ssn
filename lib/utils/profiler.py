"""
Simple profiler to track time spent in different pipeline stages.
"""

import time
import torch
from collections import defaultdict

from lib.utils.torch_device import synchronize_device


class StageProfiler:
    """
    Context manager for timing stages. Tracks both wall time and
    GPU time (via torch.cuda.Event).

    Usage:
        prof = StageProfiler(device='cuda', enabled=True)
        with prof('forward'):
            model(x)
        with prof('backward'):
            loss.backward()
        prof.report()
    """

    def __init__(self, device='cuda', enabled=True):
        self.device = device
        self.enabled = enabled
        self.times = defaultdict(list)      # stage_name → [elapsed_ms, ...]
        self.current_stage = None
        self.t_start = None
        self.event_start = None

    def __enter__(self):
        if not self.enabled:
            return self
        if self.device == 'cuda':
            synchronize_device(torch, self.device)
            self.event_start = torch.cuda.Event(enable_timing=True)
            self.event_start.record()
        elif self.device == 'mps':
            synchronize_device(torch, self.device)
        self.t_start = time.perf_counter()
        return self

    def __call__(self, stage_name):
        self.current_stage = stage_name
        return self

    def __exit__(self, *args):
        if not self.enabled:
            return
        if self.device == 'cuda':
            event_end = torch.cuda.Event(enable_timing=True)
            event_end.record()
            synchronize_device(torch, self.device)
            elapsed_ms = self.event_start.elapsed_time(event_end)
        else:
            synchronize_device(torch, self.device)
            elapsed_ms = (time.perf_counter() - self.t_start) * 1000
        self.times[self.current_stage].append(elapsed_ms)

    def report(self, top_n=10):
        """Print timing report, sorted by total time."""
        if not self.enabled or not self.times:
            return
        print("\n" + "=" * 70)
        print("PROFILER REPORT (ms per stage)")
        print("=" * 70)
        # Sort by total time
        items = sorted(
            self.times.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )[:top_n]
        for stage, times in items:
            total = sum(times)
            count = len(times)
            mean = total / count if count > 0 else 0
            print(f"  {stage:20s}  total={total:8.1f} ms  "
                  f"count={count:5d}  mean={mean:6.2f} ms")
        print("=" * 70)

    def reset(self):
        """Clear all accumulated times."""
        self.times.clear()

    def summary_dict(self):
        """Return dict {stage_name: (total_ms, count, mean_ms)}."""
        return {
            stage: (sum(times), len(times), sum(times) / len(times))
            for stage, times in self.times.items()
        }


class BatchProfiler:
    """
    Simpler version: just accumulates times per batch without fancy GPU events.
    Good for quick iteration.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.t_stage_start = None

    def start(self, stage):
        if not self.enabled:
            return
        self.t_stage_start = time.perf_counter()
        self._current_stage = stage

    def end(self):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.t_stage_start
        self.times[self._current_stage] += elapsed
        self.counts[self._current_stage] += 1

    def report(self, prefix=""):
        """Print stats for all stages tracked."""
        if not self.enabled or not self.times:
            return
        print(f"\n{prefix}Profiling (seconds):")
        total = sum(self.times.values())
        for stage in sorted(self.times.keys()):
            t = self.times[stage]
            c = self.counts[stage]
            pct = 100 * t / total if total > 0 else 0
            print(f"  {stage:25s}  {t:8.2f} s  "
                  f"({c:6d} calls)  {pct:5.1f}%")
        print(f"  {'TOTAL':25s}  {total:8.2f} s")

    def reset(self):
        self.times.clear()
        self.counts.clear()
