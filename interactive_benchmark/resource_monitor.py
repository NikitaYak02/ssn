from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass
class ResourceSnapshot:
    rss_bytes: int
    gpu_memory_mib: int


@dataclass
class ResourceUsage:
    wall_time_sec: float
    peak_rss_bytes: int
    peak_gpu_memory_mib: int
    memory_limit_bytes: int | None
    memory_limit_exceeded: bool


@dataclass
class CommandRunResult:
    returncode: int
    stdout: str
    stderr: str
    usage: ResourceUsage


def _read_children_pids(pid: int) -> list[int]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    if not children_path.exists():
        return []
    try:
        text = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not text:
        return []
    out: list[int] = []
    for token in text.split():
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _collect_descendants(root_pid: int) -> set[int]:
    pending = [int(root_pid)]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for child in _read_children_pids(current):
            if child not in seen:
                pending.append(child)
    return seen


def _read_rss_bytes(pid: int) -> int:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return 0
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except OSError:
        return 0
    return 0


def _query_gpu_memory_by_pid() -> dict[int, int]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {}
    if proc.returncode != 0:
        return {}
    usage: dict[int, int] = {}
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(parts[1])
        except ValueError:
            continue
        usage[pid] = usage.get(pid, 0) + mem
    return usage


def _collect_snapshot(root_pid: int) -> ResourceSnapshot:
    pids = _collect_descendants(root_pid)
    rss_total = sum(_read_rss_bytes(pid) for pid in pids)
    gpu_by_pid = _query_gpu_memory_by_pid()
    gpu_total = sum(int(gpu_by_pid.get(pid, 0)) for pid in pids)
    return ResourceSnapshot(rss_bytes=int(rss_total), gpu_memory_mib=int(gpu_total))


def run_monitored_command(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    memory_limit_gb: float | None = None,
    poll_interval_sec: float = 0.2,
) -> CommandRunResult:
    if not command:
        raise ValueError("command must be non-empty")

    memory_limit_bytes = None
    if memory_limit_gb is not None:
        memory_limit_bytes = int(float(memory_limit_gb) * (1024**3))

    process = subprocess.Popen(
        [str(item) for item in command],
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    state = {
        "peak_rss_bytes": 0,
        "peak_gpu_memory_mib": 0,
        "memory_limit_exceeded": False,
        "stopped": False,
    }

    def _monitor() -> None:
        while not state["stopped"]:
            snap = _collect_snapshot(process.pid)
            if snap.rss_bytes > state["peak_rss_bytes"]:
                state["peak_rss_bytes"] = snap.rss_bytes
            if snap.gpu_memory_mib > state["peak_gpu_memory_mib"]:
                state["peak_gpu_memory_mib"] = snap.gpu_memory_mib
            if memory_limit_bytes is not None and snap.rss_bytes > memory_limit_bytes:
                state["memory_limit_exceeded"] = True
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                time.sleep(1.0)
                if process.poll() is None:
                    try:
                        os.kill(process.pid, signal.SIGKILL)
                    except OSError:
                        pass
                break
            if process.poll() is not None:
                break
            time.sleep(max(0.05, float(poll_interval_sec)))

    start = time.perf_counter()
    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    stdout, stderr = process.communicate()
    state["stopped"] = True
    monitor_thread.join(timeout=2.0)
    final_snap = _collect_snapshot(process.pid)
    state["peak_rss_bytes"] = max(int(state["peak_rss_bytes"]), int(final_snap.rss_bytes))
    state["peak_gpu_memory_mib"] = max(int(state["peak_gpu_memory_mib"]), int(final_snap.gpu_memory_mib))
    elapsed = time.perf_counter() - start

    usage = ResourceUsage(
        wall_time_sec=float(elapsed),
        peak_rss_bytes=int(state["peak_rss_bytes"]),
        peak_gpu_memory_mib=int(state["peak_gpu_memory_mib"]),
        memory_limit_bytes=memory_limit_bytes,
        memory_limit_exceeded=bool(state["memory_limit_exceeded"]),
    )
    return CommandRunResult(
        returncode=int(process.returncode),
        stdout=str(stdout),
        stderr=str(stderr),
        usage=usage,
    )
