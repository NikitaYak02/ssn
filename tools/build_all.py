#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the inventory, script docs and report generators.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    return parser


def run_step(root: Path, command: list[str]) -> None:
    subprocess.run(command, cwd=root, check=True)


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.repo_root).resolve()
    run_step(root, ["python3", "tools/repo_inventory.py", "--repo-root", str(root)])
    run_step(root, ["python3", "tools/generate_script_docs.py", "--repo-root", str(root)])
    run_step(root, ["python3", "tools/generate_reports.py", "--repo-root", str(root)])
    print("Inventory, script docs and reports were rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
