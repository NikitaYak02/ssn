#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from tools.repo_conventions import MoveSpec, build_default_move_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or apply the repository layout migration manifest.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--manifest",
        default="reports/generated/migration_manifest.json",
        help="Where to write the computed manifest.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration after writing the manifest.",
    )
    parser.add_argument(
        "--no-write-manifest",
        action="store_true",
        help="Skip writing the manifest file.",
    )
    return parser


def spec_to_dict(spec: MoveSpec) -> dict[str, str]:
    return {
        "source": spec.source,
        "target": spec.target,
        "reason": spec.reason,
    }


def write_manifest(root: Path, manifest_path: Path, specs: list[MoveSpec]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "repo_root": str(root.resolve()),
        "moves": [spec_to_dict(spec) for spec in specs],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply_manifest(root: Path, specs: list[MoveSpec]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for spec in specs:
        source = root / spec.source
        target = root / spec.target
        if not source.exists():
            results.append({"source": spec.source, "target": spec.target, "status": "missing"})
            continue
        if target.exists():
            results.append({"source": spec.source, "target": spec.target, "status": "exists"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        results.append({"source": spec.source, "target": spec.target, "status": "moved"})
    out_dir = root / "out"
    if out_dir.is_dir() and not any(out_dir.iterdir()):
        out_dir.rmdir()
        results.append({"source": "out", "target": "-", "status": "removed-empty"})
    return results


def print_dry_run(specs: list[MoveSpec]) -> None:
    for spec in specs:
        print(f"{spec.source} -> {spec.target} [{spec.reason}]")


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.repo_root).resolve()
    specs = build_default_move_specs(root)
    manifest_path = root / args.manifest

    if not args.no_write_manifest:
        write_manifest(root, manifest_path, specs)
        print(f"Manifest: {manifest_path}")

    if not args.apply:
        print_dry_run(specs)
        return 0

    results = apply_manifest(root, specs)
    moved = sum(1 for item in results if item["status"] == "moved")
    skipped = len(results) - moved
    print(f"Applied migration: moved={moved} skipped={skipped}")
    for item in results:
        print(f"{item['status']}: {item['source']} -> {item['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
