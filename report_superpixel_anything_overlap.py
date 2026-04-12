#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from external_superpixels.paper_alignment import build_superpixel_anything_overlap_report


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Write a checked overlap report between repo superpixel methods and Superpixel Anything."
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "reports" / "superpixel_anything_overlap.md"),
        help="Markdown output path.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_superpixel_anything_overlap_report() + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
