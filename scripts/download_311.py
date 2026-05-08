from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.requests_311 import download_default_311_archives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYC 311 archives.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/temporal/311"),
        help="Root directory for the 311 monthly parquet files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50_000,
        help="Pagination limit for Socrata requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = download_default_311_archives(
        out_root=args.out_root,
        app_token=os.getenv("NYC_APP_TOKEN"),
        limit=args.limit,
    )
    for path in saved_paths:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
