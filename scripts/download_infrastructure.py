from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_name.infrastructure import download_infrastructure_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYC infrastructure layers.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/infrastructure"),
        help="Directory for raw CSV downloads.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/infrastructure"),
        help="Directory for processed GeoPackage outputs.",
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
    summary = download_infrastructure_layers(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        app_token=os.getenv("NYC_APP_TOKEN"),
        limit=args.limit,
    )
    for _, row in summary.iterrows():
        print(f"[saved] {row['raw_path']} rows={int(row['raw_rows']):,}")
        if row["geometry_status"] == "saved":
            print(f"[saved] {row['processed_path']} rows={int(row['geometry_rows']):,}")
    print(f"[saved] {summary.attrs.get('summary_path', args.processed_dir / 'infrastructure_download_summary.csv')}")


if __name__ == "__main__":
    main()
