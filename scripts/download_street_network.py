from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from project_name.street_network import download_lion_street_network
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "download_street_network.py requires geopandas and python-igraph."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NYC LION street layer.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/spatial/vector/streets/raw"),
        help="Directory for the raw CSV export.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/spatial/vector/streets/processed"),
        help="Directory for the GeoPackage export.",
    )
    parser.add_argument(
        "--dataset-id",
        default="2v4z-66xt",
        help="NYC Open Data dataset id for the LION street layer.",
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
    _, geodata = download_lion_street_network(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        dataset_id=args.dataset_id,
        app_token=os.getenv("NYC_APP_TOKEN"),
        limit=args.limit,
    )
    print(f"[saved] {args.raw_dir / 'lion.csv'}")
    if geodata is not None and not geodata.empty:
        print(f"[saved] {args.processed_dir / 'lion.gpkg'}")


if __name__ == "__main__":
    main()
