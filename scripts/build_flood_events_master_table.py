from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.flood_events_master import (  # noqa: E402
    MASTER_GEOPARQUET_PATH,
    MASTER_PARQUET_PATH,
    build_master_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the event-level flood modeling master table.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build the table in memory without writing parquet outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_master_table(write_outputs=not args.no_write)
    master = result.master_table

    print(f"rows={len(master)}")
    print(f"unique_segments={master['segment_id'].nunique(dropna=True)}")
    print(f"occurrence_true={int(master['occurrence'].sum())}")
    print(f"zero_prec_events={int((master['n_prec'] == 0).sum())}")
    print(f"resolved_events={int(master['resolution_bool'].sum())}")
    if not args.no_write:
        print(f"parquet={MASTER_PARQUET_PATH}")
        print(f"geoparquet={MASTER_GEOPARQUET_PATH}")


if __name__ == "__main__":
    main()
