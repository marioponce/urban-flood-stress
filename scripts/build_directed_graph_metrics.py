from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from project_name.street_network import (
        compute_directed_graph_metrics,
        load_lion_street_layer,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "build_directed_graph_metrics.py requires geopandas and python-igraph."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build directed graph metrics for the NYC street network."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/spatial/vector/streets/processed/lion.gpkg"),
        help="Input street network source.",
    )
    parser.add_argument(
        "--layer",
        default="lion",
        help="Optional layer name for multi-layer inputs such as geodatabases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spatial/vector/streets/processed/lion_metrics.gpkg"),
        help="Vector output for street-level graph metrics.",
    )
    parser.add_argument(
        "--node-output",
        type=Path,
        default=Path("data/spatial/vector/streets/processed/lion_node_metrics.csv"),
        help="CSV output for node metrics.",
    )
    parser.add_argument(
        "--snap-tolerance-ft",
        type=float,
        default=5.0,
        help="Endpoint snapping tolerance in feet.",
    )
    return parser.parse_args()


def _write_vector(frame: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.drop(columns="geometry", errors="ignore").to_csv(path, index=False)
        return

    driver = "GPKG" if path.suffix.lower() == ".gpkg" else "ESRI Shapefile"
    frame.to_file(path, driver=driver)


def main() -> None:
    args = parse_args()
    streets = load_lion_street_layer(args.input, layer=args.layer)
    metrics, node_metrics, _ = compute_directed_graph_metrics(
        streets,
        snap_tolerance_ft=args.snap_tolerance_ft,
    )

    _write_vector(
        gpd.GeoDataFrame(metrics, geometry="geometry", crs=streets.crs),
        args.output,
    )
    args.node_output.parent.mkdir(parents=True, exist_ok=True)
    node_metrics.to_csv(args.node_output, index=False)
    print(f"[saved] {args.output}")
    print(f"[saved] {args.node_output}")


if __name__ == "__main__":
    main()
