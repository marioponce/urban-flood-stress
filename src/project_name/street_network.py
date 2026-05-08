from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely import wkt

from .utils import ensure_directory, fetch_socrata_rows


DEFAULT_LION_DATASET_ID = "2v4z-66xt"
DEFAULT_LION_SOURCE_CRS = "EPSG:4326"
DEFAULT_LION_TARGET_CRS = "EPSG:2263"
SNAP_TOLERANCE_FT = 5.0
UNKNOWN_TRAFDIR_FALLBACK = "B"

VALID_FEATURE_TYPES = ("0", "6", "A", "C")
VALID_SEGMENT_TYPES = ("R", "U", "C", "E", "G", "S")


def load_lion_street_layer(path: str | Path, layer: str = "lion") -> gpd.GeoDataFrame:
    """Load a local LION geodatabase or vector file."""

    source = Path(path)
    if source.suffix.lower() in {".shp", ".gpkg", ".geojson", ".json", ".csv"}:
        return gpd.read_file(source)
    return gpd.read_file(source, layer=layer)


def detect_geometry_column(frame: pd.DataFrame) -> str | None:
    """Pick the most likely geometry column from a tabular Socrata response."""

    for candidate in ("the_geom", "shape", "geometry"):
        if candidate in frame.columns:
            return candidate

    for column in frame.columns:
        sample = frame[column].dropna()
        if sample.empty:
            continue

        value = sample.iloc[0]
        if isinstance(value, dict) and "coordinates" in value:
            return column
        if isinstance(value, str):
            stripped = value.strip().upper()
            if stripped.startswith("{") or stripped.startswith("LINESTRING"):
                return column

    return None


def _coerce_geometry(value):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, dict):
        if "geometry" in value and isinstance(value["geometry"], dict):
            return shape(value["geometry"])
        if {"type", "coordinates"} <= set(value):
            return shape(value)
        if "coordinates" in value:
            return shape({"type": "LineString", "coordinates": value["coordinates"]})

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            if "type" not in parsed and "coordinates" in parsed:
                parsed = {"type": "LineString", "coordinates": parsed["coordinates"]}
            if "coordinates" in parsed:
                return shape(parsed)

        try:
            return wkt.loads(value)
        except Exception:
            return None

    return None


def build_geodataframe(
    frame: pd.DataFrame,
    *,
    geometry_column: str | None = None,
    source_crs: str = DEFAULT_LION_SOURCE_CRS,
    target_crs: str | None = DEFAULT_LION_TARGET_CRS,
) -> gpd.GeoDataFrame | None:
    """Convert a Socrata table with geometry-like values into a GeoDataFrame."""

    if frame.empty:
        crs = target_crs or source_crs
        return gpd.GeoDataFrame(frame.copy(), geometry=[], crs=crs)

    geom_col = geometry_column or detect_geometry_column(frame)
    if geom_col is None:
        return None

    working = frame.copy()
    working["geometry"] = working[geom_col].map(_coerce_geometry)
    working = working.dropna(subset=["geometry"]).drop(columns=[geom_col])
    gdf = gpd.GeoDataFrame(working, geometry="geometry", crs=source_crs)
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf


def download_lion_street_network(
    raw_dir: str | Path = Path("data/spatial/vector/streets/raw"),
    processed_dir: str | Path = Path("data/spatial/vector/streets/processed"),
    *,
    dataset_id: str = DEFAULT_LION_DATASET_ID,
    app_token: str | None = None,
    limit: int = 50_000,
    source_crs: str = DEFAULT_LION_SOURCE_CRS,
    target_crs: str | None = DEFAULT_LION_TARGET_CRS,
    raw_name: str = "lion",
) -> tuple[pd.DataFrame, gpd.GeoDataFrame | None]:
    """Download the public LION street layer and save raw and geospatial copies."""

    raw_frame = fetch_socrata_rows(dataset_id, app_token=app_token, limit=limit)

    raw_dir = ensure_directory(Path(raw_dir))
    processed_dir = ensure_directory(Path(processed_dir))

    raw_path = raw_dir / f"{raw_name}.csv"
    raw_frame.to_csv(raw_path, index=False)

    geodata = build_geodataframe(
        raw_frame,
        source_crs=source_crs,
        target_crs=target_crs,
    )
    if geodata is not None and not geodata.empty:
        processed_path = processed_dir / f"{raw_name}.gpkg"
        geodata.to_file(processed_path, driver="GPKG")

    return raw_frame, geodata


def filter_lion_street_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the street classes used in the project graph."""

    missing = [
        column for column in ("FeatureTyp", "SegmentTyp") if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    feature_type = frame["FeatureTyp"].fillna("").astype(str)
    segment_type = frame["SegmentTyp"].fillna("").astype(str)
    mask = feature_type.isin(VALID_FEATURE_TYPES) & segment_type.isin(
        VALID_SEGMENT_TYPES
    )
    return frame.loc[mask].copy()


def _ensure_numeric_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = pd.NA
    frame[column] = pd.to_numeric(frame[column], errors="coerce")


def classify_road_class(frame: pd.DataFrame) -> pd.Series:
    """Assign a coarse road class from `RW_TYPE`."""

    if "RW_TYPE" not in frame.columns:
        return pd.Series("local", index=frame.index, name="road_class")

    rw_type = frame["RW_TYPE"].fillna("").astype(str)
    conditions = [
        rw_type.isin(["12", "13"]),
        rw_type.isin(["8", "9", "10"]),
        rw_type.isin(["5", "6", "7"]),
    ]
    choices = ["highway", "arterial", "collector"]
    values = np.select(conditions, choices, default="local")
    return pd.Series(values, index=frame.index, name="road_class")


def normalize_trafdir(value) -> tuple[str, bool]:
    cleaned = str(value).strip().upper()
    if cleaned in {"T", "W", "A"}:
        return cleaned, False
    return UNKNOWN_TRAFDIR_FALLBACK, True


def endpoint_coords(geometry):
    """Return the first and last coordinate pair for a line geometry."""

    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
        return coords[0], coords[-1]

    if geometry.geom_type == "MultiLineString":
        first = list(geometry.geoms[0].coords)[0]
        last = list(geometry.geoms[-1].coords)[-1]
        return first, last

    raise ValueError(f"Unsupported geometry type: {geometry.geom_type}")


def normalize_level(value):
    if pd.isna(value):
        return "missing"
    return str(value)


def snapped_endpoint_nodes(
    streets: pd.DataFrame,
    *,
    tolerance: float = SNAP_TOLERANCE_FT,
) -> tuple[list[int | None], list[int | None]]:
    """Snap endpoints that fall within a small tolerance and share a level."""

    required = ["NodeLevelF", "NodeLevelT", "geometry"]
    missing = [column for column in required if column not in streets.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    endpoints = []
    for row in streets.itertuples():
        start_xy, end_xy = endpoint_coords(row.geometry)
        endpoints.append(
            {
                "segment_index": row.Index,
                "endpoint": "u",
                "x": start_xy[0],
                "y": start_xy[1],
                "level": normalize_level(row.NodeLevelF),
            }
        )
        endpoints.append(
            {
                "segment_index": row.Index,
                "endpoint": "v",
                "x": end_xy[0],
                "y": end_xy[1],
                "level": normalize_level(row.NodeLevelT),
            }
        )

    parent = list(range(len(endpoints)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    buckets: dict[tuple[str, int, int], list[int]] = {}

    for idx, endpoint in enumerate(endpoints):
        cell_x = int(math.floor(endpoint["x"] / tolerance))
        cell_y = int(math.floor(endpoint["y"] / tolerance))
        level = endpoint["level"]

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_key = (level, cell_x + dx, cell_y + dy)
                for neighbor_idx in buckets.get(neighbor_key, []):
                    neighbor = endpoints[neighbor_idx]
                    if math.hypot(
                        endpoint["x"] - neighbor["x"],
                        endpoint["y"] - neighbor["y"],
                    ) <= tolerance:
                        union(idx, neighbor_idx)

        buckets.setdefault((level, cell_x, cell_y), []).append(idx)

    root_to_node_id: dict[int, int] = {}
    next_node_id = 0
    snapped_u = [None] * len(streets)
    snapped_v = [None] * len(streets)

    for idx, endpoint in enumerate(endpoints):
        root = find(idx)
        if root not in root_to_node_id:
            root_to_node_id[root] = next_node_id
            next_node_id += 1

        node_id = root_to_node_id[root]
        segment_index = endpoint["segment_index"]
        if endpoint["endpoint"] == "u":
            snapped_u[segment_index] = node_id
        else:
            snapped_v[segment_index] = node_id

    return snapped_u, snapped_v


def prepare_lion_street_network(
    frame: pd.DataFrame,
    *,
    snap_tolerance_ft: float = SNAP_TOLERANCE_FT,
) -> pd.DataFrame:
    """Filter streets and add the graph-building columns used downstream."""

    if frame.empty:
        return frame.copy()

    working = filter_lion_street_segments(frame)
    working = working.reset_index(drop=True).copy()
    working["edge_id"] = working.index
    working["road_class"] = classify_road_class(working)

    _ensure_numeric_column(working, "POSTED_SPEED")
    _ensure_numeric_column(working, "Number_Total_Lanes")

    working["POSTED_SPEED"] = working["POSTED_SPEED"] * 0.44704
    working["speed_heur"] = 6.0
    working.loc[working["Number_Total_Lanes"] >= 3, "speed_heur"] = 10.0
    working.loc[working["Number_Total_Lanes"] >= 6, "speed_heur"] = 15.0
    working["speed"] = working["POSTED_SPEED"].fillna(working["speed_heur"])
    working["length"] = working.geometry.length * 0.3048
    working["travel_time"] = working["length"] / working["speed"]

    if "TrafDir" not in working.columns:
        working["TrafDir"] = pd.NA

    trafdir_info = working["TrafDir"].map(normalize_trafdir)
    working["trafdir_resolved"] = trafdir_info.map(lambda item: item[0])
    working["trafdir_fallback_used"] = trafdir_info.map(lambda item: item[1])

    working["u"], working["v"] = snapped_endpoint_nodes(
        working, tolerance=snap_tolerance_ft
    )
    working["snapping_applied"] = True
    working["snap_tolerance_ft"] = snap_tolerance_ft
    working["in_graph"] = working["u"].notna() & working["v"].notna()
    working["u"] = working["u"].astype("Int64")
    working["v"] = working["v"].astype("Int64")

    return working


def _build_directed_graph(streets: pd.DataFrame) -> ig.Graph:
    graph_rows = streets[streets["in_graph"]].copy()
    if graph_rows.empty:
        raise ValueError("No street segments are available to build a graph.")

    node_count = int(graph_rows[["u", "v"]].to_numpy().max()) + 1
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    edge_ids: list[int] = []

    for row in graph_rows.itertuples():
        u = int(row.u)
        v = int(row.v)

        if row.trafdir_resolved in {"T", "B"}:
            edges.append((u, v))
            weights.append(row.travel_time)
            edge_ids.append(row.edge_id)

            edges.append((v, u))
            weights.append(row.travel_time)
            edge_ids.append(row.edge_id)
        elif row.trafdir_resolved == "W":
            edges.append((u, v))
            weights.append(row.travel_time)
            edge_ids.append(row.edge_id)
        elif row.trafdir_resolved == "A":
            edges.append((v, u))
            weights.append(row.travel_time)
            edge_ids.append(row.edge_id)

    graph = ig.Graph(n=node_count, edges=edges, directed=True)
    graph.es["weight"] = weights
    graph.es["edge_id"] = edge_ids
    graph.vs["node_id"] = list(range(node_count))
    return graph


def compute_directed_graph_metrics(
    frame: pd.DataFrame,
    *,
    snap_tolerance_ft: float = SNAP_TOLERANCE_FT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the directed graph and return street, node, and edge metrics."""

    prepared = prepare_lion_street_network(
        frame,
        snap_tolerance_ft=snap_tolerance_ft,
    )

    if prepared.empty:
        empty_nodes = pd.DataFrame(
            columns=[
                "node_id",
                "in_degree",
                "out_degree",
                "total_degree",
                "node_betweenness",
            ]
        )
        empty_edges = pd.DataFrame(columns=["edge_id", "edge_betweenness"])
        return prepared, empty_nodes, empty_edges

    graph = _build_directed_graph(prepared)
    node_metrics = pd.DataFrame(
        {
            "node_id": graph.vs["node_id"],
            "in_degree": graph.degree(mode="in"),
            "out_degree": graph.degree(mode="out"),
            "total_degree": graph.degree(mode="all"),
            "node_betweenness": graph.betweenness(weights="weight", directed=True),
        }
    )

    edge_metrics = pd.DataFrame(
        {
            "edge_id": graph.es["edge_id"],
            "edge_betweenness": graph.edge_betweenness(
                weights="weight", directed=True
            ),
        }
    )
    edge_metrics = (
        edge_metrics.groupby("edge_id", as_index=False)["edge_betweenness"].sum()
    )

    enriched = prepared.merge(edge_metrics, on="edge_id", how="left")
    return enriched, node_metrics, edge_metrics
