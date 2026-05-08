from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from project_name.utils import ensure_directory, fetch_socrata_rows


DEFAULT_SOCRATA_DOMAIN = "data.cityofnewyork.us"

INFRASTRUCTURE_DATASETS = {
    "catch_basins": "2w2g-fk3i",
    "green_infrastructure": "df32-vzax",
    "outfalls": "8rjn-kpsh",
}


def extract_location_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand nested Socrata location dictionaries into latitude/longitude columns."""

    expanded = frame.copy()
    for column in list(expanded.columns):
        sample = expanded[column].dropna()
        if sample.empty:
            continue
        value = sample.iloc[0]
        if not isinstance(value, dict):
            continue
        if {"latitude", "longitude"} <= set(value):
            expanded[f"{column}_latitude"] = expanded[column].map(
                lambda item: item.get("latitude") if isinstance(item, dict) else None
            )
            expanded[f"{column}_longitude"] = expanded[column].map(
                lambda item: item.get("longitude") if isinstance(item, dict) else None
            )
    return expanded


def detect_longitude_latitude(frame: pd.DataFrame) -> tuple[str, str] | None:
    """Find a usable longitude/latitude pair in a Socrata response."""

    candidates = [
        ("longitude", "latitude"),
        ("lon", "lat"),
        ("x", "y"),
        ("location_1_longitude", "location_1_latitude"),
        ("the_geom_longitude", "the_geom_latitude"),
    ]

    for lon_col, lat_col in candidates:
        if lon_col in frame.columns and lat_col in frame.columns:
            return lon_col, lat_col

    for column in frame.columns:
        if column.endswith("_longitude"):
            lat_col = column.replace("_longitude", "_latitude")
            if lat_col in frame.columns:
                return column, lat_col

    return None


def maybe_build_geodataframe(frame: pd.DataFrame) -> gpd.GeoDataFrame | None:
    """Convert a frame to points when longitude/latitude are present."""

    expanded = extract_location_columns(frame)
    detected = detect_longitude_latitude(expanded)
    if detected is None:
        return None

    lon_col, lat_col = detected
    coords = expanded.copy()
    coords[lon_col] = pd.to_numeric(coords[lon_col], errors="coerce")
    coords[lat_col] = pd.to_numeric(coords[lat_col], errors="coerce")
    coords = coords.dropna(subset=[lon_col, lat_col]).copy()
    if coords.empty:
        return None

    return gpd.GeoDataFrame(
        coords,
        geometry=gpd.points_from_xy(coords[lon_col], coords[lat_col]),
        crs="EPSG:4326",
    )


def download_infrastructure_layers(
    raw_dir: Path,
    processed_dir: Path,
    *,
    app_token: str | None = None,
    domain: str = DEFAULT_SOCRATA_DOMAIN,
    limit: int = 50_000,
) -> pd.DataFrame:
    """Download NYC infrastructure layers and save raw CSV and GeoPackage outputs."""

    raw_dir = ensure_directory(Path(raw_dir))
    processed_dir = ensure_directory(Path(processed_dir))

    rows: list[dict] = []
    for layer_name, dataset_id in INFRASTRUCTURE_DATASETS.items():
        raw = fetch_socrata_rows(
            dataset_id,
            app_token=app_token,
            domain=domain,
            limit=limit,
        )
        raw_path = raw_dir / f"{layer_name}.csv"
        raw.to_csv(raw_path, index=False)

        gdf = maybe_build_geodataframe(raw)
        processed_path = processed_dir / f"{layer_name}.gpkg"
        geometry_status = "not_detected"
        geometry_rows = 0
        if gdf is not None and not gdf.empty:
            gdf.to_file(processed_path, driver="GPKG")
            geometry_status = "saved"
            geometry_rows = len(gdf)

        rows.append(
            {
                "layer_name": layer_name,
                "dataset_id": dataset_id,
                "raw_path": str(raw_path),
                "processed_path": str(processed_path),
                "raw_rows": int(len(raw)),
                "geometry_status": geometry_status,
                "geometry_rows": int(geometry_rows),
            }
        )

    summary = pd.DataFrame.from_records(rows)
    summary_path = processed_dir / "infrastructure_download_summary.csv"
    summary.to_csv(summary_path, index=False)
    summary.attrs["summary_path"] = str(summary_path)
    return summary
