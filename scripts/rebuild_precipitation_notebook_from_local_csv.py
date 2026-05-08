from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "precipitation.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(
        markdown_cell(
            """# NYC Precipitation

Pipeline local para:

- leer `data/temporal/noaa/precipitation.csv`
- usar solo las 4 estaciones presentes en ese CSV
- construir poligonos de Thiessen recortados a NYC
- detectar eventos de lluvia horarios
- cruzar Thiessen con la red de calles

Reglas de eventos:

- separacion minima seca: `6` horas
- profundidad minima del evento: `1.1` mm
- `duration` se calcula con el ultimo timestamp humedo mas el intervalo nominal de muestreo de la estacion
"""
        )
    )

    cells.append(
        code_cell(
            """import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import MultiPoint
from shapely.ops import unary_union, voronoi_diagram

MPLCONFIGDIR = Path("/tmp/matplotlib")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    return current


ROOT = find_project_root()
LOCAL_CRS = "EPSG:2263"

PRECIP_PATH = ROOT / "data" / "temporal" / "noaa" / "precipitation.csv"
STATION_METADATA_PATH = ROOT / "data" / "temporal" / "noaa" / "4308211.csv"
NYC_BOUNDARY_PATH = ROOT / "data" / "spatial" / "vector" / "nyc_borough_boundary" / "nybb.geojson"
STREETS_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_metrics.gpkg"

STATION_POINTS_OUTPUT = ROOT / "data" / "spatial" / "vector" / "noaa" / "precipitation_station_points.geojson"
THIESSEN_OUTPUT = ROOT / "data" / "spatial" / "vector" / "noaa" / "precipitation_thiessen.geojson"
RECORDS_OUTPUT = ROOT / "data" / "processed" / "precipitation" / "hourly" / "precipitation_records.csv"
EVENTS_OUTPUT = ROOT / "data" / "processed" / "precipitation" / "events_hourly" / "precipitation_events.csv"
CROSSWALK_OUTPUT = ROOT / "data" / "spatial" / "vector" / "noaa" / "precipitation_segment_station_crosswalk.csv"

PRECIP_INPUT_UNIT = "inch"
EVENT_DRY_GAP_HOURS = 6
MIN_EVENT_DEPTH_MM = 1.1
VORONOI_BUFFER_FEET = 5000

for output_path in [
    STATION_POINTS_OUTPUT,
    THIESSEN_OUTPUT,
    RECORDS_OUTPUT,
    EVENTS_OUTPUT,
    CROSSWALK_OUTPUT,
]:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def unit_factor_to_mm(unit: str) -> float:
    unit = unit.strip().lower()
    if unit in {"inch", "inches", "in"}:
        return 25.4
    if unit in {"mm", "millimeter", "millimeters"}:
        return 1.0
    raise ValueError(f"Unsupported precipitation unit: {unit}")


def load_nyc_boundary() -> tuple[gpd.GeoDataFrame, object]:
    boroughs = gpd.read_file(NYC_BOUNDARY_PATH)
    boroughs = boroughs.to_crs(LOCAL_CRS)
    boroughs["geometry"] = boroughs.geometry.map(make_valid)
    boundary = unary_union(boroughs.geometry)
    return boroughs, boundary


def load_station_points() -> gpd.GeoDataFrame:
    stations_in_csv = pd.read_csv(PRECIP_PATH, usecols=["STATION"]).drop_duplicates()
    stations_in_csv = stations_in_csv.rename(columns={"STATION": "station_p_id"})

    metadata = pd.read_csv(
        STATION_METADATA_PATH,
        usecols=["STATION", "NAME", "LATITUDE", "LONGITUDE"],
    ).drop_duplicates()
    metadata = metadata.rename(
        columns={
            "STATION": "station_p_id",
            "NAME": "station_name",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )

    stations = stations_in_csv.merge(metadata, on="station_p_id", how="left", validate="one_to_one")
    missing = stations.loc[stations[["latitude", "longitude"]].isna().any(axis=1), "station_p_id"].tolist()
    if missing:
        raise ValueError(f"Station metadata missing for: {missing}")

    station_points = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    )
    return station_points


def load_precipitation_records() -> pd.DataFrame:
    records = pd.read_csv(
        PRECIP_PATH,
        usecols=["STATION", "DATE", "REPORT_TYPE", "SOURCE", "HourlyPrecipitation"],
        low_memory=False,
    )
    records = records.rename(
        columns={
            "STATION": "station_p_id",
            "DATE": "timestamp",
            "REPORT_TYPE": "report_type",
            "SOURCE": "source",
            "HourlyPrecipitation": "depth_raw",
        }
    )
    records["timestamp"] = pd.to_datetime(records["timestamp"], errors="coerce")

    raw = records["depth_raw"].astype("string").str.strip()
    trace_mask = raw.str.upper().isin({"T", "TRACE"})
    raw = raw.str.replace(r"[^0-9eE+\\-.]", "", regex=True)
    records["depth_raw"] = pd.to_numeric(raw, errors="coerce")
    records.loc[trace_mask, "depth_raw"] = 0.0

    factor = unit_factor_to_mm(PRECIP_INPUT_UNIT)
    records["depth"] = records["depth_raw"] * factor

    records = records.dropna(subset=["timestamp"]).copy()
    records = (
        records.groupby(["station_p_id", "timestamp"], as_index=False)
        .agg(
            depth=("depth", lambda s: s.dropna().max() if s.notna().any() else pd.NA),
            report_type=("report_type", lambda s: "|".join(sorted({str(v) for v in s.dropna() if str(v)}))),
            source=("source", "first"),
        )
        .sort_values(["station_p_id", "timestamp"])
        .reset_index(drop=True)
    )
    records["depth"] = pd.to_numeric(records["depth"], errors="coerce")
    records = conditionally_impute_precipitation(records)
    records = records.loc[records["depth"].notna()].copy()
    records = records.loc[records["depth"] >= 0].copy()
    return records


def infer_station_intervals(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station_p_id, group in records.groupby("station_p_id", sort=True):
        timestamps = group["timestamp"].sort_values().drop_duplicates()
        deltas = timestamps.diff().dropna()
        deltas = deltas[deltas > pd.Timedelta(0)]
        if deltas.empty:
            interval_minutes = 60
        else:
            minutes = (deltas.dt.total_seconds() / 60).round().astype(int)
            interval_minutes = int(minutes.mode().iloc[0]) if not minutes.mode().empty else int(round(minutes.median()))
            interval_minutes = max(interval_minutes, 1)
        rows.append(
            {
                "station_p_id": station_p_id,
                "interval_minutes": interval_minutes,
                "interval_hours": interval_minutes / 60.0,
            }
        )
    return pd.DataFrame(rows)


def conditionally_impute_precipitation(records: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for station_p_id, group in records.groupby("station_p_id", sort=True):
        group = group.sort_values("timestamp").copy()
        group["depth_observed"] = group["depth"]
        group["fill_method"] = pd.NA

        prev_value = group["depth_observed"].ffill()
        next_value = group["depth_observed"].bfill()
        interpolated = group.set_index("timestamp")["depth_observed"].interpolate(method="time", limit_area="inside")

        group["depth"] = group["depth_observed"]
        gap_mask = group["depth_observed"].isna() & prev_value.notna() & next_value.notna()
        zero_guard_mask = gap_mask & ((prev_value <= 0) | (next_value <= 0))
        positive_interp_mask = gap_mask & (prev_value > 0) & (next_value > 0)

        group.loc[zero_guard_mask, "depth"] = 0.0
        group.loc[zero_guard_mask, "fill_method"] = "zero_guard"
        group.loc[positive_interp_mask, "depth"] = interpolated.loc[group.loc[positive_interp_mask, "timestamp"]].to_numpy()
        group.loc[positive_interp_mask, "fill_method"] = "linear_positive"
        group["was_interpolated"] = group["fill_method"].notna()
        frames.append(group)

    return pd.concat(frames, ignore_index=True) if frames else records.copy()


def build_rain_events(records: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    interval_lookup = intervals.set_index("station_p_id")["interval_minutes"].to_dict()
    wet_records = records.loc[records["depth"] > 0].copy()
    event_frames = []

    for station_p_id, group in wet_records.groupby("station_p_id", sort=True):
        group = group.sort_values("timestamp").copy()
        interval_minutes = int(interval_lookup.get(station_p_id, 60))
        interval_delta = pd.Timedelta(minutes=interval_minutes)
        previous_end = group["timestamp"].shift(1) + interval_delta
        dry_gap = group["timestamp"] - previous_end
        group["event_group"] = dry_gap.isna() | (dry_gap >= pd.Timedelta(hours=EVENT_DRY_GAP_HOURS))
        group["event_group"] = group["event_group"].cumsum()

        station_events = (
            group.groupby("event_group", as_index=False)
            .agg(
                start=("timestamp", "min"),
                last_observation=("timestamp", "max"),
                depth=("depth", "sum"),
            )
            .copy()
        )
        station_events["end"] = station_events["last_observation"] + interval_delta
        station_events["duration"] = (station_events["end"] - station_events["start"]).dt.total_seconds() / 3600.0
        station_events["duration"] = station_events["duration"].clip(lower=interval_minutes / 60.0)
        station_events["intensity"] = station_events["depth"] / station_events["duration"]
        station_events["station_p_id"] = station_p_id
        station_events = station_events.loc[station_events["depth"] >= MIN_EVENT_DEPTH_MM].copy()
        event_frames.append(station_events[["station_p_id", "start", "end", "duration", "intensity", "depth"]])

    if not event_frames:
        return pd.DataFrame(columns=["event_id", "station_p_id", "start", "end", "duration", "intensity", "depth"])

    events = pd.concat(event_frames, ignore_index=True)
    events = events.sort_values(["station_p_id", "start"]).reset_index(drop=True)
    events.insert(0, "event_id", [f"P_EVENT_{i:06d}" for i in range(1, len(events) + 1)])
    return events


def build_thiessen_polygons(station_points: gpd.GeoDataFrame, nyc_boundary) -> gpd.GeoDataFrame:
    station_points_local = station_points.to_crs(LOCAL_CRS)
    source_points = MultiPoint(list(station_points_local.geometry))
    voronoi = voronoi_diagram(
        source_points,
        envelope=nyc_boundary.envelope.buffer(VORONOI_BUFFER_FEET),
        edges=False,
    )
    cells = gpd.GeoDataFrame(geometry=list(voronoi.geoms), crs=LOCAL_CRS)
    cells["geometry"] = cells.geometry.map(make_valid)
    cells["geometry"] = cells.geometry.intersection(nyc_boundary)
    cells = cells.loc[~cells.geometry.is_empty].copy()

    label_points = gpd.GeoDataFrame(
        {"cell_id": range(len(cells))},
        geometry=cells.representative_point(),
        crs=LOCAL_CRS,
    )
    assigned = gpd.sjoin_nearest(
        label_points,
        station_points_local[["station_p_id", "station_name", "latitude", "longitude", "geometry"]],
        how="left",
        distance_col="distance_to_station",
    )

    thiessen = cells.reset_index(drop=True).join(
        assigned[["station_p_id", "station_name", "latitude", "longitude", "distance_to_station"]]
    )
    thiessen = thiessen[["station_p_id", "station_name", "latitude", "longitude", "geometry"]].copy()
    thiessen = thiessen.drop_duplicates("station_p_id").reset_index(drop=True)
    return thiessen


def build_segment_crosswalk(thiessen: gpd.GeoDataFrame) -> pd.DataFrame:
    streets = gpd.read_file(STREETS_PATH, columns=["SegmentID", "geometry"]).to_crs(LOCAL_CRS)
    street_points = gpd.GeoDataFrame(
        {"segment_id": streets["SegmentID"].astype("Int64")},
        geometry=streets.representative_point(),
        crs=LOCAL_CRS,
    )

    direct = gpd.sjoin(
        street_points,
        thiessen[["station_p_id", "geometry"]],
        how="left",
        predicate="within",
    )
    direct = direct[["segment_id", "station_p_id", "geometry"]].copy()
    direct = direct.sort_values(["segment_id", "station_p_id"], kind="stable").drop_duplicates("segment_id")

    missing_mask = direct["station_p_id"].isna()
    if missing_mask.any():
        missing = direct.loc[missing_mask, ["segment_id", "geometry"]].copy()
        nearest = gpd.sjoin_nearest(
            missing,
            thiessen[["station_p_id", "geometry"]],
            how="left",
            distance_col="distance_to_polygon",
        )
        nearest = nearest[["segment_id", "station_p_id"]].drop_duplicates("segment_id")
        nearest_lookup = nearest.set_index("segment_id")["station_p_id"]
        direct.loc[missing_mask, "station_p_id"] = direct.loc[missing_mask, "segment_id"].map(nearest_lookup)

    direct = direct.drop(columns=["geometry"])

    crosswalk = direct[["segment_id", "station_p_id"]].dropna().copy()
    crosswalk["segment_id"] = crosswalk["segment_id"].astype("Int64")
    crosswalk = crosswalk.sort_values("segment_id", kind="stable").drop_duplicates("segment_id").reset_index(drop=True)
    return crosswalk
"""
        )
    )

    cells.append(
        code_cell(
            """nyc_boroughs, nyc_boundary = load_nyc_boundary()
station_points = load_station_points()
records = load_precipitation_records()
intervals = infer_station_intervals(records)

station_points.to_file(STATION_POINTS_OUTPUT, driver="GeoJSON")
records.to_csv(RECORDS_OUTPUT, index=False)

interpolation_summary = (
    records.groupby("station_p_id", as_index=False)
    .agg(
        n_rows=("station_p_id", "size"),
        n_missing_after_fill=("depth", lambda s: int(s.isna().sum())),
        n_interpolated=("was_interpolated", "sum"),
        n_zero_guard=("fill_method", lambda s: int((s == "zero_guard").sum())),
        n_linear_positive=("fill_method", lambda s: int((s == "linear_positive").sum())),
    )
    .sort_values("station_p_id")
)

print(f"Stations found in precipitation.csv: {len(station_points):,}")
display(station_points.drop(columns="geometry"))

print(f"Hourly precipitation records: {len(records):,}")
display(intervals)
display(interpolation_summary)
"""
        )
    )

    cells.append(
        code_cell(
            """events = build_rain_events(records, intervals)
events.to_csv(EVENTS_OUTPUT, index=False)

station_event_summary = (
    events.groupby("station_p_id", as_index=False)
    .agg(
        n_events=("event_id", "size"),
        mean_duration_hours=("duration", "mean"),
        mean_depth_mm=("depth", "mean"),
        mean_intensity_mm_per_hr=("intensity", "mean"),
    )
    .sort_values("n_events", ascending=False)
)

print(f"Rain events: {len(events):,}")
display(station_event_summary)
display(events.head(10))
"""
        )
    )

    cells.append(
        code_cell(
            """thiessen = build_thiessen_polygons(station_points, nyc_boundary)
thiessen.to_crs("EPSG:4326").to_file(THIESSEN_OUTPUT, driver="GeoJSON")

crosswalk = build_segment_crosswalk(thiessen)
crosswalk.to_csv(CROSSWALK_OUTPUT, index=False)

print(f"Thiessen polygons: {len(thiessen):,}")
print(f"Street crosswalk rows: {len(crosswalk):,}")
display(thiessen.drop(columns="geometry"))
display(crosswalk.head(10))
"""
        )
    )

    cells.append(
        code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)

nyc_boroughs.boundary.plot(ax=axes[0], color="black", linewidth=0.7)
thiessen.boundary.plot(ax=axes[0], color="steelblue", linewidth=1.2)
thiessen.plot(ax=axes[0], column="station_p_id", alpha=0.18, categorical=True, legend=True)
station_points.to_crs(LOCAL_CRS).plot(ax=axes[0], color="crimson", markersize=50)
for _, row in station_points.to_crs(LOCAL_CRS).iterrows():
    axes[0].annotate(row["station_p_id"], (row.geometry.x, row.geometry.y), xytext=(4, 4), textcoords="offset points", fontsize=8)
axes[0].set_title("Thiessen polygons and station points")
axes[0].set_axis_off()

events["duration"].plot(kind="hist", bins=40, ax=axes[1], color="slateblue", edgecolor="white")
axes[1].set_title("Rain event duration distribution")
axes[1].set_xlabel("Duration (hours)")
axes[1].set_ylabel("Event count")

plt.show()

print("Outputs")
print(f"- station points: {STATION_POINTS_OUTPUT}")
print(f"- hourly records: {RECORDS_OUTPUT}")
print(f"- rain events: {EVENTS_OUTPUT}")
print(f"- thiessen polygons: {THIESSEN_OUTPUT}")
print(f"- street crosswalk: {CROSSWALK_OUTPUT}")
"""
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
