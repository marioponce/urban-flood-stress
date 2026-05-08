from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


LOCAL_TIMEZONE = "America/New_York"
FT_TO_M = 0.3048

ROOT = Path(__file__).resolve().parents[2]
MODELING_DIR = ROOT / "data" / "processed" / "modeling"
DIAGNOSTICS_DIR = MODELING_DIR / "diagnostics"

FLOOD_EVENTS_PATH = ROOT / "data" / "processed" / "311" / "flood_events.csv"
INDIVIDUAL_COMPLAINTS_PATH = ROOT / "data" / "processed" / "311" / "individual_complaints.csv"

STREET_METRICS_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_metrics.gpkg"
STREET_ELEVATION_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_metrics_elevation.gpkg"
STREET_CONNECTIVITY_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_connectivity.gpkg"
STREET_CONNECTIVITY_NODES_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_connectivity_nodes.csv"
STREET_CONNECTIVITY_COMPONENTS_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_connectivity_components.csv"
STREET_SHORE_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_metrics_shore.gpkg"

PRECIP_EVENTS_PATH = ROOT / "data" / "processed" / "precipitation" / "events_hourly" / "precipitation_events.csv"
PRECIP_CROSSWALK_PATH = ROOT / "data" / "spatial" / "vector" / "noaa" / "precipitation_segment_station_crosswalk.csv"
PRECIP_THIESSEN_PATH = ROOT / "data" / "spatial" / "vector" / "noaa" / "precipitation_thiessen.geojson"

TIDE_SERIES_PATH = ROOT / "data" / "temporal" / "noaa" / "tide_series.parquet"
TIDE_CROSSWALK_PATH = ROOT / "data" / "spatial" / "vector" / "noaa" / "tide_thiessen_primary_street_crosswalk.csv"

FEMA_PATH = ROOT / "data" / "spatial" / "vector" / "fema_nfhl" / "nyc_nfhl_flood_zones.geojson"

CENSUS_DIR = ROOT / "data" / "processed" / "census"
CENSUS_METADATA_PATH = CENSUS_DIR / "census_period_metadata.csv"

INFRASTRUCTURE_SUMMARY_PATH = ROOT / "data" / "processed" / "infrastructure" / "infrastructure_download_summary.csv"
INFRASTRUCTURE_CATCH_BASINS_PATH = ROOT / "data" / "processed" / "infrastructure" / "catch_basins.gpkg"
INFRASTRUCTURE_OUTFALLS_PATH = ROOT / "data" / "processed" / "infrastructure" / "outfalls.gpkg"
INFRASTRUCTURE_GREEN_PATH = ROOT / "data" / "processed" / "infrastructure" / "green_infrastructure.gpkg"

MASTER_PARQUET_PATH = MODELING_DIR / "flood_events_master_table.parquet"
MASTER_GEOPARQUET_PATH = MODELING_DIR / "flood_events_master_table.geoparquet"


def ensure_proj_env() -> None:
    """Set PROJ_LIB when the conda environment path is known."""
    if os.environ.get("PROJ_LIB"):
        return

    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "proj")
    candidates.append(Path.home() / "miniconda3" / "envs" / "urban-flood-stress" / "share" / "proj")

    for candidate in candidates:
        if candidate.exists():
            os.environ["PROJ_LIB"] = str(candidate)
            return


def normalize_segment_id_value(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA

    if re.fullmatch(r"\d+(\.0+)?", text):
        digits = str(int(float(text)))
    else:
        digits = re.sub(r"\D", "", text)

    if not digits:
        return pd.NA

    digits = digits.lstrip("0") or "0"
    return digits


def normalize_segment_id_series(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_segment_id_value)
    return pd.Series(normalized, index=series.index, dtype="string")


def normalize_string_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    return cleaned


def normalize_borough_value(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    mapping = {
        "1": "MANHATTAN",
        "1.0": "MANHATTAN",
        "2": "BRONX",
        "2.0": "BRONX",
        "3": "BROOKLYN",
        "3.0": "BROOKLYN",
        "4": "QUEENS",
        "4.0": "QUEENS",
        "5": "STATEN ISLAND",
        "5.0": "STATEN ISLAND",
        "MN": "MANHATTAN",
        "MAN": "MANHATTAN",
        "BX": "BRONX",
        "BK": "BROOKLYN",
        "K": "BROOKLYN",
        "QN": "QUEENS",
        "Q": "QUEENS",
        "SI": "STATEN ISLAND",
        "R": "STATEN ISLAND",
    }
    return mapping.get(text, text)


def normalize_borough_series(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(normalize_borough_value), index=series.index, dtype="string")


def to_local_naive(series: pd.Series, assume_utc_when_naive: bool) -> pd.Series:
    text = series.astype("string")
    has_explicit_timezone = text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).any()
    if assume_utc_when_naive or has_explicit_timezone:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)

    parsed = pd.to_datetime(series, errors="coerce")
    tz = getattr(parsed.dt, "tz", None)
    if tz is not None:
        return parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    return parsed


def choose_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def read_segment_gdf(path: Path, columns: list[str] | None = None) -> gpd.GeoDataFrame:
    ensure_proj_env()
    gdf = gpd.read_file(path)
    segment_col = choose_existing_column(gdf, ["segment_id", "SegmentID", "SEGMENTID", "street_id"])
    if segment_col is None:
        raise KeyError(f"No segment identifier found in {path}")

    gdf["segment_id"] = normalize_segment_id_series(gdf[segment_col])
    gdf = gdf[gdf["segment_id"].notna()].copy()
    gdf = gdf.sort_values(["segment_id"], kind="stable").drop_duplicates("segment_id").reset_index(drop=True)

    if columns is not None:
        keep = ["segment_id", "geometry"] + [column for column in columns if column in gdf.columns and column not in {"segment_id", "geometry"}]
        gdf = gdf[keep].copy()
    return gdf


def load_flood_events() -> pd.DataFrame:
    events = pd.read_csv(FLOOD_EVENTS_PATH)
    events["segment_id"] = normalize_segment_id_series(events["segment_id"])
    events["event_start_local"] = to_local_naive(events["start"], assume_utc_when_naive=True)
    events["event_end_local"] = to_local_naive(events["end"], assume_utc_when_naive=True)
    events["event_window_end_local"] = events["event_end_local"]
    missing_end = events["event_window_end_local"].isna()
    events.loc[missing_end, "event_window_end_local"] = events.loc[missing_end, "event_start_local"] + pd.Timedelta(hours=24)

    events["event_window_duration_hours"] = (
        events["event_window_end_local"] - events["event_start_local"]
    ).dt.total_seconds() / 3600.0
    events["event_end_observed_local"] = pd.NaT
    closed_mask = events["status"].eq(1) & events["event_end_local"].notna()
    events.loc[closed_mask, "event_end_observed_local"] = events.loc[closed_mask, "event_end_local"]
    events["resolution_hours"] = (
        events["event_end_observed_local"] - events["event_start_local"]
    ).dt.total_seconds() / 3600.0
    events["resolution_bool"] = events["event_end_observed_local"].notna()
    events["event_end_inferred"] = events["event_end_local"].notna() & ~events["resolution_bool"]
    events["occurrence"] = True
    events["intensity"] = pd.to_numeric(events["n_complaints"], errors="coerce").fillna(0).astype("Int64")
    events["event_year"] = events["event_start_local"].dt.year.astype("Int64")
    events["event_month"] = events["event_start_local"].dt.month.astype("Int64")
    events["event_date"] = events["event_start_local"].dt.date.astype("string")
    events["flood_event_status"] = pd.to_numeric(events["status"], errors="coerce").astype("Int64")
    events["start"] = events["event_start_local"]
    events["end"] = events["event_end_observed_local"]
    events["resolution"] = events["resolution_hours"]
    return events


def load_individual_complaints() -> pd.DataFrame:
    complaints = pd.read_csv(INDIVIDUAL_COMPLAINTS_PATH)
    complaints["segment_id"] = normalize_segment_id_series(complaints["segment_id"])
    complaints["complaint_start_local"] = to_local_naive(complaints["start"], assume_utc_when_naive=False)
    complaints["match_method"] = normalize_string_series(complaints["match_method"])
    complaints["borough"] = normalize_borough_series(complaints["borough"])
    complaints["source_complaint_id"] = complaints["source_complaint_id"].astype("string")
    return complaints


def derive_segment_borough(frame: pd.DataFrame) -> pd.DataFrame:
    left = normalize_borough_series(frame.get("LBoro", pd.Series(pd.NA, index=frame.index)))
    right = normalize_borough_series(frame.get("RBoro", pd.Series(pd.NA, index=frame.index)))
    borough = left.where(left.notna(), right)
    both = left.notna() & right.notna() & left.ne(right)
    borough = borough.astype("string")
    borough.loc[both] = left.loc[both]
    frame = frame.copy()
    frame["segment_borough_left"] = left
    frame["segment_borough_right"] = right
    frame["segment_borough"] = borough
    return frame


def load_street_features() -> gpd.GeoDataFrame:
    columns = [
        "Street",
        "SAFStreetName",
        "FeatureTyp",
        "SegmentTyp",
        "TrafDir",
        "LBoro",
        "RBoro",
        "POSTED_SPEED",
        "StreetWidth_Min",
        "StreetWidth_Max",
        "Number_Travel_Lanes",
        "Number_Park_Lanes",
        "Number_Total_Lanes",
        "BikeLane",
        "edge_id",
        "road_class",
        "speed_heur",
        "speed",
        "length",
        "travel_time",
        "u",
        "v",
        "edge_betweenness",
    ]
    streets = read_segment_gdf(STREET_METRICS_PATH, columns=columns)
    streets = derive_segment_borough(streets)
    streets = streets.rename(
        columns={
            "Street": "street_name",
            "SAFStreetName": "street_name_safe",
            "FeatureTyp": "street_feature_type",
            "SegmentTyp": "street_segment_type",
            "TrafDir": "street_traffic_direction",
            "POSTED_SPEED": "posted_speed",
            "StreetWidth_Min": "street_width_min",
            "StreetWidth_Max": "street_width_max",
            "Number_Travel_Lanes": "travel_lanes",
            "Number_Park_Lanes": "park_lanes",
            "Number_Total_Lanes": "total_lanes",
            "BikeLane": "bike_lane",
            "length": "segment_length_ft",
            "travel_time": "segment_travel_time_s",
            "speed": "segment_speed_mph",
            "speed_heur": "segment_speed_heuristic_mph",
            "edge_betweenness": "segment_edge_betweenness",
        }
    )
    streets["segment_length_geom_ft"] = streets.geometry.length
    return streets


def load_elevation_features() -> pd.DataFrame:
    elevation = read_segment_gdf(
        STREET_ELEVATION_PATH,
        columns=["dem_min", "dem_max", "dem_mean", "dem_sd", "dem_start", "dem_end", "dem_slope", "elevation_processed"],
    )
    return pd.DataFrame(elevation.drop(columns="geometry"))


def load_connectivity_features() -> pd.DataFrame:
    connectivity = read_segment_gdf(
        STREET_CONNECTIVITY_PATH,
        columns=["component_id", "component_size", "in_largest_component", "is_bridge_segment", "connectivity_processed"],
    )
    return pd.DataFrame(connectivity.drop(columns="geometry"))


def load_node_features() -> pd.DataFrame:
    streets = load_street_features()[["segment_id", "u", "v"]].copy()
    nodes = pd.read_csv(STREET_CONNECTIVITY_NODES_PATH)
    streets["u"] = pd.to_numeric(streets["u"], errors="coerce").astype("Int64")
    streets["v"] = pd.to_numeric(streets["v"], errors="coerce").astype("Int64")
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")

    from_nodes = nodes.rename(
        columns={
            "node_id": "u",
            "in_degree": "node_in_degree_u",
            "out_degree": "node_out_degree_u",
            "total_degree": "node_total_degree_u",
            "node_betweenness": "node_betweenness_u",
            "component_id": "node_component_id_u",
            "component_size": "node_component_size_u",
            "in_largest_component": "node_in_largest_component_u",
            "is_articulation_node": "node_is_articulation_u",
        }
    )
    to_nodes = nodes.rename(
        columns={
            "node_id": "v",
            "in_degree": "node_in_degree_v",
            "out_degree": "node_out_degree_v",
            "total_degree": "node_total_degree_v",
            "node_betweenness": "node_betweenness_v",
            "component_id": "node_component_id_v",
            "component_size": "node_component_size_v",
            "in_largest_component": "node_in_largest_component_v",
            "is_articulation_node": "node_is_articulation_v",
        }
    )

    enriched = streets.merge(from_nodes, on="u", how="left", validate="many_to_one")
    enriched = enriched.merge(to_nodes, on="v", how="left", validate="many_to_one")

    for suffix in ["in_degree", "out_degree", "total_degree", "betweenness"]:
        left = f"node_{suffix}_u"
        right = f"node_{suffix}_v"
        if left in enriched.columns and right in enriched.columns:
            enriched[f"node_{suffix}_mean"] = enriched[[left, right]].mean(axis=1)
            enriched[f"node_{suffix}_max"] = enriched[[left, right]].max(axis=1)

    enriched["node_articulation_any"] = enriched[["node_is_articulation_u", "node_is_articulation_v"]].fillna(False).any(axis=1)
    return enriched.drop(columns=["u", "v"])


def load_shore_features() -> pd.DataFrame:
    shore = read_segment_gdf(
        STREET_SHORE_PATH,
        columns=["shore_id", "shore_graph_steps", "shore_processed", "shore_seed_edge"],
    )
    shore["shore_id"] = normalize_string_series(shore["shore_id"])
    return pd.DataFrame(shore.drop(columns="geometry"))


def load_precip_crosswalk() -> pd.DataFrame:
    crosswalk = pd.read_csv(PRECIP_CROSSWALK_PATH)
    crosswalk["segment_id"] = normalize_segment_id_series(crosswalk["segment_id"])
    crosswalk["station_p_id"] = normalize_string_series(crosswalk["station_p_id"]).str.upper()
    crosswalk = crosswalk.dropna(subset=["segment_id", "station_p_id"])
    crosswalk = crosswalk.sort_values(["segment_id", "station_p_id"], kind="stable").drop_duplicates("segment_id")
    return crosswalk.reset_index(drop=True)


def load_tide_crosswalk() -> pd.DataFrame:
    crosswalk = pd.read_csv(TIDE_CROSSWALK_PATH)
    segment_col = choose_existing_column(crosswalk, ["segment_id", "street_id", "SegmentID"])
    tide_col = choose_existing_column(crosswalk, ["tide_id", "station_id", "shore_id"])
    if segment_col is None or tide_col is None:
        raise KeyError("Tide crosswalk must contain segment and tide identifiers.")
    crosswalk["segment_id"] = normalize_segment_id_series(crosswalk[segment_col])
    crosswalk["tide_id"] = normalize_string_series(crosswalk[tide_col])
    crosswalk = crosswalk.dropna(subset=["segment_id", "tide_id"])
    crosswalk = crosswalk.sort_values(["segment_id", "tide_id"], kind="stable").drop_duplicates("segment_id")
    return crosswalk[["segment_id", "tide_id"]].reset_index(drop=True)


def load_precip_events() -> pd.DataFrame:
    events = pd.read_csv(PRECIP_EVENTS_PATH)
    events["station_p_id"] = normalize_string_series(events["station_p_id"]).str.upper()
    events["start"] = to_local_naive(events["start"], assume_utc_when_naive=False)
    events["end"] = to_local_naive(events["end"], assume_utc_when_naive=False)
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    events["intensity"] = pd.to_numeric(events["intensity"], errors="coerce")
    events["depth"] = pd.to_numeric(events["depth"], errors="coerce")
    return events


def load_tide_series() -> pd.DataFrame:
    tide = pd.read_parquet(TIDE_SERIES_PATH)
    tide["tide_id"] = normalize_string_series(tide["station_id"])
    tide["timestamp_local"] = to_local_naive(tide["timestamp"], assume_utc_when_naive=True)

    if "tide_level_m" in tide.columns:
        tide["tide_level_m"] = pd.to_numeric(tide["tide_level_m"], errors="coerce")
    elif "tide_level_ft" in tide.columns:
        tide["tide_level_m"] = pd.to_numeric(tide["tide_level_ft"], errors="coerce") * FT_TO_M
    elif "raw_value_m" in tide.columns:
        tide["tide_level_m"] = pd.to_numeric(tide["raw_value_m"], errors="coerce")
    elif "raw_value_ft" in tide.columns:
        tide["tide_level_m"] = pd.to_numeric(tide["raw_value_ft"], errors="coerce") * FT_TO_M
    else:
        raise KeyError("Tide series must expose a tide level column.")

    return tide[["tide_id", "timestamp_local", "tide_level_m"]].dropna(subset=["tide_id", "timestamp_local"]).copy()


def summarize_segment_static_features() -> gpd.GeoDataFrame:
    streets = load_street_features()
    elevation = load_elevation_features()
    connectivity = load_connectivity_features()
    node_features = load_node_features()
    shore = load_shore_features()
    precip_crosswalk = load_precip_crosswalk()
    tide_crosswalk = load_tide_crosswalk()

    streets = streets.merge(elevation, on="segment_id", how="left", validate="one_to_one")
    streets = streets.merge(connectivity, on="segment_id", how="left", validate="one_to_one")
    streets = streets.merge(node_features, on="segment_id", how="left", validate="one_to_one")
    streets = streets.merge(shore, on="segment_id", how="left", validate="one_to_one")
    streets = streets.merge(precip_crosswalk, on="segment_id", how="left", validate="one_to_one")
    streets = streets.merge(tide_crosswalk, on="segment_id", how="left", validate="one_to_one")
    return streets


def map_period_end_year(acs_period: str) -> int:
    return int(str(acs_period).split("_")[-1])


def load_census_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(CENSUS_METADATA_PATH)
    metadata["acs_period"] = metadata["acs_period"].astype("string")
    metadata["period_end_year"] = metadata["acs_period"].map(map_period_end_year)
    metadata["output_path"] = metadata["output_path"].map(Path)
    metadata = metadata.sort_values(["period_end_year", "acs_year"], kind="stable").reset_index(drop=True)
    return metadata


def select_census_period_for_year(event_year: int, metadata: pd.DataFrame) -> str:
    eligible = metadata.loc[metadata["period_end_year"] <= event_year]
    if eligible.empty:
        return str(metadata.iloc[0]["acs_period"])
    return str(eligible.iloc[-1]["acs_period"])


def compute_census_features(events: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_census_metadata()
    events = events.copy()
    events["census_period_selected"] = events["event_year"].astype(int).map(lambda year: select_census_period_for_year(year, metadata))

    feature_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    for period, subset in events.groupby("census_period_selected", sort=True):
        period_meta = metadata.loc[metadata["acs_period"] == period].iloc[0]
        layer = gpd.read_file(Path(period_meta["output_path"]))
        if layer.crs != subset.crs:
            layer = layer.to_crs(subset.crs)

        if "median_household_income" not in layer.columns:
            if "B19013_001E" in layer.columns:
                layer["median_household_income"] = pd.to_numeric(layer["B19013_001E"], errors="coerce")
            elif "DP03_0062E" in layer.columns:
                layer["median_household_income"] = pd.to_numeric(layer["DP03_0062E"], errors="coerce")
            else:
                layer["median_household_income"] = pd.NA

        keep = [
            "geometry",
            "GEOID",
            "borough",
            "poverty_rate",
            "renter_share",
            "no_vehicle_share",
            "median_household_income",
            "acs_period",
            "acs_year",
            "tiger_year",
            "census_geography_base",
        ]
        available = [column for column in keep if column in layer.columns]
        layer = layer[available].copy()

        reps = subset[["event_id", "geometry"]].copy()
        reps.geometry = reps.representative_point()
        joined = gpd.sjoin(reps, layer, how="left", predicate="within")
        duplicate_matches = int(joined["event_id"].duplicated().sum())
        joined = joined.sort_values(["event_id"], kind="stable").drop_duplicates("event_id")

        missing_mask = joined["GEOID"].isna() if "GEOID" in joined.columns else pd.Series(False, index=joined.index)
        if missing_mask.any():
            missing_points = reps[reps["event_id"].isin(joined.loc[missing_mask, "event_id"])].copy()
            nearest = gpd.sjoin_nearest(
                missing_points,
                layer,
                how="left",
                distance_col="census_nearest_distance_ft",
            )
            nearest = nearest.sort_values(["event_id", "census_nearest_distance_ft"], kind="stable").drop_duplicates("event_id")
            nearest = nearest.set_index("event_id")
            joined = joined.set_index("event_id")
            for column in [col for col in layer.columns if col != "geometry"]:
                if column in joined.columns and column in nearest.columns:
                    fill_mask = joined[column].isna()
                    if fill_mask.any():
                        joined.loc[fill_mask, column] = joined.loc[fill_mask].index.map(nearest[column])
            joined = joined.reset_index()

        renamed = joined.rename(
            columns={
                "GEOID": "census_geoid",
                "borough": "census_borough",
                "poverty_rate": "census_poverty_rate",
                "renter_share": "census_renter_share",
                "no_vehicle_share": "census_no_vehicle_share",
                "median_household_income": "census_median_household_income",
                "acs_period": "census_acs_period",
                "acs_year": "census_acs_year",
                "tiger_year": "census_tiger_year",
                "census_geography_base": "census_geography_base",
            }
        )
        feature_frames.append(pd.DataFrame(renamed.drop(columns=["geometry", "index_right"], errors="ignore")))
        diagnostic_rows.append(
            {
                "acs_period": period,
                "event_rows": int(len(subset)),
                "joined_rows": int(len(joined)),
                "null_census_geoid": int(renamed["census_geoid"].isna().sum()),
                "duplicate_point_matches": duplicate_matches,
            }
        )

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame(columns=["event_id"])
    diagnostics = pd.DataFrame(diagnostic_rows)
    return features, diagnostics


def compute_fema_features(segment_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    zones = gpd.read_file(FEMA_PATH)
    if zones.crs != segment_gdf.crs:
        zones = zones.to_crs(segment_gdf.crs)

    left = segment_gdf[["segment_id", "geometry"]].copy()
    left["segment_length_ft"] = left.geometry.length

    intersections = gpd.overlay(left, zones[["FLD_ZONE", "ZONE_SUBTY", "geometry"]], how="intersection", keep_geom_type=False)
    if intersections.empty:
        empty = pd.DataFrame(
            columns=[
                "segment_id",
                "fema_fld_zone",
                "fema_zone_subty",
                "fema_overlap_ft",
                "fema_overlap_share",
                "fema_zone_count",
                "fema_sfha_any",
            ]
        )
        return empty, pd.DataFrame([{"segments_with_intersections": 0, "intersection_rows": 0}])

    intersections["fema_overlap_ft"] = intersections.geometry.length
    intersections["fema_overlap_share"] = (
        intersections["fema_overlap_ft"] / intersections["segment_length_ft"].replace(0, np.nan)
    )
    intersections["fema_sfha_flag"] = (
        intersections["FLD_ZONE"].astype("string").str.upper().str.startswith(("A", "V"))
    )

    dominant = intersections.sort_values(
        ["segment_id", "fema_overlap_ft", "FLD_ZONE"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates("segment_id")

    counts = intersections.groupby("segment_id", as_index=False).agg(
        fema_zone_count=("FLD_ZONE", "size"),
        fema_sfha_any=("fema_sfha_flag", "max"),
    )

    features = dominant[["segment_id", "FLD_ZONE", "ZONE_SUBTY", "fema_overlap_ft", "fema_overlap_share"]].rename(
        columns={"FLD_ZONE": "fema_fld_zone", "ZONE_SUBTY": "fema_zone_subty"}
    )
    features = features.merge(counts, on="segment_id", how="left", validate="one_to_one")
    features = segment_gdf[["segment_id"]].merge(features, on="segment_id", how="left", validate="one_to_one")
    features["fema_fld_zone"] = features["fema_fld_zone"].fillna("OUTSIDE_NFHL")
    features["fema_zone_subty"] = features["fema_zone_subty"].fillna("NO_INTERSECTION")
    features["fema_overlap_ft"] = features["fema_overlap_ft"].fillna(0.0)
    features["fema_overlap_share"] = features["fema_overlap_share"].fillna(0.0)
    features["fema_zone_count"] = features["fema_zone_count"].fillna(0).astype("Int64")
    features["fema_sfha_any"] = features["fema_sfha_any"].fillna(False).astype(bool)

    diagnostics = pd.DataFrame(
        [
            {
                "segments_with_intersections": int(dominant["segment_id"].nunique()),
                "intersection_rows": int(len(intersections)),
                "sfha_segments": int(features["fema_sfha_any"].sum()),
                "segments_outside_nfhl": int(features["fema_fld_zone"].eq("OUTSIDE_NFHL").sum()),
            }
        ]
    )
    return features, diagnostics


def _count_points_within_buffers(
    segment_gdf: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    radii_ft: Iterable[int],
    prefix: str,
) -> pd.DataFrame:
    counts = pd.DataFrame({"segment_id": segment_gdf["segment_id"]})
    base = segment_gdf[["segment_id", "geometry"]].copy()

    for radius in radii_ft:
        buffered = base.copy()
        buffered.geometry = buffered.geometry.buffer(radius)
        joined = gpd.sjoin(buffered, points[["geometry"]], how="left", predicate="contains")
        summary = joined.groupby("segment_id")["index_right"].count().rename(f"{prefix}_count_{radius}ft")
        counts = counts.merge(summary, on="segment_id", how="left")
        counts[f"{prefix}_count_{radius}ft"] = counts[f"{prefix}_count_{radius}ft"].fillna(0).astype("Int64")

    return counts


def compute_infrastructure_features(segment_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics_rows: list[dict[str, object]] = []
    outputs = pd.DataFrame({"segment_id": segment_gdf["segment_id"]})

    if INFRASTRUCTURE_CATCH_BASINS_PATH.exists():
        catch = gpd.read_file(INFRASTRUCTURE_CATCH_BASINS_PATH)
        if catch.crs != segment_gdf.crs:
            catch = catch.to_crs(segment_gdf.crs)
        nearest = gpd.sjoin_nearest(
            segment_gdf[["segment_id", "geometry"]],
            catch[["geometry"]],
            how="left",
            distance_col="catch_basin_nearest_ft",
        )
        nearest = nearest[["segment_id", "catch_basin_nearest_ft"]].drop_duplicates("segment_id")
        counts = _count_points_within_buffers(segment_gdf, catch, radii_ft=[100, 250], prefix="catch_basin")
        outputs = outputs.merge(nearest, on="segment_id", how="left", validate="one_to_one")
        outputs = outputs.merge(counts, on="segment_id", how="left", validate="one_to_one")
        diagnostics_rows.append(
            {
                "layer": "catch_basins",
                "segment_rows": int(len(segment_gdf)),
                "features": int(len(catch)),
                "null_nearest_distance": int(outputs["catch_basin_nearest_ft"].isna().sum()),
            }
        )

    if INFRASTRUCTURE_OUTFALLS_PATH.exists():
        outfalls = gpd.read_file(INFRASTRUCTURE_OUTFALLS_PATH)
        if outfalls.crs != segment_gdf.crs:
            outfalls = outfalls.to_crs(segment_gdf.crs)
        nearest = gpd.sjoin_nearest(
            segment_gdf[["segment_id", "geometry"]],
            outfalls[["geometry"]],
            how="left",
            distance_col="outfall_nearest_ft",
        )
        nearest = nearest[["segment_id", "outfall_nearest_ft"]].drop_duplicates("segment_id")
        counts = _count_points_within_buffers(segment_gdf, outfalls, radii_ft=[250, 500], prefix="outfall")
        outputs = outputs.merge(nearest, on="segment_id", how="left", validate="one_to_one")
        outputs = outputs.merge(counts, on="segment_id", how="left", validate="one_to_one")
        diagnostics_rows.append(
            {
                "layer": "outfalls",
                "segment_rows": int(len(segment_gdf)),
                "features": int(len(outfalls)),
                "null_nearest_distance": int(outputs["outfall_nearest_ft"].isna().sum()),
            }
        )

    if INFRASTRUCTURE_GREEN_PATH.exists():
        diagnostics_rows.append({"layer": "green_infrastructure", "status": "available_but_not_used"})
    else:
        diagnostics_rows.append({"layer": "green_infrastructure", "status": "not_available"})

    return outputs, pd.DataFrame(diagnostics_rows)


def assign_complaints_to_events(events: pd.DataFrame, complaints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_lookup = events[["event_id", "segment_id", "event_start_local", "event_window_end_local"]].copy()
    complaint_assignments: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    events_by_segment = {segment_id: group.reset_index(drop=True) for segment_id, group in event_lookup.groupby("segment_id", sort=False)}
    complaints = complaints[complaints["segment_id"].notna() & complaints["complaint_start_local"].notna()].copy()

    for segment_id, group in complaints.groupby("segment_id", sort=False):
        segment_events = events_by_segment.get(segment_id)
        if segment_events is None or segment_events.empty:
            assigned = group.copy()
            assigned["event_id"] = pd.NA
            complaint_assignments.append(assigned)
            diagnostic_rows.append({"segment_id": segment_id, "complaints": int(len(group)), "matched": 0})
            continue

        assigned = group.copy()
        assigned["event_id"] = pd.NA
        starts = segment_events["event_start_local"].to_numpy()
        ends = segment_events["event_window_end_local"].to_numpy()

        for idx, timestamp in assigned["complaint_start_local"].items():
            mask = (starts <= timestamp) & (ends >= timestamp)
            if mask.any():
                event_id = segment_events.loc[np.flatnonzero(mask)[0], "event_id"]
                assigned.at[idx, "event_id"] = event_id

        complaint_assignments.append(assigned)
        diagnostic_rows.append(
            {
                "segment_id": segment_id,
                "complaints": int(len(group)),
                "matched": int(assigned["event_id"].notna().sum()),
            }
        )

    assigned = pd.concat(complaint_assignments, ignore_index=True) if complaint_assignments else pd.DataFrame(columns=list(complaints.columns) + ["event_id"])

    match_flags = assigned.get("match_method", pd.Series(pd.NA, index=assigned.index)).fillna("unknown")
    spatial_like = match_flags.astype("string").str.contains("spatial", case=False, na=False)
    textual_like = match_flags.astype("string").str.contains("text", case=False, na=False)

    event_summary = (
        assigned.dropna(subset=["event_id"])
        .groupby("event_id", as_index=False)
        .agg(
            complaint_count_check=("complaint_id", "size"),
            complaint_spatial_match_count=("complaint_id", lambda s: int(spatial_like.loc[s.index].sum())),
            complaint_textual_match_count=("complaint_id", lambda s: int(textual_like.loc[s.index].sum())),
            complaint_borough_mode=("borough", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else pd.NA),
        )
    )

    return event_summary, pd.DataFrame(diagnostic_rows)


def compute_precipitation_overlap_features(
    events: pd.DataFrame,
    crosswalk: pd.DataFrame,
    precip_events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = events[["event_id", "segment_id", "event_start_local", "event_window_end_local"]].merge(
        crosswalk[["segment_id", "station_p_id"]],
        on="segment_id",
        how="left",
        validate="many_to_one",
    )

    result = pd.DataFrame({"event_id": events["event_id"]}).copy()
    result["prec_station_link_count"] = (
        assignments.groupby("event_id")["station_p_id"].nunique().reindex(events["event_id"]).fillna(0).astype("Int64").to_numpy()
    )
    result["n_prec"] = 0
    result["prec_intensity_max"] = 0.0
    result["prec_intensity_mean"] = 0.0
    result["prec_depth_total"] = 0.0
    result["prec_duration_total"] = 0.0

    intensity_sums = pd.Series(0.0, index=result["event_id"])
    event_index = result.set_index("event_id")

    station_lookup = precip_events.groupby("station_p_id", sort=True)
    unmatched_station_ids: list[str] = []

    for station_p_id, flood_group in assignments.dropna(subset=["station_p_id"]).groupby("station_p_id", sort=True):
        if station_p_id not in station_lookup.groups:
            unmatched_station_ids.append(str(station_p_id))
            continue

        rain_group = station_lookup.get_group(station_p_id).sort_values("start", kind="stable").reset_index(drop=True)
        if rain_group.empty:
            continue

        rain_start = rain_group["start"].to_numpy(dtype="datetime64[ns]")
        rain_end = rain_group["end"].to_numpy(dtype="datetime64[ns]")
        rain_intensity = rain_group["intensity"].to_numpy(dtype=float)
        rain_depth = rain_group["depth"].to_numpy(dtype=float)
        rain_duration = rain_group["duration"].to_numpy(dtype=float)

        flood_subset = flood_group.sort_values("event_id", kind="stable").reset_index(drop=True)
        flood_start = flood_subset["event_start_local"].to_numpy(dtype="datetime64[ns]")
        flood_end = flood_subset["event_window_end_local"].to_numpy(dtype="datetime64[ns]")

        overlap = (rain_start[None, :] <= flood_end[:, None]) & (rain_end[None, :] >= flood_start[:, None])
        counts = overlap.sum(axis=1).astype(int)
        intensity_max = np.zeros(len(flood_subset), dtype=float)
        positive_mask = counts > 0
        if positive_mask.any():
            intensity_max[positive_mask] = np.nanmax(
                np.where(overlap[positive_mask], rain_intensity[None, :], np.nan),
                axis=1,
            )
        intensity_sum = np.nansum(np.where(overlap, rain_intensity[None, :], np.nan), axis=1)
        depth_sum = np.nansum(np.where(overlap, rain_depth[None, :], np.nan), axis=1)
        duration_sum = np.nansum(np.where(overlap, rain_duration[None, :], np.nan), axis=1)

        for idx, event_id in enumerate(flood_subset["event_id"].tolist()):
            event_index.at[event_id, "n_prec"] = int(event_index.at[event_id, "n_prec"]) + int(counts[idx])
            event_index.at[event_id, "prec_intensity_max"] = max(float(event_index.at[event_id, "prec_intensity_max"]), float(intensity_max[idx]))
            event_index.at[event_id, "prec_depth_total"] = float(event_index.at[event_id, "prec_depth_total"]) + float(depth_sum[idx])
            event_index.at[event_id, "prec_duration_total"] = float(event_index.at[event_id, "prec_duration_total"]) + float(duration_sum[idx])
            intensity_sums.at[event_id] = float(intensity_sums.at[event_id]) + float(intensity_sum[idx])

    n_prec = event_index["n_prec"].astype(int)
    event_index["prec_intensity_mean"] = np.divide(
        intensity_sums.reindex(event_index.index).to_numpy(dtype=float),
        n_prec.to_numpy(dtype=float),
        out=np.zeros(len(event_index), dtype=float),
        where=n_prec.to_numpy(dtype=float) > 0,
    )
    result = event_index.reset_index()

    diagnostics = pd.DataFrame(
        [
            {
                "events_total": int(len(events)),
                "events_without_station_crosswalk": int(assignments["station_p_id"].isna().groupby(assignments["event_id"]).all().sum()),
                "events_zero_prec_overlap": int((result["n_prec"] == 0).sum()),
                "events_one_prec_overlap": int((result["n_prec"] == 1).sum()),
                "events_multiple_prec_overlap": int((result["n_prec"] > 1).sum()),
                "unmatched_precip_station_ids": ",".join(sorted(set(unmatched_station_ids))),
            }
        ]
    )
    return result, diagnostics


def compute_tide_overlap_features(
    events: pd.DataFrame,
    crosswalk: pd.DataFrame,
    tide_series: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = events[["event_id", "segment_id", "event_start_local", "event_window_end_local"]].merge(
        crosswalk[["segment_id", "tide_id"]],
        on="segment_id",
        how="left",
        validate="many_to_one",
    )

    outputs: list[dict[str, object]] = []
    unmatched_tide_ids: list[str] = []
    tide_lookup = tide_series.groupby("tide_id", sort=True)

    for tide_id, flood_group in assignments.groupby("tide_id", dropna=False, sort=True):
        if pd.isna(tide_id):
            for row in flood_group.itertuples(index=False):
                outputs.append(
                    {
                        "event_id": row.event_id,
                        "tide_id": pd.NA,
                        "tide_obs_n": 0,
                        "tide_level_m_mean": pd.NA,
                        "tide_level_m_max": pd.NA,
                        "tide_level_m_min": pd.NA,
                        "tide_level_m_range": pd.NA,
                    }
                )
            continue

        if tide_id not in tide_lookup.groups:
            unmatched_tide_ids.append(str(tide_id))
            for row in flood_group.itertuples(index=False):
                outputs.append(
                    {
                        "event_id": row.event_id,
                        "tide_id": tide_id,
                        "tide_obs_n": 0,
                        "tide_level_m_mean": pd.NA,
                        "tide_level_m_max": pd.NA,
                        "tide_level_m_min": pd.NA,
                        "tide_level_m_range": pd.NA,
                    }
                )
            continue

        series_group = tide_lookup.get_group(tide_id).sort_values("timestamp_local", kind="stable").reset_index(drop=True)
        times = series_group["timestamp_local"].to_numpy(dtype="datetime64[ns]")
        values = series_group["tide_level_m"].to_numpy(dtype=float)

        for row in flood_group.itertuples(index=False):
            start = np.datetime64(row.event_start_local)
            end = np.datetime64(row.event_window_end_local)
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, end, side="right"))
            window = values[left:right]
            if window.size == 0:
                outputs.append(
                    {
                        "event_id": row.event_id,
                        "tide_id": tide_id,
                        "tide_obs_n": 0,
                        "tide_level_m_mean": pd.NA,
                        "tide_level_m_max": pd.NA,
                        "tide_level_m_min": pd.NA,
                        "tide_level_m_range": pd.NA,
                    }
                )
            else:
                outputs.append(
                    {
                        "event_id": row.event_id,
                        "tide_id": tide_id,
                        "tide_obs_n": int(window.size),
                        "tide_level_m_mean": float(np.nanmean(window)),
                        "tide_level_m_max": float(np.nanmax(window)),
                        "tide_level_m_min": float(np.nanmin(window)),
                        "tide_level_m_range": float(np.nanmax(window) - np.nanmin(window)),
                    }
                )

    tide_features = pd.DataFrame(outputs)
    diagnostics = pd.DataFrame(
        [
            {
                "events_total": int(len(events)),
                "events_without_tide_crosswalk": int(assignments["tide_id"].isna().groupby(assignments["event_id"]).all().sum()),
                "events_without_tide_observations": int(tide_features["tide_obs_n"].fillna(0).eq(0).sum()),
                "unmatched_tide_ids": ",".join(sorted(set(unmatched_tide_ids))),
            }
        ]
    )
    return tide_features, diagnostics


def build_missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for column in frame.columns:
        null_count = int(frame[column].isna().sum())
        rows.append(
            {
                "column": column,
                "null_count": null_count,
                "null_share": null_count / total if total else math.nan,
                "dtype": str(frame[column].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values(["null_share", "column"], ascending=[False, True], kind="stable").reset_index(drop=True)


def build_numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number, "boolean"]).copy()
    if numeric.empty:
        return pd.DataFrame(columns=["feature"])
    summary = numeric.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose().reset_index().rename(columns={"index": "feature"})
    return summary


def build_correlation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty or "intensity" not in numeric.columns:
        return pd.DataFrame(columns=["feature"])
    target_columns = [column for column in ["intensity", "event_window_duration_hours", "prec_depth_total", "tide_level_m_max"] if column in numeric.columns]
    correlations = []
    for feature in numeric.columns:
        for target in target_columns:
            if feature == target:
                continue
            valid = numeric[[feature, target]].dropna()
            if len(valid) < 3 or valid[feature].nunique(dropna=True) <= 1 or valid[target].nunique(dropna=True) <= 1:
                corr = math.nan
            else:
                corr = valid[feature].corr(valid[target])
            correlations.append({"feature": feature, "target": target, "pearson_r": corr})
    return pd.DataFrame(correlations).sort_values(["target", "pearson_r"], ascending=[True, False], kind="stable").reset_index(drop=True)


def build_validation_checks(frame: gpd.GeoDataFrame) -> pd.DataFrame:
    event_signature_duplicates = int(
        frame.duplicated(subset=["segment_id", "event_start_local", "event_window_end_local"], keep=False).sum()
    )
    checks = [
        {"check": "event_rows", "value": int(len(frame))},
        {"check": "duplicate_event_id", "value": int(frame["event_id"].duplicated().sum())},
        {"check": "null_segment_id", "value": int(frame["segment_id"].isna().sum())},
        {"check": "null_geometry", "value": int(frame.geometry.isna().sum())},
        {"check": "negative_event_window_duration", "value": int(frame["event_window_duration_hours"].lt(0).fillna(False).sum())},
        {"check": "negative_resolution_hours", "value": int(frame["resolution_hours"].lt(0).fillna(False).sum())},
        {"check": "duplicate_event_signature", "value": event_signature_duplicates},
        {"check": "events_missing_prec_station", "value": int(frame["station_p_id"].isna().sum())},
        {"check": "events_missing_tide_station", "value": int(frame["tide_id"].isna().sum())},
        {"check": "events_missing_census_geoid", "value": int(frame["census_geoid"].isna().sum()) if "census_geoid" in frame.columns else pd.NA},
        {"check": "events_missing_fema_zone", "value": int(frame["fema_fld_zone"].isna().sum()) if "fema_fld_zone" in frame.columns else pd.NA},
    ]
    return pd.DataFrame(checks)


def derive_event_coordinates(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    local = frame.copy()
    rep = local.representative_point()
    local["event_x"] = rep.x
    local["event_y"] = rep.y
    lonlat = gpd.GeoSeries(rep, crs=local.crs).to_crs(4326)
    local["event_lon"] = lonlat.x
    local["event_lat"] = lonlat.y
    return local


@dataclass
class BuildResult:
    master_table: gpd.GeoDataFrame
    diagnostics: dict[str, pd.DataFrame]


def build_master_table(write_outputs: bool = True) -> BuildResult:
    ensure_proj_env()
    MODELING_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    events = load_flood_events()
    complaints = load_individual_complaints()
    segment_features = summarize_segment_static_features()

    master = events.merge(segment_features, on="segment_id", how="left", validate="many_to_one")
    master = gpd.GeoDataFrame(master, geometry="geometry", crs=segment_features.crs)
    master = derive_event_coordinates(master)

    event_complaint_summary, complaint_assignment_diag = assign_complaints_to_events(events, complaints)
    master = master.merge(event_complaint_summary, on="event_id", how="left", validate="one_to_one")

    unique_segments = master[["segment_id", "geometry"]].drop_duplicates("segment_id").reset_index(drop=True)
    fema_features, fema_diag = compute_fema_features(unique_segments)
    infra_features, infra_diag = compute_infrastructure_features(unique_segments)
    master = master.merge(fema_features, on="segment_id", how="left", validate="many_to_one")
    master = master.merge(infra_features, on="segment_id", how="left", validate="many_to_one")

    census_features, census_diag = compute_census_features(master[["event_id", "event_year", "geometry", "census_period_selected"]] if "census_period_selected" in master.columns else master[["event_id", "event_year", "geometry"]])
    master = master.merge(census_features, on="event_id", how="left", validate="one_to_one")

    precip_crosswalk = load_precip_crosswalk()
    precip_events = load_precip_events()
    precip_features, precip_diag = compute_precipitation_overlap_features(events, precip_crosswalk, precip_events)
    master = master.merge(precip_features, on="event_id", how="left", validate="one_to_one")

    tide_crosswalk = load_tide_crosswalk()
    tide_series = load_tide_series()
    tide_features, tide_diag = compute_tide_overlap_features(events, tide_crosswalk, tide_series)
    tide_features = tide_features.rename(columns={"tide_id": "tide_id_dynamic"})
    master = master.merge(tide_features, on="event_id", how="left", validate="one_to_one")
    if "tide_id" not in master.columns and "tide_id_dynamic" in master.columns:
        master["tide_id"] = master["tide_id_dynamic"]
    elif "tide_id" in master.columns and "tide_id_dynamic" in master.columns:
        master["tide_id"] = master["tide_id"].fillna(master["tide_id_dynamic"])

    master["resolution_bool"] = master["resolution_bool"].fillna(False).astype(bool)
    master["occurrence"] = True
    master["intensity"] = pd.to_numeric(master["intensity"], errors="coerce").fillna(0).astype("Int64")
    master["segment_id"] = normalize_string_series(master["segment_id"])

    if "complaint_count_check" in master.columns:
        mismatch = master["complaint_count_check"].notna() & master["complaint_count_check"].ne(master["intensity"])
        master["complaint_count_mismatch"] = mismatch.fillna(False)
    else:
        master["complaint_count_mismatch"] = False

    master["prec_match_class"] = np.select(
        [master["n_prec"].gt(1), master["n_prec"].eq(1)],
        ["multiple", "single"],
        default="none",
    )

    diagnostics = {
        "validation_checks": build_validation_checks(master),
        "missingness_report": build_missingness_report(pd.DataFrame(master.drop(columns="geometry"))),
        "numeric_summary": build_numeric_summary(pd.DataFrame(master.drop(columns="geometry"))),
        "correlation_summary": build_correlation_summary(pd.DataFrame(master.drop(columns="geometry"))),
        "precipitation_diagnostics": precip_diag,
        "tide_diagnostics": tide_diag,
        "fema_diagnostics": fema_diag,
        "census_diagnostics": census_diag,
        "infrastructure_diagnostics": infra_diag,
        "complaint_assignment_diagnostics": complaint_assignment_diag,
        "segment_component_summary": pd.read_csv(STREET_CONNECTIVITY_COMPONENTS_PATH),
    }

    if write_outputs:
        flat_master = pd.DataFrame(master.drop(columns="geometry"))
        flat_master.to_parquet(MASTER_PARQUET_PATH, index=False)
        master.to_parquet(MASTER_GEOPARQUET_PATH, index=False)

        for name, table in diagnostics.items():
            table.to_csv(DIAGNOSTICS_DIR / f"{name}.csv", index=False)

        metadata = {
            "rows": int(len(master)),
            "unique_segments": int(master["segment_id"].nunique(dropna=True)),
            "event_start_min": str(master["event_start_local"].min()),
            "event_start_max": str(master["event_start_local"].max()),
            "parquet_path": str(MASTER_PARQUET_PATH),
            "geoparquet_path": str(MASTER_GEOPARQUET_PATH),
        }
        (DIAGNOSTICS_DIR / "build_metadata.json").write_text(json.dumps(metadata, indent=2))

    return BuildResult(master_table=master, diagnostics=diagnostics)
