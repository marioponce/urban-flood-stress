from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from project_name.flood_events_master import (
    DIAGNOSTICS_DIR,
    LOCAL_TIMEZONE,
    MASTER_GEOPARQUET_PATH,
    MASTER_PARQUET_PATH,
    MODELING_DIR,
    ROOT,
    compute_census_features,
    compute_fema_features,
    compute_infrastructure_features,
    derive_event_coordinates,
    ensure_proj_env,
    load_tide_crosswalk,
    normalize_segment_id_series,
    normalize_string_series,
    summarize_segment_static_features,
)


SHORELINE_PRIMARY_PATH = ROOT / "data" / "spatial" / "vector" / "nyc_shoreline" / "nyc_shoreline_tide_primary.geojson"
BALANCED_PARQUET_PATH = MODELING_DIR / "flood_events_balanced.parquet"
BALANCED_GEOPARQUET_PATH = MODELING_DIR / "flood_events_balanced.geoparquet"
UNREPORTED_PARQUET_PATH = MODELING_DIR / "possible_unreported_flood_candidates.parquet"
UNREPORTED_GEOPARQUET_PATH = MODELING_DIR / "possible_unreported_flood_candidates.geoparquet"
VALIDATION_MASTER_PATH = MODELING_DIR / "validation_master_table.csv"
VALIDATION_BALANCING_PATH = MODELING_DIR / "validation_balancing.csv"
VALIDATION_UNREPORTED_PATH = MODELING_DIR / "validation_unreported_candidates.csv"
BALANCING_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "balancing"
BALANCED_DIAGNOSTICS_DIR = BALANCING_DIAGNOSTICS_DIR
SEGMENT_UNIVERSE_CACHE_PATH = BALANCING_DIAGNOSTICS_DIR / "segment_universe_sampling.geoparquet"

TEMPORAL_EXCLUSION_HOURS = 6
NEGATIVE_SPATIAL_EXCLUSION_M = 100
UNREPORTED_DISTANCE_M = 20

NEGATIVE_SPATIAL_EXCLUSION_FT = NEGATIVE_SPATIAL_EXCLUSION_M / 0.304800609601219
UNREPORTED_DISTANCE_FT = UNREPORTED_DISTANCE_M / 0.304800609601219
METERS_PER_FOOT = 0.304800609601219


def normalize_borough_value(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    if not text:
        return pd.NA
    return text


def normalize_borough_series(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(normalize_borough_value), index=series.index, dtype="string")


def to_local_naive(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    tz = getattr(parsed.dt, "tz", None)
    if tz is not None:
        return parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)


def quantile_band(series: pd.Series, q: int = 10, prefix: str = "band") -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series(pd.NA, index=series.index, dtype="string")

    band = pd.qcut(valid.rank(method="first"), q=min(q, valid.nunique()), labels=False, duplicates="drop")
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out.loc[valid.index] = band.astype("Int64").astype("string").map(lambda value: f"{prefix}_{value}" if pd.notna(value) else pd.NA)
    return out


def fema_broad_class(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    if not text:
        return "UNKNOWN"
    if text == "OUTSIDE_NFHL":
        return "OUTSIDE_NFHL"
    if text.startswith("A") or text.startswith("V"):
        return "SFHA"
    if text.startswith("X"):
        return "MINIMAL"
    return "OTHER"


def road_class_group(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return str(value).strip().lower() or "UNKNOWN"


def make_storm_event_ids(events: pd.DataFrame, gap_hours: int = TEMPORAL_EXCLUSION_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = events.copy().sort_values(["event_start_local", "event_window_end_local", "event_id"], kind="stable").reset_index(drop=True)
    if frame.empty:
        frame["storm_event_id"] = pd.Series(dtype="string")
        return frame, pd.DataFrame(columns=["storm_event_id"])

    gap = pd.Timedelta(hours=gap_hours)
    current_end = frame.loc[0, "event_window_end_local"]
    current_id = 1
    storm_ids = []
    for row in frame.itertuples(index=False):
        if row.event_start_local > current_end + gap:
            current_id += 1
            current_end = row.event_window_end_local
        else:
            if pd.notna(row.event_window_end_local) and row.event_window_end_local > current_end:
                current_end = row.event_window_end_local
        storm_ids.append(f"STORM_{current_id:05d}")

    frame["storm_event_id"] = pd.Series(storm_ids, dtype="string")
    storm_summary = (
        frame.groupby("storm_event_id", as_index=False)
        .agg(
            storm_start=("event_start_local", "min"),
            storm_end=("event_window_end_local", "max"),
            n_positive_events=("event_id", "size"),
            n_positive_segments=("segment_id", "nunique"),
            n_boroughs=("segment_borough", "nunique"),
        )
    )
    storm_summary["storm_duration_hours"] = (storm_summary["storm_end"] - storm_summary["storm_start"]).dt.total_seconds() / 3600.0
    return frame, storm_summary


def compute_segment_universe_for_sampling() -> gpd.GeoDataFrame:
    ensure_proj_env()
    if SEGMENT_UNIVERSE_CACHE_PATH.exists():
        cached = gpd.read_parquet(SEGMENT_UNIVERSE_CACHE_PATH)
        cached["segment_id"] = normalize_segment_id_series(cached["segment_id"])
        cached["segment_borough"] = normalize_borough_series(cached["segment_borough"])
        cached["station_p_id"] = normalize_string_series(cached["station_p_id"]).str.upper()
        cached["tide_id"] = normalize_string_series(cached["tide_id"])
        cached["precip_polygon_id"] = normalize_string_series(cached["precip_polygon_id"]).str.upper()
        cached["tide_polygon_id"] = normalize_string_series(cached["tide_polygon_id"])
        cached["fema_zone_sampling"] = "UNAVAILABLE"
        cached["fema_broad_class"] = "UNAVAILABLE"
        return cached

    BALANCING_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    segments = summarize_segment_static_features()
    segments = segments.copy()
    segments["segment_id"] = normalize_segment_id_series(segments["segment_id"])
    segments["segment_borough"] = normalize_borough_series(segments["segment_borough"])
    segments["station_p_id"] = normalize_string_series(segments["station_p_id"]).str.upper()
    segments["tide_id"] = normalize_string_series(segments["tide_id"])
    if "shore_id" in segments.columns:
        segments["shore_id"] = normalize_string_series(segments["shore_id"])
    segments["road_class_group"] = segments["road_class"].map(road_class_group).astype("string")
    segments["shoreline_station_id"] = segments["shore_id"] if "shore_id" in segments.columns else segments["tide_id"]
    segments["shore_dist_ft"] = pd.NA
    segments["fema_zone_sampling"] = "UNAVAILABLE"
    segments["fema_broad_class"] = "UNAVAILABLE"
    segments["elevation_band"] = quantile_band(segments["dem_mean"], q=10, prefix="elev")
    segments["shore_proxy_band"] = quantile_band(segments["shore_graph_steps"], q=10, prefix="shoregraph")
    segments["shore_dist_band"] = segments["shore_proxy_band"]
    segments["precip_polygon_id"] = segments["station_p_id"]
    segments["tide_polygon_id"] = segments["tide_id"]
    segments.to_parquet(SEGMENT_UNIVERSE_CACHE_PATH, index=False)
    return segments


def augment_master_table_for_balancing(write_outputs: bool = True) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    ensure_proj_env()
    master = gpd.read_parquet(MASTER_GEOPARQUET_PATH)
    master["event_id"] = master["event_id"].astype("string")
    master["segment_id"] = normalize_segment_id_series(master["segment_id"])
    master["segment_borough"] = normalize_borough_series(master["segment_borough"])
    master["station_p_id"] = normalize_string_series(master["station_p_id"]).str.upper()
    master["tide_id"] = normalize_string_series(master["tide_id"])
    master["event_start_local"] = to_local_naive(master["event_start_local"])
    master["event_window_end_local"] = to_local_naive(master["event_window_end_local"])
    master["event_window_duration_hours"] = pd.to_numeric(master["event_window_duration_hours"], errors="coerce")

    segment_universe = compute_segment_universe_for_sampling()
    static_cols = [
        "segment_id",
        "shore_dist_ft",
        "shore_dist_band",
        "shore_proxy_band",
        "elevation_band",
        "fema_broad_class",
        "fema_zone_sampling",
        "road_class_group",
        "shoreline_station_id",
        "precip_polygon_id",
        "tide_polygon_id",
    ]
    master = master.drop(columns=[column for column in static_cols[1:] if column in master.columns], errors="ignore")
    master = master.merge(segment_universe[static_cols], on="segment_id", how="left", validate="many_to_one")
    if "fema_fld_zone" in master.columns:
        master["fema_broad_class"] = master["fema_fld_zone"].map(fema_broad_class).astype("string")

    clustered, storm_summary = make_storm_event_ids(master)
    master = clustered.sort_values(["event_id"], kind="stable").reset_index(drop=True)
    storm_lookup = storm_summary.set_index("storm_event_id")
    master["storm_event_positive_count"] = master["storm_event_id"].map(storm_lookup["n_positive_events"]).astype("Int64")
    master["storm_event_duration_hours"] = master["storm_event_id"].map(storm_lookup["storm_duration_hours"])
    master["label_definition"] = "observed_reported_flood"

    validation = pd.DataFrame(
        [
            {"metric": "rows", "value": int(len(master))},
            {"metric": "unique_segments", "value": int(master["segment_id"].nunique(dropna=True))},
            {"metric": "storm_events", "value": int(master["storm_event_id"].nunique(dropna=True))},
            {"metric": "duplicate_event_id", "value": int(master["event_id"].duplicated().sum())},
            {"metric": "null_precip_polygon_id", "value": int(master["precip_polygon_id"].isna().sum())},
            {"metric": "null_tide_polygon_id", "value": int(master["tide_polygon_id"].isna().sum())},
            {"metric": "negative_event_window_duration", "value": int(master["event_window_duration_hours"].lt(0).fillna(False).sum())},
        ]
    )

    if write_outputs:
        pd.DataFrame(master.drop(columns="geometry")).to_parquet(MASTER_PARQUET_PATH, index=False)
        master.to_parquet(MASTER_GEOPARQUET_PATH, index=False)
        validation.to_csv(VALIDATION_MASTER_PATH, index=False)
        storm_summary.to_csv(BALANCING_DIAGNOSTICS_DIR.parent / "storm_event_summary.csv", index=False)

    return master, segment_universe, storm_summary


def segment_event_windows(master: gpd.GeoDataFrame) -> dict[str, pd.DataFrame]:
    windows = {}
    for segment_id, group in master.groupby("segment_id", sort=False):
        windows[str(segment_id)] = group[["event_start_local", "event_window_end_local", "storm_event_id"]].sort_values("event_start_local", kind="stable").reset_index(drop=True)
    return windows


def build_storm_positive_segment_sets(master: gpd.GeoDataFrame) -> dict[str, set[str]]:
    return {
        str(storm_event_id): set(group["segment_id"].astype(str))
        for storm_event_id, group in master.groupby("storm_event_id", sort=False)
    }


def has_temporal_conflict(segment_windows: dict[str, pd.DataFrame], segment_id: str, start: pd.Timestamp, end: pd.Timestamp, buffer_hours: int = TEMPORAL_EXCLUSION_HOURS) -> bool:
    windows = segment_windows.get(str(segment_id))
    if windows is None or windows.empty:
        return False
    start_buffer = start - pd.Timedelta(hours=buffer_hours)
    end_buffer = end + pd.Timedelta(hours=buffer_hours)
    overlap = (windows["event_start_local"] <= end_buffer) & (windows["event_window_end_local"] >= start_buffer)
    return bool(overlap.any())


def build_storm_buffer_geometries(master: gpd.GeoDataFrame, distance_ft: float) -> dict[str, object]:
    buffers = {}
    for storm_event_id, group in master.groupby("storm_event_id", sort=False):
        buffered = group.geometry.buffer(distance_ft)
        buffers[str(storm_event_id)] = buffered.union_all()
    return buffers


def build_storm_spatial_exclusion_sets(
    master: gpd.GeoDataFrame,
    segment_universe: gpd.GeoDataFrame,
    distance_ft: float,
) -> dict[str, set[str]]:
    spatial_index = segment_universe.sindex
    excluded: dict[str, set[str]] = {}
    for storm_event_id, group in master.groupby("storm_event_id", sort=False):
        storm_buffer = group.geometry.buffer(distance_ft).union_all()
        candidate_idx = list(spatial_index.query(storm_buffer, predicate="intersects"))
        if not candidate_idx:
            excluded[str(storm_event_id)] = set()
            continue
        candidates = segment_universe.iloc[candidate_idx][["segment_id", "geometry"]].copy()
        candidates = candidates[candidates.geometry.intersects(storm_buffer)]
        excluded[str(storm_event_id)] = set(candidates["segment_id"].astype(str))
    return excluded


def build_possible_unreported_candidates(
    master: gpd.GeoDataFrame,
    segment_universe: gpd.GeoDataFrame,
    storm_summary: pd.DataFrame,
    storm_positive_segments: dict[str, set[str]] | None = None,
    write_outputs: bool = True,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    positives_by_storm = {storm: group.copy() for storm, group in master.groupby("storm_event_id", sort=False)}
    spatial_index = segment_universe.sindex
    if storm_positive_segments is None:
        storm_positive_segments = build_storm_positive_segment_sets(master)

    for storm_event_id, positive_group in positives_by_storm.items():
        storm_buffer = positive_group.geometry.buffer(UNREPORTED_DISTANCE_FT).union_all()
        candidate_idx = list(spatial_index.query(storm_buffer, predicate="intersects"))
        if not candidate_idx:
            continue
        candidates = segment_universe.iloc[candidate_idx].copy()
        candidates = candidates[candidates.geometry.intersects(storm_buffer)].copy()
        if candidates.empty:
            continue

        positive_segments = set(positive_group["segment_id"].astype(str))
        candidates = candidates[~candidates["segment_id"].astype(str).isin(positive_segments)].copy()
        if candidates.empty:
            continue

        nearest = gpd.sjoin_nearest(
            candidates[["segment_id", "segment_borough", "station_p_id", "tide_id", "geometry"]],
            positive_group[["event_id", "segment_id", "segment_borough", "station_p_id", "tide_id", "event_start_local", "event_window_end_local", "geometry"]].rename(columns={"segment_id": "reported_segment_id"}),
            how="left",
            distance_col="distance_ft",
            lsuffix="cand",
            rsuffix="rep",
        )
        nearest = nearest.sort_values(["segment_id", "distance_ft"], kind="stable").drop_duplicates("segment_id")
        nearest["distance_to_reported_event_m"] = nearest["distance_ft"] * METERS_PER_FOOT
        nearest["same_tide_polygon"] = nearest["tide_id_cand"].eq(nearest["tide_id_rep"])
        nearest["same_precip_polygon"] = nearest["station_p_id_cand"].eq(nearest["station_p_id_rep"])
        nearest["same_borough"] = nearest["segment_borough_cand"].eq(nearest["segment_borough_rep"])
        nearest["possible_unreported_flood"] = True
        nearest["storm_event_id"] = storm_event_id
        nearest["time_delta_to_reported_event_hours"] = 0.0
        nearest["candidate_segment_id"] = nearest["segment_id"]
        nearest["nearest_reported_event_id"] = nearest["event_id"]

        nearest["unreported_confidence_level"] = np.select(
            [
                nearest["distance_to_reported_event_m"].le(20)
                & nearest["same_tide_polygon"]
                & nearest["same_precip_polygon"],
                nearest["distance_to_reported_event_m"].le(20)
                & nearest["same_borough"]
                & (nearest["same_tide_polygon"] | nearest["same_precip_polygon"]),
            ],
            ["high", "medium"],
            default="low",
        )

        storm_positive_segment_set = storm_positive_segments.get(str(storm_event_id), set())
        nearest["candidate_has_reported_event_in_storm"] = nearest["candidate_segment_id"].astype(str).isin(storm_positive_segment_set)
        nearest = nearest[~nearest["candidate_has_reported_event_in_storm"]].copy()
        if nearest.empty:
            continue

        keep = [
            "candidate_segment_id",
            "nearest_reported_event_id",
            "reported_segment_id",
            "storm_event_id",
            "distance_to_reported_event_m",
            "time_delta_to_reported_event_hours",
            "same_tide_polygon",
            "same_precip_polygon",
            "same_borough",
            "possible_unreported_flood",
            "unreported_confidence_level",
            "geometry",
        ]
        if "segment_borough_cand" in nearest.columns:
            nearest = nearest.rename(columns={"segment_borough_cand": "segment_borough"})
            keep.append("segment_borough")
        if "station_p_id_cand" in nearest.columns:
            nearest = nearest.rename(columns={"station_p_id_cand": "station_p_id"})
            keep.append("station_p_id")
        if "tide_id_cand" in nearest.columns:
            nearest = nearest.rename(columns={"tide_id_cand": "tide_id"})
            keep.append("tide_id")
        records.extend(nearest[keep].to_dict("records"))

    unreported = gpd.GeoDataFrame(records, geometry="geometry", crs=segment_universe.crs)
    if not unreported.empty:
        unreported = unreported.sort_values(
            ["storm_event_id", "candidate_segment_id", "distance_to_reported_event_m"],
            kind="stable",
        ).drop_duplicates(["storm_event_id", "candidate_segment_id"])

    diagnostics = pd.DataFrame(
        [
            {
                "n_possible_unreported_candidates": int(len(unreported)),
                "mean_distance_m": float(unreported["distance_to_reported_event_m"].mean()) if not unreported.empty else math.nan,
                "median_distance_m": float(unreported["distance_to_reported_event_m"].median()) if not unreported.empty else math.nan,
                "high_confidence": int(unreported["unreported_confidence_level"].eq("high").sum()) if not unreported.empty else 0,
                "medium_confidence": int(unreported["unreported_confidence_level"].eq("medium").sum()) if not unreported.empty else 0,
                "low_confidence": int(unreported["unreported_confidence_level"].eq("low").sum()) if not unreported.empty else 0,
            }
        ]
    )

    if write_outputs:
        if unreported.empty:
            pd.DataFrame(columns=["candidate_segment_id"]).to_parquet(UNREPORTED_PARQUET_PATH, index=False)
            unreported.to_parquet(UNREPORTED_GEOPARQUET_PATH, index=False)
        else:
            pd.DataFrame(unreported.drop(columns="geometry")).to_parquet(UNREPORTED_PARQUET_PATH, index=False)
            unreported.to_parquet(UNREPORTED_GEOPARQUET_PATH, index=False)
        diagnostics.to_csv(VALIDATION_UNREPORTED_PATH, index=False)

    return unreported, diagnostics


def _score_candidates(candidates: pd.DataFrame, positive_row: pd.Series, usage_counts: dict[str, int]) -> pd.Series:
    elevation_diff = (pd.to_numeric(candidates["dem_mean"], errors="coerce") - float(positive_row.get("dem_mean", np.nan))).abs().fillna(1e9)
    shore_diff = (pd.to_numeric(candidates["shore_graph_steps"], errors="coerce") - float(positive_row.get("shore_graph_steps", np.nan))).abs().fillna(1e9)
    length_diff = (pd.to_numeric(candidates["segment_length_geom_ft"], errors="coerce") - float(positive_row.get("segment_length_geom_ft", np.nan))).abs().fillna(1e9)
    usage_penalty = candidates["segment_id"].astype(str).map(lambda value: usage_counts.get(value, 0)).astype(float)
    return elevation_diff + (shore_diff / 10.0) + (length_diff / 1000.0) + (usage_penalty * 10.0)


def _key_part(value: object) -> str:
    if pd.isna(value):
        return "__NA__"
    text = str(value)
    return text if text else "__NA__"


def _make_lookup_key(*values: object) -> tuple[str, ...]:
    return tuple(_key_part(value) for value in values)


def build_negative_samples(
    master: gpd.GeoDataFrame,
    segment_universe: gpd.GeoDataFrame,
    unreported: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_frame = pd.DataFrame(segment_universe.drop(columns="geometry")).copy()
    segment_frame["segment_id"] = segment_frame["segment_id"].astype(str)
    segment_frame = segment_frame.sort_values("segment_id", kind="stable").reset_index(drop=True)
    segment_lookup = segment_universe.set_index(segment_universe["segment_id"].astype(str), drop=False)

    group4_lookup: dict[tuple[str, ...], list[str]] = {}
    group4_set_lookup: dict[tuple[str, ...], set[str]] = {}
    level1_lookup: dict[tuple[str, ...], list[str]] = {}
    level2_lookup: dict[tuple[str, ...], list[str]] = {}

    for row in segment_frame.itertuples(index=False):
        base_key = _make_lookup_key(row.segment_borough, row.tide_polygon_id, row.precip_polygon_id)
        level1_key = _make_lookup_key(
            row.segment_borough,
            row.tide_polygon_id,
            row.precip_polygon_id,
            row.road_class_group,
            row.elevation_band,
            row.shore_dist_band,
        )
        level2_key = _make_lookup_key(
            row.segment_borough,
            row.tide_polygon_id,
            row.precip_polygon_id,
            row.elevation_band,
            row.shore_dist_band,
        )
        segment_id = str(row.segment_id)
        group4_lookup.setdefault(base_key, []).append(segment_id)
        level1_lookup.setdefault(level1_key, []).append(segment_id)
        level2_lookup.setdefault(level2_key, []).append(segment_id)

    group4_set_lookup = {key: set(values) for key, values in group4_lookup.items()}
    storm_positive_segments = build_storm_positive_segment_sets(master)
    storm_spatial_excluded_segments = build_storm_spatial_exclusion_sets(master, segment_universe, NEGATIVE_SPATIAL_EXCLUSION_FT)
    unreported_by_storm = {
        storm_event_id: set(group["candidate_segment_id"].astype(str))
        for storm_event_id, group in unreported.groupby("storm_event_id", sort=False)
    }

    segment_usage_counts: dict[str, int] = {}
    used_negative_signatures: set[tuple[str, pd.Timestamp, pd.Timestamp]] = set()
    negative_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    for positive in master.sort_values(["event_start_local", "event_id"], kind="stable").itertuples(index=False):
        base_key = _make_lookup_key(positive.segment_borough, positive.tide_polygon_id, positive.precip_polygon_id)
        base_ids = group4_lookup.get(base_key)
        base_id_set = group4_set_lookup.get(base_key)
        if base_ids is None or base_id_set is None:
            diagnostics_rows.append(
                {
                    "positive_event_id": positive.event_id,
                    "matched_negative_event_id": pd.NA,
                    "candidate_pool_size_before_filtering": 0,
                    "candidate_pool_size_after_filtering": 0,
                    "excluded_by_same_segment": 0,
                    "excluded_by_temporal_overlap": 0,
                    "excluded_by_spatial_buffer": 0,
                    "excluded_by_spatiotemporal_buffer": 0,
                    "excluded_by_possible_unreported": 0,
                    "negative_sampling_level": pd.NA,
                    "negative_sampling_relaxed": pd.NA,
                    "negative_sampling_notes": "no_base_pool",
                    "sampling_success": False,
                }
            )
            continue

        before = len(base_ids)
        same_segment_id = str(positive.segment_id)
        storm_positive_segment_set = storm_positive_segments.get(str(positive.storm_event_id), set())
        storm_spatial_segment_set = storm_spatial_excluded_segments.get(str(positive.storm_event_id), set())
        unreported_segments = unreported_by_storm.get(str(positive.storm_event_id), set())
        same_segment_set = {same_segment_id}
        temporal_excluded_set = set(storm_positive_segment_set) - same_segment_set
        spatial_excluded_set = set(storm_spatial_segment_set) - same_segment_set
        spatiotemporal_excluded_set = temporal_excluded_set & spatial_excluded_set
        excluded_union = same_segment_set | temporal_excluded_set | spatial_excluded_set | set(unreported_segments)
        after_filtering = len(base_id_set - excluded_union)

        levels = [
            (
                1,
                level1_lookup.get(
                    _make_lookup_key(
                        positive.segment_borough,
                        positive.tide_polygon_id,
                        positive.precip_polygon_id,
                        road_class_group(positive.road_class),
                        getattr(positive, "elevation_band", pd.NA),
                        getattr(positive, "shore_dist_band", pd.NA),
                    ),
                    [],
                ),
                "same_road_same_elev_same_shore",
            ),
            (
                2,
                level2_lookup.get(
                    _make_lookup_key(
                        positive.segment_borough,
                        positive.tide_polygon_id,
                        positive.precip_polygon_id,
                        getattr(positive, "elevation_band", pd.NA),
                        getattr(positive, "shore_dist_band", pd.NA),
                    ),
                    [],
                ),
                "same_elev_same_shore",
            ),
            (
                3,
                base_ids,
                "same_borough_same_tide_same_precip",
            ),
        ]

        chosen = None
        chosen_level = None
        chosen_note = None
        for level, candidate_ids, note in levels:
            if not candidate_ids:
                continue
            best_candidate_id: str | None = None
            best_key: tuple[float, str] | None = None
            for candidate_id in candidate_ids:
                if candidate_id in excluded_union:
                    continue
                signature = (candidate_id, positive.event_start_local, positive.event_window_end_local)
                if signature in used_negative_signatures:
                    continue
                candidate_key = (float(segment_usage_counts.get(candidate_id, 0)), candidate_id)
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_candidate_id = candidate_id
            if best_candidate_id is None or best_key is None:
                continue
            chosen = segment_lookup.loc[best_candidate_id].copy()
            chosen["negative_match_score"] = float(best_key[0])
            chosen_level = level
            chosen_note = note
            if chosen is not None:
                break

        diagnostics = {
            "positive_event_id": positive.event_id,
            "candidate_pool_size_before_filtering": int(before),
            "candidate_pool_size_after_filtering": int(after_filtering),
            "excluded_by_same_segment": int(same_segment_id in base_id_set),
            "excluded_by_temporal_overlap": int(len(base_id_set & temporal_excluded_set)),
            "excluded_by_spatial_buffer": int(len(base_id_set & spatial_excluded_set)),
            "excluded_by_spatiotemporal_buffer": int(len(base_id_set & spatiotemporal_excluded_set)),
            "excluded_by_possible_unreported": int(len(base_id_set & set(unreported_segments))),
        }

        if chosen is None:
            diagnostics.update(
                {
                    "matched_negative_event_id": pd.NA,
                    "negative_sampling_level": pd.NA,
                    "negative_sampling_relaxed": pd.NA,
                    "negative_sampling_notes": "no_candidate_after_relaxation",
                    "sampling_success": False,
                }
            )
            diagnostics_rows.append(diagnostics)
            continue

        negative_event_id = f"NEG_{positive.event_id}"
        signature = (str(chosen.segment_id), positive.event_start_local, positive.event_window_end_local)
        used_negative_signatures.add(signature)
        segment_usage_counts[str(chosen.segment_id)] = segment_usage_counts.get(str(chosen.segment_id), 0) + 1

        negative_rows.append(
            {
                "event_id": negative_event_id,
                "matched_positive_event_id": str(positive.event_id),
                "segment_id": str(chosen.segment_id),
                "segment_borough": chosen.segment_borough,
                "station_p_id": chosen.station_p_id,
                "tide_id": chosen.tide_id,
                "precip_polygon_id": chosen.precip_polygon_id,
                "tide_polygon_id": chosen.tide_polygon_id,
                "event_start_local": positive.event_start_local,
                "event_window_end_local": positive.event_window_end_local,
                "event_window_duration_hours": positive.event_window_duration_hours,
                "event_year": positive.event_year,
                "event_month": positive.event_month,
                "event_date": positive.event_date,
                "storm_event_id": positive.storm_event_id,
                "storm_event_positive_count": positive.storm_event_positive_count,
                "storm_event_duration_hours": positive.storm_event_duration_hours,
                "occurrence": False,
                "intensity": 0,
                "resolution_hours": np.nan,
                "resolution_bool": False,
                "resolution": np.nan,
                "start": positive.event_start_local,
                "end": pd.NaT,
                "event_end_local": pd.NaT,
                "event_end_observed_local": pd.NaT,
                "event_end_inferred": False,
                "n_complaints": 0,
                "status": 0,
                "flood_event_status": 0,
                "label_definition": "matched_no_observed_reported_flood",
                "negative_sampling_level": chosen_level,
                "negative_sampling_relaxed": bool(chosen_level > 1),
                "negative_sampling_notes": chosen_note,
                "negative_match_score": float(chosen.negative_match_score),
                "negative_source_segment_id": str(chosen.segment_id),
                "geometry": chosen.geometry,
            }
        )

        diagnostics.update(
            {
                "matched_negative_event_id": negative_event_id,
                "negative_sampling_level": chosen_level,
                "negative_sampling_relaxed": bool(chosen_level > 1),
                "negative_sampling_notes": chosen_note,
                "sampling_success": True,
            }
        )
        diagnostics_rows.append(diagnostics)

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    negative_df = pd.DataFrame(negative_rows)
    return negative_df, diagnostics_df


def enrich_negative_events(
    negative_df: pd.DataFrame,
    master: gpd.GeoDataFrame,
    segment_universe: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if negative_df.empty:
        return gpd.GeoDataFrame(negative_df, geometry="geometry", crs=segment_universe.crs)

    negative = gpd.GeoDataFrame(negative_df, geometry="geometry", crs=segment_universe.crs)
    static_keep = [
        column
        for column in segment_universe.columns
        if column not in {"geometry", "station_p_id", "tide_id", "segment_borough", "precip_polygon_id", "tide_polygon_id"}
    ]
    negative = negative.drop(columns=[column for column in static_keep if column in negative.columns and column not in {"segment_id", "geometry"}], errors="ignore")
    static_lookup = pd.DataFrame(segment_universe.drop(columns="geometry"))
    negative = negative.merge(static_lookup[static_keep], on="segment_id", how="left", validate="many_to_one")

    positive_copy_cols = [
        "matched_positive_event_id",
        "n_prec",
        "prec_station_link_count",
        "prec_intensity_max",
        "prec_intensity_mean",
        "prec_depth_total",
        "prec_duration_total",
        "tide_obs_n",
        "tide_level_m_mean",
        "tide_level_m_max",
        "tide_level_m_min",
        "tide_level_m_range",
        "complaint_spatial_match_count",
        "complaint_textual_match_count",
        "complaint_borough_mode",
    ]
    positive_lookup = master[["event_id"] + [col for col in positive_copy_cols if col in master.columns]].copy()
    negative = negative.merge(
        positive_lookup.rename(columns={"event_id": "matched_positive_event_id"}),
        on="matched_positive_event_id",
        how="left",
        validate="many_to_one",
    )

    negative = derive_event_coordinates(negative)
    negative["census_period_selected"] = negative["event_year"].astype(int).astype(str)
    census_features, _ = compute_census_features(negative[["event_id", "event_year", "geometry"]].copy())
    negative = negative.merge(census_features, on="event_id", how="left", validate="one_to_one")

    unique_segments = negative[["segment_id", "geometry"]].drop_duplicates("segment_id").reset_index(drop=True)
    fema_features, _ = compute_fema_features(unique_segments)
    infra_features, _ = compute_infrastructure_features(unique_segments)
    negative = negative.merge(fema_features, on="segment_id", how="left", validate="many_to_one")
    negative = negative.merge(infra_features, on="segment_id", how="left", validate="many_to_one")

    negative["duration_hours"] = np.nan
    negative["resolution"] = np.nan
    negative["prec_match_class"] = np.select(
        [negative["n_prec"].gt(1), negative["n_prec"].eq(1)],
        ["multiple", "single"],
        default="none",
    )
    negative["complaint_count_mismatch"] = False
    negative["dataset_split_role"] = "negative"
    return negative


def build_balanced_dataset(write_outputs: bool = True) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    BALANCING_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    master, segment_universe, storm_summary = augment_master_table_for_balancing(write_outputs=True)
    storm_positive_segments = build_storm_positive_segment_sets(master)
    unreported, unreported_validation = build_possible_unreported_candidates(
        master,
        segment_universe,
        storm_summary,
        storm_positive_segments=storm_positive_segments,
        write_outputs=True,
    )
    negative_df, sampling_diagnostics = build_negative_samples(master, segment_universe, unreported)
    negatives = enrich_negative_events(negative_df, master, segment_universe)

    positives = master.copy()
    positives["event_id"] = positives["event_id"].astype("string")
    positives["matched_positive_event_id"] = positives["event_id"]
    positives["negative_sampling_level"] = pd.NA
    positives["negative_sampling_relaxed"] = pd.NA
    positives["negative_sampling_notes"] = pd.NA
    positives["negative_match_score"] = pd.NA
    positives["negative_source_segment_id"] = pd.NA
    positives["dataset_split_role"] = "positive"

    balanced = pd.concat([positives, negatives], ignore_index=True, sort=False)
    balanced = gpd.GeoDataFrame(balanced, geometry="geometry", crs=master.crs)
    for column in [
        "event_id",
        "matched_positive_event_id",
        "segment_id",
        "segment_borough",
        "station_p_id",
        "tide_id",
        "precip_polygon_id",
        "tide_polygon_id",
        "label_definition",
        "negative_sampling_notes",
        "negative_source_segment_id",
        "dataset_split_role",
        "storm_event_id",
    ]:
        if column in balanced.columns:
            balanced[column] = balanced[column].astype("string")
    if "negative_sampling_level" in balanced.columns:
        balanced["negative_sampling_level"] = pd.to_numeric(balanced["negative_sampling_level"], errors="coerce").astype("Int64")
    if "negative_sampling_relaxed" in balanced.columns:
        balanced["negative_sampling_relaxed"] = balanced["negative_sampling_relaxed"].astype("boolean")

    distribution_rows: list[dict[str, object]] = []
    compare_columns = [
        "segment_borough",
        "tide_polygon_id",
        "precip_polygon_id",
        "road_class",
        "fema_fld_zone",
        "elevation_band",
        "shore_dist_band",
    ]
    for column in compare_columns:
        if column not in balanced.columns:
            continue
        summary = (
            balanced.groupby(["dataset_split_role", column], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        summary["feature"] = column
        distribution_rows.extend(summary.to_dict("records"))
    distribution_df = pd.DataFrame(distribution_rows)

    numeric_distribution_rows: list[dict[str, object]] = []
    numeric_compare_columns = [
        "dem_mean",
        "shore_graph_steps",
        "prec_depth_total",
        "prec_intensity_max",
        "prec_intensity_mean",
        "tide_level_m_max",
        "segment_edge_betweenness",
        "node_total_degree_mean",
    ]
    for column in numeric_compare_columns:
        if column not in balanced.columns:
            continue
        summary = (
            balanced.groupby("dataset_split_role", dropna=False)[column]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        summary["feature"] = column
        numeric_distribution_rows.extend(summary.to_dict("records"))
    numeric_distribution_df = pd.DataFrame(numeric_distribution_rows)

    sampling_level_distribution = (
        sampling_diagnostics["negative_sampling_level"]
        .value_counts(dropna=False)
        .rename_axis("negative_sampling_level")
        .reset_index(name="count")
    )

    unreported_confidence_distribution = (
        unreported["unreported_confidence_level"]
        .value_counts(dropna=False)
        .rename_axis("unreported_confidence_level")
        .reset_index(name="count")
        if not unreported.empty
        else pd.DataFrame(columns=["unreported_confidence_level", "count"])
    )
    unreported_by_borough = (
        unreported["segment_borough"]
        .value_counts(dropna=False)
        .rename_axis("segment_borough")
        .reset_index(name="count")
        if not unreported.empty
        else pd.DataFrame(columns=["segment_borough", "count"])
    )
    unreported_by_tide = (
        unreported["tide_id"]
        .value_counts(dropna=False)
        .rename_axis("tide_id")
        .reset_index(name="count")
        if not unreported.empty
        else pd.DataFrame(columns=["tide_id", "count"])
    )
    unreported_by_precip = (
        unreported["station_p_id"]
        .value_counts(dropna=False)
        .rename_axis("station_p_id")
        .reset_index(name="count")
        if not unreported.empty
        else pd.DataFrame(columns=["station_p_id", "count"])
    )

    validation_balancing = pd.DataFrame(
        [
            {
                "n_positive_events": int(len(positives)),
                "n_negative_events": int(len(negatives)),
                "n_failed_negative_samples": int((~sampling_diagnostics["sampling_success"]).sum()),
                "balance_ratio": float(len(negatives) / len(positives)) if len(positives) else math.nan,
                "candidate_pool_size_before_filtering": float(sampling_diagnostics["candidate_pool_size_before_filtering"].mean()),
                "candidate_pool_size_after_filtering": float(sampling_diagnostics["candidate_pool_size_after_filtering"].mean()),
                "excluded_by_same_segment": int(sampling_diagnostics["excluded_by_same_segment"].sum()),
                "negative_sampling_relaxed_count": int(sampling_diagnostics["negative_sampling_relaxed"].fillna(False).sum()),
                "excluded_by_temporal_overlap": int(sampling_diagnostics["excluded_by_temporal_overlap"].sum()),
                "excluded_by_spatial_buffer": int(sampling_diagnostics["excluded_by_spatial_buffer"].sum()),
                "excluded_by_spatiotemporal_buffer": int(sampling_diagnostics["excluded_by_spatiotemporal_buffer"].sum()),
                "excluded_by_possible_unreported": int(sampling_diagnostics["excluded_by_possible_unreported"].sum()),
                "duplicate_negative_event_count": int(negatives["event_id"].duplicated().sum()),
                "duplicate_segment_time_count": int(negatives.duplicated(subset=["segment_id", "event_start_local", "event_window_end_local"]).sum()),
            }
        ]
    )

    positive_segment_lookup = master[["event_id", "segment_id"]].rename(columns={"event_id": "matched_positive_event_id", "segment_id": "positive_segment_id"})
    negatives = negatives.merge(positive_segment_lookup, on="matched_positive_event_id", how="left", validate="many_to_one")
    temporal_index = segment_event_windows(master)

    leakage_checks = pd.DataFrame(
        [
            {
                "check": "negative_same_segment_as_positive",
                "value": int((negatives["segment_id"].astype(str) == negatives["positive_segment_id"].astype(str)).sum()),
            },
            {
                "check": "negative_temporal_conflict_with_observed",
                "value": int(
                    negatives.apply(
                        lambda row: has_temporal_conflict(
                            temporal_index,
                            str(row["segment_id"]),
                            row["event_start_local"],
                            row["event_window_end_local"],
                            buffer_hours=TEMPORAL_EXCLUSION_HOURS,
                        ),
                        axis=1,
                    ).sum()
                ) if not negatives.empty else 0,
            },
        ]
    )

    if write_outputs:
        pd.DataFrame(master.drop(columns="geometry")).to_parquet(MASTER_PARQUET_PATH, index=False)
        master.to_parquet(MASTER_GEOPARQUET_PATH, index=False)
        pd.DataFrame(balanced.drop(columns="geometry")).to_parquet(BALANCED_PARQUET_PATH, index=False)
        balanced.to_parquet(BALANCED_GEOPARQUET_PATH, index=False)
        validation_balancing.to_csv(VALIDATION_BALANCING_PATH, index=False)
        unreported_validation.to_csv(VALIDATION_UNREPORTED_PATH, index=False)
        sampling_diagnostics.to_csv(BALANCING_DIAGNOSTICS_DIR / "negative_sampling_event_diagnostics.csv", index=False)
        distribution_df.to_csv(BALANCING_DIAGNOSTICS_DIR / "positive_negative_distribution_checks.csv", index=False)
        numeric_distribution_df.to_csv(BALANCING_DIAGNOSTICS_DIR / "positive_negative_numeric_distribution_checks.csv", index=False)
        sampling_level_distribution.to_csv(BALANCING_DIAGNOSTICS_DIR / "negative_sampling_level_distribution.csv", index=False)
        leakage_checks.to_csv(BALANCING_DIAGNOSTICS_DIR / "leakage_checks.csv", index=False)
        storm_summary.to_csv(BALANCING_DIAGNOSTICS_DIR / "storm_event_summary.csv", index=False)
        unreported_confidence_distribution.to_csv(BALANCING_DIAGNOSTICS_DIR / "unreported_confidence_distribution.csv", index=False)
        unreported_by_borough.to_csv(BALANCING_DIAGNOSTICS_DIR / "unreported_by_borough.csv", index=False)
        unreported_by_tide.to_csv(BALANCING_DIAGNOSTICS_DIR / "unreported_by_tide_polygon.csv", index=False)
        unreported_by_precip.to_csv(BALANCING_DIAGNOSTICS_DIR / "unreported_by_precip_polygon.csv", index=False)

        metadata = {
            "natural_rows": int(len(master)),
            "balanced_rows": int(len(balanced)),
            "positive_rows": int(len(positives)),
            "negative_rows": int(len(negatives)),
            "unreported_candidates": int(len(unreported)),
            "temporal_exclusion_hours": TEMPORAL_EXCLUSION_HOURS,
            "negative_spatial_exclusion_m": NEGATIVE_SPATIAL_EXCLUSION_M,
            "unreported_distance_m": UNREPORTED_DISTANCE_M,
            "resolution_policy_for_negatives": "NaN",
        }
        (BALANCING_DIAGNOSTICS_DIR / "balancing_metadata.json").write_text(json.dumps(metadata, indent=2))

    return balanced, unreported, validation_balancing, sampling_diagnostics
