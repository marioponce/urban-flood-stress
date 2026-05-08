from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.flood_events_master import ensure_proj_env  # noqa: E402


CITY_TZ = "America/New_York"
LOCAL_311_DIR = ROOT / "data" / "temporal" / "311"
STREETS_PATH = ROOT / "data" / "spatial" / "vector" / "streets" / "processed" / "lion_metrics.gpkg"
INDIVIDUAL_OUTPUT = ROOT / "data" / "processed" / "311" / "individual_complaints.csv"
EVENTS_OUTPUT = ROOT / "data" / "processed" / "311" / "flood_events.csv"

FLOOD_DESCRIPTOR_PATTERNS = [
    "STREET FLOODING (SJ)",
    "FLOODING ON STREET",
    "HIGHWAY FLOODING (SH)",
]
SPATIAL_DISTANCE_THRESHOLD_FT = 150.0

BOROUGH_CODE_MAP = {
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
    "BKLYN": "BROOKLYN",
    "QN": "QUEENS",
    "SI": "STATEN ISLAND",
    "R": "STATEN ISLAND",
    "KINGS": "BROOKLYN",
    "KINGS COUNTY": "BROOKLYN",
}


def normalize_segment_id_value(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    digits = re.sub(r"\D", "", text)
    if not digits:
        return pd.NA
    return digits.lstrip("0") or "0"


def normalize_segment_id_series(series: pd.Series) -> pd.Series:
    return pd.Series(series.map(normalize_segment_id_value), index=series.index, dtype="string")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(CITY_TZ)


def clean_text(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    text = text.replace("&", " AND ")
    text = re.sub(r"[\.,;/_@\-\(\)]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    replacements = {
        " ST ": " STREET ",
        " AVE ": " AVENUE ",
        " BLVD ": " BOULEVARD ",
        " RD ": " ROAD ",
        " DR ": " DRIVE ",
        " PL ": " PLACE ",
        " LN ": " LANE ",
    }
    padded = f" {text} "
    for src, dst in replacements.items():
        padded = padded.replace(src, dst)
    return re.sub(r"\s+", " ", padded).strip()


def normalize_borough(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = clean_text(value)
    if pd.isna(text):
        return pd.NA
    return BOROUGH_CODE_MAP.get(str(text), str(text))


def load_citywide_street_flooding_raw() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = [
        "unique_key",
        "created_date",
        "closed_date",
        "status",
        "borough",
        "street_name",
        "latitude",
        "longitude",
        "complaint_type",
        "descriptor",
    ]

    for path in sorted(LOCAL_311_DIR.glob("311_*/*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        descriptor = frame["descriptor"].astype("string").fillna("").str.upper()
        mask = False
        for pattern in FLOOD_DESCRIPTOR_PATTERNS:
            mask = mask | descriptor.eq(pattern)

        subset = frame.loc[mask].copy()
        if subset.empty:
            continue

        subset["source_id"] = path.stem.split("_")[0]
        subset["source_window"] = "_".join(path.stem.split("_")[1:3])
        frames.append(subset)

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns + ["source_id", "source_window"])
    raw = raw.drop_duplicates(subset=["unique_key"], keep="last").reset_index(drop=True)
    raw["created_date"] = parse_timestamp_series(raw["created_date"])
    raw["closed_date"] = parse_timestamp_series(raw["closed_date"])
    raw["status_text"] = raw["status"].astype("string")
    raw["borough_normalized"] = raw["borough"].map(normalize_borough).astype("string")
    raw["normalized_street_name"] = raw["street_name"].map(clean_text).astype("string")
    raw["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce")
    raw["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce")
    raw["has_valid_coordinates"] = raw["latitude"].between(40.45, 40.95) & raw["longitude"].between(-74.3, -73.65)
    valid_temporal = raw["closed_date"].isna() | raw["created_date"].isna() | raw["closed_date"].ge(raw["created_date"])
    raw = raw.loc[valid_temporal].reset_index(drop=True)
    return raw


def load_lion_reference() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_proj_env()
    lion = gpd.read_file(STREETS_PATH)
    lion["segment_id"] = normalize_segment_id_series(lion["SegmentID"])
    lion = lion[lion["segment_id"].notna()].copy()
    lion = lion.sort_values(["segment_id"], kind="stable").drop_duplicates("segment_id").reset_index(drop=True)

    lion["normalized_street_name"] = lion["Street"].map(clean_text).astype("string")
    lion["borough_left"] = lion["LBoro"].astype("string").map(normalize_borough).astype("string")
    lion["borough_right"] = lion["RBoro"].astype("string").map(normalize_borough).astype("string")

    name_col = "normalized_street_name"
    text_candidates = pd.concat(
        [
            lion[["segment_id", name_col, "borough_left"]].rename(columns={"borough_left": "borough"}),
            lion[["segment_id", name_col, "borough_right"]].rename(columns={"borough_right": "borough"}),
        ],
        ignore_index=True,
    ).dropna(subset=[name_col, "borough"])

    borough_lookup = (
        text_candidates.groupby([name_col, "borough"], as_index=False)
        .agg(
            n_segments=("segment_id", "nunique"),
            segment_id=("segment_id", lambda s: s.iloc[0] if s.nunique() == 1 else pd.NA),
        )
    )
    borough_lookup = borough_lookup[borough_lookup["segment_id"].notna()].copy()

    global_lookup = (
        lion.groupby(name_col, as_index=False)
        .agg(
            n_segments=("segment_id", "nunique"),
            segment_id=("segment_id", lambda s: s.iloc[0] if s.nunique() == 1 else pd.NA),
        )
    )
    global_lookup = global_lookup[global_lookup["segment_id"].notna()].copy()
    return lion, borough_lookup, global_lookup


def spatial_match(raw: pd.DataFrame, lion: gpd.GeoDataFrame) -> pd.DataFrame:
    points = raw[raw["has_valid_coordinates"]].copy()
    gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points["longitude"], points["latitude"]),
        crs="EPSG:4326",
    ).to_crs(lion.crs)

    nearest = gpd.sjoin_nearest(
        gdf[["unique_key", "geometry"]],
        lion[["segment_id", "geometry"]],
        how="left",
        distance_col="match_distance",
    )
    nearest = nearest[["unique_key", "segment_id", "match_distance"]].dropna(subset=["segment_id"]).copy()
    nearest = nearest.sort_values(["unique_key", "match_distance", "segment_id"], kind="stable").drop_duplicates("unique_key")
    nearest["match_method"] = "spatial_nearest"

    far_mask = nearest["match_distance"].gt(SPATIAL_DISTANCE_THRESHOLD_FT)
    nearest.loc[far_mask, ["segment_id", "match_method"]] = [pd.NA, pd.NA]
    return nearest


def textual_match(raw: pd.DataFrame, borough_lookup: pd.DataFrame, global_lookup: pd.DataFrame) -> pd.DataFrame:
    out = raw[["unique_key", "normalized_street_name", "borough_normalized"]].copy()
    borough_join = out.merge(
        borough_lookup.rename(columns={"borough": "borough_normalized"}),
        on=["normalized_street_name", "borough_normalized"],
        how="left",
    )
    borough_join["match_method"] = np.where(
        borough_join["segment_id"].notna(),
        "textual_borough_unique",
        pd.NA,
    )

    global_join = out.merge(global_lookup, on="normalized_street_name", how="left")
    global_join["match_method"] = np.where(global_join["segment_id"].notna(), "textual_global_unique", pd.NA)

    merged = borough_join[["unique_key", "segment_id", "match_method"]].copy()
    needs_global = merged["segment_id"].isna()
    merged.loc[needs_global, ["segment_id", "match_method"]] = global_join.loc[needs_global, ["segment_id", "match_method"]].to_numpy()
    merged["match_distance"] = np.nan
    return merged


def prepare_individual_complaints(raw: pd.DataFrame, lion: gpd.GeoDataFrame, borough_lookup: pd.DataFrame, global_lookup: pd.DataFrame) -> pd.DataFrame:
    spatial = spatial_match(raw, lion)
    textual = textual_match(raw, borough_lookup, global_lookup)

    matched = raw.merge(
        spatial,
        on="unique_key",
        how="left",
        suffixes=("", "_spatial"),
    )

    needs_text = matched["segment_id"].isna()
    matched.loc[needs_text, "segment_id"] = textual.loc[needs_text, "segment_id"].to_numpy()
    matched.loc[needs_text, "match_method"] = textual.loc[needs_text, "match_method"].to_numpy()
    matched.loc[needs_text, "match_distance"] = textual.loc[needs_text, "match_distance"].to_numpy()

    matched["segment_id"] = normalize_segment_id_series(matched["segment_id"])
    matched["status"] = np.where(matched["closed_date"].notna(), 1, 0).astype("int8")
    matched["complaint_id"] = np.arange(1, len(matched) + 1, dtype="int64")
    matched["source_complaint_id"] = matched["unique_key"].astype("string")
    matched["start"] = matched["created_date"]
    matched["end"] = matched["closed_date"]
    matched["borough"] = matched["borough_normalized"]
    matched["street_name_raw"] = matched["street_name"].astype("string")
    matched["coordinate_source"] = "latlon"

    out = matched[
        [
            "complaint_id",
            "source_complaint_id",
            "start",
            "end",
            "segment_id",
            "status",
            "match_method",
            "match_distance",
            "normalized_street_name",
            "borough_normalized",
            "borough",
            "street_name_raw",
            "status_text",
            "coordinate_source",
            "has_valid_coordinates",
            "latitude",
            "longitude",
            "source_id",
            "source_window",
        ]
    ].copy()

    out = out.sort_values(["start", "complaint_id"], kind="stable").reset_index(drop=True)
    out["complaint_id"] = np.arange(1, len(out) + 1, dtype="int64")
    return out


def merge_segment_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["segment_id", "start", "end", "duration_hours", "n_complaints", "status"]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    group = frame.copy()
    group["start"] = pd.to_datetime(group["start"], errors="coerce")
    group["end"] = pd.to_datetime(group["end"], errors="coerce")
    group = group.dropna(subset=["start"]).sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    if group.empty:
        return pd.DataFrame(columns=output_columns)

    start_ns = group["start"].astype("int64").to_numpy()
    end_ns = group["end"].astype("int64").to_numpy()
    open_ended = group["end"].isna().to_numpy()
    end_for_merging = np.where(open_ended, np.iinfo(np.int64).max, end_ns)

    running_max = np.maximum.accumulate(end_for_merging)
    previous_max = np.empty_like(running_max)
    previous_max[0] = np.iinfo(np.int64).min
    previous_max[1:] = running_max[:-1]
    new_event = start_ns > previous_max
    group["local_event_id"] = new_event.cumsum() - 1

    aggregated = (
        group.groupby("local_event_id", sort=True)
        .agg(
            segment_id=("segment_id", "first"),
            start=("start", "min"),
            end=("end", lambda s: pd.NaT if s.isna().any() else s.max()),
            n_complaints=("segment_id", "size"),
            n_open_complaints=("status", lambda s: int((s == 0).sum())),
            n_closed_complaints=("status", lambda s: int((s == 1).sum())),
        )
        .reset_index(drop=True)
    )

    aggregated["start"] = pd.to_datetime(aggregated["start"], errors="coerce", utc=True).dt.tz_convert(CITY_TZ)
    aggregated["end"] = pd.to_datetime(aggregated["end"], errors="coerce", utc=True).dt.tz_convert(CITY_TZ)
    aggregated["duration_hours"] = (
        aggregated["end"] - aggregated["start"]
    ).dt.total_seconds() / 3600.0
    aggregated.loc[aggregated["end"].isna(), "duration_hours"] = pd.NA
    aggregated["status"] = np.select(
        [
            aggregated["n_closed_complaints"].eq(0),
            aggregated["n_open_complaints"].eq(0),
        ],
        [0, 1],
        default=2,
    ).astype("int8")
    return aggregated[output_columns]


def prepare_flood_events(individual: pd.DataFrame) -> pd.DataFrame:
    usable = individual[individual["segment_id"].notna()].copy()
    usable = usable.sort_values(["segment_id", "start"], kind="stable").reset_index(drop=True)
    parts = []
    for _, group in usable.groupby("segment_id", sort=False):
        merged = merge_segment_intervals(group)
        if not merged.empty:
            parts.append(merged)

    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["segment_id", "start", "end", "duration_hours", "n_complaints", "status"])
    events = events.sort_values(["segment_id", "start"], kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1, dtype="int64")
    return events[["event_id", "start", "end", "duration_hours", "segment_id", "n_complaints", "status"]].copy()


def main() -> None:
    raw = load_citywide_street_flooding_raw()
    lion, borough_lookup, global_lookup = load_lion_reference()
    individual = prepare_individual_complaints(raw, lion, borough_lookup, global_lookup)
    events = prepare_flood_events(individual)

    INDIVIDUAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    individual.to_csv(INDIVIDUAL_OUTPUT, index=False)
    events.to_csv(EVENTS_OUTPUT, index=False)

    print(f"raw_rows={len(raw)}")
    print(f"individual_rows={len(individual)}")
    print(f"event_rows={len(events)}")
    print("borough_counts:")
    print(individual["borough"].fillna("NA").value_counts().to_string())
    print("match_method_counts:")
    print(individual["match_method"].fillna("NA").value_counts().to_string())


if __name__ == "__main__":
    main()
