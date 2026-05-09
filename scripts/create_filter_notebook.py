from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "12_filter_.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Filter and Analysis Views

This notebook prepares the **analysis-ready views** of the flood-event modeling data.

It keeps three parallel products:
- `strict_main`: only observed positives + matched negatives
- `main_plus_high_conf_possible`: `strict_main` plus high-confidence possible unreported flood candidates
- `semi_supervised_pool`: `strict_main` plus all possible unreported candidates as weak-label rows

Important:
- `possible_unreported_flood` rows are **not confirmed positives**
- they should not be mixed into standard supervised training unless you explicitly want a weak-label experiment
- the last section builds `has_<field>` indicators for the dataset you actually want to analyze"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODELING_DIR = ROOT / "data" / "processed" / "modeling"
FILTERED_DIR = MODELING_DIR / "filtered"
FILTERED_DIR.mkdir(parents=True, exist_ok=True)

BALANCED_PATH = MODELING_DIR / "flood_events_balanced.parquet"
NATURAL_PATH = MODELING_DIR / "flood_events_master_table.parquet"
POSSIBLE_PATH = MODELING_DIR / "possible_unreported_flood_candidates.parquet"

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """balanced_df = pd.read_parquet(BALANCED_PATH)
natural_df = pd.read_parquet(NATURAL_PATH)
possible_df = pd.read_parquet(POSSIBLE_PATH)

balanced_df["event_id"] = balanced_df["event_id"].astype("string")
balanced_df["segment_id"] = balanced_df["segment_id"].astype("string")
balanced_df["dataset_split_role"] = balanced_df["dataset_split_role"].astype("string")
balanced_df["label_definition"] = balanced_df["label_definition"].astype("string")

natural_df["event_id"] = natural_df["event_id"].astype("string")
natural_df["segment_id"] = natural_df["segment_id"].astype("string")

possible_df["candidate_segment_id"] = possible_df["candidate_segment_id"].astype("string")
possible_df["nearest_reported_event_id"] = possible_df["nearest_reported_event_id"].astype("string")
possible_df["reported_segment_id"] = possible_df["reported_segment_id"].astype("string")
possible_df["storm_event_id"] = possible_df["storm_event_id"].astype("string")

print(f"balanced rows: {len(balanced_df):,}")
print(f"natural rows: {len(natural_df):,}")
print(f"possible rows: {len(possible_df):,}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Core Summaries"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """summary = pd.DataFrame(
    [
        {
            "dataset": "strict_main_source",
            "rows": len(balanced_df),
            "positive_rows": int((balanced_df["occurrence"] == True).sum()),  # noqa: E712
            "negative_rows": int((balanced_df["occurrence"] == False).sum()),  # noqa: E712
            "possible_rows": 0,
        },
        {
            "dataset": "possible_unreported_source",
            "rows": len(possible_df),
            "positive_rows": 0,
            "negative_rows": 0,
            "possible_rows": len(possible_df),
        },
    ]
)

display(summary)
display(possible_df["unreported_confidence_level"].value_counts(dropna=False).rename_axis("unreported_confidence_level").reset_index(name="count"))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Build Weak-Label Possible Rows

These rows borrow:
- the event window from the nearest reported event
- static street/segment attributes from the candidate segment

They do **not** become confirmed positives.
They stay marked with:
- `possible_unreported_flood = True`
- `weak_occurrence = True`
- `include_in_supervised_training = False`"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """EVENT_TEMPLATE_COLS = [
    "event_start_local",
    "event_end_local",
    "event_window_end_local",
    "event_window_duration_hours",
    "event_end_observed_local",
    "event_end_inferred",
    "event_year",
    "event_month",
    "event_date",
    "storm_event_id",
    "storm_event_positive_count",
    "storm_event_duration_hours",
    "prec_depth_total",
    "prec_duration_total",
    "prec_intensity_max",
    "prec_intensity_mean",
    "n_prec",
    "tide_obs_n",
    "tide_level_m_mean",
    "tide_level_m_max",
    "tide_level_m_min",
    "tide_level_m_range",
]

STATIC_PRIORITY_COLS = [
    "segment_id",
    "segment_borough",
    "station_p_id",
    "tide_id",
    "precip_polygon_id",
    "tide_polygon_id",
    "road_class",
    "street_name",
    "street_name_safe",
    "street_feature_type",
    "street_segment_type",
    "street_traffic_direction",
    "segment_length_geom_ft",
    "segment_speed_mph",
    "segment_travel_time_s",
    "dem_min",
    "dem_max",
    "dem_mean",
    "dem_sd",
    "dem_start",
    "dem_end",
    "dem_slope",
    "component_id",
    "component_size",
    "in_largest_component",
    "is_bridge_segment",
    "shore_id",
    "shore_graph_steps",
    "shore_dist_band",
    "elevation_band",
    "fema_fld_zone",
    "fema_zone_subty",
    "fema_overlap_ft",
    "fema_overlap_share",
    "fema_sfha_any",
    "census_geoid",
    "census_borough",
    "census_poverty_rate",
    "census_renter_share",
    "census_no_vehicle_share",
    "census_median_household_income",
    "catch_basin_nearest_ft",
    "catch_basin_count_100ft",
    "catch_basin_count_250ft",
    "outfall_nearest_ft",
    "outfall_count_250ft",
    "outfall_count_500ft",
]

segment_static_lookup = (
    balanced_df.sort_values(["segment_id", "dataset_split_role", "occurrence"], kind="stable")
    .groupby("segment_id", as_index=False)
    .first()
)

event_template_lookup = natural_df[["event_id"] + [c for c in EVENT_TEMPLATE_COLS if c in natural_df.columns]].copy()

possible_event_like = possible_df.rename(columns={"candidate_segment_id": "segment_id"}).copy()
possible_event_like = possible_event_like.merge(
    event_template_lookup.rename(columns={"event_id": "nearest_reported_event_id"}),
    on="nearest_reported_event_id",
    how="left",
    validate="many_to_one",
)
possible_event_like = possible_event_like.merge(
    segment_static_lookup[[c for c in STATIC_PRIORITY_COLS if c in segment_static_lookup.columns]],
    on="segment_id",
    how="left",
    validate="many_to_one",
)

possible_event_like = possible_event_like.sort_values(
    ["nearest_reported_event_id", "distance_to_reported_event_m", "segment_id"],
    kind="stable",
).reset_index(drop=True)

possible_event_like["event_id"] = (
    "POSSIBLE_"
    + possible_event_like["nearest_reported_event_id"].astype("string")
    + "_"
    + (possible_event_like.groupby("nearest_reported_event_id").cumcount() + 1).astype(str).str.zfill(4)
)
possible_event_like["start"] = possible_event_like["event_start_local"]
possible_event_like["end"] = pd.NaT
possible_event_like["duration_hours"] = np.nan
possible_event_like["resolution"] = np.nan
possible_event_like["resolution_hours"] = np.nan
possible_event_like["resolution_bool"] = False
possible_event_like["occurrence"] = pd.NA
possible_event_like["weak_occurrence"] = True
possible_event_like["intensity"] = 0
possible_event_like["n_complaints"] = 0
possible_event_like["status"] = 0
possible_event_like["flood_event_status"] = pd.NA
possible_event_like["dataset_split_role"] = "possible_candidate"
possible_event_like["label_definition"] = "possible_unreported_flood_candidate"
possible_event_like["include_in_supervised_training"] = False
possible_event_like["label_strength"] = possible_event_like["unreported_confidence_level"].map(
    {"high": "weak_high", "medium": "weak_medium", "low": "weak_low"}
)

missing_cols = [col for col in balanced_df.columns if col not in possible_event_like.columns]
if missing_cols:
    possible_event_like = pd.concat(
        [
            possible_event_like,
            pd.DataFrame({col: pd.NA for col in missing_cols}, index=possible_event_like.index),
        ],
        axis=1,
    )

possible_event_like = possible_event_like[[col for col in balanced_df.columns] + [
    "possible_unreported_flood",
    "unreported_confidence_level",
    "distance_to_reported_event_m",
    "time_delta_to_reported_event_hours",
    "nearest_reported_event_id",
    "reported_segment_id",
    "same_tide_polygon",
    "same_precip_polygon",
    "same_borough",
    "weak_occurrence",
    "include_in_supervised_training",
    "label_strength",
]]

display(possible_event_like.head())"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Analysis Views"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """strict_main_df = balanced_df.copy()
strict_main_df["possible_unreported_flood"] = False
strict_main_df["weak_occurrence"] = strict_main_df["occurrence"]
strict_main_df["include_in_supervised_training"] = True
strict_main_df["label_strength"] = np.where(
    strict_main_df["occurrence"].eq(True),
    "observed_positive",
    "matched_negative",
)

high_conf_possible_df = possible_event_like.loc[
    possible_event_like["unreported_confidence_level"].eq("high")
].copy()

main_plus_high_conf_possible_df = pd.concat(
    [strict_main_df, high_conf_possible_df],
    ignore_index=True,
    sort=False,
)

semi_supervised_pool_df = pd.concat(
    [strict_main_df, possible_event_like],
    ignore_index=True,
    sort=False,
)

views_summary = pd.DataFrame(
    [
        {
            "view_name": "strict_main",
            "rows": len(strict_main_df),
            "observed_positive": int(strict_main_df["occurrence"].eq(True).sum()),
            "matched_negative": int(strict_main_df["occurrence"].eq(False).sum()),
            "possible_rows": int(strict_main_df["possible_unreported_flood"].fillna(False).sum()),
        },
        {
            "view_name": "main_plus_high_conf_possible",
            "rows": len(main_plus_high_conf_possible_df),
            "observed_positive": int(main_plus_high_conf_possible_df["occurrence"].eq(True).sum()),
            "matched_negative": int(main_plus_high_conf_possible_df["occurrence"].eq(False).sum()),
            "possible_rows": int(main_plus_high_conf_possible_df["possible_unreported_flood"].fillna(False).sum()),
        },
        {
            "view_name": "semi_supervised_pool",
            "rows": len(semi_supervised_pool_df),
            "observed_positive": int(semi_supervised_pool_df["occurrence"].eq(True).sum()),
            "matched_negative": int(semi_supervised_pool_df["occurrence"].eq(False).sum()),
            "possible_rows": int(semi_supervised_pool_df["possible_unreported_flood"].fillna(False).sum()),
        },
    ]
)

display(views_summary)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """STRICT_MAIN_PATH = FILTERED_DIR / "strict_main.parquet"
MAIN_PLUS_HIGH_PATH = FILTERED_DIR / "main_plus_high_conf_possible.parquet"
SEMI_SUPERVISED_PATH = FILTERED_DIR / "semi_supervised_pool.parquet"

strict_main_df.to_parquet(STRICT_MAIN_PATH, index=False)
main_plus_high_conf_possible_df.to_parquet(MAIN_PLUS_HIGH_PATH, index=False)
semi_supervised_pool_df.to_parquet(SEMI_SUPERVISED_PATH, index=False)

print(STRICT_MAIN_PATH)
print(MAIN_PLUS_HIGH_PATH)
print(SEMI_SUPERVISED_PATH)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Canonical Variable Frame

This section maps the current pipeline outputs into the conceptual variable names you want to use for real analysis.

Rules:
- names are simplified and stable
- time variables are derived from the event window
- `resolution` is stored in **hours**
- `max_tide` is the event-window maximum tide level in **meters**
- `elevation` is mean street elevation
- `shore_dist` is the currently available shoreline distance field
- unavailable variables are kept as explicit columns with missing values so you can decide later whether to keep or drop them"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """VIEW_LOOKUP = {
    "strict_main": strict_main_df,
    "main_plus_high_conf_possible": main_plus_high_conf_possible_df,
    "semi_supervised_pool": semi_supervised_pool_df,
}

def month_to_season(series: pd.Series) -> pd.Series:
    mapping = {
        12: "DJF", 1: "DJF", 2: "DJF",
        3: "MAM", 4: "MAM", 5: "MAM",
        6: "JJA", 7: "JJA", 8: "JJA",
        9: "SON", 10: "SON", 11: "SON",
    }
    return pd.to_numeric(series, errors="coerce").map(mapping).astype("string")


SELECTED_VIEW_NAME = "strict_main"
analysis_source_df = VIEW_LOOKUP[SELECTED_VIEW_NAME].copy()

edge_betweenness_numeric = pd.to_numeric(analysis_source_df.get("segment_edge_betweenness"), errors="coerce")
edge_betweenness_threshold = edge_betweenness_numeric.quantile(0.95) if edge_betweenness_numeric.notna().any() else np.nan

analysis_feature_df = pd.DataFrame(
    {
        "event_id": analysis_source_df["event_id"].astype("string"),
        "segment_id": analysis_source_df["segment_id"].astype("string"),
        "analysis_view_name": SELECTED_VIEW_NAME,
        "dataset_split_role": analysis_source_df["dataset_split_role"].astype("string"),
        "label_definition": analysis_source_df["label_definition"].astype("string"),
        "possible_unreported_flood": analysis_source_df["possible_unreported_flood"].fillna(False).astype("boolean"),
        "occurrence": analysis_source_df["occurrence"],
        "intensity": pd.to_numeric(analysis_source_df["intensity"], errors="coerce"),
        "resolution": pd.to_numeric(analysis_source_df["resolution_hours"], errors="coerce"),
        "resolution_bool": analysis_source_df["resolution_bool"].astype("boolean"),
        "start": pd.to_datetime(analysis_source_df["event_start_local"], errors="coerce"),
        "end": pd.to_datetime(analysis_source_df["event_end_local"], errors="coerce"),
        "duration": pd.to_numeric(analysis_source_df["event_window_duration_hours"], errors="coerce"),
        "month": pd.to_numeric(analysis_source_df["event_month"], errors="coerce").astype("Int64"),
        "season": month_to_season(analysis_source_df["event_month"]),
        "hour": pd.to_datetime(analysis_source_df["event_start_local"], errors="coerce").dt.hour.astype("Int64"),
        "day_of_week": pd.to_datetime(analysis_source_df["event_start_local"], errors="coerce").dt.dayofweek.astype("Int64"),
        "storm_event_id": analysis_source_df["storm_event_id"].astype("string"),
        "n_prec": pd.to_numeric(analysis_source_df["n_prec"], errors="coerce"),
        "prec_intensity_max": pd.to_numeric(analysis_source_df["prec_intensity_max"], errors="coerce"),
        "prec_intensity_mean": pd.to_numeric(analysis_source_df["prec_intensity_mean"], errors="coerce"),
        "prec_depth_total": pd.to_numeric(analysis_source_df["prec_depth_total"], errors="coerce"),
        "prec_duration_total": pd.to_numeric(analysis_source_df["prec_duration_total"], errors="coerce"),
        "max_tide": pd.to_numeric(analysis_source_df["tide_level_m_max"], errors="coerce"),
        "elevation": pd.to_numeric(analysis_source_df["dem_mean"], errors="coerce"),
        "slope": pd.to_numeric(analysis_source_df["dem_slope"], errors="coerce"),
        "shore_dist": pd.to_numeric(analysis_source_df["shore_dist_ft"], errors="coerce"),
        "fema_fld_zone": analysis_source_df["fema_fld_zone"].astype("string"),
        "fema_zone_subty": analysis_source_df["fema_zone_subty"].astype("string"),
        "fema_overlap_ft": pd.to_numeric(analysis_source_df["fema_overlap_ft"], errors="coerce"),
        "fema_overlap_share": pd.to_numeric(analysis_source_df["fema_overlap_share"], errors="coerce"),
        "fema_sfha_any": analysis_source_df["fema_sfha_any"].astype("boolean"),
        "road_class": analysis_source_df["road_class"].astype("string"),
        "travel_time": pd.to_numeric(analysis_source_df["segment_travel_time_s"], errors="coerce"),
        "edge_betweenness": pd.to_numeric(analysis_source_df["segment_edge_betweenness"], errors="coerce"),
        "component_count_after_removal": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "giant_component_size_after_removal": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "giant_component_size_loss": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "additional_disconnected_node_pairs": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "pct_giant_component_loss": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "high_network_criticality": edge_betweenness_numeric.ge(edge_betweenness_threshold).astype("boolean"),
        "drainage_catch_basin_nearest_ft": pd.to_numeric(analysis_source_df["catch_basin_nearest_ft"], errors="coerce"),
        "drainage_catch_basin_count_100ft": pd.to_numeric(analysis_source_df["catch_basin_count_100ft"], errors="coerce"),
        "drainage_catch_basin_count_250ft": pd.to_numeric(analysis_source_df["catch_basin_count_250ft"], errors="coerce"),
        "outfall_nearest_ft": pd.to_numeric(analysis_source_df["outfall_nearest_ft"], errors="coerce"),
        "outfall_count_250ft": pd.to_numeric(analysis_source_df["outfall_count_250ft"], errors="coerce"),
        "outfall_count_500ft": pd.to_numeric(analysis_source_df["outfall_count_500ft"], errors="coerce"),
        "pump_nearest_ft": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "critical_infra_exposure": pd.Series(pd.NA, index=analysis_source_df.index, dtype="Float64"),
        "census_geoid": analysis_source_df["census_geoid"].astype("string"),
        "census_borough": analysis_source_df["census_borough"].astype("string"),
        "census_poverty_rate": pd.to_numeric(analysis_source_df["census_poverty_rate"], errors="coerce"),
        "census_renter_share": pd.to_numeric(analysis_source_df["census_renter_share"], errors="coerce"),
        "census_no_vehicle_share": pd.to_numeric(analysis_source_df["census_no_vehicle_share"], errors="coerce"),
        "census_median_household_income": pd.to_numeric(analysis_source_df["census_median_household_income"], errors="coerce"),
        "gov_city": pd.Series("NEW YORK CITY", index=analysis_source_df.index, dtype="string"),
        "gov_borough": analysis_source_df["segment_borough"].astype("string"),
        "borough": analysis_source_df["segment_borough"].astype("string"),
        "tide_polygon": analysis_source_df["tide_polygon_id"].astype("string"),
        "tide_station": analysis_source_df["tide_id"].astype("string"),
        "precipitation_polygon": analysis_source_df["precip_polygon_id"].astype("string"),
        "precipitation_station": analysis_source_df["station_p_id"].astype("string"),
    }
)

VARIABLE_GROUPS = {
    "targets": ["occurrence", "intensity", "resolution", "resolution_bool"],
    "event_time": ["start", "end", "duration", "month", "season", "hour", "day_of_week", "storm_event_id"],
    "hydrometeorology": ["n_prec", "prec_intensity_max", "prec_intensity_mean", "prec_depth_total", "prec_duration_total", "max_tide"],
    "terrain_coastal": ["elevation", "slope", "shore_dist", "fema_fld_zone", "fema_zone_subty", "fema_overlap_ft", "fema_overlap_share", "fema_sfha_any"],
    "network": ["road_class", "travel_time", "edge_betweenness", "component_count_after_removal", "giant_component_size_after_removal", "giant_component_size_loss", "additional_disconnected_node_pairs", "pct_giant_component_loss", "high_network_criticality"],
    "infrastructure": ["drainage_catch_basin_nearest_ft", "drainage_catch_basin_count_100ft", "drainage_catch_basin_count_250ft", "outfall_nearest_ft", "outfall_count_250ft", "outfall_count_500ft", "pump_nearest_ft", "critical_infra_exposure"],
    "socioeconomic": ["census_geoid", "census_borough", "census_poverty_rate", "census_renter_share", "census_no_vehicle_share", "census_median_household_income"],
    "governance": ["gov_city", "gov_borough"],
    "spatial_controls": ["borough", "tide_polygon", "tide_station", "precipitation_polygon", "precipitation_station"],
}

field_catalog_rows = []
for group_name, fields in VARIABLE_GROUPS.items():
    for field in fields:
        series = analysis_feature_df[field]
        availability = float(series.notna().mean()) if len(series) else np.nan
        field_catalog_rows.append(
            {
                "group": group_name,
                "field": field,
                "available_in_table": field in analysis_feature_df.columns,
                "non_null_share": availability,
            }
        )

field_catalog_df = pd.DataFrame(field_catalog_rows)
display(field_catalog_df)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Final Dataset to Analyze

Set the following booleans to choose the variables you really want to keep.

Convention:
- `has_<field> = True` means keep that variable in the saved dataset
- all are `True` by default
- if a variable exists but is currently unavailable in the pipeline, it will still appear and you can switch it off if it is not useful for the current analysis"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """has_occurrence = True
has_intensity = True
has_resolution = True
has_resolution_bool = True

has_start = True
has_end = True
has_duration = True
has_month = True
has_season = True
has_hour = True
has_day_of_week = True
has_storm_event_id = True

has_n_prec = True
has_prec_intensity_max = True
has_prec_intensity_mean = True
has_prec_depth_total = True
has_prec_duration_total = True
has_max_tide = True

has_elevation = True
has_slope = True
has_shore_dist = True
has_fema_fld_zone = True
has_fema_zone_subty = True
has_fema_overlap_ft = True
has_fema_overlap_share = True
has_fema_sfha_any = True

has_road_class = True
has_travel_time = True
has_edge_betweenness = True
has_component_count_after_removal = True
has_giant_component_size_after_removal = True
has_giant_component_size_loss = True
has_additional_disconnected_node_pairs = True
has_pct_giant_component_loss = True
has_high_network_criticality = True

has_drainage_catch_basin_nearest_ft = True
has_drainage_catch_basin_count_100ft = True
has_drainage_catch_basin_count_250ft = True
has_outfall_nearest_ft = True
has_outfall_count_250ft = True
has_outfall_count_500ft = True
has_pump_nearest_ft = True
has_critical_infra_exposure = True

has_census_geoid = True
has_census_borough = True
has_census_poverty_rate = True
has_census_renter_share = True
has_census_no_vehicle_share = True
has_census_median_household_income = True

has_gov_city = True
has_gov_borough = True

has_borough = True
has_tide_polygon = True
has_tide_station = True
has_precipitation_polygon = True
has_precipitation_station = True

DROP_ROWS_WITH_MISSING_SELECTED_FIELDS = False

FIELD_SELECTION = {
    "occurrence": has_occurrence,
    "intensity": has_intensity,
    "resolution": has_resolution,
    "resolution_bool": has_resolution_bool,
    "start": has_start,
    "end": has_end,
    "duration": has_duration,
    "month": has_month,
    "season": has_season,
    "hour": has_hour,
    "day_of_week": has_day_of_week,
    "storm_event_id": has_storm_event_id,
    "n_prec": has_n_prec,
    "prec_intensity_max": has_prec_intensity_max,
    "prec_intensity_mean": has_prec_intensity_mean,
    "prec_depth_total": has_prec_depth_total,
    "prec_duration_total": has_prec_duration_total,
    "max_tide": has_max_tide,
    "elevation": has_elevation,
    "slope": has_slope,
    "shore_dist": has_shore_dist,
    "fema_fld_zone": has_fema_fld_zone,
    "fema_zone_subty": has_fema_zone_subty,
    "fema_overlap_ft": has_fema_overlap_ft,
    "fema_overlap_share": has_fema_overlap_share,
    "fema_sfha_any": has_fema_sfha_any,
    "road_class": has_road_class,
    "travel_time": has_travel_time,
    "edge_betweenness": has_edge_betweenness,
    "component_count_after_removal": has_component_count_after_removal,
    "giant_component_size_after_removal": has_giant_component_size_after_removal,
    "giant_component_size_loss": has_giant_component_size_loss,
    "additional_disconnected_node_pairs": has_additional_disconnected_node_pairs,
    "pct_giant_component_loss": has_pct_giant_component_loss,
    "high_network_criticality": has_high_network_criticality,
    "drainage_catch_basin_nearest_ft": has_drainage_catch_basin_nearest_ft,
    "drainage_catch_basin_count_100ft": has_drainage_catch_basin_count_100ft,
    "drainage_catch_basin_count_250ft": has_drainage_catch_basin_count_250ft,
    "outfall_nearest_ft": has_outfall_nearest_ft,
    "outfall_count_250ft": has_outfall_count_250ft,
    "outfall_count_500ft": has_outfall_count_500ft,
    "pump_nearest_ft": has_pump_nearest_ft,
    "critical_infra_exposure": has_critical_infra_exposure,
    "census_geoid": has_census_geoid,
    "census_borough": has_census_borough,
    "census_poverty_rate": has_census_poverty_rate,
    "census_renter_share": has_census_renter_share,
    "census_no_vehicle_share": has_census_no_vehicle_share,
    "census_median_household_income": has_census_median_household_income,
    "gov_city": has_gov_city,
    "gov_borough": has_gov_borough,
    "borough": has_borough,
    "tide_polygon": has_tide_polygon,
    "tide_station": has_tide_station,
    "precipitation_polygon": has_precipitation_polygon,
    "precipitation_station": has_precipitation_station,
}

selected_fields = [field for field, keep in FIELD_SELECTION.items() if keep]
selection_summary = pd.DataFrame(
    {
        "field": list(FIELD_SELECTION.keys()),
        "selected": list(FIELD_SELECTION.values()),
    }
).merge(field_catalog_df, on="field", how="left")

analysis_df = analysis_feature_df.copy()
analysis_df["analysis_view_name"] = SELECTED_VIEW_NAME

mandatory_fields = [
    "event_id",
    "segment_id",
    "analysis_view_name",
    "dataset_split_role",
    "label_definition",
    "possible_unreported_flood",
]

def value_is_present(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.notna() & series.astype("string").str.strip().ne("")
    return series.notna()

present_columns = []
for field in selected_fields:
    present_col = f"present_{field}"
    analysis_df[present_col] = value_is_present(analysis_df[field]).astype("boolean")
    present_columns.append(present_col)

analysis_df["has_all_selected_fields"] = analysis_df[present_columns].fillna(False).all(axis=1)

if DROP_ROWS_WITH_MISSING_SELECTED_FIELDS:
    analysis_df = analysis_df.loc[analysis_df["has_all_selected_fields"]].copy()

final_columns = mandatory_fields + selected_fields + present_columns + ["has_all_selected_fields"]
final_columns = [col for col in final_columns if col in analysis_df.columns]
final_analysis_df = analysis_df[final_columns].copy()

FINAL_ANALYSIS_PATH = FILTERED_DIR / f"final_analysis_{SELECTED_VIEW_NAME}.parquet"
final_analysis_df.to_parquet(FINAL_ANALYSIS_PATH, index=False)

display(selection_summary)
display(final_analysis_df.head(20))
print(FINAL_ANALYSIS_PATH)
print(f"rows saved: {len(final_analysis_df):,}")"""
        )
    )

    nb["cells"] = cells
    NOTEBOOK_PATH.write_text(nbf.writes(nb))


if __name__ == "__main__":
    main()
