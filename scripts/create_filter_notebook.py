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

for col in balanced_df.columns:
    if col not in possible_event_like.columns:
        possible_event_like[col] = pd.NA

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
            """## Final Dataset to Analyze

Edit:
- `SELECTED_VIEW_NAME`
- `FIELDS_TO_ANALYZE`
- `DROP_ROWS_WITH_MISSING_SELECTED_FIELDS`

This last step adds:
- `has_<field>` for every selected field
- `has_all_selected_fields`

and saves the version you really want to analyze."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """VIEW_LOOKUP = {
    "strict_main": strict_main_df,
    "main_plus_high_conf_possible": main_plus_high_conf_possible_df,
    "semi_supervised_pool": semi_supervised_pool_df,
}

SELECTED_VIEW_NAME = "strict_main"
FIELDS_TO_ANALYZE = [
    "segment_id",
    "segment_borough",
    "event_start_local",
    "event_window_duration_hours",
    "dem_mean",
    "shore_graph_steps",
    "fema_fld_zone",
    "census_median_household_income",
    "prec_depth_total",
    "tide_level_m_max",
    "component_size",
    "catch_basin_nearest_ft",
]
DROP_ROWS_WITH_MISSING_SELECTED_FIELDS = False

def has_value(series: pd.Series) -> pd.Series:
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        return series.notna() & series.astype("string").str.strip().ne("")
    return series.notna()

analysis_df = VIEW_LOOKUP[SELECTED_VIEW_NAME].copy()
analysis_df["analysis_view_name"] = SELECTED_VIEW_NAME

has_columns = []
for field in FIELDS_TO_ANALYZE:
    has_col = f"has_{field}"
    if field in analysis_df.columns:
        analysis_df[has_col] = has_value(analysis_df[field]).astype("boolean")
    else:
        analysis_df[has_col] = False
    has_columns.append(has_col)

analysis_df["has_all_selected_fields"] = analysis_df[has_columns].fillna(False).all(axis=1)

if DROP_ROWS_WITH_MISSING_SELECTED_FIELDS:
    analysis_df = analysis_df.loc[analysis_df["has_all_selected_fields"]].copy()

FINAL_ANALYSIS_PATH = FILTERED_DIR / f"final_analysis_{SELECTED_VIEW_NAME}.parquet"
analysis_df.to_parquet(FINAL_ANALYSIS_PATH, index=False)

display(analysis_df[["event_id", "segment_id", "occurrence", "possible_unreported_flood", "analysis_view_name", "has_all_selected_fields"] + has_columns].head(20))
print(FINAL_ANALYSIS_PATH)
print(f"rows saved: {len(analysis_df):,}")"""
        )
    )

    nb["cells"] = cells
    NOTEBOOK_PATH.write_text(nbf.writes(nb))


if __name__ == "__main__":
    main()
