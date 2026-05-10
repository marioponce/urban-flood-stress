from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge, TweedieRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    train_test_split,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from project_name.flood_event_balancing import (
    BALANCED_GEOPARQUET_PATH,
    BALANCED_PARQUET_PATH,
    UNREPORTED_PARQUET_PATH,
)
from project_name.flood_events_master import MASTER_GEOPARQUET_PATH, MASTER_PARQUET_PATH


ROOT = Path(__file__).resolve().parents[2]
MODELING_DIR = ROOT / "data" / "processed" / "modeling"
FILTERED_DIR = MODELING_DIR / "filtered"
MODELING_DIAGNOSTICS_DIR = MODELING_DIR / "diagnostics" / "modeling_stage"
MODELING_CHECKPOINT_DIR = MODELING_DIAGNOSTICS_DIR / "checkpoints"
CLUSTERING_DIR = MODELING_DIR / "clustering"
ML_DIR = MODELING_DIR / "ml"
BAYESIAN_DIR = MODELING_DIR / "bayesian"

STRICT_MAIN_PATH = FILTERED_DIR / "final_analysis_strict_main.parquet"
MAIN_PLUS_POSSIBLE_PATH = FILTERED_DIR / "final_analysis_main_plus_high_conf_possible.parquet"
SEMI_SUPERVISED_PATH = FILTERED_DIR / "final_analysis_semi_supervised_pool.parquet"

CLUSTERING_ARCHETYPES_PATH = MODELING_DIR / "clustering_event_archetypes.parquet"
CLUSTERING_ARCHETYPES_GEOPARQUET_PATH = MODELING_DIR / "clustering_event_archetypes.geoparquet"
ANOMALY_SCORES_PATH = MODELING_DIR / "anomaly_event_scores.parquet"
CLUSTERING_MODEL_SELECTION_PATH = MODELING_DIR / "clustering_model_selection.csv"

OCCURRENCE_RESULTS_PATH = MODELING_DIR / "results_occurrence_ml.csv"
OCCURRENCE_PREDICTIONS_PATH = MODELING_DIR / "predictions_occurrence_ml.parquet"
OCCURRENCE_FEATURE_IMPORTANCE_PATH = MODELING_DIR / "feature_importance_occurrence.csv"
OCCURRENCE_BEST_MODEL_PATH = MODELING_DIR / "best_occurrence_model.joblib"
OCCURRENCE_LEAKAGE_AUDIT_PATH = MODELING_DIR / "leakage_audit_occurrence.csv"
OCCURRENCE_BIAS_DIAGNOSTICS_PATH = MODELING_DIR / "bias_diagnostics_occurrence.csv"
OCCURRENCE_CALIBRATION_PATH = MODELING_DIR / "calibration_occurrence_ml.csv"

INTENSITY_RESULTS_PATH = MODELING_DIR / "results_intensity_ml.csv"
INTENSITY_PREDICTIONS_PATH = MODELING_DIR / "predictions_intensity_ml.parquet"
INTENSITY_FEATURE_IMPORTANCE_PATH = MODELING_DIR / "feature_importance_intensity.csv"
INTENSITY_BEST_MODEL_PATH = MODELING_DIR / "best_intensity_model.joblib"
INTENSITY_LEAKAGE_AUDIT_PATH = MODELING_DIR / "leakage_audit_intensity.csv"
INTENSITY_BIAS_DIAGNOSTICS_PATH = MODELING_DIR / "bias_diagnostics_intensity.csv"

RESOLUTION_CLOSURE_RESULTS_PATH = MODELING_DIR / "results_resolution_closure_ml.csv"
RESOLUTION_TIME_RESULTS_PATH = MODELING_DIR / "results_resolution_time_ml.csv"
RESOLUTION_PREDICTIONS_PATH = MODELING_DIR / "predictions_resolution_ml.parquet"
RESOLUTION_FEATURE_IMPORTANCE_PATH = MODELING_DIR / "feature_importance_resolution.csv"
RESOLUTION_BEST_CLOSURE_MODEL_PATH = MODELING_DIR / "best_resolution_closure_model.joblib"
RESOLUTION_BEST_TIME_MODEL_PATH = MODELING_DIR / "best_resolution_time_model.joblib"
RESOLUTION_LEAKAGE_AUDIT_PATH = MODELING_DIR / "leakage_audit_resolution.csv"
RESOLUTION_BIAS_DIAGNOSTICS_PATH = MODELING_DIR / "bias_diagnostics_resolution.csv"

BAYESIAN_EDGES_PATH = MODELING_DIR / "bayesian_network_edges.csv"
BAYESIAN_MODEL_SELECTION_PATH = MODELING_DIR / "bayesian_network_model_selection.csv"
BAYESIAN_NODE_SUMMARY_PATH = MODELING_DIR / "bayesian_network_node_summary.csv"

RANDOM_STATE = 42
MAX_CLUSTER_SAMPLE = 12000
MAX_OCSVM_SAMPLE = 5000


EXPECTED_FEATURE_GROUPS: dict[str, list[str]] = {
    "targets": ["occurrence", "intensity", "resolution", "resolution_bool"],
    "event_time": ["start", "end", "duration", "month", "season", "hour", "day_of_week", "storm_event_id"],
    "hydrometeorology": [
        "n_prec",
        "prec_intensity_max",
        "prec_intensity_mean",
        "prec_depth_total",
        "prec_duration_total",
        "max_tide",
    ],
    "terrain_coastal": [
        "elevation",
        "slope",
        "shore_dist",
        "fema_fld_zone",
        "fema_zone_subty",
        "fema_overlap_ft",
        "fema_overlap_share",
        "fema_sfha_any",
    ],
    "network": [
        "road_class",
        "travel_time",
        "edge_betweenness",
        "component_count_after_removal",
        "giant_component_size_after_removal",
        "giant_component_size_loss",
        "additional_disconnected_node_pairs",
        "pct_giant_component_loss",
        "high_network_criticality",
    ],
    "infrastructure": [
        "drainage_catch_basin_nearest_ft",
        "outfall_nearest_ft",
        "pump_nearest_ft",
        "critical_infra_exposure",
    ],
    "socioeconomic": [
        "census_poverty_rate",
        "census_renter_share",
        "census_no_vehicle_share",
        "census_median_household_income",
    ],
    "governance": ["gov_city", "gov_borough", "mayoral_administration"],
    "spatial_controls": ["borough", "tide_polygon", "tide_station", "precipitation_polygon", "precipitation_station"],
}

FEMA_VARIABLES = ["fema_fld_zone", "fema_zone_subty", "fema_overlap_ft", "fema_overlap_share", "fema_sfha_any"]

RAW_IDENTIFIER_COLUMNS = [
    "event_id",
    "segment_id",
    "analysis_view_name",
    "dataset_split_role",
    "label_definition",
]

DERIVED_COMPLETENESS_COLUMNS = ["possible_unreported_flood", "has_all_selected_fields"]

LEAKAGE_ALWAYS_DROP = RAW_IDENTIFIER_COLUMNS + DERIVED_COMPLETENESS_COLUMNS
TECHNICAL_DUPLICATE_COLUMNS = [
    "event_start_local",
    "event_end_local",
    "event_window_end_local",
    "event_window_duration_hours",
    "event_month",
    "segment_borough",
    "segment_edge_betweenness",
    "segment_travel_time_s",
    "shore_dist_ft",
    "shore_graph_steps",
    "station_p_id",
    "tide_id",
    "tide_polygon_id",
    "precip_polygon_id",
    "dem_mean",
    "dem_slope",
    "catch_basin_nearest_ft",
]


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]
    task: str
    optional_dependency: str | None = None
    notes: str | None = None


@dataclass
class SplitDefinition:
    name: str
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    stratify_labels: pd.Series | None = None
    notes: str | None = None


def ensure_output_dirs() -> None:
    for path in [MODELING_DIR, FILTERED_DIR, MODELING_DIAGNOSTICS_DIR, MODELING_CHECKPOINT_DIR, CLUSTERING_DIR, ML_DIR, BAYESIAN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def optional_module_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def normalize_boolean(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "boolean":
        return series
    if str(series.dtype) == "bool":
        return series.astype("boolean")
    mapped = series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            1: True,
            0: False,
        }
    )
    return pd.Series(mapped, index=series.index, dtype="boolean")


def season_from_month(series: pd.Series) -> pd.Series:
    mapping = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
    }
    return series.map(mapping).astype("string")


def mayoral_administration_from_start(series: pd.Series) -> pd.Series:
    dates = to_naive_datetime(series)
    labels = pd.Series("unknown_mayoral_administration", index=series.index, dtype="string")
    labels.loc[dates.lt(pd.Timestamp("2014-01-01"))] = "michael_bloomberg_2002_2013"
    labels.loc[dates.ge(pd.Timestamp("2014-01-01")) & dates.lt(pd.Timestamp("2022-01-01"))] = "bill_de_blasio_2014_2021"
    labels.loc[dates.ge(pd.Timestamp("2022-01-01")) & dates.lt(pd.Timestamp("2026-01-01"))] = "eric_adams_2022_2025"
    labels.loc[dates.ge(pd.Timestamp("2026-01-01"))] = "zohran_mamdani_2026_present"
    labels.loc[dates.isna()] = "unknown_mayoral_administration"
    return labels.astype("string")


def to_naive_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    tz = getattr(parsed.dt, "tz", None)
    if tz is not None:
        return parsed.dt.tz_localize(None)
    return parsed


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _analysis_path_from_view(view_name: str) -> Path:
    mapping = {
        "strict_main": STRICT_MAIN_PATH,
        "main_plus_high_conf_possible": MAIN_PLUS_POSSIBLE_PATH,
        "semi_supervised_pool": SEMI_SUPERVISED_PATH,
    }
    if view_name not in mapping:
        raise KeyError(f"Unsupported analysis view: {view_name}")
    return mapping[view_name]


def _load_analysis_view(view_name: str = "strict_main") -> pd.DataFrame:
    path = _analysis_path_from_view(view_name)
    frame = pd.read_parquet(path)
    frame["event_id"] = frame["event_id"].astype("string")
    frame["segment_id"] = frame["segment_id"].astype("string")
    frame["occurrence"] = normalize_boolean(frame["occurrence"]).fillna(False).astype(bool)
    frame["resolution_bool"] = normalize_boolean(frame["resolution_bool"])
    if "possible_unreported_flood" in frame.columns:
        frame["possible_unreported_flood"] = normalize_boolean(frame["possible_unreported_flood"])
    for column in ["start", "end"]:
        if column in frame.columns:
            frame[column] = to_naive_datetime(frame[column])
    return frame


def _load_upstream_lookup(include_geometry: bool = False) -> pd.DataFrame:
    upstream_path = BALANCED_GEOPARQUET_PATH if include_geometry else BALANCED_PARQUET_PATH
    upstream = gpd.read_parquet(upstream_path) if include_geometry else pd.read_parquet(upstream_path)
    upstream["event_id"] = upstream["event_id"].astype("string")
    upstream["segment_id"] = upstream["segment_id"].astype("string")
    keep = [
        "event_id",
        "segment_id",
        "storm_event_id",
        "event_start_local",
        "event_end_local",
        "event_window_end_local",
        "event_window_duration_hours",
        "event_month",
        "prec_intensity_mean",
        "precip_polygon_id",
        "tide_polygon_id",
        "station_p_id",
        "tide_id",
        "segment_borough",
        "segment_travel_time_s",
        "segment_edge_betweenness",
        "component_size",
        "shore_graph_steps",
        "shore_dist_ft",
        "dem_mean",
        "dem_slope",
        "fema_zone_subty",
        "fema_overlap_ft",
        "fema_overlap_share",
        "fema_sfha_any",
        "catch_basin_nearest_ft",
        "outfall_nearest_ft",
        "census_poverty_rate",
        "census_renter_share",
        "census_no_vehicle_share",
        "census_median_household_income",
        "gov_city",
        "geometry",
    ]
    keep = [column for column in keep if column in upstream.columns]
    lookup = upstream[keep].copy()
    for column in ["event_start_local", "event_end_local", "event_window_end_local"]:
        if column in lookup.columns:
            lookup[column] = to_naive_datetime(lookup[column])
    return lookup


def load_modeling_frame(
    view_name: str = "strict_main",
    include_geometry: bool = False,
    observed_only: bool | None = None,
) -> pd.DataFrame | gpd.GeoDataFrame:
    ensure_output_dirs()
    frame = _load_analysis_view(view_name=view_name)
    lookup = _load_upstream_lookup(include_geometry=include_geometry)

    overlap_cols = [column for column in lookup.columns if column in frame.columns and column not in {"event_id", "segment_id", "geometry"}]
    if overlap_cols:
        lookup = lookup.drop(columns=overlap_cols)

    merged = frame.merge(lookup, on=["event_id", "segment_id"], how="left", validate="one_to_one")

    if "storm_event_id" not in merged.columns:
        merged["storm_event_id"] = pd.NA
    merged["storm_event_id"] = merged["storm_event_id"].astype("string")

    if "travel_time" not in merged.columns:
        merged["travel_time"] = safe_numeric(merged.get("segment_travel_time_s"))
    if "tide_polygon" not in merged.columns:
        merged["tide_polygon"] = merged.get("tide_polygon_id", pd.Series(pd.NA, index=merged.index)).astype("string")
    if "tide_station" not in merged.columns:
        merged["tide_station"] = merged.get("tide_id", pd.Series(pd.NA, index=merged.index)).astype("string")
    if "precipitation_polygon" not in merged.columns:
        merged["precipitation_polygon"] = merged.get("precip_polygon_id", pd.Series(pd.NA, index=merged.index)).astype("string")
    if "precipitation_station" not in merged.columns:
        merged["precipitation_station"] = merged.get("station_p_id", pd.Series(pd.NA, index=merged.index)).astype("string")
    if "gov_borough" not in merged.columns:
        merged["gov_borough"] = merged.get("segment_borough", merged.get("borough", pd.Series(pd.NA, index=merged.index))).astype("string")
    else:
        merged["gov_borough"] = merged["gov_borough"].astype("string")
    if "mayoral_administration" not in merged.columns:
        if "start" in merged.columns:
            merged["mayoral_administration"] = mayoral_administration_from_start(merged["start"])
        else:
            merged["mayoral_administration"] = pd.Series("unknown_mayoral_administration", index=merged.index, dtype="string")
    else:
        merged["mayoral_administration"] = merged["mayoral_administration"].astype("string")
    if "borough" in merged.columns:
        merged["borough"] = merged["borough"].astype("string")
    if "season" not in merged.columns and "month" in merged.columns:
        merged["season"] = season_from_month(merged["month"])
    if "month" not in merged.columns and "start" in merged.columns:
        merged["month"] = merged["start"].dt.month.astype("Int64")
    if "hour" not in merged.columns and "start" in merged.columns:
        merged["hour"] = merged["start"].dt.hour.astype("Int64")
    if "day_of_week" not in merged.columns and "start" in merged.columns:
        merged["day_of_week"] = merged["start"].dt.dayofweek.astype("Int64")
    if "duration" not in merged.columns:
        merged["duration"] = safe_numeric(merged.get("event_window_duration_hours"))
    if "max_tide" not in merged.columns and "tide_level_m_max" in merged.columns:
        merged["max_tide"] = safe_numeric(merged["tide_level_m_max"])
    if "shore_dist" not in merged.columns:
        if "shore_dist_ft" in merged.columns:
            merged["shore_dist"] = safe_numeric(merged["shore_dist_ft"])
        elif "shore_graph_steps" in merged.columns:
            merged["shore_dist"] = safe_numeric(merged["shore_graph_steps"])
    if "edge_betweenness" not in merged.columns:
        merged["edge_betweenness"] = safe_numeric(merged.get("segment_edge_betweenness"))
    if "elevation" not in merged.columns:
        merged["elevation"] = safe_numeric(merged.get("dem_mean"))
    if "slope" not in merged.columns:
        merged["slope"] = safe_numeric(merged.get("dem_slope"))

    if observed_only is True:
        merged = merged[merged["occurrence"]].copy()
    elif observed_only is False:
        merged = merged[~merged["occurrence"]].copy()

    if include_geometry:
        return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:2263")
    return merged


def feature_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen = set()
    for group_name, columns in EXPECTED_FEATURE_GROUPS.items():
        for column in columns:
            if column in seen:
                continue
            seen.add(column)
            available = column in frame.columns
            records.append(
                {
                    "feature_group": group_name,
                    "feature": column,
                    "available": available,
                    "dtype": str(frame[column].dtype) if available else pd.NA,
                    "non_null_rate": float(frame[column].notna().mean()) if available else np.nan,
                    "n_unique": int(frame[column].nunique(dropna=True)) if available else 0,
                }
            )
    extra_columns = [
        column
        for column in frame.columns
        if column not in seen and not column.startswith("present_") and column not in LEAKAGE_ALWAYS_DROP
    ]
    for column in extra_columns:
        records.append(
            {
                "feature_group": "extra",
                "feature": column,
                "available": True,
                "dtype": str(frame[column].dtype),
                "non_null_rate": float(frame[column].notna().mean()),
                "n_unique": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(records).sort_values(["feature_group", "feature"], kind="stable").reset_index(drop=True)


def summarize_feature_groups(frame: pd.DataFrame) -> pd.DataFrame:
    catalog = feature_catalog(frame)
    summary = (
        catalog.groupby("feature_group", as_index=False)
        .agg(
            expected_features=("feature", "size"),
            available_features=("available", "sum"),
            mean_non_null_rate=("non_null_rate", "mean"),
        )
    )
    summary["missing_features"] = summary["expected_features"] - summary["available_features"]
    return summary.sort_values("feature_group", kind="stable").reset_index(drop=True)


def unavailable_expected_features(frame: pd.DataFrame) -> pd.DataFrame:
    catalog = feature_catalog(frame)
    return catalog[~catalog["available"]].copy().reset_index(drop=True)


def leakage_rules_for_task(task_name: str) -> dict[str, str]:
    rules = {column: "identifier_or_derived_selection" for column in LEAKAGE_ALWAYS_DROP}
    rules.update({column: "technical_duplicate_use_canonical_feature_name_instead" for column in TECHNICAL_DUPLICATE_COLUMNS})
    rules.update({column: "presence_indicator_or_selection_metadata" for column in frame_like_presence_columns(task_name=None)})

    if task_name == "occurrence":
        rules.update(
            {
                "occurrence": "target_column",
                "intensity": "post_event_complaint_count",
                "resolution": "post_event_service_outcome",
                "resolution_bool": "post_event_service_outcome",
                "end": "event_close_time_not_available_at_prediction",
                "duration": "event_window_outcome_not_available_at_prediction",
                "storm_event_id": "grouping_only_not_predictor",
                "start": "raw_timestamp_use_calendar_derivatives_instead",
            }
        )
    elif task_name == "intensity":
        rules.update(
            {
                "intensity": "target_column",
                "occurrence": "constant_or_label_column",
                "resolution": "post_event_service_outcome",
                "resolution_bool": "post_event_service_outcome",
                "end": "close_time_not_available_at_event_start",
                "duration": "event_window_outcome_not_available_at_event_start",
                "storm_event_id": "grouping_only_not_predictor",
                "start": "raw_timestamp_use_calendar_derivatives_instead",
            }
        )
    elif task_name == "resolution_closure":
        rules.update(
            {
                "resolution_bool": "target_column",
                "resolution": "target_magnitude_or_post_event_outcome",
                "end": "close_time_target_related",
                "duration": "derived_from_event_end_and_target_related",
                "occurrence": "constant_label_for_positive_events",
                "intensity": "observed_complaint_count_accumulates_during_event",
                "storm_event_id": "grouping_only_not_predictor",
                "start": "raw_timestamp_use_calendar_derivatives_instead",
            }
        )
    elif task_name == "resolution_time":
        rules.update(
            {
                "resolution": "target_column",
                "resolution_bool": "target_closure_indicator",
                "end": "close_time_target_related",
                "duration": "derived_from_event_end_and_target_related",
                "occurrence": "constant_label_for_positive_events",
                "intensity": "observed_complaint_count_accumulates_during_event",
                "storm_event_id": "grouping_only_not_predictor",
                "start": "raw_timestamp_use_calendar_derivatives_instead",
            }
        )
    return rules


def frame_like_presence_columns(task_name: str | None) -> list[str]:
    del task_name
    # The filtered analysis dataset stores completeness flags as present_<field>.
    return []


def build_leakage_audit(frame: pd.DataFrame, task_name: str) -> pd.DataFrame:
    rules = leakage_rules_for_task(task_name)
    records: list[dict[str, Any]] = []
    for column in frame.columns:
        excluded = column.startswith("present_") or column in rules
        reason = "presence_indicator_or_selection_metadata" if column.startswith("present_") else rules.get(column)
        records.append(
            {
                "feature": column,
                "excluded_from_predictors": bool(excluded),
                "reason": reason if excluded else pd.NA,
                "dtype": str(frame[column].dtype),
                "non_null_rate": float(frame[column].notna().mean()),
            }
        )
    return pd.DataFrame(records).sort_values(["excluded_from_predictors", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)


def get_predictor_columns(frame: pd.DataFrame, task_name: str) -> list[str]:
    audit = build_leakage_audit(frame, task_name)
    available = audit[~audit["excluded_from_predictors"]]["feature"].tolist()
    predictor_cols = []
    for column in available:
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.notna().sum() == 0:
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        non_null_sample = series.dropna().head(100)
        if not non_null_sample.empty and non_null_sample.map(lambda value: isinstance(value, pd.Timestamp)).any():
            continue
        if str(series.dtype) == "string" and series.nunique(dropna=True) <= 1:
            continue
        predictor_cols.append(column)
    return predictor_cols


def split_columns_by_type(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for column in columns:
        dtype = frame[column].dtype
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric_cols.append(column)
        else:
            categorical_cols.append(column)
    return numeric_cols, categorical_cols


def cast_frame_to_object(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(object)


def build_preprocessor(frame: pd.DataFrame, predictor_columns: list[str], scale_numeric: bool = True) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols, categorical_cols = split_columns_by_type(frame, predictor_columns)

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical_steps: list[tuple[str, Any]] = [
        ("cast_object", FunctionTransformer(cast_frame_to_object, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encoder", categorical_encoder),
    ]

    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer, numeric_cols, categorical_cols


def _binned_numeric_labels(series: pd.Series, q: int = 5, prefix: str = "bin") -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series("MISSING", index=series.index, dtype="string")
    valid = numeric.dropna()
    if valid.nunique() <= 1:
        labels.loc[valid.index] = f"{prefix}_constant"
        return labels
    bins = pd.qcut(valid.rank(method="first"), q=min(q, valid.nunique()), labels=False, duplicates="drop")
    labels.loc[valid.index] = bins.astype("Int64").astype("string").map(lambda value: f"{prefix}_{value}")
    return labels


def _target_strata(frame: pd.DataFrame, target_col: str | None = None) -> pd.Series:
    if target_col and target_col in frame.columns:
        target = frame[target_col]
    elif "occurrence" in frame.columns and frame["occurrence"].nunique(dropna=True) > 1:
        target = frame["occurrence"]
        target_col = "occurrence"
    elif "resolution" in frame.columns and pd.to_numeric(frame["resolution"], errors="coerce").notna().sum() > 0:
        target = frame["resolution"]
        target_col = "resolution"
    elif "intensity" in frame.columns:
        target = frame["intensity"]
        target_col = "intensity"
    else:
        return pd.Series("all", index=frame.index, dtype="string")

    if target_col in {"occurrence", "resolution_bool"}:
        return target.astype("string").fillna("MISSING")
    return _binned_numeric_labels(target, q=5, prefix=str(target_col))


def _valid_strata(labels: pd.Series, min_count: int = 2) -> bool:
    counts = labels.astype("string").value_counts(dropna=False)
    return len(counts) > 1 and bool(counts.ge(min_count).all())


def _stratification_labels(frame: pd.DataFrame, target_col: str | None = None) -> pd.Series | None:
    target_labels = _target_strata(frame, target_col=target_col)
    borough_labels = frame["borough"].astype("string").fillna("UNKNOWN") if "borough" in frame.columns else None
    if "mayoral_administration" in frame.columns:
        mayor_labels = frame["mayoral_administration"].astype("string").fillna("UNKNOWN_MAYOR")
    elif "gov_city" in frame.columns:
        mayor_labels = frame["gov_city"].astype("string").fillna("UNKNOWN_MAYOR")
    else:
        mayor_labels = None

    if mayor_labels is not None and borough_labels is not None:
        combined = (
            target_labels.astype("string")
            + "|"
            + mayor_labels.astype("string")
            + "|"
            + borough_labels.astype("string")
        ).astype("string")
        if _valid_strata(combined, min_count=20):
            return combined

    if mayor_labels is not None:
        combined = (target_labels.astype("string") + "|" + mayor_labels.astype("string")).astype("string")
        if _valid_strata(combined, min_count=12):
            return combined

    if borough_labels is not None:
        combined = (target_labels.astype("string") + "|" + borough_labels.astype("string")).astype("string")
        if _valid_strata(combined, min_count=20):
            return combined

    if _valid_strata(target_labels, min_count=4):
        return target_labels.astype("string")

    if borough_labels is not None and _valid_strata(borough_labels, min_count=4):
        return borough_labels.astype("string")
    return None


def _temporal_split_by_administration(
    frame: pd.DataFrame,
    target_col: str | None,
) -> SplitDefinition | None:
    """Build a forward-looking temporal split aligned with mayoral administrations.

    Train  = bloomberg era  (events with start < 2014-01-01)
    Val    = de blasio era  (2014-01-01 <= start < 2022-01-01)
    Test   = adams/mamdani  (start >= 2022-01-01)

    Falls back to None when any partition is too small to be useful (< 30 rows)
    or when the target has only one class in a partition.
    """
    if "start" not in frame.columns:
        return None

    dates = to_naive_datetime(frame["start"])
    cut_val = pd.Timestamp("2014-01-01")
    cut_test = pd.Timestamp("2022-01-01")

    train_mask = dates.lt(cut_val)
    val_mask = dates.ge(cut_val) & dates.lt(cut_test)
    test_mask = dates.ge(cut_test)

    train_index = frame.index[train_mask].to_numpy()
    validation_index = frame.index[val_mask].to_numpy()
    test_index = frame.index[test_mask].to_numpy()

    min_rows = 30
    if len(train_index) < min_rows or len(validation_index) < min_rows or len(test_index) < min_rows:
        return None

    if target_col and target_col in frame.columns:
        for idx_arr, label in [(train_index, "train"), (validation_index, "validation"), (test_index, "test")]:
            partition_target = frame.loc[idx_arr, target_col]
            if target_col in {"occurrence", "resolution_bool"}:
                if partition_target.astype("string").nunique(dropna=True) < 2:
                    return None
            else:
                if pd.to_numeric(partition_target, errors="coerce").notna().sum() < min_rows:
                    return None

    return SplitDefinition(
        name="temporal_by_administration",
        train_index=train_index,
        validation_index=validation_index,
        test_index=test_index,
        stratify_labels=None,
        notes=(
            "Forward-looking temporal split by mayoral administration: "
            "train=Bloomberg(<2014), val=de Blasio(2014-2021), test=Adams/Mamdani(>=2022). "
            "No shuffle — strictly respects time ordering to prevent leakage. "
            "Hyperparameter search uses train/validation; test is final holdout."
        ),
    )


def make_split_definitions(
    frame: pd.DataFrame,
    include_spatial: bool = True,
    target_col: str | None = None,
) -> list[SplitDefinition]:
    del include_spatial
    splits: list[SplitDefinition] = []

    # --- Split 1: forward-looking temporal split by mayoral administration ---
    temporal_split = _temporal_split_by_administration(frame, target_col=target_col)
    if temporal_split is not None:
        splits.append(temporal_split)

    # --- Split 2: stratified random 75/15/10 (classic baseline) ---
    labels = _stratification_labels(frame, target_col=target_col)
    indices = frame.index.to_numpy()
    stratify = labels.loc[indices] if labels is not None else None

    train_index, holdout_index = train_test_split(
        indices,
        train_size=0.75,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=stratify,
    )

    holdout_labels = labels.loc[holdout_index] if labels is not None else None
    if holdout_labels is not None and not _valid_strata(holdout_labels, min_count=2):
        holdout_labels = _target_strata(frame.loc[holdout_index], target_col=target_col)
        if not _valid_strata(holdout_labels, min_count=2):
            holdout_labels = None

    validation_index, test_index = train_test_split(
        holdout_index,
        train_size=0.60,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=holdout_labels,
    )

    splits.append(
        SplitDefinition(
            name="stratified_75_15_10",
            train_index=np.asarray(train_index),
            validation_index=np.asarray(validation_index),
            test_index=np.asarray(test_index),
            stratify_labels=labels,
            notes=(
                "Classic 75/15/10 stratified-random split. "
                "Stratification prioritizes target balance plus mayoral_administration when feasible; "
                "hyperparameter search uses train/validation, test split is final holdout. "
                "Shuffle=True — compare against temporal_by_administration to detect leakage."
            ),
        )
    )

    return splits


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
        metrics["brier_score"] = brier_score_loss(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["brier_score"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)
    metrics["false_positive_rate"] = float(fp / max(fp + tn, 1))
    metrics["false_negative_rate"] = float(fn / max(fn + tp, 1))
    return metrics


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_series = pd.Series(y_true).astype(float)
    y_pred_series = pd.Series(y_pred).astype(float)
    metrics = {
        "rmse": root_mean_squared_error(y_true_series, y_pred_series),
        "mae": mean_absolute_error(y_true_series, y_pred_series),
        "r2": r2_score(y_true_series, y_pred_series),
        "median_absolute_error": median_absolute_error(y_true_series, y_pred_series),
        "spearman_correlation": y_true_series.corr(y_pred_series, method="spearman"),
        "pearson_correlation": y_true_series.corr(y_pred_series, method="pearson"),
    }
    finite_mask = np.isfinite(y_true_series) & np.isfinite(y_pred_series)
    if finite_mask.all() and np.all(y_true_series.to_numpy() != 0):
        try:
            metrics["mape"] = mean_absolute_percentage_error(y_true_series, y_pred_series)
        except Exception:
            metrics["mape"] = np.nan
    else:
        metrics["mape"] = np.nan
    return metrics


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def primary_metric_value(metrics: dict[str, float], task: str) -> float:
    if task == "classification":
        return float(metrics.get("pr_auc", np.nan))
    return float(-metrics.get("mae", np.nan))


def overfitting_flags(record: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "classification":
        train_score = record.get("train_pr_auc", np.nan)
        test_score = record.get("test_pr_auc", np.nan)
        validation_score = record.get("cv_validation_primary_score", np.nan)
        gap_threshold = 0.10
    else:
        # Lower MAE is better, so the gaps use train minus validation/test on the negative-MAE scale.
        train_score = -record.get("train_mae", np.nan)
        test_score = -record.get("test_mae", np.nan)
        validation_score = record.get("cv_validation_primary_score", np.nan)
        gap_threshold = 0.15

    train_validation_gap = train_score - validation_score if pd.notna(train_score) and pd.notna(validation_score) else np.nan
    validation_test_gap = validation_score - test_score if pd.notna(validation_score) and pd.notna(test_score) else np.nan
    return {
        "train_vs_validation_gap": train_validation_gap,
        "validation_vs_test_gap": validation_test_gap,
        "possible_overfitting": bool(pd.notna(train_validation_gap) and train_validation_gap > gap_threshold),
        "possible_test_degradation": bool(pd.notna(validation_test_gap) and validation_test_gap > gap_threshold),
    }


def make_classification_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
            param_grid={
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            task="classification",
        ),
        ModelSpec(
            name="decision_tree",
            estimator=__import__("sklearn.tree", fromlist=["DecisionTreeClassifier"]).DecisionTreeClassifier(random_state=RANDOM_STATE),
            param_grid={
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_leaf": [1, 5, 10, 25],
                "model__criterion": ["gini", "entropy"],
            },
            task="classification",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__max_features": ["sqrt", "log2"],
                "model__class_weight": [None, "balanced"],
            },
            task="classification",
        ),
        ModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0],
            },
            task="classification",
        ),
    ]

    if optional_module_available("xgboost"):
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
                param_grid={
                    "model__n_estimators": [200, 500],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                },
                task="classification",
                optional_dependency="xgboost",
            )
        )
    if optional_module_available("catboost"):
        from catboost import CatBoostClassifier

        specs.append(
            ModelSpec(
                name="catboost",
                estimator=CatBoostClassifier(verbose=False, random_state=RANDOM_STATE),
                param_grid={
                    "model__depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.1],
                    "model__iterations": [200, 500],
                },
                task="classification",
                optional_dependency="catboost",
            )
        )
    if optional_module_available("lightgbm"):
        from lightgbm import LGBMClassifier

        specs.append(
            ModelSpec(
                name="lightgbm",
                estimator=LGBMClassifier(random_state=RANDOM_STATE),
                param_grid={
                    "model__n_estimators": [200, 500],
                    "model__max_depth": [3, 5, 7, -1],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                },
                task="classification",
                optional_dependency="lightgbm",
            )
        )
    return specs


def make_regression_model_specs(include_count_models: bool = True) -> list[ModelSpec]:
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression

    specs = [
        ModelSpec(
            name="dummy_regressor",
            estimator=DummyRegressor(strategy="median"),
            param_grid={},
            task="regression",
        ),
        ModelSpec(
            name="linear_regression",
            estimator=LinearRegression(),
            param_grid={},
            task="regression",
        ),
        ModelSpec(
            name="ridge",
            estimator=Ridge(random_state=RANDOM_STATE),
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            task="regression",
        ),
        ModelSpec(
            name="random_forest_regressor",
            estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__max_depth": [5, 10, 20, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__max_features": ["sqrt", "log2"],
            },
            task="regression",
        ),
        ModelSpec(
            name="gradient_boosting_regressor",
            estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0],
            },
            task="regression",
        ),
    ]
    if include_count_models:
        specs.extend(
            [
                ModelSpec(
                    name="poisson_regressor",
                    estimator=PoissonRegressor(max_iter=1000),
                    param_grid={"model__alpha": [0.0, 0.1, 1.0]},
                    task="regression",
                ),
                ModelSpec(
                    name="tweedie_regressor",
                    estimator=TweedieRegressor(max_iter=1000, power=1.5),
                    param_grid={"model__alpha": [0.0, 0.1, 1.0], "model__power": [1.1, 1.5, 1.9]},
                    task="regression",
                ),
            ]
        )
    if optional_module_available("xgboost"):
        from xgboost import XGBRegressor

        specs.append(
            ModelSpec(
                name="xgboost_regressor",
                estimator=XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                param_grid={
                    "model__n_estimators": [200, 500],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                    "model__colsample_bytree": [0.8, 1.0],
                },
                task="regression",
                optional_dependency="xgboost",
            )
        )
    if optional_module_available("catboost"):
        from catboost import CatBoostRegressor

        specs.append(
            ModelSpec(
                name="catboost_regressor",
                estimator=CatBoostRegressor(verbose=False, random_state=RANDOM_STATE),
                param_grid={
                    "model__depth": [4, 6, 8],
                    "model__learning_rate": [0.03, 0.1],
                    "model__iterations": [200, 500],
                },
                task="regression",
                optional_dependency="catboost",
            )
        )
    if optional_module_available("lightgbm"):
        from lightgbm import LGBMRegressor

        specs.append(
            ModelSpec(
                name="lightgbm_regressor",
                estimator=LGBMRegressor(random_state=RANDOM_STATE),
                param_grid={
                    "model__n_estimators": [200, 500],
                    "model__max_depth": [3, 5, 7, -1],
                    "model__learning_rate": [0.03, 0.1],
                    "model__subsample": [0.8, 1.0],
                },
                task="regression",
                optional_dependency="lightgbm",
            )
        )
    return specs


def _make_grid_search(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    task: str,
    validation_fold: np.ndarray,
) -> GridSearchCV:
    if task == "classification":
        scoring = "average_precision"
    else:
        scoring = "neg_mean_absolute_error"

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=PredefinedSplit(validation_fold),
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )


def _safe_column_values(frame: pd.DataFrame, column: str, default: Any = pd.NA) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.full(len(frame), default, dtype=object)


def _prediction_frame(
    source_df: pd.DataFrame,
    split_name: str,
    model_name: str,
    target_col: str,
    set_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "event_id": _safe_column_values(source_df, "event_id"),
            "segment_id": _safe_column_values(source_df, "segment_id"),
            "borough": _safe_column_values(source_df, "borough"),
            "gov_city": _safe_column_values(source_df, "gov_city"),
            "mayoral_administration": _safe_column_values(source_df, "mayoral_administration"),
            "intensity": _safe_column_values(source_df, "intensity", np.nan),
            "census_poverty_rate": _safe_column_values(source_df, "census_poverty_rate", np.nan),
            "split_strategy": split_name,
            "model_name": model_name,
            "target": target_col,
            "set_name": set_name,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
        }
    )
    frame["y_score"] = np.asarray(y_score) if y_score is not None else np.nan
    return frame


def _safe_checkpoint_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _save_model_checkpoint(
    task_name: str,
    split_name: str,
    model_name: str,
    results_records: list[dict[str, Any]],
    pred_frame: pd.DataFrame | None = None,
    estimator: Pipeline | None = None,
) -> None:
    checkpoint_dir = MODELING_CHECKPOINT_DIR / task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_split = _safe_checkpoint_name(split_name)
    safe_model = _safe_checkpoint_name(model_name)

    pd.DataFrame(results_records).to_csv(checkpoint_dir / "results_so_far.csv", index=False)
    if pred_frame is not None and not pred_frame.empty:
        pred_frame.to_parquet(checkpoint_dir / f"{safe_split}__{safe_model}__predictions.parquet", index=False)
    if estimator is not None:
        joblib.dump(estimator, checkpoint_dir / f"{safe_split}__{safe_model}.joblib")


def fit_supervised_models(
    frame: pd.DataFrame,
    target_col: str,
    task_name: str,
    model_specs: list[ModelSpec],
    split_definitions: list[SplitDefinition],
    regression_target_transform: str | None = None,
    closed_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Pipeline]:
    ensure_output_dirs()
    work = frame.copy()

    if closed_only:
        work = work[work[target_col].notna()].copy()

    predictor_cols = get_predictor_columns(work, task_name)
    audit = build_leakage_audit(work, task_name)
    # Bug-4 fix: do NOT fit a global preprocessor here.  The scaler/imputer
    # must be fit exclusively on each split's train partition so that val/test
    # statistics never leak into the transformation of train data.  We keep a
    # prototype (unfitted) preprocessor and clone it inside the split loop.
    preprocessor_proto, _, _ = build_preprocessor(work, predictor_cols, scale_numeric=True)

    results_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_importance_records: list[dict[str, Any]] = []
    best_bundle: tuple[float, Pipeline] | None = None

    for split in split_definitions:
        train_df = work.loc[split.train_index].copy()
        validation_df = work.loc[split.validation_index].copy()
        test_df = work.loc[split.test_index].copy()
        y_train = train_df[target_col].copy()
        y_validation = validation_df[target_col].copy()
        y_test = test_df[target_col].copy()

        if task_name in {"occurrence", "resolution_closure"}:
            if (
                pd.Series(y_train).nunique(dropna=True) < 2
                or pd.Series(y_validation).nunique(dropna=True) < 2
                or pd.Series(y_test).nunique(dropna=True) < 2
            ):
                for spec in model_specs:
                    results_records.append(
                        {
                            "task_name": task_name,
                            "split_strategy": split.name,
                            "model_name": spec.name,
                            "status": "skipped",
                            "error": "split_has_single_class_in_train_validation_or_test",
                            "n_train": len(train_df),
                            "n_validation": len(validation_df),
                            "n_test": len(test_df),
                            "notes": split.notes,
                        }
                    )
                    _save_model_checkpoint(task_name, split.name, spec.name, results_records)
                continue

        if regression_target_transform == "log1p":
            y_train_fit = np.log1p(y_train.astype(float))
            y_validation_fit = np.log1p(y_validation.astype(float))
        else:
            y_train_fit = y_train
            y_validation_fit = y_validation

        search_df = pd.concat([train_df, validation_df], axis=0)
        y_search_fit = pd.concat(
            [
                pd.Series(y_train_fit, index=train_df.index),
                pd.Series(y_validation_fit, index=validation_df.index),
            ],
            axis=0,
        )
        validation_fold = np.r_[
            np.full(len(train_df), -1, dtype=int),
            np.zeros(len(validation_df), dtype=int),
        ]

        for spec in model_specs:
            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor_proto)),
                    ("model", clone(spec.estimator)),
                ]
            )
            search = _make_grid_search(pipeline, spec.param_grid or {}, spec.task, validation_fold)

            try:
                search.fit(search_df[predictor_cols], y_search_fit)
            except Exception as exc:
                results_records.append(
                    {
                        "task_name": task_name,
                        "split_strategy": split.name,
                        "model_name": spec.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                _save_model_checkpoint(task_name, split.name, spec.name, results_records)
                continue

            best_estimator = search.best_estimator_
            if spec.task == "classification":
                y_train_pred = best_estimator.predict(train_df[predictor_cols])
                y_validation_pred = best_estimator.predict(validation_df[predictor_cols])
                y_pred = best_estimator.predict(test_df[predictor_cols])
                if hasattr(best_estimator, "predict_proba"):
                    y_train_score = best_estimator.predict_proba(train_df[predictor_cols])[:, 1]
                    y_validation_score = best_estimator.predict_proba(validation_df[predictor_cols])[:, 1]
                    y_score = best_estimator.predict_proba(test_df[predictor_cols])[:, 1]
                elif hasattr(best_estimator, "decision_function"):
                    raw_train_score = best_estimator.decision_function(train_df[predictor_cols])
                    raw_validation_score = best_estimator.decision_function(validation_df[predictor_cols])
                    raw_score = best_estimator.decision_function(test_df[predictor_cols])
                    y_train_score = 1.0 / (1.0 + np.exp(-raw_train_score))
                    y_validation_score = 1.0 / (1.0 + np.exp(-raw_validation_score))
                    y_score = 1.0 / (1.0 + np.exp(-raw_score))
                else:
                    y_train_score = None
                    y_validation_score = None
                    y_score = None
                train_metrics = classification_metrics(y_train.astype(bool), y_train_pred.astype(bool), y_train_score)
                validation_metrics = classification_metrics(y_validation.astype(bool), y_validation_pred.astype(bool), y_validation_score)
                test_metrics = classification_metrics(y_test.astype(bool), y_pred.astype(bool), y_score)
                metrics = test_metrics
                selection_score = primary_metric_value(validation_metrics, "classification")
                pred_frame = pd.concat(
                    [
                        _prediction_frame(train_df, split.name, spec.name, target_col, "train", y_train.astype(bool), y_train_pred.astype(bool), y_train_score),
                        _prediction_frame(validation_df, split.name, spec.name, target_col, "validation", y_validation.astype(bool), y_validation_pred.astype(bool), y_validation_score),
                        _prediction_frame(test_df, split.name, spec.name, target_col, "test", y_test.astype(bool), y_pred.astype(bool), y_score),
                    ],
                    ignore_index=True,
                )
            else:
                y_train_pred_fit = best_estimator.predict(train_df[predictor_cols])
                y_validation_pred_fit = best_estimator.predict(validation_df[predictor_cols])
                y_pred_fit = best_estimator.predict(test_df[predictor_cols])
                if regression_target_transform == "log1p":
                    y_train_pred = np.expm1(y_train_pred_fit)
                    y_validation_pred = np.expm1(y_validation_pred_fit)
                    y_pred = np.expm1(y_pred_fit)
                else:
                    y_train_pred = y_train_pred_fit
                    y_validation_pred = y_validation_pred_fit
                    y_pred = y_pred_fit
                train_metrics = regression_metrics(y_train.astype(float), y_train_pred)
                validation_metrics = regression_metrics(y_validation.astype(float), y_validation_pred)
                test_metrics = regression_metrics(y_test.astype(float), y_pred)
                metrics = test_metrics
                selection_score = primary_metric_value(validation_metrics, "regression")
                pred_frame = pd.concat(
                    [
                        _prediction_frame(train_df, split.name, spec.name, target_col, "train", y_train.astype(float), y_train_pred),
                        _prediction_frame(validation_df, split.name, spec.name, target_col, "validation", y_validation.astype(float), y_validation_pred),
                        _prediction_frame(test_df, split.name, spec.name, target_col, "test", y_test.astype(float), y_pred),
                    ],
                    ignore_index=True,
                )

            record = {
                "task_name": task_name,
                "split_strategy": split.name,
                "model_name": spec.name,
                "status": "ok",
                "n_train": len(train_df),
                "n_validation": len(validation_df),
                "n_test": len(test_df),
                "n_features": len(predictor_cols),
                "best_params": json.dumps(search.best_params_),
                "validation_primary_score": selection_score,
                "cv_validation_primary_score": selection_score,
                "cv_train_primary_score": primary_metric_value(train_metrics, spec.task),
                "notes": split.notes,
            }
            record.update(metrics)
            record.update(prefix_metrics(train_metrics, "train"))
            record.update(prefix_metrics(validation_metrics, "validation"))
            record.update(prefix_metrics(test_metrics, "test"))
            record.update(overfitting_flags(record, spec.task))
            results_records.append(record)
            prediction_frames.append(pred_frame)
            _save_model_checkpoint(task_name, split.name, spec.name, results_records, pred_frame, best_estimator)

            if best_bundle is None or selection_score > best_bundle[0]:
                best_bundle = (selection_score, best_estimator)

    if best_bundle is None:
        raise RuntimeError(f"No model succeeded for task {task_name}")

    best_model = best_bundle[1]
    try:
        check_is_fitted(best_model)
    except Exception as exc:
        raise RuntimeError(f"Best model for task {task_name} is not fitted") from exc

    transformed_feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        importances = np.abs(coef.ravel()) if coef.ndim > 1 else np.abs(coef)
    else:
        importances = np.full(len(transformed_feature_names), np.nan)

    for feature_name, importance in zip(transformed_feature_names, importances, strict=False):
        feature_importance_records.append(
            {
                "task_name": task_name,
                "feature": feature_name,
                "importance": float(importance) if pd.notna(importance) else np.nan,
            }
        )

    results_df = pd.DataFrame(results_records)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    feature_importance_df = pd.DataFrame(feature_importance_records).sort_values("importance", ascending=False, kind="stable")
    return results_df, predictions_df, feature_importance_df, best_model


def compute_bias_diagnostics(predictions: pd.DataFrame, task_name: str) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    frame = predictions.copy()
    records: list[dict[str, Any]] = []

    if task_name in {"occurrence", "resolution_closure"}:
        frame["prediction_error"] = frame["y_pred"].astype(bool).ne(frame["y_true"].astype(bool))
        frame["false_negative"] = frame["y_true"].astype(bool) & ~frame["y_pred"].astype(bool)
        frame["false_positive"] = ~frame["y_true"].astype(bool) & frame["y_pred"].astype(bool)

        for group_col in ["borough", "mayoral_administration"]:
            if group_col in frame.columns:
                group_keys = ["set_name", group_col] if "set_name" in frame.columns else [group_col]
                for group_values, group in frame.groupby(group_keys, dropna=False):
                    if len(group_keys) == 2:
                        set_name, group_value = group_values
                    else:
                        set_name, group_value = "all", group_values
                    records.append(
                        {
                            "task_name": task_name,
                            "set_name": set_name,
                            "bias_dimension": group_col,
                            "group_value": group_value,
                            "n": len(group),
                            "error_rate": float(group["prediction_error"].mean()),
                            "false_negative_rate": float(group["false_negative"].sum() / max(group["y_true"].astype(bool).sum(), 1)),
                            "false_positive_rate": float(group["false_positive"].sum() / max((~group["y_true"].astype(bool)).sum(), 1)),
                            "mean_score": float(pd.to_numeric(group.get("y_score"), errors="coerce").mean()) if "y_score" in group else np.nan,
                        }
                    )

        if "intensity" in frame.columns and pd.to_numeric(frame["intensity"], errors="coerce").notna().any():
            frame["intensity_band"] = pd.qcut(pd.to_numeric(frame["intensity"], errors="coerce").rank(method="first"), q=4, duplicates="drop").astype("string")
            group_keys = ["set_name", "intensity_band"] if "set_name" in frame.columns else ["intensity_band"]
            for group_values, group in frame.groupby(group_keys, dropna=False):
                if len(group_keys) == 2:
                    set_name, group_value = group_values
                else:
                    set_name, group_value = "all", group_values
                records.append(
                    {
                        "task_name": task_name,
                        "set_name": set_name,
                        "bias_dimension": "event_intensity_band",
                        "group_value": group_value,
                        "n": len(group),
                        "error_rate": float(group["prediction_error"].mean()),
                        "false_negative_rate": float(group["false_negative"].sum() / max(group["y_true"].astype(bool).sum(), 1)),
                        "false_positive_rate": float(group["false_positive"].sum() / max((~group["y_true"].astype(bool)).sum(), 1)),
                        "mean_score": float(pd.to_numeric(group.get("y_score"), errors="coerce").mean()) if "y_score" in group else np.nan,
                    }
                )

    else:
        frame["residual"] = pd.to_numeric(frame["y_true"], errors="coerce") - pd.to_numeric(frame["y_pred"], errors="coerce")
        frame["absolute_error"] = frame["residual"].abs()

        for group_col in ["borough", "mayoral_administration"]:
            if group_col in frame.columns:
                group_keys = ["set_name", group_col] if "set_name" in frame.columns else [group_col]
                for group_values, group in frame.groupby(group_keys, dropna=False):
                    if len(group_keys) == 2:
                        set_name, group_value = group_values
                    else:
                        set_name, group_value = "all", group_values
                    records.append(
                        {
                            "task_name": task_name,
                            "set_name": set_name,
                            "bias_dimension": group_col,
                            "group_value": group_value,
                            "n": len(group),
                            "mean_residual": float(group["residual"].mean()),
                            "median_residual": float(group["residual"].median()),
                            "mae": float(group["absolute_error"].mean()),
                            "underprediction_rate": float(group["residual"].gt(0).mean()),
                        }
                    )

        if "intensity" in frame.columns and pd.to_numeric(frame["intensity"], errors="coerce").notna().any():
            frame["intensity_band"] = pd.qcut(pd.to_numeric(frame["intensity"], errors="coerce").rank(method="first"), q=4, duplicates="drop").astype("string")
            group_keys = ["set_name", "intensity_band"] if "set_name" in frame.columns else ["intensity_band"]
            for group_values, group in frame.groupby(group_keys, dropna=False):
                if len(group_keys) == 2:
                    set_name, group_value = group_values
                else:
                    set_name, group_value = "all", group_values
                records.append(
                    {
                        "task_name": task_name,
                        "set_name": set_name,
                        "bias_dimension": "event_intensity_band",
                        "group_value": group_value,
                        "n": len(group),
                        "mean_residual": float(group["residual"].mean()),
                        "median_residual": float(group["residual"].median()),
                        "mae": float(group["absolute_error"].mean()),
                        "underprediction_rate": float(group["residual"].gt(0).mean()),
                    }
                )

        if "census_poverty_rate" in frame.columns and pd.to_numeric(frame["census_poverty_rate"], errors="coerce").notna().any():
            frame["poverty_band"] = pd.qcut(pd.to_numeric(frame["census_poverty_rate"], errors="coerce").rank(method="first"), q=4, duplicates="drop").astype("string")
            group_keys = ["set_name", "poverty_band"] if "set_name" in frame.columns else ["poverty_band"]
            for group_values, group in frame.groupby(group_keys, dropna=False):
                if len(group_keys) == 2:
                    set_name, group_value = group_values
                else:
                    set_name, group_value = "all", group_values
                records.append(
                    {
                        "task_name": task_name,
                        "set_name": set_name,
                        "bias_dimension": "vulnerability_poverty_band",
                        "group_value": group_value,
                        "n": len(group),
                        "mean_residual": float(group["residual"].mean()),
                        "median_residual": float(group["residual"].median()),
                        "mae": float(group["absolute_error"].mean()),
                        "underprediction_rate": float(group["residual"].gt(0).mean()),
                    }
                )

    return pd.DataFrame(records)


def calibration_table(predictions: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    if predictions.empty or "y_score" not in predictions.columns:
        return pd.DataFrame()
    frame = predictions[predictions["y_score"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["score_bin"] = pd.qcut(pd.to_numeric(frame["y_score"], errors="coerce").rank(method="first"), q=n_bins, duplicates="drop")
    return (
        frame.groupby(["target", "split_strategy", "model_name", "set_name", "score_bin"], observed=False)
        .agg(
            n=("event_id", "size"),
            mean_predicted_probability=("y_score", "mean"),
            observed_rate=("y_true", lambda s: pd.Series(s).astype(bool).mean()),
        )
        .reset_index()
    )


def _cluster_matrix(frame: pd.DataFrame, feature_cols: list[str], sample_size: int | None = None) -> tuple[pd.DataFrame, np.ndarray, Pipeline]:
    work = frame[feature_cols].copy()
    if sample_size is not None and len(work) > sample_size:
        sampled_index = work.sample(n=sample_size, random_state=RANDOM_STATE).index
        work = work.loc[sampled_index].copy()
    preprocessor, _, _ = build_preprocessor(work, feature_cols, scale_numeric=True)
    pipeline = Pipeline([("preprocess", preprocessor)])
    matrix = pipeline.fit_transform(work)
    return work, np.asarray(matrix), pipeline


def clustering_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    sets = {
        "physical_forcing": [
            "max_tide",
            "n_prec",
            "prec_intensity_max",
            "prec_intensity_mean",
            "prec_depth_total",
            "prec_duration_total",
            "duration",
            "shore_dist",
            "elevation",
            "fema_fld_zone",
            "fema_overlap_share",
            "fema_sfha_any",
        ],
        "infrastructure_network": [
            "road_class",
            "travel_time",
            "edge_betweenness",
            "component_count_after_removal",
            "giant_component_size_loss",
            "additional_disconnected_node_pairs",
            "pct_giant_component_loss",
            "high_network_criticality",
            "drainage_catch_basin_nearest_ft",
            "outfall_nearest_ft",
            "pump_nearest_ft",
            "critical_infra_exposure",
        ],
        "social_governance": [
            "census_poverty_rate",
            "census_renter_share",
            "census_no_vehicle_share",
            "census_median_household_income",
            "gov_city",
            "gov_borough",
            "borough",
        ],
    }
    combined = []
    for columns in sets.values():
        combined.extend(columns)
    sets["combined"] = list(dict.fromkeys(combined))
    return {name: [column for column in columns if column in frame.columns and frame[column].notna().sum() > 0] for name, columns in sets.items()}


def evaluate_clustering_algorithms(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    observed = frame[frame["occurrence"]].copy()
    feature_sets = clustering_feature_sets(observed)

    selection_records: list[dict[str, Any]] = []
    label_frames: list[pd.DataFrame] = []

    for feature_set_name, columns in feature_sets.items():
        if len(columns) < 2:
            continue
        sample_df, matrix, transform = _cluster_matrix(observed, columns, sample_size=MAX_CLUSTER_SAMPLE)
        full_work, full_matrix, _ = _cluster_matrix(observed, columns, sample_size=None)

        for k in range(2, 11):
            algorithms: list[tuple[str, Any]] = [
                ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)),
                ("agglomerative", AgglomerativeClustering(n_clusters=k)),
                ("gaussian_mixture", GaussianMixture(n_components=k, random_state=RANDOM_STATE)),
            ]

            for algorithm_name, algorithm in algorithms:
                try:
                    if algorithm_name == "gaussian_mixture":
                        labels = algorithm.fit_predict(matrix)
                        score_labels = labels
                        aic = algorithm.aic(matrix)
                        bic = algorithm.bic(matrix)
                    else:
                        labels = algorithm.fit_predict(matrix)
                        score_labels = labels
                        aic = np.nan
                        bic = np.nan

                    valid_mask = pd.Series(score_labels).ge(0).to_numpy()
                    n_clusters = len(set(score_labels[valid_mask])) if valid_mask.any() else 0
                    if n_clusters >= 2 and valid_mask.sum() > n_clusters:
                        sil = silhouette_score(matrix[valid_mask], score_labels[valid_mask])
                        ch = calinski_harabasz_score(matrix[valid_mask], score_labels[valid_mask])
                        db = davies_bouldin_score(matrix[valid_mask], score_labels[valid_mask])
                    else:
                        sil = np.nan
                        ch = np.nan
                        db = np.nan
                    selection_records.append(
                        {
                            "feature_set": feature_set_name,
                            "algorithm": algorithm_name,
                            "k": k,
                            "n_clusters": int(n_clusters),
                            "noise_fraction": float((pd.Series(score_labels) == -1).mean()),
                            "silhouette_score": sil,
                            "calinski_harabasz_score": ch,
                            "davies_bouldin_score": db,
                            "aic": aic,
                            "bic": bic,
                            "sample_n": len(sample_df),
                        }
                    )
                except Exception as exc:
                    selection_records.append(
                        {
                            "feature_set": feature_set_name,
                            "algorithm": algorithm_name,
                            "k": k,
                            "n_clusters": 0,
                            "noise_fraction": np.nan,
                            "silhouette_score": np.nan,
                            "calinski_harabasz_score": np.nan,
                            "davies_bouldin_score": np.nan,
                            "aic": np.nan,
                            "bic": np.nan,
                            "sample_n": len(sample_df),
                            "error": str(exc),
                        }
                    )

        dbscan_grid = [(0.5, 10), (1.0, 10), (1.5, 15), (2.0, 20)]
        for eps, min_samples in dbscan_grid:
            try:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
                valid_mask = pd.Series(labels).ge(0).to_numpy()
                n_clusters = len(set(labels[valid_mask])) if valid_mask.any() else 0
                if n_clusters >= 2 and valid_mask.sum() > n_clusters:
                    sil = silhouette_score(matrix[valid_mask], labels[valid_mask])
                    ch = calinski_harabasz_score(matrix[valid_mask], labels[valid_mask])
                    db = davies_bouldin_score(matrix[valid_mask], labels[valid_mask])
                else:
                    sil = np.nan
                    ch = np.nan
                    db = np.nan
                selection_records.append(
                    {
                        "feature_set": feature_set_name,
                        "algorithm": "dbscan",
                        "k": np.nan,
                        "eps": eps,
                        "min_samples": min_samples,
                        "n_clusters": int(n_clusters),
                        "noise_fraction": float((pd.Series(labels) == -1).mean()),
                        "silhouette_score": sil,
                        "calinski_harabasz_score": ch,
                        "davies_bouldin_score": db,
                        "aic": np.nan,
                        "bic": np.nan,
                        "sample_n": len(sample_df),
                    }
                )
            except Exception as exc:
                selection_records.append(
                    {
                        "feature_set": feature_set_name,
                        "algorithm": "dbscan",
                        "eps": eps,
                        "min_samples": min_samples,
                        "n_clusters": 0,
                        "sample_n": len(sample_df),
                        "error": str(exc),
                    }
                )

        # Fit a stable primary KMeans on the full positive dataset for downstream profiling.
        best_kmeans = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
        full_labels = best_kmeans.fit_predict(full_matrix)
        label_frame = observed.loc[full_work.index, ["event_id", "segment_id", "occurrence", "intensity", "resolution", "resolution_bool"]].copy()
        label_frame[f"{feature_set_name}_cluster_id"] = full_labels
        label_frames.append(label_frame)

    selection_df = pd.DataFrame(selection_records)
    label_df = observed[["event_id", "segment_id"]].copy()
    for part in label_frames:
        label_df = label_df.merge(part.drop(columns=["occurrence", "intensity", "resolution", "resolution_bool"]), on=["event_id", "segment_id"], how="left")

    # Bug-3 guard: warn when events were silently dropped by _cluster_matrix
    # because all their clustering features were NaN. These events get NaN
    # cluster_id and are invisible in downstream diagnostics (NB 14, 15, 16).
    cluster_id_cols = [c for c in label_df.columns if c.endswith("_cluster_id")]
    if cluster_id_cols:
        n_total = len(label_df)
        n_no_cluster = int(label_df[cluster_id_cols].isna().all(axis=1).sum())
        if n_no_cluster > 0:
            warnings.warn(
                f"evaluate_clustering_algorithms: {n_no_cluster}/{n_total} observed events "
                f"({100 * n_no_cluster / max(n_total, 1):.1f}%) have no cluster assignment "
                f"because all their clustering features were NaN. "
                f"These events will be missing from downstream cluster diagnostics "
                f"in notebooks 14, 15, and 16. "
                f"Check feature availability with unavailable_expected_features().",
                stacklevel=2,
            )

    combined_cluster_col = "combined_cluster_id"
    if combined_cluster_col in label_df.columns:
        profiles = observed.merge(label_df[["event_id", combined_cluster_col]], on="event_id", how="left")
        cluster_names = name_combined_clusters(profiles, combined_cluster_col)
        label_df["cluster_label"] = label_df[combined_cluster_col].map(cluster_names).astype("string")
    else:
        label_df["cluster_label"] = pd.NA

    selection_df.to_csv(CLUSTERING_MODEL_SELECTION_PATH, index=False)
    archetypes = observed.merge(label_df, on=["event_id", "segment_id"], how="left")
    archetypes.to_parquet(CLUSTERING_ARCHETYPES_PATH, index=False)

    try:
        geo = load_modeling_frame(view_name="strict_main", include_geometry=True, observed_only=True)
        geo = geo.merge(label_df, on=["event_id", "segment_id"], how="left")
        gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:2263").to_parquet(CLUSTERING_ARCHETYPES_GEOPARQUET_PATH, index=False)
    except Exception:
        pass

    anomalies = detect_anomalies(observed)
    anomalies.to_parquet(ANOMALY_SCORES_PATH, index=False)
    return selection_df, archetypes, anomalies


def name_combined_clusters(frame: pd.DataFrame, cluster_col: str) -> dict[int, str]:
    summary = (
        frame.groupby(cluster_col, as_index=True)
        .agg(
            max_tide=("max_tide", "mean"),
            prec_depth_total=("prec_depth_total", "mean"),
            elevation=("elevation", "mean"),
            shore_dist=("shore_dist", "mean"),
            high_network_criticality=("high_network_criticality", lambda series: pd.Series(series).astype("boolean").mean()),
            census_poverty_rate=("census_poverty_rate", "mean"),
        )
    )
    mapping: dict[int, str] = {}
    for cluster_id, row in summary.iterrows():
        labels = []
        if pd.notna(row["max_tide"]) and row["max_tide"] >= summary["max_tide"].quantile(0.75):
            labels.append("coastal-tide dominated")
        if pd.notna(row["prec_depth_total"]) and row["prec_depth_total"] >= summary["prec_depth_total"].quantile(0.75):
            labels.append("heavy-precipitation pluvial")
        if pd.notna(row["elevation"]) and row["elevation"] <= summary["elevation"].quantile(0.25):
            labels.append("low-elevation FEMA-exposed")
        if pd.notna(row["shore_dist"]) and row["shore_dist"] <= summary["shore_dist"].quantile(0.25):
            labels.append("shore-proximate")
        if pd.notna(row["high_network_criticality"]) and row["high_network_criticality"] >= 0.5:
            labels.append("high-network-criticality")
        if pd.notna(row["census_poverty_rate"]) and row["census_poverty_rate"] >= summary["census_poverty_rate"].quantile(0.75):
            labels.append("social-vulnerability")
        mapping[int(cluster_id)] = " / ".join(labels[:2]) if labels else f"mixed-archetype-{int(cluster_id)}"
    return mapping


def detect_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        column
        for column in [
            "max_tide",
            "n_prec",
            "prec_intensity_max",
            "prec_depth_total",
            "prec_duration_total",
            "duration",
            "shore_dist",
            "elevation",
            "slope",
            "edge_betweenness",
            "travel_time",
            "drainage_catch_basin_nearest_ft",
            "outfall_nearest_ft",
            "census_poverty_rate",
            "census_renter_share",
            "census_no_vehicle_share",
            "census_median_household_income",
        ]
        if column in frame.columns and frame[column].notna().sum() > 0
    ]
    work, matrix, transform = _cluster_matrix(frame, feature_cols, sample_size=None)
    result = frame.loc[work.index, ["event_id", "segment_id", "occurrence", "intensity", "resolution", "resolution_bool"]].copy()

    iso = IsolationForest(random_state=RANDOM_STATE, contamination="auto")
    iso.fit(matrix)
    result["anomaly_score_isolation_forest"] = -iso.score_samples(matrix)
    result["anomaly_flag_isolation_forest"] = iso.predict(matrix) == -1

    lof = LocalOutlierFactor(n_neighbors=35, contamination="auto")
    lof_labels = lof.fit_predict(matrix)
    result["anomaly_score_lof"] = -lof.negative_outlier_factor_
    result["anomaly_flag_lof"] = lof_labels == -1

    try:
        dbscan_labels = DBSCAN(eps=1.5, min_samples=15).fit_predict(matrix)
        result["anomaly_flag_dbscan_noise"] = dbscan_labels == -1
    except Exception:
        result["anomaly_flag_dbscan_noise"] = False

    z_matrix = np.asarray(matrix)
    robust_center = np.nanmedian(z_matrix, axis=0)
    mad = np.nanmedian(np.abs(z_matrix - robust_center), axis=0)
    mad[mad == 0] = 1.0
    robust_z = np.abs((z_matrix - robust_center) / mad)
    result["anomaly_score_robust_z"] = np.nanmean(robust_z, axis=1)
    result["anomaly_flag_robust_z"] = result["anomaly_score_robust_z"] >= np.nanquantile(result["anomaly_score_robust_z"], 0.98)

    flag_cols = [column for column in result.columns if column.startswith("anomaly_flag_")]
    result["anomaly_method_agreement"] = result[flag_cols].sum(axis=1)
    score_cols = [column for column in result.columns if column.startswith("anomaly_score_")]
    result["anomaly_score"] = result[score_cols].rank(pct=True).mean(axis=1)
    result["anomaly_flag"] = result["anomaly_method_agreement"] >= 2
    return result.sort_values("anomaly_score", ascending=False, kind="stable").reset_index(drop=True)


def pca_embedding(frame: pd.DataFrame, feature_cols: list[str], n_components: int = 2) -> pd.DataFrame:
    work, matrix, _ = _cluster_matrix(frame, feature_cols, sample_size=None)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    embedded = pca.fit_transform(matrix)
    out = frame.loc[work.index, ["event_id", "segment_id"]].copy()
    for idx in range(n_components):
        out[f"pc{idx + 1}"] = embedded[:, idx]
    out["explained_variance_ratio"] = np.nan
    out.loc[out.index[0], "explained_variance_ratio"] = pca.explained_variance_ratio_.sum()
    return out


def run_occurrence_ml(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = load_modeling_frame("strict_main") if frame is None else frame.copy()
    predictor_frame = work.copy()
    splits = make_split_definitions(predictor_frame, include_spatial=True, target_col="occurrence")
    results, predictions, feature_importance, best_model = fit_supervised_models(
        predictor_frame,
        target_col="occurrence",
        task_name="occurrence",
        model_specs=make_classification_model_specs(),
        split_definitions=splits,
    )
    results.to_csv(OCCURRENCE_RESULTS_PATH, index=False)
    predictions.to_parquet(OCCURRENCE_PREDICTIONS_PATH, index=False)
    feature_importance.to_csv(OCCURRENCE_FEATURE_IMPORTANCE_PATH, index=False)
    compute_bias_diagnostics(predictions, "occurrence").to_csv(OCCURRENCE_BIAS_DIAGNOSTICS_PATH, index=False)
    calibration_table(predictions).to_csv(OCCURRENCE_CALIBRATION_PATH, index=False)
    build_leakage_audit(predictor_frame, "occurrence").to_csv(OCCURRENCE_LEAKAGE_AUDIT_PATH, index=False)
    joblib.dump(best_model, OCCURRENCE_BEST_MODEL_PATH)
    return results, predictions, feature_importance


def run_intensity_ml(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = load_modeling_frame("strict_main", observed_only=True) if frame is None else frame.copy()
    work = work[work["occurrence"]].copy()
    splits = make_split_definitions(work, include_spatial=True, target_col="intensity")
    results, predictions, feature_importance, best_model = fit_supervised_models(
        work,
        target_col="intensity",
        task_name="intensity",
        model_specs=make_regression_model_specs(include_count_models=True),
        split_definitions=splits,
        regression_target_transform="log1p",
    )
    results.to_csv(INTENSITY_RESULTS_PATH, index=False)
    predictions.to_parquet(INTENSITY_PREDICTIONS_PATH, index=False)
    feature_importance.to_csv(INTENSITY_FEATURE_IMPORTANCE_PATH, index=False)
    compute_bias_diagnostics(predictions, "intensity").to_csv(INTENSITY_BIAS_DIAGNOSTICS_PATH, index=False)
    build_leakage_audit(work, "intensity").to_csv(INTENSITY_LEAKAGE_AUDIT_PATH, index=False)
    joblib.dump(best_model, INTENSITY_BEST_MODEL_PATH)
    return results, predictions, feature_importance


def run_resolution_time_ml(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = load_modeling_frame("strict_main", observed_only=True) if frame is None else frame.copy()
    work = work[work["occurrence"]].copy()
    closed_work = work[work["resolution"].notna()].copy()
    time_splits = make_split_definitions(closed_work, include_spatial=True, target_col="resolution")
    time_results, time_predictions, time_importance, time_model = fit_supervised_models(
        closed_work,
        target_col="resolution",
        task_name="resolution_time",
        model_specs=make_regression_model_specs(include_count_models=False),
        split_definitions=time_splits,
        regression_target_transform="log1p",
        closed_only=True,
    )
    time_results.to_csv(RESOLUTION_TIME_RESULTS_PATH, index=False)
    time_predictions.to_parquet(RESOLUTION_PREDICTIONS_PATH, index=False)
    time_importance.assign(subtask="resolution_time_regression").to_csv(RESOLUTION_FEATURE_IMPORTANCE_PATH, index=False)
    compute_bias_diagnostics(time_predictions, "resolution_time").to_csv(RESOLUTION_BIAS_DIAGNOSTICS_PATH, index=False)
    build_leakage_audit(closed_work, "resolution_time").to_csv(RESOLUTION_LEAKAGE_AUDIT_PATH, index=False)
    joblib.dump(time_model, RESOLUTION_BEST_TIME_MODEL_PATH)
    return time_results, time_predictions, time_importance


def run_resolution_ml(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = load_modeling_frame("strict_main", observed_only=True) if frame is None else frame.copy()
    work = work[work["occurrence"]].copy()
    splits = make_split_definitions(work, include_spatial=True, target_col="resolution_bool")

    closure_results, closure_predictions, closure_importance, closure_model = fit_supervised_models(
        work,
        target_col="resolution_bool",
        task_name="resolution_closure",
        model_specs=make_classification_model_specs(),
        split_definitions=splits,
    )

    closed_work = work[work["resolution"].notna()].copy()
    time_splits = make_split_definitions(closed_work, include_spatial=True, target_col="resolution")
    time_results, time_predictions, time_importance, time_model = fit_supervised_models(
        closed_work,
        target_col="resolution",
        task_name="resolution_time",
        model_specs=make_regression_model_specs(include_count_models=False),
        split_definitions=time_splits,
        regression_target_transform="log1p",
        closed_only=True,
    )

    closure_results.to_csv(RESOLUTION_CLOSURE_RESULTS_PATH, index=False)
    time_results.to_csv(RESOLUTION_TIME_RESULTS_PATH, index=False)
    # Cast y_true/y_pred to float before combining classification (bool) and regression
    # (float) frames — pd.to_numeric leaves bool as bool, so concat yields object dtype
    # which breaks Arrow parquet serialization.
    for _col in ("y_true", "y_pred"):
        closure_predictions[_col] = closure_predictions[_col].astype(float)
    predictions = pd.concat([closure_predictions, time_predictions], ignore_index=True)
    predictions.to_parquet(RESOLUTION_PREDICTIONS_PATH, index=False)
    importance = pd.concat(
        [
            closure_importance.assign(subtask="closure_classification"),
            time_importance.assign(subtask="resolution_time_regression"),
        ],
        ignore_index=True,
    )
    importance.to_csv(RESOLUTION_FEATURE_IMPORTANCE_PATH, index=False)
    compute_bias_diagnostics(time_predictions, "resolution_time").to_csv(RESOLUTION_BIAS_DIAGNOSTICS_PATH, index=False)
    build_leakage_audit(work, "resolution_closure").to_csv(RESOLUTION_LEAKAGE_AUDIT_PATH, index=False)
    joblib.dump(closure_model, RESOLUTION_BEST_CLOSURE_MODEL_PATH)
    joblib.dump(time_model, RESOLUTION_BEST_TIME_MODEL_PATH)
    return closure_results, time_results, predictions, importance


def discrete_bayesian_frame(frame: pd.DataFrame) -> pd.DataFrame:
    bayes = frame.copy()
    if "occurrence" in bayes.columns:
        bayes["occurrence"] = bayes["occurrence"].map({True: "reported", False: "matched_non_flood"}).astype("string")
    if "resolution_bool" in bayes.columns:
        bayes["resolution_bool"] = bayes["resolution_bool"].map({True: "closed", False: "open_or_unresolved"}).astype("string")
    if "intensity" in bayes.columns:
        bayes["intensity_bin"] = pd.qcut(safe_numeric(bayes["intensity"]).rank(method="first"), q=4, duplicates="drop").astype("string")
    if "resolution" in bayes.columns:
        closed_mask = bayes["resolution"].notna()
        bayes.loc[closed_mask, "resolution_bin"] = pd.qcut(safe_numeric(bayes.loc[closed_mask, "resolution"]).rank(method="first"), q=4, duplicates="drop").astype("string")
        bayes["resolution_bin"] = bayes["resolution_bin"].astype("string").fillna("unresolved_censored")
    for column in ["max_tide", "prec_depth_total", "elevation", "shore_dist", "edge_betweenness", "travel_time", "census_poverty_rate", "census_median_household_income"]:
        if column in bayes.columns:
            valid = safe_numeric(bayes[column])
            if valid.notna().sum() > 10:
                bayes[f"{column}_bin"] = pd.qcut(valid.rank(method="first"), q=4, duplicates="drop").astype("string")
    for column in ["road_class", "fema_fld_zone", "borough", "tide_polygon", "precipitation_polygon", "high_network_criticality"]:
        if column in bayes.columns:
            bayes[column] = bayes[column].astype("string").fillna("MISSING")
    columns = [
        column
        for column in [
            "occurrence",
            "resolution_bool",
            "intensity_bin",
            "resolution_bin",
            "max_tide_bin",
            "prec_depth_total_bin",
            "elevation_bin",
            "shore_dist_bin",
            "edge_betweenness_bin",
            "travel_time_bin",
            "census_poverty_rate_bin",
            "census_median_household_income_bin",
            "road_class",
            "fema_fld_zone",
            "borough",
            "tide_polygon",
            "precipitation_polygon",
            "high_network_criticality",
        ]
        if column in bayes.columns
    ]
    return bayes[columns].dropna(axis=0, how="any").copy()


def run_bayesian_network(frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    try:
        from pgmpy.estimators import BicScore, HillClimbSearch, MaximumLikelihoodEstimator
        from pgmpy.models import BayesianNetwork
    except Exception as exc:
        raise RuntimeError("pgmpy is required for Bayesian-network analysis") from exc

    work = load_modeling_frame("strict_main") if frame is None else frame.copy()
    discrete = discrete_bayesian_frame(work)
    if discrete.empty:
        raise RuntimeError("No rows available after discretization for Bayesian-network analysis")

    scorer = BicScore(discrete)
    search = HillClimbSearch(discrete)
    dag = search.estimate(scoring_method=scorer, max_indegree=3, show_progress=False)
    model = BayesianNetwork(dag.edges())
    model.fit(discrete, estimator=MaximumLikelihoodEstimator)

    edges_df = pd.DataFrame(list(model.edges()), columns=["source", "target"])
    edges_df["edge_type"] = "learned_structure"
    edges_df.to_csv(BAYESIAN_EDGES_PATH, index=False)

    selection_df = pd.DataFrame(
        [
            {
                "metric": "bic_score",
                "value": scorer.score(model),
            },
            {
                "metric": "n_nodes",
                "value": len(model.nodes()),
            },
            {
                "metric": "n_edges",
                "value": len(model.edges()),
            },
            {
                "metric": "n_rows_used",
                "value": len(discrete),
            },
        ]
    )
    selection_df.to_csv(BAYESIAN_MODEL_SELECTION_PATH, index=False)

    node_summary = pd.DataFrame(
        [
            {
                "node": node,
                "parents": ", ".join(model.get_parents(node)) if model.get_parents(node) else "",
                "children": ", ".join(model.get_children(node)) if model.get_children(node) else "",
                "n_states": discrete[node].nunique(dropna=True),
            }
            for node in model.nodes()
        ]
    )
    node_summary.to_csv(BAYESIAN_NODE_SUMMARY_PATH, index=False)
    return edges_df, selection_df, node_summary


def compact_modeling_summary(results_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    keep_cols = [column for column in ["task_name", "split_strategy", "model_name", "status"] + metric_cols + ["best_params", "notes"] if column in results_df.columns]
    return results_df[keep_cols].sort_values(metric_cols[0], ascending=False, kind="stable").reset_index(drop=True)