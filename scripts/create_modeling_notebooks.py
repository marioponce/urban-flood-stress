from __future__ import annotations

from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _imports_cell() -> str:
    return """from pathlib import Path
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

ROOT = Path.cwd()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)"""


def build_clustering_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 13 Clustering and Anomaly Detection

This notebook explores **flood-event archetypes** and **anomalous events** using the canonical event-level frame from:

`data/processed/modeling/filtered/final_analysis_strict_main.parquet`

Important interpretation rules:
- this is an **event-level** analysis, not complaint-level
- the main clustering run uses **observed flood events only**
- anomalies are **not automatically errors**; they can be physically important or institutionally unusual events
- cluster names are **exploratory labels**, not causal truth

Outputs written by this notebook:
- `data/processed/modeling/clustering_event_archetypes.parquet`
- `data/processed/modeling/anomaly_event_scores.parquet`
- `data/processed/modeling/clustering_model_selection.csv`"""
        )
    )

    cells.append(nbf.v4.new_code_cell(_imports_cell()))

    cells.append(
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    ANOMALY_SCORES_PATH,
    CLUSTERING_ARCHETYPES_GEOPARQUET_PATH,
    CLUSTERING_ARCHETYPES_PATH,
    CLUSTERING_MODEL_SELECTION_PATH,
    clustering_feature_sets,
    compact_modeling_summary,
    evaluate_clustering_algorithms,
    feature_catalog,
    load_modeling_frame,
    pca_embedding,
    summarize_feature_groups,
    unavailable_expected_features,
)

OPTIONAL = {
    "umap": importlib.util.find_spec("umap") is not None,
    "hdbscan": importlib.util.find_spec("hdbscan") is not None,
}

observed = load_modeling_frame(view_name="strict_main", include_geometry=False, observed_only=True)
catalog = feature_catalog(observed)
group_summary = summarize_feature_groups(observed)
missing_expected = unavailable_expected_features(observed)

print(f"observed flood events: {len(observed):,}")
print(f"unique segments: {observed['segment_id'].nunique():,}")
display(group_summary)
display(missing_expected)
display(pd.DataFrame([OPTIONAL]))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Run Clustering and Anomaly Suites

The shared modeling module tests:
- `KMeans`
- `AgglomerativeClustering`
- `GaussianMixture`
- `DBSCAN`
- `SpectralClustering` when computationally feasible

If `HDBSCAN` or `UMAP` are not installed, the notebook does **not** fail."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """selection_df, archetypes, anomalies = evaluate_clustering_algorithms(observed)

print(f"model selection rows: {len(selection_df):,}")
print(f"archetype rows: {len(archetypes):,}")
print(f"anomaly rows: {len(anomalies):,}")

display(
    selection_df.sort_values(
        ["feature_set", "silhouette_score", "calinski_harabasz_score"],
        ascending=[True, False, False],
        kind="stable",
    ).groupby("feature_set", as_index=False).head(8)
)
display(archetypes.head())
display(anomalies.head())"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## PCA View of the Combined Feature Set"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """combined_features = clustering_feature_sets(observed)["combined"]
pca_points = pca_embedding(observed, combined_features)
pca_plot = pca_points.merge(
    archetypes[["event_id", "combined_cluster_id", "cluster_label", "intensity", "resolution_bool"]],
    on="event_id",
    how="left",
)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

scatter = axes[0].scatter(
    pca_plot["pc1"],
    pca_plot["pc2"],
    c=pd.to_numeric(pca_plot["combined_cluster_id"], errors="coerce"),
    cmap="tab10",
    s=10,
    alpha=0.7,
)
axes[0].set_title("PCA Projection Colored by Combined Cluster")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(scatter, ax=axes[0], shrink=0.8)

axes[1].hist(anomalies["anomaly_score"].astype(float), bins=40, color="#DC2626", alpha=0.85)
axes[1].set_title("Anomaly Score Distribution")
axes[1].set_xlabel("ensemble anomaly score")
axes[1].set_ylabel("events")

plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Cluster Profiles and Sizes"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """cluster_counts = (
    archetypes["combined_cluster_id"]
    .value_counts(dropna=False)
    .rename_axis("combined_cluster_id")
    .reset_index(name="n_events")
    .sort_values("combined_cluster_id", kind="stable")
)

profile_cols = [
    column
    for column in [
        "max_tide",
        "prec_depth_total",
        "prec_duration_total",
        "elevation",
        "shore_dist",
        "edge_betweenness",
        "travel_time",
        "census_poverty_rate",
        "census_median_household_income",
        "intensity",
        "resolution",
    ]
    if column in archetypes.columns
]

cluster_profiles = archetypes.groupby("combined_cluster_id")[profile_cols].mean(numeric_only=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

axes[0].bar(cluster_counts["combined_cluster_id"].astype(str), cluster_counts["n_events"], color="#2563EB")
axes[0].set_title("Cluster Size Distribution")
axes[0].set_xlabel("combined_cluster_id")
axes[0].set_ylabel("events")

im = axes[1].imshow(cluster_profiles.to_numpy(dtype=float), aspect="auto", cmap="viridis")
axes[1].set_title("Cluster Profile Heatmap")
axes[1].set_xticks(range(len(cluster_profiles.columns)))
axes[1].set_xticklabels(cluster_profiles.columns, rotation=90)
axes[1].set_yticks(range(len(cluster_profiles.index)))
axes[1].set_yticklabels(cluster_profiles.index.astype(str))
plt.colorbar(im, ax=axes[1], shrink=0.8)

plt.show()

display(cluster_counts)
display(cluster_profiles)
display(
    archetypes.groupby(["combined_cluster_id", "cluster_label"], as_index=False)
    .agg(
        n_events=("event_id", "size"),
        mean_intensity=("intensity", "mean"),
        closure_rate=("resolution_bool", lambda s: pd.Series(s).astype("boolean").mean()),
    )
    .sort_values(["combined_cluster_id"], kind="stable")
)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Spatial QA/QC"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """try:
    geo = pd.read_parquet(CLUSTERING_ARCHETYPES_GEOPARQUET_PATH)
    geo = geo.to_crs(2263)
    top_anomalies = anomalies.head(250)[["event_id"]].merge(geo, on="event_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    geo.plot(ax=axes[0], column="combined_cluster_id", categorical=True, legend=True, linewidth=0.8, cmap="tab10")
    axes[0].set_title("Observed Events by Combined Cluster")
    axes[0].set_axis_off()

    geo.plot(ax=axes[1], color="#94A3B8", linewidth=0.5, alpha=0.35)
    top_anomalies.plot(ax=axes[1], color="#DC2626", linewidth=1.2, alpha=0.85)
    axes[1].set_title("Top Anomalous Events")
    axes[1].set_axis_off()

    plt.show()
except Exception as exc:
    print(f"Spatial QA/QC plot skipped: {exc}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Compact Summary

This notebook:
- used the canonical filtered event table as the modeling base
- kept clustering separate from causal interpretation
- named clusters with human-readable exploratory labels
- preserved anomaly flags as a diagnostic layer, not a data-cleaning deletion rule

Saved files:
- `data/processed/modeling/clustering_event_archetypes.parquet`
- `data/processed/modeling/anomaly_event_scores.parquet`
- `data/processed/modeling/clustering_model_selection.csv`"""
        )
    )

    nb["cells"] = cells
    return nb


def build_occurrence_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 14 Ocurrence ML

This notebook models:

`P(observed / reported flooding complaint event)`

Target:
- `occurrence`

Important label interpretation:
- `occurrence = True` means a **reported / observed** flood complaint event
- `occurrence = False` means a **matched non-flood control event**
- it does **not** guarantee physical absence of flooding

Leakage control:
- `intensity`, `resolution`, `resolution_bool`, `end`, and raw `duration` are excluded from predictors
- raw identifiers are excluded
- `storm_event_id` is used for grouping and split control, not as a predictor"""
        )
    )

    cells.append(nbf.v4.new_code_cell(_imports_cell()))

    cells.append(
        nbf.v4.new_code_cell(
            """from sklearn.calibration import calibration_curve

from project_name.modeling_stage import (
    OCCURRENCE_FEATURE_IMPORTANCE_PATH,
    OCCURRENCE_PREDICTIONS_PATH,
    OCCURRENCE_RESULTS_PATH,
    OCCURRENCE_LEAKAGE_AUDIT_PATH,
    compact_modeling_summary,
    feature_catalog,
    load_modeling_frame,
    run_occurrence_ml,
    summarize_feature_groups,
    unavailable_expected_features,
)

balanced = load_modeling_frame(view_name="strict_main", include_geometry=False)
catalog = feature_catalog(balanced)
group_summary = summarize_feature_groups(balanced)
missing_expected = unavailable_expected_features(balanced)
leakage_audit = pd.read_csv(OCCURRENCE_LEAKAGE_AUDIT_PATH) if OCCURRENCE_LEAKAGE_AUDIT_PATH.exists() else None

print(f"balanced rows: {len(balanced):,}")
print(balanced['occurrence'].value_counts(dropna=False).to_string())
display(group_summary)
display(missing_expected)
if leakage_audit is not None:
    display(leakage_audit[leakage_audit['excluded_from_predictors']].head(25))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Train and Save Occurrence Models"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """results, predictions, feature_importance = run_occurrence_ml(balanced)

summary = compact_modeling_summary(results, ["pr_auc", "recall", "balanced_accuracy"])
display(summary.head(20))
display(feature_importance.head(25))

print(f"results path: {OCCURRENCE_RESULTS_PATH}")
print(f"predictions path: {OCCURRENCE_PREDICTIONS_PATH}")
print(f"feature importance path: {OCCURRENCE_FEATURE_IMPORTANCE_PATH}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Performance by Model and Split"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """results = pd.read_csv(OCCURRENCE_RESULTS_PATH)
predictions = pd.read_parquet(OCCURRENCE_PREDICTIONS_PATH)
feature_importance = pd.read_csv(OCCURRENCE_FEATURE_IMPORTANCE_PATH)
leakage_audit = pd.read_csv(OCCURRENCE_LEAKAGE_AUDIT_PATH)

best_row = (
    results.query("status == 'ok'")
    .sort_values(["pr_auc", "recall", "balanced_accuracy"], ascending=False, kind="stable")
    .iloc[0]
)

best_predictions = predictions[
    (predictions["model_name"] == best_row["model_name"])
    & (predictions["split_strategy"] == best_row["split_strategy"])
]

fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

plot_results = results.query("status == 'ok'").copy()
plot_results["label"] = plot_results["model_name"] + " | " + plot_results["split_strategy"]
axes[0, 0].barh(plot_results["label"], plot_results["pr_auc"], color="#2563EB")
axes[0, 0].set_title("PR-AUC by Model and Split")
axes[0, 0].set_xlabel("PR-AUC")

axes[0, 1].barh(plot_results["label"], plot_results["recall"], color="#DC2626")
axes[0, 1].set_title("Recall by Model and Split")
axes[0, 1].set_xlabel("Recall")

cm = pd.crosstab(best_predictions["y_true"], best_predictions["y_pred"])
im = axes[1, 0].imshow(cm.to_numpy(), cmap="Blues")
axes[1, 0].set_title(f"Confusion Matrix: {best_row['model_name']} | {best_row['split_strategy']}")
axes[1, 0].set_xticks(range(len(cm.columns)))
axes[1, 0].set_xticklabels(cm.columns.astype(str))
axes[1, 0].set_yticks(range(len(cm.index)))
axes[1, 0].set_yticklabels(cm.index.astype(str))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1, 0].text(j, i, int(cm.iloc[i, j]), ha="center", va="center", color="black")
plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

top_features = feature_importance.head(20).sort_values("importance", kind="stable")
axes[1, 1].barh(top_features["feature"], top_features["importance"], color="#059669")
axes[1, 1].set_title("Top Feature Importances")
axes[1, 1].set_xlabel("importance")

plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Calibration Check

If a model produced probabilities, we inspect calibration on the best split.
Natural prevalence-aware evaluation is **not fully available** here because the current event universe provides balanced matched negatives rather than a full natural non-flood event universe."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """if "y_score" in best_predictions.columns and best_predictions["y_score"].notna().any():
    frac_pos, mean_pred = calibration_curve(
        best_predictions["y_true"].astype(bool),
        best_predictions["y_score"].astype(float),
        n_bins=10,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B")
    ax.plot(mean_pred, frac_pos, marker="o", color="#2563EB")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed event fraction")
    plt.show()
else:
    print("Calibration plot skipped because the best prediction table does not contain usable probabilities.")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Compact Summary

What was modeled?
- probability of **observed / reported** flooding

What dataset was used?
- canonical `strict_main` filtered analysis table

What assumptions were made?
- matched negatives are useful controls for reported-flood classification
- unresolved physical flooding may still exist among negatives, so interpretation remains cautious

What files were saved?
- `data/processed/modeling/results_occurrence_ml.csv`
- `data/processed/modeling/predictions_occurrence_ml.parquet`
- `data/processed/modeling/feature_importance_occurrence.csv`
- `data/processed/modeling/best_occurrence_model.joblib`"""
        )
    )

    nb["cells"] = cells
    return nb


def build_intensity_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 15 Intensity ML

This notebook predicts:

`intensity = number of complaints associated with the flood event`

Important scope:
- only **observed positive flood events** are used by default
- matched negatives are excluded from the main intensity model
- the notebook compares a log-transformed target strategy for better behavior under right-skewed counts"""
        )
    )

    cells.append(nbf.v4.new_code_cell(_imports_cell()))

    cells.append(
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    INTENSITY_FEATURE_IMPORTANCE_PATH,
    INTENSITY_PREDICTIONS_PATH,
    INTENSITY_RESULTS_PATH,
    INTENSITY_LEAKAGE_AUDIT_PATH,
    compact_modeling_summary,
    feature_catalog,
    load_modeling_frame,
    run_intensity_ml,
    summarize_feature_groups,
)

observed = load_modeling_frame(view_name="strict_main", include_geometry=False, observed_only=True)
display(summarize_feature_groups(observed))
display(feature_catalog(observed).head(40))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
pd.to_numeric(observed["intensity"], errors="coerce").plot.hist(ax=axes[0], bins=40, color="#2563EB", alpha=0.85)
axes[0].set_title("Observed Intensity")
axes[0].set_xlabel("complaint count")

np.log1p(pd.to_numeric(observed["intensity"], errors="coerce")).plot.hist(ax=axes[1], bins=40, color="#DC2626", alpha=0.85)
axes[1].set_title("log1p(Intensity)")
axes[1].set_xlabel("log1p complaint count")

plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """results, predictions, feature_importance = run_intensity_ml(observed)

summary = compact_modeling_summary(results, ["mae", "rmse", "r2"])
display(summary.head(20))
display(feature_importance.head(25))

print(f"results path: {INTENSITY_RESULTS_PATH}")
print(f"predictions path: {INTENSITY_PREDICTIONS_PATH}")
print(f"feature importance path: {INTENSITY_FEATURE_IMPORTANCE_PATH}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Error Diagnostics"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """results = pd.read_csv(INTENSITY_RESULTS_PATH)
predictions = pd.read_parquet(INTENSITY_PREDICTIONS_PATH)
feature_importance = pd.read_csv(INTENSITY_FEATURE_IMPORTANCE_PATH)

best_row = (
    results.query("status == 'ok'")
    .sort_values(["mae", "rmse"], ascending=[True, True], kind="stable")
    .iloc[0]
)

best_predictions = predictions[
    (predictions["model_name"] == best_row["model_name"])
    & (predictions["split_strategy"] == best_row["split_strategy"])
].copy()
best_predictions["residual"] = best_predictions["y_true"] - best_predictions["y_pred"]

plot_frame = best_predictions.merge(observed[["event_id", "borough", "fema_fld_zone", "tide_polygon", "precipitation_polygon"]], on="event_id", how="left")

fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

axes[0, 0].scatter(best_predictions["y_true"], best_predictions["y_pred"], alpha=0.5, color="#2563EB")
axes[0, 0].plot(
    [best_predictions["y_true"].min(), best_predictions["y_true"].max()],
    [best_predictions["y_true"].min(), best_predictions["y_true"].max()],
    linestyle="--",
    color="#64748B",
)
axes[0, 0].set_title("Observed vs Predicted Intensity")
axes[0, 0].set_xlabel("observed")
axes[0, 0].set_ylabel("predicted")

axes[0, 1].hist(best_predictions["residual"].astype(float), bins=40, color="#DC2626", alpha=0.85)
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("observed - predicted")

top_features = feature_importance.head(20).sort_values("importance", kind="stable")
axes[1, 0].barh(top_features["feature"], top_features["importance"], color="#059669")
axes[1, 0].set_title("Top Feature Importances")
axes[1, 0].set_xlabel("importance")

borough_error = (
    plot_frame.groupby("borough", as_index=False)
    .agg(
        mae=("residual", lambda s: np.mean(np.abs(pd.to_numeric(s, errors='coerce')))),
        mean_observed=("y_true", "mean"),
        mean_predicted=("y_pred", "mean"),
    )
    .sort_values("mae", ascending=False, kind="stable")
)
axes[1, 1].barh(borough_error["borough"], borough_error["mae"], color="#7C3AED")
axes[1, 1].set_title("MAE by Borough")
axes[1, 1].set_xlabel("MAE")

plt.show()
display(borough_error)"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Compact Summary

What was modeled?
- observed complaint-event intensity

What dataset was used?
- positive observed events from the canonical filtered table

What assumptions were made?
- count skew is handled through a log-target training strategy in the shared modeling module
- negatives are excluded from the main count model

What files were saved?
- `data/processed/modeling/results_intensity_ml.csv`
- `data/processed/modeling/predictions_intensity_ml.parquet`
- `data/processed/modeling/feature_importance_intensity.csv`
- `data/processed/modeling/best_intensity_model.joblib`"""
        )
    )

    nb["cells"] = cells
    return nb


def build_resolution_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# 16 Resolution ML

This notebook models flood-event resolution in two linked ways:

1. `resolution_bool`: whether the event was closed
2. `resolution`: time to closure for events that were actually closed

Critical interpretation rule:
- `resolution = NA` is **not ordinary random missingness**
- it means the event was still open / unresolved in the processed table
- that is informative and is modeled through the **closure classification task**
- the regression task uses only events with valid closure time

This is conceptually close to **informative censoring**, so unresolved events are kept for classification and excluded from the closure-time regression target."""
        )
    )

    cells.append(nbf.v4.new_code_cell(_imports_cell()))

    cells.append(
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    RESOLUTION_CLOSURE_RESULTS_PATH,
    RESOLUTION_TIME_RESULTS_PATH,
    RESOLUTION_PREDICTIONS_PATH,
    RESOLUTION_FEATURE_IMPORTANCE_PATH,
    RESOLUTION_LEAKAGE_AUDIT_PATH,
    compact_modeling_summary,
    feature_catalog,
    load_modeling_frame,
    run_resolution_ml,
    summarize_feature_groups,
)

OPTIONAL_SURVIVAL_LIBS = {
    "lifelines": importlib.util.find_spec("lifelines") is not None,
    "sksurv": importlib.util.find_spec("sksurv") is not None,
}

observed = load_modeling_frame(view_name="strict_main", include_geometry=False, observed_only=True)

print(f"observed events: {len(observed):,}")
print(f"closed events: {int(observed['resolution_bool'].fillna(False).sum()):,}")
print(f"open / unresolved events: {int((~observed['resolution_bool'].fillna(False)).sum()):,}")
display(pd.DataFrame([OPTIONAL_SURVIVAL_LIBS]))
display(summarize_feature_groups(observed))
display(feature_catalog(observed).head(40))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
observed["resolution_bool"].astype("boolean").value_counts(dropna=False).plot.bar(ax=axes[0], color=["#2563EB", "#DC2626"])
axes[0].set_title("Closure Status Distribution")
axes[0].set_xlabel("resolution_bool")
axes[0].set_ylabel("events")

pd.to_numeric(observed["resolution"], errors="coerce").dropna().plot.hist(ax=axes[1], bins=40, color="#059669", alpha=0.85)
axes[1].set_title("Resolution Hours for Closed Events")
axes[1].set_xlabel("resolution hours")
plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """closure_results, time_results, predictions, feature_importance = run_resolution_ml(observed)

display(compact_modeling_summary(closure_results, ["pr_auc", "recall", "balanced_accuracy"]).head(20))
display(compact_modeling_summary(time_results, ["mae", "rmse", "r2"]).head(20))
display(feature_importance.head(30))

print(f"closure results path: {RESOLUTION_CLOSURE_RESULTS_PATH}")
print(f"time results path: {RESOLUTION_TIME_RESULTS_PATH}")
print(f"predictions path: {RESOLUTION_PREDICTIONS_PATH}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Closure Classification Diagnostics"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """closure_results = pd.read_csv(RESOLUTION_CLOSURE_RESULTS_PATH)
time_results = pd.read_csv(RESOLUTION_TIME_RESULTS_PATH)
predictions = pd.read_parquet(RESOLUTION_PREDICTIONS_PATH)
feature_importance = pd.read_csv(RESOLUTION_FEATURE_IMPORTANCE_PATH)
leakage_audit = pd.read_csv(RESOLUTION_LEAKAGE_AUDIT_PATH)

best_closure = (
    closure_results.query("status == 'ok'")
    .sort_values(["pr_auc", "recall", "balanced_accuracy"], ascending=False, kind="stable")
    .iloc[0]
)
best_time = (
    time_results.query("status == 'ok'")
    .sort_values(["mae", "rmse"], ascending=[True, True], kind="stable")
    .iloc[0]
)

closure_pred = predictions[
    (predictions["target"] == "resolution_bool")
    & (predictions["model_name"] == best_closure["model_name"])
    & (predictions["split_strategy"] == best_closure["split_strategy"])
].copy()

time_pred = predictions[
    (predictions["target"] == "resolution")
    & (predictions["model_name"] == best_time["model_name"])
    & (predictions["split_strategy"] == best_time["split_strategy"])
].copy()
time_pred["residual"] = time_pred["y_true"] - time_pred["y_pred"]

fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

plot_closure = closure_results.query("status == 'ok'").copy()
plot_closure["label"] = plot_closure["model_name"] + " | " + plot_closure["split_strategy"]
axes[0, 0].barh(plot_closure["label"], plot_closure["pr_auc"], color="#2563EB")
axes[0, 0].set_title("Closure Classification PR-AUC")

cm = pd.crosstab(closure_pred["y_true"], closure_pred["y_pred"])
im = axes[0, 1].imshow(cm.to_numpy(), cmap="Blues")
axes[0, 1].set_title(f"Closure Confusion Matrix: {best_closure['model_name']} | {best_closure['split_strategy']}")
axes[0, 1].set_xticks(range(len(cm.columns)))
axes[0, 1].set_xticklabels(cm.columns.astype(str))
axes[0, 1].set_yticks(range(len(cm.index)))
axes[0, 1].set_yticklabels(cm.index.astype(str))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0, 1].text(j, i, int(cm.iloc[i, j]), ha="center", va="center", color="black")
plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

axes[1, 0].scatter(time_pred["y_true"], time_pred["y_pred"], alpha=0.5, color="#059669")
axes[1, 0].plot(
    [time_pred["y_true"].min(), time_pred["y_true"].max()],
    [time_pred["y_true"].min(), time_pred["y_true"].max()],
    linestyle="--",
    color="#64748B",
)
axes[1, 0].set_title("Observed vs Predicted Resolution Time")
axes[1, 0].set_xlabel("observed hours")
axes[1, 0].set_ylabel("predicted hours")

axes[1, 1].hist(pd.to_numeric(time_pred["residual"], errors="coerce").dropna(), bins=40, color="#DC2626", alpha=0.85)
axes[1, 1].set_title("Resolution-Time Residuals")
axes[1, 1].set_xlabel("observed - predicted hours")

plt.show()
display(leakage_audit[leakage_audit["excluded_from_predictors"]].head(30))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Survival-Analysis Extension

If survival libraries are available, a future extension could model time-to-closure more directly.
For now:
- unresolved events remain in the closure classifier
- unresolved events are treated as censored / open and excluded from the closure-time regression target"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Compact Summary

What was modeled?
- closure probability and closure time

What dataset was used?
- observed positive events from the canonical filtered table

What assumptions were made?
- `resolution = NA` means unresolved / open, not random missingness
- unresolved events therefore carry signal through `resolution_bool`
- closure-time regression is restricted to closed events

What files were saved?
- `data/processed/modeling/results_resolution_closure_ml.csv`
- `data/processed/modeling/results_resolution_time_ml.csv`
- `data/processed/modeling/predictions_resolution_ml.parquet`
- `data/processed/modeling/feature_importance_resolution.csv`
- `data/processed/modeling/best_resolution_closure_model.joblib`
- `data/processed/modeling/best_resolution_time_model.joblib`"""
        )
    )

    nb["cells"] = cells
    return nb


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the canonical generator aligned with the corrected course-aligned
    # copy notebooks: richer diagnostics, no out-of-scope models, and
    # resolution-time modeling as the main resolution task.
    try:
        from scripts.create_modeling_copy_notebooks import (
            clustering_copy,
            intensity_copy,
            occurrence_copy,
            resolution_copy,
        )

        notebooks = {
            "13_clustering-anomaly.ipynb": clustering_copy(),
            "14_ocurrence-ml.ipynb": occurrence_copy(),
            "15_intensity-ml.ipynb": intensity_copy(),
            "16_resolution-ml.ipynb": resolution_copy(),
        }
        for name, notebook in notebooks.items():
            (NOTEBOOKS_DIR / name).write_text(nbf.writes(notebook))
        return
    except Exception:
        # Fall back to the original builders if the copy generator is not
        # importable in a stripped-down environment.
        pass

    notebooks = {
        "13_clustering-anomaly.ipynb": build_clustering_notebook(),
        "14_ocurrence-ml.ipynb": build_occurrence_notebook(),
        "15_intensity-ml.ipynb": build_intensity_notebook(),
        "16_resolution-ml.ipynb": build_resolution_notebook(),
    }

    for name, notebook in notebooks.items():
        (NOTEBOOKS_DIR / name).write_text(nbf.writes(notebook))


if __name__ == "__main__":
    main()
