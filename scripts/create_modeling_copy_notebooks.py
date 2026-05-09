from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def imports_cell() -> str:
    return """from pathlib import Path
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


def clustering_copy() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 13 Clustering and Anomaly Detection

This copy notebook keeps the unsupervised stage aligned with the course workflow.

Allowed methods used here:
- KMeans
- Agglomerative clustering
- DBSCAN
- Gaussian Mixture Models
- Isolation Forest
- Local Outlier Factor

The unit of analysis is the flood event. Clusters are exploratory archetypes, and anomaly flags are diagnostics rather than data-cleaning rules."""
        ),
        nbf.v4.new_code_cell(imports_cell()),
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    ANOMALY_SCORES_PATH,
    CLUSTERING_ARCHETYPES_GEOPARQUET_PATH,
    CLUSTERING_ARCHETYPES_PATH,
    CLUSTERING_MODEL_SELECTION_PATH,
    clustering_feature_sets,
    evaluate_clustering_algorithms,
    feature_catalog,
    load_modeling_frame,
    pca_embedding,
    summarize_feature_groups,
    unavailable_expected_features,
)

observed = load_modeling_frame(view_name="strict_main", include_geometry=False, observed_only=True)
print(f"observed flood events: {len(observed):,}")
print(f"unique segments: {observed['segment_id'].nunique():,}")
display(summarize_feature_groups(observed))
display(unavailable_expected_features(observed))
display(feature_catalog(observed).head(40))"""
        ),
        nbf.v4.new_code_cell(
            """selection_df, archetypes, anomalies = evaluate_clustering_algorithms(observed)

display(
    selection_df.sort_values(
        ["feature_set", "silhouette_score", "calinski_harabasz_score"],
        ascending=[True, False, False],
        kind="stable",
    ).groupby("feature_set", as_index=False).head(8)
)
display(archetypes.head())
display(anomalies.head())

print(f"saved: {CLUSTERING_MODEL_SELECTION_PATH}")
print(f"saved: {CLUSTERING_ARCHETYPES_PATH}")
print(f"saved: {ANOMALY_SCORES_PATH}")"""
        ),
        nbf.v4.new_code_cell(
            """combined_features = clustering_feature_sets(observed)["combined"]
pca_points = pca_embedding(observed, combined_features)
pca_plot = pca_points.merge(
    archetypes[["event_id", "combined_cluster_id", "cluster_label", "intensity", "resolution"]],
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

axes[1].hist(pd.to_numeric(anomalies["anomaly_score"], errors="coerce").dropna(), bins=40, color="#DC2626", alpha=0.85)
axes[1].set_title("Anomaly Score Distribution")
axes[1].set_xlabel("ensemble anomaly score")
axes[1].set_ylabel("events")
plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """profile_cols = [
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
cluster_counts = archetypes["combined_cluster_id"].value_counts(dropna=False).sort_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
cluster_counts.plot.bar(ax=axes[0], color="#2563EB")
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

display(cluster_profiles)
display(
    archetypes.groupby(["combined_cluster_id", "cluster_label"], as_index=False)
    .agg(
        n_events=("event_id", "size"),
        mean_intensity=("intensity", "mean"),
        mean_resolution_hours=("resolution", "mean"),
    )
)"""
        ),
        nbf.v4.new_code_cell(
            """try:
    geo = pd.read_parquet(CLUSTERING_ARCHETYPES_GEOPARQUET_PATH).to_crs(2263)
    top_anomalies = anomalies.head(250)[["event_id"]].merge(geo, on="event_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    geo.plot(ax=axes[0], column="combined_cluster_id", categorical=True, legend=True, linewidth=0.8, cmap="tab10")
    axes[0].set_title("Observed Events by Cluster")
    axes[0].set_axis_off()

    geo.plot(ax=axes[1], color="#94A3B8", linewidth=0.5, alpha=0.35)
    top_anomalies.plot(ax=axes[1], color="#DC2626", linewidth=1.2, alpha=0.85)
    axes[1].set_title("Top Anomalous Events")
    axes[1].set_axis_off()
    plt.show()
except Exception as exc:
    print(f"Spatial QA/QC plot skipped: {exc}")"""
        ),
    ]
    return nb


def occurrence_copy() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 14 Ocurrence ML

This copy notebook models `P(observed / reported flooding complaint event)`.

The negative label means no observed 311 flood complaint during the matched event window. It is not proof of physical no-flooding.

Split rule:
- one classic stratified `75/15/10` train/validation/test split
- stratification prioritizes the target plus `mayoral_administration` when feasible
- model selection is a hyperparameter-grid sweep on train/validation; test is untouched final evaluation

Models are restricted to course-aligned supervised methods:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- optional XGBoost / LightGBM / CatBoost only if installed and already part of the workflow"""
        ),
        nbf.v4.new_code_cell(imports_cell()),
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    OCCURRENCE_BIAS_DIAGNOSTICS_PATH,
    OCCURRENCE_CALIBRATION_PATH,
    OCCURRENCE_FEATURE_IMPORTANCE_PATH,
    OCCURRENCE_LEAKAGE_AUDIT_PATH,
    OCCURRENCE_PREDICTIONS_PATH,
    OCCURRENCE_RESULTS_PATH,
    compact_modeling_summary,
    feature_catalog,
    load_modeling_frame,
    run_occurrence_ml,
    summarize_feature_groups,
    unavailable_expected_features,
)

balanced = load_modeling_frame(view_name="strict_main", include_geometry=False)
print(f"balanced rows: {len(balanced):,}")
print(balanced["occurrence"].value_counts(dropna=False).to_string())
display(summarize_feature_groups(balanced))
display(unavailable_expected_features(balanced))
display(feature_catalog(balanced).head(40))"""
        ),
        nbf.v4.new_code_cell(
            """results, predictions, feature_importance = run_occurrence_ml(balanced)

display(compact_modeling_summary(results, ["pr_auc", "recall", "balanced_accuracy"]).head(30))
display(results.sort_values(["pr_auc", "recall"], ascending=False, kind="stable").head(30))
display(feature_importance.head(30))

print(f"saved: {OCCURRENCE_RESULTS_PATH}")
print(f"saved: {OCCURRENCE_PREDICTIONS_PATH}")
print(f"saved: {OCCURRENCE_FEATURE_IMPORTANCE_PATH}")
print(f"saved: {OCCURRENCE_BIAS_DIAGNOSTICS_PATH}")
print(f"saved: {OCCURRENCE_CALIBRATION_PATH}")
print("incremental checkpoints: data/processed/modeling/diagnostics/modeling_stage/checkpoints/occurrence")"""
        ),
        nbf.v4.new_code_cell(
            """results = pd.read_csv(OCCURRENCE_RESULTS_PATH)
predictions = pd.read_parquet(OCCURRENCE_PREDICTIONS_PATH)
feature_importance = pd.read_csv(OCCURRENCE_FEATURE_IMPORTANCE_PATH)
bias = pd.read_csv(OCCURRENCE_BIAS_DIAGNOSTICS_PATH)
calibration = pd.read_csv(OCCURRENCE_CALIBRATION_PATH)
leakage_audit = pd.read_csv(OCCURRENCE_LEAKAGE_AUDIT_PATH)

metric_cols = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
    "brier_score",
    "train_pr_auc",
    "validation_pr_auc",
    "test_pr_auc",
    "cv_validation_primary_score",
    "train_vs_validation_gap",
    "validation_vs_test_gap",
    "possible_overfitting",
    "possible_test_degradation",
]
display(results[["model_name", "split_strategy", "status"] + [c for c in metric_cols if c in results.columns]].query("status == 'ok'"))
display(bias.head(40))
display(leakage_audit[leakage_audit["excluded_from_predictors"]].head(30))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Archetype Diagnostics for False Negatives

Clusters were learned from observed positive flood events only, so they are not used as occurrence predictors.
They are still useful for asking which reported-event archetypes are missed by the classifier."""
        ),
        nbf.v4.new_code_cell(
            """CLUSTER_PATH = ROOT / "data" / "processed" / "modeling" / "clustering_event_archetypes.parquet"
ANOMALY_PATH = ROOT / "data" / "processed" / "modeling" / "anomaly_event_scores.parquet"

if CLUSTER_PATH.exists() and ANOMALY_PATH.exists():
    clusters = pd.read_parquet(CLUSTER_PATH)[
        ["event_id", "combined_cluster_id", "cluster_label"]
    ].copy()
    anomalies = pd.read_parquet(ANOMALY_PATH)[
        ["event_id", "anomaly_score", "anomaly_flag", "anomaly_method_agreement"]
    ].copy()
    occurrence_cluster_diag = (
        predictions.merge(clusters, on="event_id", how="left")
        .merge(anomalies, on="event_id", how="left")
    )
    occurrence_cluster_diag = occurrence_cluster_diag[occurrence_cluster_diag["set_name"].eq("test")].copy()
    occurrence_cluster_diag = occurrence_cluster_diag[occurrence_cluster_diag["y_true"].astype(bool)].copy()
    occurrence_cluster_diag["false_negative"] = ~occurrence_cluster_diag["y_pred"].astype(bool)
    false_negative_by_cluster = (
        occurrence_cluster_diag.groupby(["combined_cluster_id", "cluster_label"], dropna=False)
        .agg(
            n_positive_test_events=("event_id", "size"),
            false_negative_count=("false_negative", "sum"),
            false_negative_rate=("false_negative", "mean"),
            mean_anomaly_score=("anomaly_score", "mean"),
        )
        .reset_index()
        .sort_values("false_negative_rate", ascending=False, kind="stable")
    )
    false_negative_by_cluster.to_csv(
        ROOT / "data" / "processed" / "modeling" / "diagnostics_occurrence_by_cluster.csv",
        index=False,
    )
    display(false_negative_by_cluster)
else:
    print("Cluster/anomaly outputs are not available yet. Run 13_clustering-anomaly first.")"""
        ),
        nbf.v4.new_code_cell(
            """best_row = (
    results.query("status == 'ok'")
    .sort_values(["pr_auc", "recall", "balanced_accuracy"], ascending=False, kind="stable")
    .iloc[0]
)
best_predictions = predictions[
    (predictions["model_name"] == best_row["model_name"])
    & (predictions["split_strategy"] == best_row["split_strategy"])
    & (predictions["set_name"] == "test")
]

fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
plot_results = results.query("status == 'ok'").copy()
plot_results["label"] = plot_results["model_name"] + " | " + plot_results["split_strategy"]
axes[0, 0].barh(plot_results["label"], plot_results["pr_auc"], color="#2563EB")
axes[0, 0].set_title("PR-AUC by Model and Split")
axes[0, 0].set_xlabel("PR-AUC")

axes[0, 1].barh(plot_results["label"], plot_results["false_negative_rate"], color="#DC2626")
axes[0, 1].set_title("False Negative Rate by Model and Split")
axes[0, 1].set_xlabel("FNR")

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
        ),
        nbf.v4.new_code_cell(
            """if not calibration.empty:
    best_calibration = calibration[
        (calibration["model_name"] == best_row["model_name"])
        & (calibration["split_strategy"] == best_row["split_strategy"])
        & (calibration["set_name"] == "test")
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B")
    ax.plot(
        best_calibration["mean_predicted_probability"],
        best_calibration["observed_rate"],
        marker="o",
        color="#2563EB",
    )
    ax.set_title("Calibration Curve")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed event fraction")
    plt.show()
else:
    print("Calibration table is empty.")"""
        ),
    ]
    return nb


def intensity_copy() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 15 Intensity ML

This copy notebook predicts event intensity, defined as the number of complaints in an observed flood event.

Only reported positive events are used in the default model. Matched negatives are not part of the count model.

Split rule:
- one classic stratified `75/15/10` train/validation/test split
- stratification uses intensity bands plus `mayoral_administration` when feasible
- model selection is a hyperparameter-grid sweep on train/validation; test is untouched final evaluation"""
        ),
        nbf.v4.new_code_cell(imports_cell()),
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    INTENSITY_BIAS_DIAGNOSTICS_PATH,
    INTENSITY_FEATURE_IMPORTANCE_PATH,
    INTENSITY_LEAKAGE_AUDIT_PATH,
    INTENSITY_PREDICTIONS_PATH,
    INTENSITY_RESULTS_PATH,
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
        ),
        nbf.v4.new_code_cell(
            """results, predictions, feature_importance = run_intensity_ml(observed)

display(compact_modeling_summary(results, ["mae", "rmse", "r2"]).head(30))
display(results.sort_values(["mae", "rmse"], ascending=[True, True], kind="stable").head(30))
display(feature_importance.head(30))

print(f"saved: {INTENSITY_RESULTS_PATH}")
print(f"saved: {INTENSITY_PREDICTIONS_PATH}")
print(f"saved: {INTENSITY_FEATURE_IMPORTANCE_PATH}")
print(f"saved: {INTENSITY_BIAS_DIAGNOSTICS_PATH}")
print("incremental checkpoints: data/processed/modeling/diagnostics/modeling_stage/checkpoints/intensity")"""
        ),
        nbf.v4.new_code_cell(
            """results = pd.read_csv(INTENSITY_RESULTS_PATH)
predictions = pd.read_parquet(INTENSITY_PREDICTIONS_PATH)
feature_importance = pd.read_csv(INTENSITY_FEATURE_IMPORTANCE_PATH)
bias = pd.read_csv(INTENSITY_BIAS_DIAGNOSTICS_PATH)
leakage_audit = pd.read_csv(INTENSITY_LEAKAGE_AUDIT_PATH)

metric_cols = [
    "rmse",
    "mae",
    "r2",
    "spearman_correlation",
    "pearson_correlation",
    "median_absolute_error",
    "train_mae",
    "validation_mae",
    "test_mae",
    "cv_validation_primary_score",
    "train_vs_validation_gap",
    "validation_vs_test_gap",
    "possible_overfitting",
    "possible_test_degradation",
]
display(results[["model_name", "split_strategy", "status"] + [c for c in metric_cols if c in results.columns]].query("status == 'ok'"))
display(bias.head(50))
display(leakage_audit[leakage_audit["excluded_from_predictors"]].head(30))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Archetype and Anomaly Error Diagnostics

Cluster labels are used here for residual diagnostics, not as a default predictor.
This helps identify whether heavy-event, compound, or unusual event regimes are systematically underpredicted."""
        ),
        nbf.v4.new_code_cell(
            """CLUSTER_PATH = ROOT / "data" / "processed" / "modeling" / "clustering_event_archetypes.parquet"
ANOMALY_PATH = ROOT / "data" / "processed" / "modeling" / "anomaly_event_scores.parquet"

if CLUSTER_PATH.exists() and ANOMALY_PATH.exists():
    clusters = pd.read_parquet(CLUSTER_PATH)[
        ["event_id", "combined_cluster_id", "cluster_label"]
    ].copy()
    anomalies = pd.read_parquet(ANOMALY_PATH)[
        ["event_id", "anomaly_score", "anomaly_flag", "anomaly_method_agreement"]
    ].copy()
    intensity_cluster_diag = (
        predictions.merge(clusters, on="event_id", how="left")
        .merge(anomalies, on="event_id", how="left")
    )
    intensity_cluster_diag = intensity_cluster_diag[intensity_cluster_diag["set_name"].eq("test")].copy()
    intensity_cluster_diag["residual"] = intensity_cluster_diag["y_true"] - intensity_cluster_diag["y_pred"]
    intensity_cluster_diag["absolute_error"] = intensity_cluster_diag["residual"].abs()
    intensity_by_cluster = (
        intensity_cluster_diag.groupby(["combined_cluster_id", "cluster_label"], dropna=False)
        .agg(
            n_events=("event_id", "size"),
            mae=("absolute_error", "mean"),
            mean_residual=("residual", "mean"),
            underprediction_rate=("residual", lambda s: pd.Series(s).gt(0).mean()),
            mean_anomaly_score=("anomaly_score", "mean"),
            anomaly_share=("anomaly_flag", lambda s: pd.Series(s).fillna(False).astype(bool).mean()),
        )
        .reset_index()
        .sort_values("mae", ascending=False, kind="stable")
    )
    intensity_by_cluster.to_csv(
        ROOT / "data" / "processed" / "modeling" / "diagnostics_intensity_by_cluster.csv",
        index=False,
    )
    display(intensity_by_cluster)
else:
    print("Cluster/anomaly outputs are not available yet. Run 13_clustering-anomaly first.")"""
        ),
        nbf.v4.new_code_cell(
            """best_row = (
    results.query("status == 'ok'")
    .sort_values(["mae", "rmse"], ascending=[True, True], kind="stable")
    .iloc[0]
)
best_predictions = predictions[
    (predictions["model_name"] == best_row["model_name"])
    & (predictions["split_strategy"] == best_row["split_strategy"])
    & (predictions["set_name"] == "test")
].copy()
best_predictions["residual"] = best_predictions["y_true"] - best_predictions["y_pred"]

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

axes[0, 1].hist(pd.to_numeric(best_predictions["residual"], errors="coerce").dropna(), bins=40, color="#DC2626", alpha=0.85)
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("observed - predicted")

top_features = feature_importance.head(20).sort_values("importance", kind="stable")
axes[1, 0].barh(top_features["feature"], top_features["importance"], color="#059669")
axes[1, 0].set_title("Top Feature Importances")
axes[1, 0].set_xlabel("importance")

bias.query("bias_dimension == 'borough'").plot.barh(x="group_value", y="mae", ax=axes[1, 1], color="#7C3AED", legend=False)
axes[1, 1].set_title("MAE by Borough")
axes[1, 1].set_xlabel("MAE")
plt.show()"""
        ),
    ]
    return nb


def resolution_copy() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# 16 Resolution Time ML

This copy notebook focuses on **time-to-resolution inference**.

We do not train a main model for `resolution_bool` here. Whether a case closed is useful metadata, but the modeling question for this stage is the length of resolution time among events with an observed closure.

Important treatment:
- `resolution = NA` means open / unresolved / censored
- unresolved events are summarized as context
- unresolved events are excluded from the regression target because no final resolution time exists

Split rule:
- one classic stratified `75/15/10` train/validation/test split
- stratification uses resolution-time bands plus `mayoral_administration` when feasible
- model selection is a hyperparameter-grid sweep on train/validation; test is untouched final evaluation"""
        ),
        nbf.v4.new_code_cell(imports_cell()),
        nbf.v4.new_code_cell(
            """from project_name.modeling_stage import (
    RESOLUTION_BIAS_DIAGNOSTICS_PATH,
    RESOLUTION_FEATURE_IMPORTANCE_PATH,
    RESOLUTION_LEAKAGE_AUDIT_PATH,
    RESOLUTION_PREDICTIONS_PATH,
    RESOLUTION_TIME_RESULTS_PATH,
    compact_modeling_summary,
    feature_catalog,
    load_modeling_frame,
    run_resolution_time_ml,
    summarize_feature_groups,
)

observed = load_modeling_frame(view_name="strict_main", include_geometry=False, observed_only=True)
closed = observed[observed["resolution"].notna()].copy()
open_or_censored = observed[observed["resolution"].isna()].copy()

print(f"observed flood events: {len(observed):,}")
print(f"closed with resolution time: {len(closed):,}")
print(f"open / unresolved / censored: {len(open_or_censored):,}")
display(summarize_feature_groups(observed))
display(feature_catalog(observed).head(40))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
observed["resolution_bool"].astype("boolean").value_counts(dropna=False).plot.bar(ax=axes[0], color=["#2563EB", "#DC2626"])
axes[0].set_title("Resolution Status")
axes[0].set_xlabel("resolution_bool")
axes[0].set_ylabel("events")

pd.to_numeric(closed["resolution"], errors="coerce").plot.hist(ax=axes[1], bins=40, color="#059669", alpha=0.85)
axes[1].set_title("Resolution Time for Closed Events")
axes[1].set_xlabel("hours")
plt.show()"""
        ),
        nbf.v4.new_code_cell(
            """results, predictions, feature_importance = run_resolution_time_ml(observed)

display(compact_modeling_summary(results, ["mae", "rmse", "r2"]).head(30))
display(results.sort_values(["mae", "rmse"], ascending=[True, True], kind="stable").head(30))
display(feature_importance.head(30))

print(f"saved: {RESOLUTION_TIME_RESULTS_PATH}")
print(f"saved: {RESOLUTION_PREDICTIONS_PATH}")
print(f"saved: {RESOLUTION_FEATURE_IMPORTANCE_PATH}")
print(f"saved: {RESOLUTION_BIAS_DIAGNOSTICS_PATH}")
print("incremental checkpoints: data/processed/modeling/diagnostics/modeling_stage/checkpoints/resolution_time")"""
        ),
        nbf.v4.new_code_cell(
            """results = pd.read_csv(RESOLUTION_TIME_RESULTS_PATH)
predictions = pd.read_parquet(RESOLUTION_PREDICTIONS_PATH)
feature_importance = pd.read_csv(RESOLUTION_FEATURE_IMPORTANCE_PATH)
bias = pd.read_csv(RESOLUTION_BIAS_DIAGNOSTICS_PATH)
leakage_audit = pd.read_csv(RESOLUTION_LEAKAGE_AUDIT_PATH)

metric_cols = [
    "rmse",
    "mae",
    "r2",
    "spearman_correlation",
    "pearson_correlation",
    "median_absolute_error",
    "train_mae",
    "validation_mae",
    "test_mae",
    "cv_validation_primary_score",
    "train_vs_validation_gap",
    "validation_vs_test_gap",
    "possible_overfitting",
    "possible_test_degradation",
]
display(results[["model_name", "split_strategy", "status"] + [c for c in metric_cols if c in results.columns]].query("status == 'ok'"))
display(bias.head(50))
display(leakage_audit[leakage_audit["excluded_from_predictors"]].head(30))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Archetype and Anomaly Error Diagnostics

The cluster output is especially useful for resolution-time inference because some event regimes may create larger service delays.
We use clusters and anomaly scores for diagnostics after prediction, not as the default regression target construction."""
        ),
        nbf.v4.new_code_cell(
            """CLUSTER_PATH = ROOT / "data" / "processed" / "modeling" / "clustering_event_archetypes.parquet"
ANOMALY_PATH = ROOT / "data" / "processed" / "modeling" / "anomaly_event_scores.parquet"

if CLUSTER_PATH.exists() and ANOMALY_PATH.exists():
    clusters = pd.read_parquet(CLUSTER_PATH)[
        ["event_id", "combined_cluster_id", "cluster_label"]
    ].copy()
    anomalies = pd.read_parquet(ANOMALY_PATH)[
        ["event_id", "anomaly_score", "anomaly_flag", "anomaly_method_agreement"]
    ].copy()
    resolution_cluster_diag = (
        predictions.merge(clusters, on="event_id", how="left")
        .merge(anomalies, on="event_id", how="left")
    )
    resolution_cluster_diag = resolution_cluster_diag[resolution_cluster_diag["set_name"].eq("test")].copy()
    resolution_cluster_diag["residual"] = resolution_cluster_diag["y_true"] - resolution_cluster_diag["y_pred"]
    resolution_cluster_diag["absolute_error"] = resolution_cluster_diag["residual"].abs()
    resolution_by_cluster = (
        resolution_cluster_diag.groupby(["combined_cluster_id", "cluster_label"], dropna=False)
        .agg(
            n_events=("event_id", "size"),
            mae_hours=("absolute_error", "mean"),
            mean_residual_hours=("residual", "mean"),
            underprediction_rate=("residual", lambda s: pd.Series(s).gt(0).mean()),
            mean_anomaly_score=("anomaly_score", "mean"),
            anomaly_share=("anomaly_flag", lambda s: pd.Series(s).fillna(False).astype(bool).mean()),
        )
        .reset_index()
        .sort_values("mae_hours", ascending=False, kind="stable")
    )
    resolution_by_cluster.to_csv(
        ROOT / "data" / "processed" / "modeling" / "diagnostics_resolution_by_cluster.csv",
        index=False,
    )
    display(resolution_by_cluster)
else:
    print("Cluster/anomaly outputs are not available yet. Run 13_clustering-anomaly first.")"""
        ),
        nbf.v4.new_code_cell(
            """best_row = (
    results.query("status == 'ok'")
    .sort_values(["mae", "rmse"], ascending=[True, True], kind="stable")
    .iloc[0]
)
best_predictions = predictions[
    (predictions["model_name"] == best_row["model_name"])
    & (predictions["split_strategy"] == best_row["split_strategy"])
    & (predictions["set_name"] == "test")
].copy()
best_predictions["residual"] = best_predictions["y_true"] - best_predictions["y_pred"]

fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
axes[0, 0].scatter(best_predictions["y_true"], best_predictions["y_pred"], alpha=0.5, color="#059669")
axes[0, 0].plot(
    [best_predictions["y_true"].min(), best_predictions["y_true"].max()],
    [best_predictions["y_true"].min(), best_predictions["y_true"].max()],
    linestyle="--",
    color="#64748B",
)
axes[0, 0].set_title("Observed vs Predicted Resolution Time")
axes[0, 0].set_xlabel("observed hours")
axes[0, 0].set_ylabel("predicted hours")

axes[0, 1].hist(pd.to_numeric(best_predictions["residual"], errors="coerce").dropna(), bins=40, color="#DC2626", alpha=0.85)
axes[0, 1].set_title("Residual Distribution")
axes[0, 1].set_xlabel("observed - predicted hours")

top_features = feature_importance.head(20).sort_values("importance", kind="stable")
axes[1, 0].barh(top_features["feature"], top_features["importance"], color="#2563EB")
axes[1, 0].set_title("Top Feature Importances")
axes[1, 0].set_xlabel("importance")

bias.query("bias_dimension == 'borough'").plot.barh(x="group_value", y="mae", ax=axes[1, 1], color="#7C3AED", legend=False)
axes[1, 1].set_title("Resolution MAE by Borough")
axes[1, 1].set_xlabel("MAE hours")
plt.show()"""
        ),
    ]
    return nb


def main() -> None:
    notebooks = {
        "13_clustering-anomaly copy.ipynb": clustering_copy(),
        "14_ocurrence-ml copy.ipynb": occurrence_copy(),
        "15_intensity-ml copy.ipynb": intensity_copy(),
        "16_resolution-ml copy.ipynb": resolution_copy(),
    }
    for name, notebook in notebooks.items():
        (NOTEBOOKS_DIR / name).write_text(nbf.writes(notebook))


if __name__ == "__main__":
    main()
