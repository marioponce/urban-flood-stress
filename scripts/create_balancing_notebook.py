from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "11_balancing.ipynb"
ALT_NOTEBOOK_PATH = ROOT / "notebooks" / "balancing.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            """# Balanced Flood Event Dataset

This notebook builds the **balanced event-level dataset** for supervised modeling of **observed / reported flooding complaints**.

Important label rule:
- Positive class: observed and reported flood event from 311.
- Negative class: no observed flood complaint on the sampled segment during the matched event window.
- Negative class does **not** guarantee physical absence of flooding.

This workflow preserves three products:
- natural dataset: `data/processed/modeling/flood_events_master_table.parquet`
- balanced dataset: `data/processed/modeling/flood_events_balanced.parquet`
- possible unreported flood candidates: `data/processed/modeling/possible_unreported_flood_candidates.parquet`

Leakage controls:
- same `segment_id` never reused as its own negative
- conservative storm-cluster temporal exclusion
- 100 m spatial exclusion around observed positives within the same storm cluster
- exclusion of possible unreported flood candidates from the negative pool"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.flood_event_balancing import (
    BALANCED_DIAGNOSTICS_DIR,
    BALANCED_GEOPARQUET_PATH,
    BALANCED_PARQUET_PATH,
    UNREPORTED_GEOPARQUET_PATH,
    UNREPORTED_PARQUET_PATH,
    VALIDATION_BALANCING_PATH,
    VALIDATION_MASTER_PATH,
    VALIDATION_UNREPORTED_PATH,
    build_balanced_dataset,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """balanced, unreported, validation_balancing, sampling_diagnostics = build_balanced_dataset(write_outputs=True)

master_path = ROOT / "data" / "processed" / "modeling" / "flood_events_master_table.parquet"
master = pd.read_parquet(master_path)

print(f"natural rows: {len(master):,}")
print(f"balanced rows: {len(balanced):,}")
print(f"positive rows: {(balanced['occurrence'] == True).sum():,}")  # noqa: E712
print(f"negative rows: {(balanced['occurrence'] == False).sum():,}")  # noqa: E712
print(f"unreported candidates: {len(unreported):,}")
print(f"balanced parquet: {BALANCED_PARQUET_PATH}")
print(f"balanced geoparquet: {BALANCED_GEOPARQUET_PATH}")
print(f"unreported parquet: {UNREPORTED_PARQUET_PATH}")
print(f"unreported geoparquet: {UNREPORTED_GEOPARQUET_PATH}")"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Validation and Sampling Diagnostics"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """display(pd.read_csv(VALIDATION_MASTER_PATH))
display(pd.read_csv(VALIDATION_BALANCING_PATH))
display(pd.read_csv(VALIDATION_UNREPORTED_PATH))
display(sampling_diagnostics.head(20))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Negative Sampling Quality

`negative_sampling_level` indicates how much we had to relax the hard-negative matching criteria:
- `1`: same road class, elevation band, and shoreline-connectivity band
- `2`: same elevation band and shoreline-connectivity band
- `3`: same borough, tide polygon, and precipitation polygon only

Note:
- FEMA is still included in the final enriched balanced table.
- It is not used in the candidate-pool strata because the fast all-segment sampler works from precomputed street-level layers where FEMA broad class is not cached for every segment."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """sampling_level_distribution = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'negative_sampling_level_distribution.csv')
distribution_checks = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'positive_negative_distribution_checks.csv')
numeric_distribution_checks = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'positive_negative_numeric_distribution_checks.csv')
leakage_checks = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'leakage_checks.csv')

display(sampling_level_distribution)
display(leakage_checks)
display(distribution_checks.head(40))
display(numeric_distribution_checks.head(40))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Class Balance and Distribution Checks"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

balanced['dataset_split_role'].value_counts().plot.bar(ax=axes[0, 0], color=['#2563EB', '#DC2626'])
axes[0, 0].set_title('Balanced Class Counts')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('events')

borough_plot = balanced.groupby(['dataset_split_role', 'segment_borough']).size().unstack(fill_value=0).T
borough_plot.plot.bar(ax=axes[0, 1])
axes[0, 1].set_title('Events by Borough and Class')
axes[0, 1].set_xlabel('borough')
axes[0, 1].set_ylabel('events')

balanced.groupby('dataset_split_role')['dem_mean'].plot.hist(
    ax=axes[1, 0],
    bins=30,
    alpha=0.65,
    legend=True,
)
axes[1, 0].set_title('Elevation Distribution')
axes[1, 0].set_xlabel('mean elevation')

balanced.groupby('dataset_split_role')['shore_graph_steps'].plot.hist(
    ax=axes[1, 1],
    bins=30,
    alpha=0.65,
    legend=True,
)
axes[1, 1].set_title('Shore Connectivity Proxy Distribution')
axes[1, 1].set_xlabel('graph steps')

plt.show()"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Possible Unreported Flood Candidates

These segments are **not** confirmed positives.
They are nearby, environmentally similar segments that may plausibly have flooded without a 311 report.
They are excluded from the negative sampling pool and should be treated as a separate sensitivity-analysis layer."""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """unreported_confidence = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'unreported_confidence_distribution.csv')
unreported_borough = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'unreported_by_borough.csv')
unreported_tide = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'unreported_by_tide_polygon.csv')
unreported_precip = pd.read_csv(BALANCED_DIAGNOSTICS_DIR / 'unreported_by_precip_polygon.csv')

display(unreported_confidence)
display(unreported_borough.head(20))
display(unreported_tide.head(20))
display(unreported_precip.head(20))
display(unreported.head(20))"""
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            """## Spatial QA/QC"""
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            """plot_frame = gpd.read_parquet(BALANCED_GEOPARQUET_PATH).to_crs(2263).copy()
plot_frame['occurrence_label'] = plot_frame['occurrence'].map({True: 'reported positive', False: 'matched negative'})

fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

plot_frame.plot(ax=axes[0], column='occurrence_label', categorical=True, legend=True, linewidth=1.0, cmap='Set1')
axes[0].set_title('Balanced Events by Label')
axes[0].set_axis_off()

if len(unreported):
    unreported_plot = gpd.read_parquet(UNREPORTED_GEOPARQUET_PATH).to_crs(2263)
    plot_frame[plot_frame['occurrence']].plot(ax=axes[1], color='#2563EB', linewidth=0.8, alpha=0.45)
    unreported_plot.plot(ax=axes[1], color='#DC2626', linewidth=1.0, alpha=0.85)
    axes[1].set_title('Possible Unreported Flood Candidates')
    axes[1].set_axis_off()
else:
    axes[1].text(0.5, 0.5, 'No unreported candidates generated', ha='center', va='center')
    axes[1].set_axis_off()

plt.show()"""
        )
    )

    nb["cells"] = cells
    text = nbf.writes(nb)
    NOTEBOOK_PATH.write_text(text)
    ALT_NOTEBOOK_PATH.write_text(text)


if __name__ == "__main__":
    main()
